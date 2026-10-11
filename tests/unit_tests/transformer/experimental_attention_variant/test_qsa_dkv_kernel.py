# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""QSA dk/dv backward: KV-tile kernel vs the original per-block kernel.

The ``"tile"`` implementation (one program per ``tile_blocks`` consecutive selection
blocks, iterating the union of their queries with a per-entry block bitmask) must
reproduce the ``"block"`` implementation within bf16 tolerance, be no less accurate than it
against an fp32 dense reference, and be bitwise repeatable. Shapes cover GQA group sizes
that are not powers of two, packed THD documents with ragged and empty blocks, block counts
that are not a multiple of the tile, a context-parallel style query shard (TQ < TK), and
the split-column (fp32 partial) path.
"""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.ops import triton_qsa as tq

BS = tq.BS


def _rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _make_case(doc_lens, hq, h, d, budget_blocks, dtype, seed=0, q_rows=None):
    """Packed documents; every query selects its own block plus a random subset of at
    most ``budget_blocks`` visible complete blocks of its document (global block ids)."""
    g = torch.Generator().manual_seed(seed)
    s = sum(doc_lens)
    bases, ends, rows = [], [], []
    tok0 = blk0 = 0
    for sd in doc_lens:
        nb_d = sd // BS + 1
        b_d = [tok0 + BS * j for j in range(nb_d)]
        bases += b_d
        ends += [min(x + BS, tok0 + sd) for x in b_d]
        for t in range(sd):
            m_t = (t + 1) // BS
            k = min(budget_blocks, m_t)
            picked = torch.randperm(m_t, generator=g)[:k].tolist() if m_t else []
            rows.append([blk0 + m_t] + [blk0 + j for j in picked])
        tok0 += sd
        blk0 += nb_d
    S = 1 + budget_blocks
    idx = torch.zeros(1, s, S, dtype=torch.long)
    counts = torch.zeros(1, s, dtype=torch.long)
    for t, r in enumerate(rows):
        idx[0, t, : len(r)] = torch.tensor(r)
        # garbage beyond the count must be ignored
        idx[0, t, len(r):] = torch.randint(0, blk0, (S - len(r),), generator=g)
        counts[0, t] = len(r)
    q_pos = torch.arange(s)
    if q_rows is not None:  # CP-style shard: local query rows against the full K/V
        idx, counts, q_pos = idx[:, q_rows], counts[:, q_rows], q_pos[q_rows]
    tq_ = q_pos.numel()
    q = torch.randn(1, tq_, hq, d, generator=g).to(dtype)
    k = torch.randn(1, s, h, d, generator=g).to(dtype)
    v = torch.randn(1, s, h, d, generator=g).to(dtype)
    do = torch.randn(1, tq_, hq, d, generator=g).to(dtype)
    dev = "cuda"
    return dict(
        idx=idx.to(dev),
        counts=counts.to(dev),
        bases=torch.tensor(bases, dtype=torch.int32, device=dev),
        ends=torch.tensor(ends, dtype=torch.int32, device=dev),
        q_pos=q_pos.to(dev),
        q=q.to(dev),
        k=k.to(dev),
        v=v.to(dev),
        do=do.to(dev),
    )


def _grads(c, dkv_impl, cfg=None, static_shapes=True):
    q, k, v = (c[n].clone().requires_grad_(True) for n in ("q", "k", "v"))
    out = tq.qsa_sparse_attention(
        q,
        k,
        v,
        block_indices=c["idx"],
        block_counts=c["counts"],
        block_bases=c["bases"],
        block_ends=c["ends"],
        q_positions=c["q_pos"],
        static_shapes=static_shapes,
        dkv_impl=dkv_impl,
        dkv_tile_config=cfg,
    )
    out.backward(c["do"])
    return out.detach(), q.grad, k.grad, v.grad


def _dense_reference(c):
    """fp32 dense masked attention dk/dv."""
    q, k, v, do = c["q"].float(), c["k"].float(), c["v"].float(), c["do"].float()
    B, TQ, HQ, D = q.shape
    TK, H = k.shape[1], k.shape[2]
    G = HQ // H
    bases, ends = c["bases"].long(), c["ends"].long()
    NB = bases.numel()
    pos = bases[:, None] + torch.arange(BS, device=q.device)[None, :]
    valid = pos < ends[:, None]
    tok_blk = torch.full((TK,), NB, dtype=torch.long, device=q.device)
    tok_blk[pos[valid]] = torch.arange(NB, device=q.device)[:, None].expand(NB, BS)[valid]
    S = c["idx"].shape[-1]
    slot_valid = torch.arange(S, device=q.device) < c["counts"].unsqueeze(-1)
    sel = torch.zeros(B, TQ, NB + 1, dtype=torch.bool, device=q.device)
    sel.scatter_(2, torch.where(slot_valid, c["idx"], NB), True)
    sel[..., NB] = False
    mask = sel[:, :, tok_blk] & (
        torch.arange(TK, device=q.device)[None, None, :] <= c["q_pos"][None, :, None]
    )
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    sc = torch.einsum("bthd,bshd->bhts", q, k.repeat_interleave(G, 2)) * D**-0.5
    p = torch.softmax(sc.masked_fill(~mask[:, None], float("-inf")), -1).nan_to_num(0.0)
    torch.einsum("bhts,bshd->bthd", p, v.repeat_interleave(G, 2)).backward(do)
    return k.grad, v.grad


CASES = {
    # name: (doc_lens, hq, h, d, budget_blocks, q_rows)
    "proxy_like_g12": ([1024], 24, 2, 256, 64, None),
    "g1": ([300], 2, 2, 64, 16, None),
    "g3_hd128": ([333], 6, 2, 128, 16, None),
    "thd_ragged": ([3, 101, 0, 216, 1, 2, 57], 8, 2, 64, 8, None),
    "cp_shard": ([512], 8, 2, 64, 16, list(range(0, 128)) + list(range(384, 512))),
}
CONFIGS = {
    "default": None,
    "trans": tq.QSADkvTileConfig(trans=True),
    "kb8_split2": tq.QSADkvTileConfig(tile_blocks=8, nsplit=2),
    "kb16_split4_cq32": tq.QSADkvTileConfig(tile_blocks=16, cq=32, nsplit=4, num_warps=8),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestQSADkvTileKernel:
    def test_tile_csr_matches_brute_force(self):
        c = _make_case([3, 101, 0, 216, 1, 2, 57], 4, 2, 32, 8, torch.bfloat16, seed=5)
        nb = c["bases"].numel()
        idx, counts = c["idx"].cpu(), c["counts"].cpu()
        for kb in (1, 4, 5, 16):
            csr = tq.build_qsa_kv_tile_csr(c["idx"], c["counts"], nb, kb)
            nt = (nb + kb - 1) // kb
            assert csr.n_tiles == nt
            want = [dict() for _ in range(nt)]
            for t in range(idx.shape[1]):
                for j in idx[0, t, : counts[0, t]].tolist():
                    want[j // kb][t] = want[j // kb].get(t, 0) | (1 << (j % kb))
            off = csr.tile_offsets.cpu()
            eq, eb = csr.entry_query.cpu(), csr.entry_bits.cpu()
            for i in range(nt):
                got_q = eq[off[i] : off[i + 1]].tolist()
                assert got_q == sorted(want[i]), f"kb={kb} tile {i} queries"
                assert eb[off[i] : off[i + 1]].tolist() == [want[i][t] for t in got_q]

    @pytest.mark.parametrize("case", list(CASES))
    @pytest.mark.parametrize("cfg_name", list(CONFIGS))
    def test_tile_matches_block_bf16(self, case, cfg_name):
        doc_lens, hq, h, d, budget, q_rows = CASES[case]
        c = _make_case(doc_lens, hq, h, d, budget, torch.bfloat16, q_rows=q_rows)
        cfg = CONFIGS[cfg_name]
        o_b, dq_b, dk_b, dv_b = _grads(c, "block")
        o_t, dq_t, dk_t, dv_t = _grads(c, "tile", cfg)
        # forward and dq do not depend on the dkv implementation
        assert torch.equal(o_b, o_t)
        assert torch.equal(dq_b, dq_t)
        assert _rel_l2(dk_t, dk_b) <= 1e-3 and _rel_l2(dv_t, dv_b) <= 1e-3, (
            f"dk {_rel_l2(dk_t, dk_b):.2e} dv {_rel_l2(dv_t, dv_b):.2e}"
        )
        # no less accurate than the original kernel against fp32
        rdk, rdv = _dense_reference(c)
        for name, got, base, ref in (("dk", dk_t, dk_b, rdk), ("dv", dv_t, dv_b, rdv)):
            assert _rel_l2(got, ref) <= 1.25 * _rel_l2(base, ref) + 1e-4, name

    @pytest.mark.parametrize("case", ["g3_hd128", "thd_ragged", "cp_shard"])
    def test_tile_matches_reference_fp32(self, case):
        doc_lens, hq, h, d, budget, q_rows = CASES[case]
        c = _make_case(doc_lens, hq, h, d, budget, torch.float32, q_rows=q_rows)
        _, _, dk_b, dv_b = _grads(c, "block")
        _, _, dk_t, dv_t = _grads(c, "tile")
        rdk, rdv = _dense_reference(c)
        for name, got, base, ref in (("dk", dk_t, dk_b, rdk), ("dv", dv_t, dv_b, rdv)):
            assert _rel_l2(got, ref) <= 1.25 * _rel_l2(base, ref) + 1e-4, name

    @pytest.mark.parametrize("cfg_name", list(CONFIGS))
    def test_bitwise_repeatable(self, cfg_name):
        c = _make_case([600, 424], 24, 2, 256, 32, torch.bfloat16, seed=3)
        cfg = CONFIGS[cfg_name]
        first = _grads(c, "tile", cfg)
        for _ in range(2):
            again = _grads(c, "tile", cfg)
            for a, b in zip(first, again):
                assert torch.equal(a, b)

    def test_dynamic_and_static_builders_agree(self):
        c = _make_case([3, 101, 0, 216, 1, 2, 57], 8, 2, 64, 8, torch.bfloat16)
        a = _grads(c, "tile", static_shapes=False)
        b = _grads(c, "tile", static_shapes=True)
        for x, y in zip(a, b):
            assert torch.equal(x, y)

    def test_cuda_graph_capture_replays_eager(self):
        c = _make_case([700, 324], 24, 2, 256, 32, torch.bfloat16, seed=7)
        q, k, v = (c[n].clone().requires_grad_(True) for n in ("q", "k", "v"))

        def step():
            for t in (q, k, v):
                t.grad = None
            out = tq.qsa_sparse_attention(
                q,
                k,
                v,
                block_indices=c["idx"],
                block_counts=c["counts"],
                block_bases=c["bases"],
                block_ends=c["ends"],
                q_positions=c["q_pos"],
                static_shapes=True,
                dkv_impl="tile",
            )
            out.backward(c["do"])
            return k.grad, v.grad

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            eager = [t.clone() for t in step()]
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gk, gv = step()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(gk, eager[0]) and torch.equal(gv, eager[1])

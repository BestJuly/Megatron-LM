# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA sparse GQA attention through cuDNN Frontend's QSA training kernels.

:func:`cudnn_qsa_sparse_attention` takes exactly the selection that
:func:`~.cute_qsa_attn.qsa_sparse_attention` takes and runs

* the forward with ``cudnn.ops.qsa_forward`` (cudnn-frontend PR #1765, "Miles" packed contract:
  a ``[T, NB]`` block bitmap plus per-query token bounds and per-block token spans), and
* the backward with one of two implementations (``qsa_cudnn_bwd_impl``):

  - ``qsa_api``: ``cudnn.ops.qsa_backward`` (PR #1765), on the same bitmap and the forward's
    fp32 output.
  - ``sparse_gqa``: ``cudnn.sdpa.bwd.sparse_gqa_bwd_wrapper_sm100``, the index-list QSA backward
    (``topk_idxs [T, 2051]``: 512 selected 4-token blocks and the ragged causal tail of one
    sequence that starts at token 0). The bitmap selection is converted to index lists here,
    and the kernel runs once per sequence (batch row or packed document).

The Megatron selection is document-local blocks of 4 tokens, laid out per document as
``sd // 4 + 1`` attention blocks (the last one possibly empty) whose global ids are offset by
``block_offsets``. That is the Miles block numbering when ``blk_base`` is the document's block
offset and ``tok_base`` its first token, so the bitmap is a scatter of the selected ids. Bounds
are ``lo = tok_base`` and ``hi = position`` (causal, inclusive). A slot whose block starts after
its query (the own block of a query that ends a block, or the empty block past a document) is
dropped from the bitmap: it has no visible token, and Miles spans must be non-empty.

Batches (BSHD, ``B > 1``) are packed into one ``T = B * S`` sequence space with every batch row
its own sequence, so a ``[T, NB]`` bitmap grows as ``B**2``; Qwen4 trains with micro-batch 1.
The bitmap and cuDNN's compact-index scratch are ``O(T * NB)``: at ``T = 131072`` the cuDNN
scratch alone is 16 GiB.

A cuDNN Frontend with the index-list entry points (``cudnn.ops.qsa_forward_ids`` /
``qsa_backward_ids``) is used when present: the selection goes in as the per-query ascending
block lists that ``qsa_forward`` compacts its bitmap into (bitwise the same results, ``O(T * K)``
memory), and the queries may be a context-parallel rank's local tokens against the gathered
keys (``Tq != Tk``, ``qsa_api`` backward).
"""

from typing import List, NamedTuple, Optional, Sequence, Tuple

import torch

QSA_CUDNN_HEAD_DIM = 256
QSA_CUDNN_BLOCK_SIZE = 4
# (query heads, KV heads) per rank served by cudnn.ops.qsa_forward / qsa_backward.
QSA_CUDNN_HEAD_GEOMETRIES = ((24, 2), (12, 1))
# The sparse_gqa backward: GQA ratios, its fixed 512-block selection and the index-list width.
SPARSE_GQA_RATIOS = (12, 16)
SPARSE_GQA_TOPK_BLOCKS = 512
SPARSE_GQA_WIDTH = SPARSE_GQA_TOPK_BLOCKS * QSA_CUDNN_BLOCK_SIZE + QSA_CUDNN_BLOCK_SIZE - 1
QSA_CUDNN_BWD_IMPLS = ("qsa_api", "sparse_gqa")


def cudnn_qsa_support_error(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    block_size: int,
    num_selected_blocks: int,
    bwd_impl: str,
    device_capability: Optional[Tuple[int, int]],
) -> Optional[str]:
    """Why the cuDNN QSA backend cannot serve this per-rank geometry, or None if it can."""
    if bwd_impl not in QSA_CUDNN_BWD_IMPLS:
        return f"qsa_cudnn_bwd_impl must be one of {QSA_CUDNN_BWD_IMPLS}, got {bwd_impl!r}"
    if device_capability not in ((10, 0), (10, 3), (10, 7)):
        return (
            "the cuDNN QSA kernels need an SM100, SM103 or SM107 GPU, got "
            f"{'no CUDA device' if device_capability is None else device_capability}"
        )
    if dtype != torch.bfloat16:
        return f"the cuDNN QSA kernels are bf16 only, got {dtype}"
    if head_dim != QSA_CUDNN_HEAD_DIM:
        return f"the cuDNN QSA kernels need head dim {QSA_CUDNN_HEAD_DIM}, got {head_dim}"
    if block_size != QSA_CUDNN_BLOCK_SIZE:
        return f"the cuDNN QSA kernels need 4-token blocks, got compress ratio {block_size}"
    if (num_q_heads, num_kv_heads) not in QSA_CUDNN_HEAD_GEOMETRIES:
        return (
            "cudnn.ops.qsa_forward serves 24 query / 2 KV heads or 12 / 1 per rank, got "
            f"{num_q_heads} / {num_kv_heads}"
        )
    if bwd_impl == "sparse_gqa":
        group = num_q_heads // num_kv_heads
        if group not in SPARSE_GQA_RATIOS or num_kv_heads % 2:
            return (
                f"the sparse_gqa backward needs a GQA ratio in {SPARSE_GQA_RATIOS} and an even "
                f"number of KV heads per rank, got {num_q_heads} / {num_kv_heads}; use "
                "qsa_cudnn_bwd_impl='qsa_api'"
            )
        if num_selected_blocks != SPARSE_GQA_TOPK_BLOCKS:
            return (
                f"the sparse_gqa backward selects exactly {SPARSE_GQA_TOPK_BLOCKS} blocks, got "
                f"qsa_indexer_budget / qsa_indexer_compress_ratio = {num_selected_blocks}; use "
                "qsa_cudnn_bwd_impl='qsa_api'"
            )
    return None


class QSAMilesMetadata(NamedTuple):
    """The cudnn.ops.qsa_forward / qsa_backward selection and bounds (Miles contract)."""

    sel_blocks: torch.Tensor  # [T, NB] uint8
    lo: torch.Tensor  # [T] int32
    hi: torch.Tensor  # [T] int32
    blk_base: torch.Tensor  # [T] int32
    tok_base: torch.Tensor  # [T] int32
    block_first: torch.Tensor  # [NB] int32
    block_last: torch.Tensor  # [NB] int32


def build_miles_metadata(
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    block_bases: torch.Tensor,
    block_ends: torch.Tensor,
    block_offsets: torch.Tensor,
    token_offsets: Optional[torch.Tensor],
) -> QSAMilesMetadata:
    """Translate a QSA selection to the packed Miles metadata. Device-only (no host sync).

    Args:
        block_indices: ``[B, S, K]`` per-row global block ids (slot 0 the own block).
        block_counts: ``[B, S]`` valid leading slots.
        block_bases / block_ends: ``[NBr]`` first token and one-past-last token of each block,
            in row-local token coordinates.
        block_offsets: ``[S]`` id of the first block of each query's document.
        token_offsets: ``[S]`` first token of each query's document (None for BSHD: 0).
    """
    B, S, K = block_indices.shape
    device = block_indices.device
    nbr = block_bases.shape[0]
    T, NB = B * S, B * nbr
    row = torch.arange(B, device=device, dtype=torch.int64)
    pos = torch.arange(S, device=device, dtype=torch.int64)
    tok0 = (
        torch.zeros(S, dtype=torch.int64, device=device)
        if token_offsets is None
        else token_offsets.to(torch.int64)
    )
    # Pack the rows: row b's tokens start at b * S and its blocks at b * nbr.
    hi = (row[:, None] * S + pos[None, :]).reshape(T)
    tok_base = (row[:, None] * S + tok0[None, :]).reshape(T)
    blk_base = (row[:, None] * nbr + block_offsets.to(torch.int64)[None, :]).reshape(T)
    bases = block_bases.to(torch.int64)
    block_first = (row[:, None] * S + bases[None, :]).reshape(NB)
    block_last = (row[:, None] * S + block_ends.to(torch.int64)[None, :] - 1).reshape(NB)

    ids = block_indices.to(torch.int64)
    slot = torch.arange(K, device=device)
    # A slot counts if it is a valid leading slot and its block starts at or before the query
    # (else none of its tokens is visible: the own block of a query that ends a block).
    valid = (slot[None, None, :] < block_counts[..., None]) & (
        bases.index_select(0, ids.clamp(0, nbr - 1).reshape(-1)).view(B, S, K) <= pos[None, :, None]
    )
    gid = (ids + row[:, None, None] * nbr).reshape(T, K)
    sel = torch.zeros(T, NB + 1, dtype=torch.uint8, device=device)
    # Invalid slots write into a discarded spill column.
    sel.scatter_(1, torch.where(valid.reshape(T, K), gid, NB), 1)
    sel = sel[:, :NB].contiguous()
    i32 = lambda t: t.to(torch.int32).contiguous()  # noqa: E731
    return QSAMilesMetadata(
        sel_blocks=sel,
        lo=i32(tok_base),
        hi=i32(hi),
        blk_base=i32(blk_base),
        tok_base=i32(tok_base),
        block_first=i32(block_first),
        block_last=i32(block_last),
    )


class QSAIndexLists(NamedTuple):
    """Selection for ``cudnn.ops.qsa_forward_ids`` / ``qsa_backward_ids``: per-query block id lists."""

    block_ids: torch.Tensor  # [Tq, K] int32, the first block_counts[t] entries valid, ascending
    block_counts: torch.Tensor  # [Tq] int32
    lo: torch.Tensor  # [Tq] int32, first visible key token (key-token space)
    hi: torch.Tensor  # [Tq] int32, query position (key-token space)
    block_first: torch.Tensor  # [NB] int32
    block_last: torch.Tensor  # [NB] int32


def build_index_lists(
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    block_bases: torch.Tensor,
    block_ends: torch.Tensor,
    key_len: int,
    q_positions: Optional[torch.Tensor],
    token_offsets: Optional[torch.Tensor],
) -> QSAIndexLists:
    """Translate a QSA selection to per-query block id lists. Device-only, ``O(Tq * K)`` memory.

    The lists are exactly what ``cudnn.ops.qsa_forward`` compacts its ``[T, NB]`` bitmap into
    (ascending ids whose token span meets the query's ``[lo, hi]``), so the results match the
    bitmap entry point bitwise, without its ``O(T * NB)`` bitmap and compaction scratch.

    Args:
        block_indices: ``[B, S, K]`` global block ids (in the key blocks of each row).
        block_counts: ``[B, S]`` valid leading slots.
        block_bases / block_ends: ``[NBr]`` first / one-past-last key token of each block.
        key_len: key tokens per row (``S`` without context parallelism, the gathered length with).
        q_positions: ``[S]`` key-token position of each query row (None: ``arange(S)``).
        token_offsets: ``[S]`` first key token of each query's document (None: 0).
    """
    B, S, K = block_indices.shape
    device = block_indices.device
    nbr = block_bases.shape[0]
    Tq, NB = B * S, B * nbr
    row = torch.arange(B, device=device, dtype=torch.int64)
    pos = (
        torch.arange(S, device=device, dtype=torch.int64)
        if q_positions is None
        else q_positions.to(torch.int64)
    )
    tok0 = (
        torch.zeros(S, dtype=torch.int64, device=device)
        if token_offsets is None
        else token_offsets.to(torch.int64)
    )
    hi = (row[:, None] * key_len + pos[None, :]).reshape(Tq)
    lo = (row[:, None] * key_len + tok0[None, :]).reshape(Tq)
    bases = block_bases.to(torch.int64)
    ends = block_ends.to(torch.int64)
    block_first = (row[:, None] * key_len + bases[None, :]).reshape(NB)
    block_last = (row[:, None] * key_len + ends[None, :] - 1).reshape(NB)
    ids = block_indices.to(torch.int64).clamp(0, nbr - 1)
    slot = torch.arange(K, device=device)
    # The bitmap path's slot validity (a leading slot whose block starts at or before the query)
    # and the compaction's span test (the block's tokens meet [lo, hi]).
    b0 = bases.index_select(0, ids.reshape(-1)).view(B, S, K)
    b1 = ends.index_select(0, ids.reshape(-1)).view(B, S, K) - 1
    keep = (
        (slot[None, None, :] < block_counts[..., None])
        & (b0 <= pos[None, :, None])
        & (b1 >= tok0[None, :, None])
        & (tok0 <= pos)[None, :, None]
    )
    gid = (ids + row[:, None, None] * nbr).reshape(Tq, K)
    keep = keep.reshape(Tq, K)
    # Ascending ids first (each query selects a block at most once), invalid slots pushed last.
    # int32 keys (NB < 2**31, checked by the kernels): the int64 sort costs ~2x at 128K tokens.
    key = torch.where(keep, gid, NB).to(torch.int32)
    block_ids = torch.sort(key, dim=1).values
    counts = keep.sum(1)
    i32 = lambda t: t.to(torch.int32).contiguous()  # noqa: E731
    return QSAIndexLists(
        block_ids=torch.where(block_ids < NB, block_ids, 0).contiguous(),
        block_counts=i32(counts),
        lo=i32(lo),
        hi=i32(hi),
        block_first=i32(block_first),
        block_last=i32(block_last),
    )


def has_index_list_kernels() -> bool:
    """Whether the installed cuDNN Frontend has the index-list (and Tq != Tk) QSA entry points."""
    try:
        import cudnn.ops as cudnn_ops

        cudnn_ops.qsa_forward_ids, cudnn_ops.qsa_backward_ids  # pylint: disable=pointless-statement
        return True
    except (ImportError, AttributeError):
        return False


def build_sparse_gqa_index_lists(
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    block_offsets: torch.Tensor,
    token_offsets: Optional[torch.Tensor],
) -> torch.Tensor:
    """``[B * S, 2051]`` int32 index lists, in the token coordinates of each query's sequence.

    The layout of ``cudnn.sdpa.bwd.sparse_gqa_sm100``: the 4 tokens of every selected complete
    block, then the ``(t + 1) % 4`` tokens of the ragged tail, then -1. The own block (slot 0)
    is the tail: a complete own block is selected through the top-k like any other.
    """
    B, S, K = block_indices.shape
    device = block_indices.device
    r = QSA_CUDNN_BLOCK_SIZE
    tok0 = (
        torch.zeros(S, dtype=torch.int64, device=device)
        if token_offsets is None
        else token_offsets.to(torch.int64)
    )
    t_local = torch.arange(S, device=device, dtype=torch.int64) - tok0  # document-local position
    # Document-local ids of the top-k slots (slot 0, the own block, is handled as the tail).
    local = block_indices[..., 1:].to(torch.int64) - block_offsets.to(torch.int64)[None, :, None]
    nsel = (block_counts.to(torch.int64) - 1).clamp(min=0)  # [B, S]
    k = K - 1
    slot_ok = torch.arange(k, device=device)[None, None, :] < nsel[..., None]
    tokens = local[..., None] * r + torch.arange(r, device=device)
    tokens = torch.where(slot_ok[..., None], tokens, -1).reshape(B * S, k * r)
    out = torch.full((B * S, SPARSE_GQA_WIDTH + 1), -1, dtype=torch.int64, device=device)
    out[:, : k * r] = tokens
    j = torch.arange(r - 1, device=device)
    ntail = ((t_local + 1) % r).repeat(B)  # [B * S]
    in_tail = j[None, :] < ntail[:, None]
    tail_slot = torch.where(in_tail, nsel.reshape(-1, 1) * r + j, SPARSE_GQA_WIDTH)
    tail_tok = torch.where(in_tail, ((t_local + 1) // r * r).repeat(B)[:, None] + j, -1)
    out.scatter_(1, tail_slot, tail_tok)
    return out[:, :SPARSE_GQA_WIDTH].to(torch.int32).contiguous()


def sequence_spans(
    batch: int, seq_len: int, cu_seqlens: Optional[torch.Tensor]
) -> List[Tuple[int, int]]:
    """``(first token, length)`` of every sequence in the packed ``[B * S]`` token space.

    Packed THD needs the document boundaries on the host (one sync), so the ``sparse_gqa``
    backward cannot be captured into a CUDA graph with THD inputs.
    """
    if cu_seqlens is None:
        return [(b * seq_len, seq_len) for b in range(batch)]
    bounds = cu_seqlens.tolist()
    return [(s0, s1 - s0) for s0, s1 in zip(bounds[:-1], bounds[1:]) if s1 > s0]


def _sparse_gqa_backward(q, k, v, out, dout, lse_t, topk_idxs, spans, scale):
    from cudnn.sdpa.bwd import sparse_gqa_bwd_wrapper_sm100

    # Tokens outside every sequence (static THD padding) take no gradient.
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    for s0, n in spans:
        rows = slice(s0, s0 + n)
        sparse_gqa_bwd_wrapper_sm100(
            q[rows],
            k[rows],
            v[rows],
            out[rows],
            dout[rows],
            lse_t[rows],
            topk_idxs[rows],
            dq=dq[rows],
            dk=dk[rows],
            dv=dv[rows],
            softmax_scale=scale,
        )
    return dq, dk, dv


class CudnnQSAAttnFunction(torch.autograd.Function):
    """cuDNN FE QSA forward; ``qsa_api`` or ``sparse_gqa`` backward. Packed ``[T, H, D]``."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        meta: QSAMilesMetadata,
        scale: float,
        bwd_impl: str,
        index_list_args: Optional[tuple],
        spans: Optional[Sequence[Tuple[int, int]]],
    ):
        """Run ``cudnn.ops.qsa_forward`` and save what the chosen backward consumes."""
        from cudnn.ops import qsa_forward

        out, out_fp32, lse = qsa_forward(q, k, v, *meta, scale=scale, block_size=4)
        ctx.scale = scale
        ctx.bwd_impl = bwd_impl
        if bwd_impl == "qsa_api":
            ctx.save_for_backward(q, k, v, out_fp32, lse, *meta)
        else:
            # The index-list backward consumes the bf16 output and a [T, H] LSE.
            ctx.index_list_args = index_list_args
            ctx.spans = spans
            ctx.save_for_backward(q, k, v, out, lse.t().contiguous())
        return out

    @staticmethod
    def backward(ctx, dout):
        """dq, dk, dv through the configured cuDNN QSA backward."""
        dout = dout.contiguous()
        if ctx.bwd_impl == "qsa_api":
            from cudnn.ops import qsa_backward

            q, k, v, out_fp32, lse, *meta = ctx.saved_tensors
            dq, dk, dv = qsa_backward(
                dout, q, k, v, out_fp32, lse, *meta, scale=ctx.scale, block_size=4
            )
        else:
            q, k, v, out, lse_t = ctx.saved_tensors
            topk_idxs = build_sparse_gqa_index_lists(*ctx.index_list_args)
            dq, dk, dv = _sparse_gqa_backward(
                q, k, v, out, dout, lse_t, topk_idxs, ctx.spans, ctx.scale
            )
        return dq, dk, dv, None, None, None, None, None


class CudnnQSAIdsAttnFunction(torch.autograd.Function):
    """cuDNN FE QSA through the index-list entry points (no ``[T, NB]`` bitmap); ``Tq`` queries
    against ``Tk`` keys. Backward: ``qsa_backward_ids`` or, without context parallelism,
    the ``sparse_gqa`` backward."""

    @staticmethod
    def forward(ctx, q, k, v, lists: QSAIndexLists, scale, bwd_impl, index_list_args, spans):
        """Run ``cudnn.ops.qsa_forward_ids`` and save what the chosen backward consumes."""
        from cudnn.ops import qsa_forward_ids

        out, out_fp32, lse = qsa_forward_ids(q, k, v, *lists, scale=scale, block_size=4)
        ctx.scale = scale
        ctx.bwd_impl = bwd_impl
        if bwd_impl == "qsa_api":
            ctx.save_for_backward(q, k, v, out_fp32, lse, *lists)
        else:
            ctx.index_list_args = index_list_args
            ctx.spans = spans
            ctx.save_for_backward(q, k, v, out, lse.t().contiguous())
        return out

    @staticmethod
    def backward(ctx, dout):
        """dq, dk, dv through the configured cuDNN QSA backward."""
        dout = dout.contiguous()
        if ctx.bwd_impl == "qsa_api":
            from cudnn.ops import qsa_backward_ids

            q, k, v, out_fp32, lse, *lists = ctx.saved_tensors
            dq, dk, dv = qsa_backward_ids(
                dout, q, k, v, out_fp32, lse, *lists, scale=ctx.scale, block_size=4
            )
        else:
            q, k, v, out, lse_t = ctx.saved_tensors
            topk_idxs = build_sparse_gqa_index_lists(*ctx.index_list_args)
            dq, dk, dv = _sparse_gqa_backward(
                q, k, v, out, dout, lse_t, topk_idxs, ctx.spans, ctx.scale
            )
        return dq, dk, dv, None, None, None, None, None


def cudnn_qsa_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    block_bases: torch.Tensor,
    block_ends: torch.Tensor,
    block_offsets: torch.Tensor,
    token_offsets: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    bwd_impl: str = "sparse_gqa",
    q_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sparse GQA attention over per-query selected 4-token blocks, on cuDNN Frontend.

    Takes the selection of :func:`~.cute_qsa_attn.qsa_sparse_attention` (``q`` ``[B, S, HQ, D]``,
    ``k``/``v`` ``[B, S, H, D]``, global ``block_indices`` ``[B, S, K]`` with slot 0 the own
    block, ``block_counts`` ``[B, S]``, per-block token ranges) plus each query's document
    block offset and first token. ``cu_seqlens`` (packed THD) is only read by the ``sparse_gqa``
    backward. Returns ``[B, S, HQ, D]``.

    With a cuDNN Frontend that has the index-list entry points (``qsa_forward_ids``), the
    selection is passed as per-query block lists (``O(T * K)`` memory instead of the bitmap's
    ``O(T * NB)``) and ``k``/``v`` may be the gathered keys of context parallelism
    (``[B, Sk, H, D]`` with ``q_positions`` the queries' key-token positions; ``qsa_api``
    backward only).
    """
    B, S, HQ, D = q.shape
    Sk = k.shape[1]
    use_ids = has_index_list_kernels()
    if Sk != S or q_positions is not None:
        assert use_ids, (
            "context-parallel cuDNN QSA needs a cuDNN Frontend with cudnn.ops.qsa_forward_ids"
        )
        assert bwd_impl == "qsa_api", (
            "context-parallel cuDNN QSA (queries against gathered keys) needs "
            "qsa_cudnn_bwd_impl='qsa_api': the sparse_gqa backward runs per sequence."
        )
    if scale is None:
        scale = D**-0.5
    if (
        bwd_impl == "sparse_gqa"
        and cu_seqlens is not None
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(
            "qsa_cudnn_bwd_impl='sparse_gqa' runs one backward per packed document and needs the "
            "document boundaries on the host, so it cannot be captured into a CUDA graph with "
            "packed (THD) inputs. Use qsa_cudnn_bwd_impl='qsa_api'."
        )
    index_list_args = spans = None
    if bwd_impl == "sparse_gqa":
        index_list_args = (block_indices, block_counts, block_offsets, token_offsets)
        spans = sequence_spans(B, S, cu_seqlens)
    packed = lambda t: t.reshape(-1, *t.shape[2:]).contiguous()  # noqa: E731
    if use_ids:
        fn = CudnnQSAIdsAttnFunction
        sel = build_index_lists(
            block_indices, block_counts, block_bases, block_ends, Sk, q_positions, token_offsets
        )
    else:
        fn = CudnnQSAAttnFunction
        sel = build_miles_metadata(
            block_indices, block_counts, block_bases, block_ends, block_offsets, token_offsets
        )
    out = fn.apply(
        packed(q), packed(k), packed(v), sel, float(scale), bwd_impl, index_list_args, spans
    )
    return out.view(B, S, HQ, out.shape[-1])
    assert Sk == S and q_positions is None, (
        "context-parallel cuDNN QSA needs a cuDNN Frontend with cudnn.ops.qsa_forward_ids"
    )
    if scale is None:
        scale = D**-0.5
    if (
        bwd_impl == "sparse_gqa"
        and cu_seqlens is not None
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(
            "qsa_cudnn_bwd_impl='sparse_gqa' runs one backward per packed document and needs the "
            "document boundaries on the host, so it cannot be captured into a CUDA graph with "
            "packed (THD) inputs. Use qsa_cudnn_bwd_impl='qsa_api'."
        )
    meta = build_miles_metadata(
        block_indices, block_counts, block_bases, block_ends, block_offsets, token_offsets
    )
    index_list_args = spans = None
    if bwd_impl == "sparse_gqa":
        index_list_args = (block_indices, block_counts, block_offsets, token_offsets)
        spans = sequence_spans(B, S, cu_seqlens)
    packed = lambda t: t.reshape(B * S, *t.shape[2:]).contiguous()  # noqa: E731
    out = CudnnQSAAttnFunction.apply(
        packed(q), packed(k), packed(v), meta, float(scale), bwd_impl, index_list_args, spans
    )
    return out.view(B, S, HQ, out.shape[-1])

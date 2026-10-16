# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Portions of this file (the tcgen05 / TMEM / mbarrier-pipeline structure of the warp-specialized
# kernels, the cp.async row-gather into swizzled shared memory, and the TMEM load/store idioms)
# are adapted from flash-attention's CuTe DSL kernels (flash_attn/cute: flash_fwd_sm100.py,
# sm100_hd256_2cta_fmha_forward.py, topk_gather_kv.py), which carry the following notice:
#
#   Copyright (c) 2025, Tri Dao, Ted Zadouri and the flash-attention contributors.
#   BSD 3-Clause License. Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#   1. Redistributions of source code must retain the above copyright notice, this list of
#      conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright notice, this list of
#      conditions and the following disclaimer in the documentation and/or other materials
#      provided with the distribution.
#   3. Neither the name of the copyright holder nor the names of its contributors may be used
#      to endorse or promote products derived from this software without specific prior
#      written permission.
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
#   OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
#   BE LIABLE FOR ANY DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE.

"""CuTe DSL (SM100: tcgen05 MMA, tensor memory, warp specialization) kernels for the QSA sparse
GQA attention. Torch-free: the host wrappers / autograd live in :mod:`cute_qsa_attn`.

The kernels consume the sparse-attention metadata built in :mod:`triton_qsa`: the query-tile
shared superset (``tile_offsets`` / ``entry_base`` / ``entry_end`` / ``membership``) for the
forward and dq, and the KV-tile query CSR for dk/dv.

Forward (:class:`QSAFwdSm100`), one CTA per (query tile, KV head):

* rows: the flattened ``q_loc * G + g`` (query, group head) pairs of ``BQ = 128 // G`` queries
  (G = 12 -> 120 of 128 MMA rows used, no power-of-two head padding);
* columns: the tile's union entries, ``N / 4`` entries (``N`` gathered KV rows) per step;
* warp roles: 4 softmax warps (one TMEM lane / S row per thread; online softmax, O rescale,
  epilogue), 1 MMA warp (S = Q K^T and O += P V on tcgen05, S/P double-buffered in TMEM, O
  resident in TMEM), 4 loader warps (cp.async row gather of Q / K / V into the swizzled UMMA
  layouts, plus a per-step (query, column) mask word built with warp ballots);
* the LSE is stored in the log2 domain like the Triton kernels.
"""

import math

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Boolean, Float32, Int32, Int64, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05

LOG2E = 1.4426950408889634


def _gather_copy(dtype, num_threads: int, head_dim: int):
    """cp.async (16 B) tiled copy: ``num_threads`` threads, 8 threads per 128 B row segment."""
    elems = 128 // dtype.width  # 8 bf16 per 16 B
    blk = math.gcd(head_dim, 128 // (dtype.width // 8))  # 64 bf16 columns per row segment
    tpr = blk // elems
    atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), dtype, num_bits_per_copy=128
    )
    thr_layout = cute.make_ordered_layout((num_threads // tpr, tpr), order=(1, 0))
    val_layout = cute.make_layout((1, elems))
    return cute.make_tiled_copy_tv(atom, thr_layout, val_layout), tpr, elems


class QSAFwdSm100:
    """Sparse GQA forward over the query-tile shared superset (see module docstring)."""

    def __init__(self, head_dim: int, group: int, tile_queries: int, n_block: int = 64,
                 kv_stage: int = 0):
        assert head_dim in (64, 128, 256)
        assert 1 <= group and tile_queries * group <= 128
        assert n_block in (64, 128)
        self.D = head_dim
        self.G = group
        self.BQ = tile_queries
        self.M = 128
        self.N = n_block
        self.EPS = n_block // 4  # union entries per step (4-token blocks)
        # K / V share one ring of (N x D) slots: 128 KB of shared memory
        self.kv_stage = kv_stage or min(8, (128 * 1024) // (n_block * head_dim * 2))
        self.s_stage = 2
        self.mask_stage = self.kv_stage
        self.mask_words = n_block // 32
        self.softmax_warps = (0, 1, 2, 3)
        self.mma_warp = 4
        self.load_warps = (8, 9, 10, 11)
        self.num_threads = 32 * 12
        self.num_load_threads = 128
        self.tmem_s_off = (0, n_block)
        self.tmem_p_off = (n_block // 2, n_block + n_block // 2)
        self.tmem_o_off = 2 * n_block
        self.tmem_cols = 512
        assert self.tmem_o_off + head_dim <= self.tmem_cols
        # setmaxnreg budget (sum <= 3 x 168 per thread slot): chosen so ptxas does not spill
        self.regs_softmax = 200
        self.regs_load = 152
        self.regs_other = 152

    # ------------------------------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # [B, TQ, HQ, D]
        mK: cute.Tensor,  # [B, TK, H, D]
        mV: cute.Tensor,  # [B, TK, H, D]
        mO: cute.Tensor,  # [B, TQ, HQ, D]
        mLSE: cute.Tensor,  # [B, TQ, HQ] fp32 (log2 domain)
        mQpos: cute.Tensor,  # [TQ] int64
        mTileOff: cute.Tensor,  # [B * n_tiles + 1] int64
        mEBase: cute.Tensor,  # [E] int32
        mEEnd: cute.Tensor,  # [E] int32
        mMem: cute.Tensor,  # [E, BQ] uint8
        scale_log2: Float32,
        n_tiles: Int32,
        stream,
    ):
        self.dtype = mQ.element_type
        dt = self.dtype
        M, N, D = self.M, self.N, self.D
        K_MAJ = tcgen05.OperandMajorMode.K
        MN_MAJ = tcgen05.OperandMajorMode.MN
        one = tcgen05.CtaGroup.ONE
        tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(dt, dt, K_MAJ, K_MAJ, Float32, one, (M, N))
        tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
            dt, dt, K_MAJ, MN_MAJ, Float32, one, (M, D), tcgen05.OperandSource.TMEM
        )
        self.mma_tiler_qk = (M, N, D)
        self.mma_tiler_pv = (M, D, N)
        sQ_layout = sm100_utils.make_smem_layout_a(tiled_mma_qk, self.mma_tiler_qk, dt, 1)
        sK_layout = sm100_utils.make_smem_layout_b(tiled_mma_qk, self.mma_tiler_qk, dt, self.kv_stage)
        sV_layout = sm100_utils.make_smem_layout_b(tiled_mma_pv, self.mma_tiler_pv, dt, self.kv_stage)
        tP_layout = sm100_utils.make_smem_layout_a(tiled_mma_pv, self.mma_tiler_pv, dt, self.s_stage)

        sKV_size = cutlass.max(cute.cosize(sK_layout), cute.cosize(sV_layout))
        mask_size = self.mask_stage * 128 * self.mask_words

        @cute.struct
        class SharedStorage:
            mbar_q: cute.struct.MemRange[Int64, 2]
            mbar_kv: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_mask: cute.struct.MemRange[Int64, self.mask_stage * 2]
            mbar_s: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_p: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_o: cute.struct.MemRange[Int64, 2]
            tmem_holding_buf: Int32
            sQpos: cute.struct.MemRange[Int64, 128]
            sMask: cute.struct.MemRange[Uint32, mask_size]
            sQ: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sQ_layout)], 1024]
            sKV: cute.struct.Align[cute.struct.MemRange[dt, sKV_size], 1024]

        self.shared_storage = SharedStorage

        grid = (mTileOff.shape[0] - 1, mK.shape[2], 1)
        self.kernel(
            mQ, mK, mV, mO, mLSE, mQpos, mTileOff, mEBase, mEEnd, mMem,
            scale_log2, n_tiles,
            tiled_mma_qk, tiled_mma_pv, sQ_layout, sK_layout, sV_layout, tP_layout,
        ).launch(grid=grid, block=[self.num_threads, 1, 1], stream=stream, min_blocks_per_mp=1)

    # ------------------------------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mQpos: cute.Tensor,
        mTileOff: cute.Tensor,
        mEBase: cute.Tensor,
        mEEnd: cute.Tensor,
        mMem: cute.Tensor,
        scale_log2: Float32,
        n_tiles: Int32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
    ):
        M, N, D, G, BQ, EPS = self.M, self.N, self.D, self.G, self.BQ, self.EPS
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bx, h_kv, _ = cute.arch.block_idx()
        n_bt = cute.arch.grid_dim()[0]
        # Longest-processing-time first: later query tiles have the largest unions.
        bt = n_bt - 1 - bx
        b = bt // n_tiles
        tile = bt - b * n_tiles
        TQ = mQ.shape[1]
        TK = mK.shape[1]
        HQ = mQ.shape[2]
        H = mK.shape[2]
        e0 = Int32(mTileOff[bt])
        e1 = Int32(mTileOff[bt + 1])
        n_steps = (e1 - e0 + EPS - 1) // EPS

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        Thr = pipeline.CooperativeGroup
        Agent = pipeline.Agent
        mma_grp = Thr(Agent.Thread, 1)
        load_grp = Thr(Agent.Thread, self.num_load_threads)
        sm_grp = Thr(Agent.Thread, 128)
        pipe_q = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_q.data_ptr(), num_stages=1,
            producer_group=load_grp, consumer_group=mma_grp, defer_sync=True,
        )
        pipe_kv = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_kv.data_ptr(), num_stages=self.kv_stage,
            producer_group=load_grp, consumer_group=mma_grp, defer_sync=True,
        )
        pipe_mask = pipeline.PipelineAsync.create(
            barrier_storage=storage.mbar_mask.data_ptr(), num_stages=self.mask_stage,
            producer_group=load_grp, consumer_group=sm_grp, defer_sync=True,
        )
        pipe_s = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_s.data_ptr(), num_stages=self.s_stage,
            producer_group=mma_grp, consumer_group=sm_grp, defer_sync=True,
        )
        pipe_p = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_p.data_ptr(), num_stages=self.s_stage,
            producer_group=sm_grp, consumer_group=mma_grp, defer_sync=True,
        )
        pipe_o = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_o.data_ptr(), num_stages=1,
            producer_group=mma_grp, consumer_group=sm_grp, defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.num_threads)
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_bar,
            allocator_warp_id=self.mma_warp,
        )
        tmem.allocate(self.tmem_cols)
        tmem.wait_for_alloc()  # also a CTA-wide sync after the mbarrier init
        tmem_ptr = tmem.retrieve_ptr(Float32)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sKV.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        sMask = storage.sMask.get_tensor(cute.make_layout((self.mask_words, 128, self.mask_stage)))
        sQpos = storage.sQpos.get_tensor(cute.make_layout((128,)))

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)
        tSrQ = thr_mma_qk.make_fragment_A(sQ)
        tSrK = thr_mma_qk.make_fragment_B(sK)
        tOrV = thr_mma_pv.make_fragment_B(sV)
        qk_acc_shape = thr_mma_qk.partition_shape_C((M, N))
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        tStS = cute.make_tensor(tmem_ptr + self.tmem_s_off[0], tStS.layout)
        pv_acc_shape = thr_mma_pv.partition_shape_C((M, D))
        tOtO = thr_mma_pv.make_fragment_C(pv_acc_shape)
        tOtO = cute.make_tensor(tmem_ptr + self.tmem_o_off, tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP0 = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        p_ratio = Float32.width // self.dtype.width
        p_stage_stride = (self.tmem_p_off[1] - self.tmem_p_off[0]) * p_ratio
        tOrP = cute.make_tensor(
            tOrP0.iterator + self.tmem_p_off[0] * p_ratio,
            cute.append(tOrP0.layout, cute.make_layout((self.s_stage,), stride=(p_stage_stride,))),
        )

        # (computed before the role branches: the MMA warp mutates the tiled MMA's fields)
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor((M, N)))[(None, None), 0, 0]

        Prod = pipeline.PipelineUserType.Producer
        Cons = pipeline.PipelineUserType.Consumer

        # ======================================================================================
        # Loader warps: Q / K / V row gather (cp.async) + per-step mask words
        # ======================================================================================
        if warp_idx >= self.load_warps[0]:
            cute.arch.warpgroup_reg_dealloc(self.regs_load)
            ltid = tidx - self.load_warps[0] * 32
            lwarp = ltid // 32
            lane = ltid % 32
            gcopy, tpr, celems = _gather_copy(self.dtype, self.num_load_threads, D)
            gthr = gcopy.get_slice(ltid)
            rows_per_pass = self.num_load_threads // tpr
            col0 = (ltid % tpr) * celems

            # query positions of the tile (shared by every step's causal test)
            if ltid < BQ:
                tq = tile * BQ + ltid
                qp = Int64(-1)
                if tq < TQ:
                    qp = Int64(mQpos[tq])
                sQpos[ltid] = qp
            load_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=self.num_load_threads)
            load_bar.arrive_and_wait()

            # ---- Q: rows r = q_loc * G + g ----
            q_prod = pipeline.make_pipeline_state(Prod, 1)
            kv_prod = pipeline.make_pipeline_state(Prod, self.kv_stage)
            m_prod = pipeline.make_pipeline_state(Prod, self.mask_stage)
            if n_steps > 0:
                pipe_q.producer_acquire(q_prod)
                sQ_nd = cute.composition(
                    sQ[None, None, None, 0], cute.make_ordered_layout((M, D), order=(0, 1))
                )
                tQsQ = gthr.partition_D(sQ_nd)
                for m in cutlass.range_constexpr(M // rows_per_pass):
                    r = ltid // tpr + m * rows_per_pass
                    q_loc = r // G
                    g = r - q_loc * G
                    tq = tile * BQ + q_loc
                    ok = (r < BQ * G) and (tq < TQ)
                    off = Int64(0)
                    if ok:
                        off = ((Int64(b) * TQ + tq) * HQ + h_kv * G + g) * D
                    self._copy_row(gcopy, mQ, off, col0, tQsQ, m)
                cute.arch.cp_async_commit_group()
                pipe_q.sync_object_full.arrive_cp_async_mbarrier(q_prod.index)

                rows = cute.make_rmem_tensor((N // rows_per_pass,), Int64)
                rows_prev = cute.make_rmem_tensor((N // rows_per_pass,), Int64)
                for i in cutlass.range(n_steps, unroll=1):
                    ebase = e0 + i * EPS
                    # -- mask words for step i (warps 0..mask_words-1 of the loader group) --
                    pipe_mask.producer_acquire(m_prod)
                    if lwarp < self.mask_words:
                        c = ltid  # column of the step
                        e = ebase + c // 4
                        tok = Int32(0)
                        ok_tok = False
                        if e < e1:
                            tok = mEBase[e] + c % 4
                            ok_tok = tok < mEEnd[e]
                        for qi in cutlass.range(BQ, unroll=1):
                            bit = False
                            if ok_tok:
                                bit = (mMem[e, qi] != 0) and (Int64(tok) <= sQpos[qi])
                            word = cute.arch.vote_ballot_sync(bit)
                            if lane == 0:
                                sMask[lwarp, qi, m_prod.index] = Uint32(word)
                    pipe_mask.producer_commit(m_prod)
                    m_prod.advance()
                    # -- K rows of step i --
                    for m in cutlass.range_constexpr(N // rows_per_pass):
                        rows_prev[m] = rows[m]
                        r = ltid // tpr + m * rows_per_pass
                        e = ebase + r // 4
                        tok = Int32(0)
                        if e < e1:
                            t0 = mEBase[e] + r % 4
                            if t0 < mEEnd[e]:
                                tok = t0
                        rows[m] = ((Int64(b) * TK + tok) * H + h_kv) * D
                    kv_prod = self._load_kv(gthr, gcopy, mK, sK, rows, col0, pipe_kv, kv_prod,
                                            transpose=False)
                    if i > 0:
                        kv_prod = self._load_kv(gthr, gcopy, mV, sV, rows_prev, col0, pipe_kv,
                                                kv_prod, transpose=True)
                kv_prod = self._load_kv(gthr, gcopy, mV, sV, rows, col0, pipe_kv, kv_prod,
                                        transpose=True)

        # ======================================================================================
        # MMA warp
        # ======================================================================================
        if warp_idx > self.mma_warp and warp_idx < self.load_warps[0]:
            # idle warps of the MMA warpgroup (setmaxnreg is warpgroup-uniform)
            cute.arch.warpgroup_reg_dealloc(self.regs_other)
        if warp_idx == self.mma_warp:
            cute.arch.warpgroup_reg_dealloc(self.regs_other)
            q_cons = pipeline.make_pipeline_state(Cons, 1)
            kv_cons = pipeline.make_pipeline_state(Cons, self.kv_stage)
            s_prod = pipeline.make_pipeline_state(Prod, self.s_stage)
            p_cons = pipeline.make_pipeline_state(Cons, self.s_stage)
            o_prod = pipeline.make_pipeline_state(Prod, 1)
            if n_steps > 0:
                pipe_q.consumer_wait(q_cons)
                tSrQ0 = tSrQ[None, None, None, 0]
                for i in cutlass.range(n_steps, unroll=1):
                    pipe_s.producer_acquire(s_prod)
                    pipe_kv.consumer_wait(kv_cons)
                    tiled_mma_qk = self._gemm(tiled_mma_qk, tStS[None, None, None, s_prod.index],
                                              tSrQ0, tSrK[None, None, None, kv_cons.index], True)
                    pipe_kv.consumer_release(kv_cons)
                    kv_cons.advance()
                    pipe_s.producer_commit(s_prod)
                    s_prod.advance()
                    if i > 0:
                        kv_cons, p_cons, o_prod, tiled_mma_pv = self._pv(
                            tiled_mma_pv, tOtO, tOrP, tOrV, pipe_kv, pipe_p, pipe_o,
                            kv_cons, p_cons, o_prod, i == 1)
                kv_cons, p_cons, o_prod, tiled_mma_pv = self._pv(
                    tiled_mma_pv, tOtO, tOrP, tOrV, pipe_kv, pipe_p, pipe_o,
                    kv_cons, p_cons, o_prod, n_steps == 1)

        # ======================================================================================
        # Softmax warps: online softmax, O rescale, epilogue
        # ======================================================================================
        if warp_idx < self.mma_warp:
            cute.arch.warpgroup_reg_alloc(self.regs_softmax)
            r = tidx  # TMEM lane == S/O row
            q_loc = r // G
            g = r - q_loc * G
            tq = tile * BQ + q_loc
            row_ok = (r < BQ * G) and (tq < TQ)
            q_loc_c = cutlass.min(q_loc, BQ - 1)

            s_cons = pipeline.make_pipeline_state(Cons, self.s_stage)
            m_cons = pipeline.make_pipeline_state(Cons, self.mask_stage)
            p_prod = pipeline.make_pipeline_state(Prod, self.s_stage)
            o_cons = pipeline.make_pipeline_state(Cons, 1)

            # TMEM <-> register copies (one row per thread)
            tStS0 = tStS[(None, None), 0, 0, 0]
            ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), Float32)
            tiled_ld_s = tcgen05.make_tmem_copy(ld_atom, tStS0)
            thr_ld_s = tiled_ld_s.get_slice(tidx)
            tSc = thr_ld_s.partition_D(tScS)
            rS = cute.make_rmem_tensor(tSc.shape, Float32)

            P_COLS = N * self.dtype.width // 32
            tStP_layout = cute.composition(tStS0.layout, cute.make_layout((M, P_COLS)))
            st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(P_COLS)), Float32)
            tStP0 = cute.make_tensor(tStS0.iterator + (self.tmem_p_off[0] - self.tmem_s_off[0]),
                                     tStP_layout)
            tiled_st_p = tcgen05.make_tmem_copy(st_atom, tStP0)
            thr_st_p = tiled_st_p.get_slice(tidx)
            tScP = cute.composition(tScS, cute.make_layout((M, P_COLS)))
            tPcP = thr_st_p.partition_S(tScP)
            rP = cute.make_rmem_tensor(tSc.shape, self.dtype)
            rP32 = cute.make_tensor(cute.recast_ptr(rP.iterator, dtype=Float32), tPcP.shape)

            # O in 32-column chunks
            OCH = 32
            tOtO0 = tOtO[(None, None), 0, 0]
            tOch_layout = cute.composition(tOtO0.layout, cute.make_layout((M, OCH)))
            tOch0 = cute.make_tensor(tOtO0.iterator, tOch_layout)
            ldo_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(OCH)), Float32)
            sto_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(OCH)), Float32)
            tiled_ld_o = tcgen05.make_tmem_copy(ldo_atom, tOch0)
            tiled_st_o = tcgen05.make_tmem_copy(sto_atom, tOch0)
            thr_ld_o = tiled_ld_o.get_slice(tidx)
            thr_st_o = tiled_st_o.get_slice(tidx)
            cO = cute.make_identity_tensor((M, OCH))
            tOcO = thr_ld_o.partition_D(cO)
            rO = cute.make_rmem_tensor(tOcO.shape, Float32)

            row_max = -Float32.inf
            row_sum = Float32(0.0)
            for i in cutlass.range(n_steps, unroll=1):
                # ---- S_i -> registers ----
                pipe_s.consumer_wait(s_cons)
                tSt = thr_ld_s.partition_S(tStS[(None, None), 0, 0, s_cons.index])
                cute.copy(tiled_ld_s, tSt, rS)
                cute.arch.fence_view_async_tmem_load()
                pipe_s.consumer_release(s_cons)
                s_cons.advance()
                # ---- mask ----
                pipe_mask.consumer_wait(m_cons)
                words = cute.make_rmem_tensor((self.mask_words,), Uint32)
                for w in cutlass.range_constexpr(self.mask_words):
                    words[w] = sMask[w, q_loc_c, m_cons.index]
                pipe_mask.consumer_release(m_cons)
                m_cons.advance()
                # ---- masked scaled scores, running max ----
                m_new = row_max
                for j in cutlass.range_constexpr(N):
                    bit = (words[j // 32] >> Uint32(j % 32)) & Uint32(1)
                    sv = rS[j] * scale_log2
                    if bit == Uint32(0):
                        sv = -Float32.inf
                    rS[j] = sv
                    m_new = cute.arch.fmax(m_new, sv)
                m_safe = m_new
                if m_new == -Float32.inf:
                    m_safe = Float32(0.0)
                alpha = cute.math.exp2(row_max - m_safe, fastmath=True)
                psum = Float32(0.0)
                for j in cutlass.range_constexpr(N):
                    pj = cute.math.exp2(rS[j] - m_safe, fastmath=True)
                    rS[j] = pj
                    psum += pj
                row_sum = row_sum * alpha + psum
                row_max = m_new
                rP.store(rS.load().to(self.dtype))
                # ---- P_i -> TMEM ----
                pipe_p.producer_acquire(p_prod)
                tPt = thr_st_p.partition_D(
                    cute.make_tensor(tStP0.iterator + (self.tmem_p_off[1] - self.tmem_p_off[0])
                                     * p_prod.index, tStP_layout))
                cute.copy(tiled_st_p, rP32, tPt)
                cute.arch.fence_view_async_tmem_store()
                # ---- rescale O (after PV(i-1) completed) ----
                if i > 0:
                    pipe_o.consumer_wait(o_cons)
                    need = cute.arch.vote_any_sync(alpha < Float32(1.0))
                    if need:
                        for ch in cutlass.range_constexpr(D // OCH):
                            tO_ch = cute.make_tensor(tOch0.iterator + ch * OCH, tOch_layout)
                            cute.copy(tiled_ld_o, thr_ld_o.partition_S(tO_ch), rO)
                            for j in cutlass.range_constexpr(cute.size(rO)):
                                rO[j] = rO[j] * alpha
                            cute.copy(tiled_st_o, rO, thr_st_o.partition_D(tO_ch))
                        cute.arch.fence_view_async_tmem_store()
                    pipe_o.consumer_release(o_cons)
                    o_cons.advance()
                pipe_p.producer_commit(p_prod)
                p_prod.advance()

            # ---- epilogue ----
            inv = Float32(0.0)
            if row_sum > Float32(0.0):
                inv = Float32(1.0) / row_sum
            lse = Float32(0.0)
            if row_max != -Float32.inf:
                lse = row_max + cute.math.log2(cutlass.max(row_sum, Float32(1e-10)), fastmath=False)
            o_off = Int64(0)
            if row_ok:
                o_off = ((Int64(b) * TQ + tq) * HQ + h_kv * G + g) * D
                mLSE[b, tq, h_kv * G + g] = lse
            if n_steps > 0:
                pipe_o.consumer_wait(o_cons)
            rOb = cute.make_rmem_tensor(tOcO.shape, self.dtype)
            for ch in cutlass.range_constexpr(D // OCH):
                if n_steps > 0:
                    tO_ch = cute.make_tensor(tOch0.iterator + ch * OCH, tOch_layout)
                    cute.copy(tiled_ld_o, thr_ld_o.partition_S(tO_ch), rO)
                else:
                    rO.fill(0.0)
                for j in cutlass.range_constexpr(cute.size(rO)):
                    rO[j] = rO[j] * inv
                rOb.store(rO.load().to(self.dtype))
                if row_ok:
                    gO = cute.make_tensor(
                        cute.make_ptr(self.dtype, (mO.iterator + o_off + ch * OCH).toint(),
                                      cute.AddressSpace.gmem, assumed_align=16),
                        cute.make_layout((OCH,)),
                    )
                    cute.autovec_copy(rOb, gO)
            if n_steps > 0:
                cute.arch.fence_view_async_tmem_load()
                pipe_o.consumer_release(o_cons)

        # ======================================================================================
        tmem_bar.arrive_and_wait()
        if warp_idx == self.mma_warp:
            tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    # ------------------------------------------------------------------------------------------
    @cute.jit
    def _gemm(self, tiled_mma, acc, tA, tB, zero_init):
        for k in cutlass.range_constexpr(cute.size(tA, mode=[2])):
            if const_expr(k == 0):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, not zero_init)
            else:
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(tiled_mma, acc, tA[None, None, k], tB[None, None, k], acc)
        return tiled_mma

    @cute.jit
    def _pv(self, tiled_mma_pv, tOtO, tOrP, tOrV, pipe_kv, pipe_p, pipe_o, kv_cons, p_cons,
            o_prod, first):
        pipe_p.consumer_wait(p_cons)
        pipe_o.producer_acquire(o_prod)
        pipe_kv.consumer_wait(kv_cons)
        tiled_mma_pv = self._gemm(tiled_mma_pv, tOtO, tOrP[None, None, None, p_cons.index],
                                  tOrV[None, None, None, kv_cons.index], first)
        pipe_kv.consumer_release(kv_cons)
        kv_cons.advance()
        pipe_o.producer_commit(o_prod)
        o_prod.advance()
        pipe_p.consumer_release(p_cons)
        p_cons.advance()
        return kv_cons, p_cons, o_prod, tiled_mma_pv

    @cute.jit
    def _copy_row(self, gcopy, mX, off, col0, tXsX, m):
        """cp.async one gathered row segment set: row base element offset ``off``."""
        for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
            src = cute.make_tensor(
                cute.make_ptr(mX.element_type, (mX.iterator + off + col0 + k * 64).toint(),
                              cute.AddressSpace.gmem, assumed_align=16),
                tXsX[None, m, k].layout,
            )
            cute.copy(gcopy, src, tXsX[None, m, k])

    @cute.jit
    def _load_kv(self, gthr, gcopy, mX, sX, rows, col0, pipe_kv, kv_prod, transpose):
        pipe_kv.producer_acquire(kv_prod)
        order = (1, 0) if const_expr(transpose) else (0, 1)
        sX_nd = cute.composition(
            sX[None, None, None, kv_prod.index],
            cute.make_ordered_layout((self.N, self.D), order=order),
        )
        tXsX = gthr.partition_D(sX_nd)
        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            self._copy_row(gcopy, mX, rows[m], col0, tXsX, m)
        cute.arch.cp_async_commit_group()
        pipe_kv.sync_object_full.arrive_cp_async_mbarrier(kv_prod.index)
        kv_prod.advance()
        return kv_prod

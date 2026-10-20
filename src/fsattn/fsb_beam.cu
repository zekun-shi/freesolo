// >>> SECTION: _HELPERS <<<

typedef long long i64;
#define FULL 0xffffffffu
__device__ __forceinline__ float f_add(float a, float b) { float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_sub(float a, float b) { float r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_mul(float a, float b) { float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_fma_ftz(float a, float b, float c) {
    float r; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float f_div_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_sqrt_ftz(float a) { float r; asm("sqrt.rn.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_abs_ftz(float a) { float r; asm("abs.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_ex2_ftz(float a) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_bits(unsigned u) { return __int_as_float((int)u); }
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }

// tl.math.erf: libdevice __nv_erff as Triton links it (nvvm-reflect ftz), instruction for instruction
__device__ __forceinline__ float erf_tl(float x) {
    const float a = f_abs_ftz(x);
    const bool big = a >= f_bits(0x3F8060FEu);
    const float u = big ? a : f_mul(x, x);
    float r = f_fma_ftz(big ? f_bits(0x3789CA3Cu) : f_bits(0x38B1E96Au), u, big ? f_bits(0xB9F560B9u) : f_bits(0xBA574D20u));
    r = f_fma_ftz(r, u, big ? f_bits(0x3BAC840Bu) : f_bits(0x3BAAD5EAu));
    r = f_fma_ftz(r, u, big ? f_bits(0xBD0C8162u) : f_bits(0xBCDC1BE7u));
    r = f_fma_ftz(r, u, big ? f_bits(0x3E1CF906u) : f_bits(0x3DE718AFu));
    r = f_fma_ftz(r, u, big ? f_bits(0x3F6A937Eu) : f_bits(0xBEC093ACu));
    r = f_fma_ftz(r, u, big ? f_bits(0x3F20D842u) : f_bits(0x3E0375D3u));
    const float s = big ? -u : x;
    r = f_fma_ftz(r, s, s);
    if (big) {
        r = f_sub(1.0f, f_ex2_ftz(r));
        r = __int_as_float((__float_as_int(x) & (int)0x80000000) | __float_as_int(r));
    }
    return r;
}
// >>> SECTION: _KERNEL <<<

#define PER (DHP / 32)
#define NEG_INF __int_as_float(0xff800000)
__device__ float g_zero[DHP + 16];                // node-sized zeros: the address idle child slots read

// the forest's node tables are split: means in a bf16 slab (each stored mean is the value every beam agrees on; the
// upcast is a 16-bit shift), the 9 aux fields in fp32. Spine rows keep the packed fp32 layout (NC floats) and are
// always fp32 in every kernel below.
__device__ __forceinline__ const void* node_mu_at(const unsigned short* cmub, const float* sbase, const int r) {
    return r >= 0 ? (const void*)(cmub + (i64)r * DH) : (const void*)(sbase + (i64)(-r - 2) * NC);
}
__device__ __forceinline__ const float* aux_row(const float* cax, const float* sbase, const int r) {
    return r >= 0 ? cax + (i64)r * NA : sbase + (i64)(-r - 2) * NC + DH;
}

template <int H>
__device__ __forceinline__ void halve_step(float* x) {
#pragma unroll
    for (int k = 0; k < H; k++) x[k] = f_add(x[k], x[k + H]);
}

// in-lane part of _halve_rows: x[k] is column lane + 32 k; pairs (k, k + h) for h = PER/2 .. 1 leave column lane of
// the 32-wide partial sums
__device__ __forceinline__ float lane_halve(float* x) {
    if constexpr (PER >= 8) halve_step<4>(x);
    if constexpr (PER >= 4) halve_step<2>(x);
    if constexpr (PER >= 2) halve_step<1>(x);
    return x[0];
}

// one reduce-scatter round: lanes with bit o keep the upper M values, the others the lower M, each adding its
// partner's copy of the kept children (the halving step that pairs columns j and j ^ o)
template <int M>
__device__ __forceinline__ void rs_round(float* v, const int lane, const int o) {
    const bool up = (lane & o) != 0;
#pragma unroll
    for (int i = 0; i < M; i++) {
        const float send = up ? v[i] : v[i + M];
        const float keep = up ? v[i + M] : v[i];
        v[i] = f_add(keep, __shfl_xor_sync(FULL, send, o));
    }
}

__device__ __forceinline__ const void* node_at(const unsigned short* cmub, const float* sbase, const int r) {
    return node_mu_at(cmub, sbase, r);
}

// onebeam table descriptor for the paired kernels (passed by value)
struct BeamP {
    const i64* ROOT; i64* CELLS; float* NVIS; const i64* WBEAM; const i64* BUDGET; const i64* TIDX;
    const unsigned short* CHUNK; const float* CHUNK_AX; const float* SPINE; const i64* LEAF; const i64* ADM;
    i64 Ntot; i64 Sp; i64 Ltot; i64 Mm; i64 rows; int KMAXP1; int pad_;
};

// per-warp scratch of the warp-per-query descent
struct __align__(16) WarpS {
    float sc[32];
    float cnt[32], chl[32], chr[32];
    const void* ptr[32];
    const float* aptr[32];
    int cref[32];
    int nref[WM];
    float ncnt[WM], nchl[WM], nchr[WM];
};

// scores of the n compacted children (n <= CH): q.mu as a reduce-scatter over the 32 lanes (offsets 16, 8, ...: the
// remaining halving steps, in order), then the butterfly within each group of 32 / CH lanes; lane l ends with child
// l / (32 / CH), whose fields it also loads. Idle child slots read g_zero (the caller stores it in sptr[n..31], so
// the loads are unconditional) and score -inf, which no child's rank counts.
template <int CH>
__device__ __forceinline__ void score_children(const int lane, const int n, const float* qv, const float qn,
                                               const float qpos, const float lam, const float sig2, const float scale,
                                               const float cell_size, WarpS& S) {
    constexpr int G = 32 / CH;
    const int c = lane / G;
    const bool cv = c < n;
    const float* pa = S.aptr[c];
    const float rad = __ldg(pa), amax = __ldg(pa + 1), pmin = __ldg(pa + 2);
    const float cnt = __ldg(pa + 3), chl = __ldg(pa + 4), chr = __ldg(pa + 5);
    float v[CH];
    if (__all_sync(FULL, c >= n || S.cref[c] >= 0)) {
        // the common whole-chunk level: the bf16 means load in one branchless batch (issue pattern of the
        // pre-split kernel, u16 lanes), spine-free
#pragma unroll
        for (int i = 0; i < CH; i++) {
            const unsigned short* p16 = reinterpret_cast<const unsigned short*>(S.ptr[i]);
            float x[PER];
#pragma unroll
            for (int k = 0; k < PER; k++) {
                const int col = lane + 32 * k;
                x[k] = f_mul(qv[k], (col < DH) ? __uint_as_float(((unsigned)__ldg(p16 + col)) << 16) : 0.0f);
            }
            v[i] = lane_halve(x);
        }
    } else {
#pragma unroll
        for (int i = 0; i < CH; i++) {
            float x[PER];
            if (S.cref[i] >= 0) {
                const unsigned short* p16 = reinterpret_cast<const unsigned short*>(S.ptr[i]);
#pragma unroll
                for (int k = 0; k < PER; k++) {
                    const int col = lane + 32 * k;
                    x[k] = f_mul(qv[k], (col < DH) ? __uint_as_float(((unsigned)__ldg(p16 + col)) << 16) : 0.0f);
                }
            } else {
                const float* p32 = reinterpret_cast<const float*>(S.ptr[i]);
#pragma unroll
                for (int k = 0; k < PER; k++) {
                    const int col = lane + 32 * k;
                    x[k] = f_mul(qv[k], (col < DH) ? __ldg(p32 + col) : 0.0f);
                }
            }
            v[i] = lane_halve(x);
        }
    }
    rs_round<CH / 2>(v, lane, 16);
    rs_round<CH / 4>(v, lane, 8);
    rs_round<CH / 8>(v, lane, 4);
    if constexpr (CH >= 16) rs_round<CH / 16>(v, lane, 2);
    if constexpr (CH >= 32) rs_round<CH / 32>(v, lane, 1);
#pragma unroll
    for (int o = G / 2; o >= 1; o >>= 1) v[0] = f_add(v[0], __shfl_xor_sync(FULL, v[0], o));
    float content = f_mul(v[0], scale);
#if BOUND
    content = f_add(content, f_mul(f_mul(qn, rad), scale));
#endif
    const float pmin_pos = f_mul(f_add(pmin, 0.5f), cell_size);
    const float dx = __int_as_float(__float_as_int(f_sub(qpos, pmin_pos)) & 0x7fffffff);
    const float score = f_add(content, f_mul(f_mul(lam, amax), erf_tl(f_div_full(dx, sig2))));
    if ((lane % G) == 0) {
        S.sc[c] = cv ? score : NEG_INF;
        if (cv) { S.cnt[c] = cnt; S.chl[c] = chl; S.chr[c] = chr; }
    }
}

// the descent of one query by one warp from level lev0 (lane l: child slot l, frontier slot l / 2, whose node f_ref
// and its count and children the lane holds) with nvis children visited before, then its cells
__device__ __forceinline__ void warp_descend(WarpS& S, const int lane, const float* qv, const float qn, const float qpos,
                                             const float lam, const float sig2, const float bud, const i64 wbeam,
                                             int f_ref, float f_cnt, float f_chl, float f_chr, const int lev0,
                                             int nvis, const int KMAXP1, const float scale, const float cell_size,
                                             const unsigned short* cmub, const float* cax, const float* sbase,
                                             const i64* leafb,
                                             const i64* admb, const i64 Ltot, const i64 Mm, i64* out,
                                             float* nvis_out) {
    const int side = lane & 1, slot = lane >> 1;
    const bool in_w = lane < 2 * WM;
    const unsigned lt = (1u << lane) - 1u;
    for (int lev = lev0; lev < KMAXP1; lev++) {
        const bool active = in_w && f_ref != -1;
        const bool expand = active && (f_cnt > bud);
        const bool carried = active && (f_cnt <= bud);
        const unsigned car = __ballot_sync(FULL, carried && side == 0);
        const int ncar = __popc(car);
        // every active slot carried and kept: the frontier is a fixed point and no child is visited
        if (__ballot_sync(FULL, active && !carried) == 0u && ncar <= wbeam) break;
        const int child = expand ? (int)(i64)(side ? f_chr : f_chl) : -1;
        const unsigned vmask = __ballot_sync(FULL, child != -1);
        const int n = __popc(vmask);
        nvis += n;
        if (child != -1) {
            const int pos = __popc(vmask & lt);
            S.ptr[pos] = node_at(cmub, sbase, child);
            S.aptr[pos] = aux_row(cax, sbase, child);
            S.cref[pos] = child;
            prefetch_l2(S.ptr[pos]);
            prefetch_l2(reinterpret_cast<const char*>(S.ptr[pos]) + 128);
            prefetch_l2(S.aptr[pos]);
        }
        if (lane >= n) { S.ptr[lane] = g_zero; S.aptr[lane] = g_zero; }
        __syncwarp();
        if (n <= 8)
            score_children<8>(lane, n, qv, qn, qpos, lam, sig2, scale, cell_size, S);
        else if (n <= 16)
            score_children<16>(lane, n, qv, qn, qpos, lam, sig2, scale, cell_size, S);
        else
            score_children<32>(lane, n, qv, qn, qpos, lam, sig2, scale, cell_size, S);
        __syncwarp();
        // carried slots keep their order ahead of every child; children rank by (score desc, slot asc): the children
        // scoring strictly higher (idle slots hold -inf and never count) plus the equal scores (+0 == -0) at a lower
        // slot, i.e. (s_m > s_c) || (s_m == s_c && m < c) over m < n for every child that can be kept (not NaN)
        const int nr = n <= 8 ? 8 : (n <= 16 ? 16 : 32);
        const float sc = (lane < nr) ? S.sc[lane] : 0.0f;
        int crank = ncar;
        {
            int r = 0;
            const float4* s4p = reinterpret_cast<const float4*>(S.sc);
#pragma unroll 2
            for (int m4 = 0; m4 < nr / 4; m4++) {
                const float4 s4 = s4p[m4];
                r += (int)(s4.x > sc) + (int)(s4.y > sc) + (int)(s4.z > sc) + (int)(s4.w > sc);
            }
            const unsigned key = (sc == 0.0f) ? 0u : __float_as_uint(sc);
            const unsigned valid = (n >= 32) ? FULL : ((1u << n) - 1u);
            crank += r + __popc(__match_any_sync(FULL, key) & valid & lt);
        }
        const bool c_ok = lane < n && sc > -5e8f && crank < wbeam && crank < WM;
        const int ncb = __popc(car & lt);
        const bool f_ok = carried && side == 0 && ncb < wbeam && ncb < WM;
        if (lane < WM) S.nref[lane] = -1;
        __syncwarp();
        if (f_ok) { S.nref[ncb] = f_ref; S.ncnt[ncb] = f_cnt; S.nchl[ncb] = f_chl; S.nchr[ncb] = f_chr; }
        if (c_ok) {
            S.nref[crank] = S.cref[lane]; S.ncnt[crank] = S.cnt[lane]; S.nchl[crank] = S.chl[lane];
            S.nchr[crank] = S.chr[lane];
        }
        __syncwarp();
        if (in_w) { f_ref = S.nref[slot]; f_cnt = S.ncnt[slot]; f_chl = S.nchl[slot]; f_chr = S.nchr[slot]; }
        __syncwarp();
    }
    // cells epilogue: src/off/len of each frontier node, then the cell ids from the leaf (src == 1) or admission table
    if (in_w && side == 0) S.nref[slot] = f_ref;
    __syncwarp();
#pragma unroll
    for (int it = 0; it < (WM * R + 31) / 32; it++) {
        const int idx = lane + 32 * it;
        if (WM * R % 32 != 0 && idx >= WM * R) break;
        const int w = idx / R, tr = idx - w * R;
        const int ref = S.nref[w];
        i64 cell = -1;
        if (ref != -1) {
            const float* p = aux_row(cax, sbase, ref);
            const float src = p[6], off = p[7], ln = p[8];
            if (src > 0.0f && (float)tr < ln) {
                const i64 pos = (i64)off + tr;
                if (src == 1.0f) cell = leafb[min(max(pos, (i64)0), Ltot - 1)];
                else cell = admb[min(max(pos, (i64)0), Mm - 1)];
            }
        }
        out[idx] = cell;
    }
    if (lane == 0) *nvis_out = (float)nvis;
}

// the whole descent of one query by one warp, from the root
__device__ __forceinline__ void warp_query(WarpS& S, const int lane, const float* qp, const float rcq, const float qpos,
                                           const float lam, const float sig2, const float bud, const i64 wbeam,
                                           const int root, const int KMAXP1, const float scale, const float cell_size,
                                           const unsigned short* cmub, const float* cax, const float* sbase,
                                           const i64* leafb, const i64* admb,
                                           const i64 Ltot, const i64 Mm, i64* out, float* nvis_out,
                                           const int q_bf16 = 0) {
    float qv[PER], x2[PER];
#pragma unroll
    for (int k = 0; k < PER; k++) {
        const int col = lane + 32 * k;
        qv[k] = (col < DH) ? (q_bf16 ? __uint_as_float(((unsigned)reinterpret_cast<const unsigned short*>(qp)[col]) << 16)
                                     : qp[col])
                           : 0.0f;
        x2[k] = f_mul(qv[k], qv[k]);
    }
    float qs = lane_halve(x2);
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) qs = f_add(qs, __shfl_xor_sync(FULL, qs, o));
    const float qn = f_mul(f_sqrt_ftz(qs), rcq);
    int f_ref = -1;
    float f_cnt = 0.0f, f_chl = -1.0f, f_chr = -1.0f;
    if ((lane >> 1) == 0) {
        f_ref = root;
        if (f_ref != -1) {
            const float* p = aux_row(cax, sbase, f_ref);
            f_cnt = p[3]; f_chl = p[4]; f_chr = p[5];
        }
    }
    warp_descend(S, lane, qv, qn, qpos, lam, sig2, bud, wbeam, f_ref, f_cnt, f_chl, f_chr, 0, 0, KMAXP1, scale,
                 cell_size, cmub, cax, sbase, leafb, admb, Ltot, Mm, out, nvis_out);
}
// >>> SECTION: _WARP_KERNEL <<<

extern "C" __global__ void __launch_bounds__(NWARP * 32)
fsb_beam_cuda(const i64* __restrict__ ROOT, i64* __restrict__ CELLS, float* __restrict__ NVIS,
              const float* __restrict__ Q, const float* __restrict__ RC, const float rc_s,
              const float* __restrict__ QPOS, const float* __restrict__ LAM, const float* __restrict__ SIG,
              const i64* __restrict__ WBEAM, const i64* __restrict__ BUDGET, const i64* __restrict__ TIDX,
              const unsigned short* __restrict__ CHUNK, const float* __restrict__ CHUNK_AX,
              const float* __restrict__ SPINE,
              const i64* __restrict__ LEAF, const i64* __restrict__ ADM,
              const i64 Ntot, const i64 Sp, const i64 Ltot, const i64 Mm, const i64 Bq, const i64 nq,
              const float scale, const float cell_size, const int KMAXP1, const int q_bf16) {
    __shared__ WarpS ws[NWARP];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const i64 pid = (i64)blockIdx.x * NWARP + wid;
    if (pid >= nq) return;
    const i64 t = pid / Bq;
    const i64 tidx = TIDX[t];
#if RC_T
    const float rcq = RC[t];
#else
    const float rcq = rc_s;
#endif
    const float* qp_row = q_bf16 ? reinterpret_cast<const float*>(reinterpret_cast<const unsigned short*>(Q) + pid * DH)
                                 : (Q + pid * DH);
    warp_query(ws[wid], lane, qp_row, rcq, QPOS[pid], LAM[t], f_mul(2.0f, SIG[t]), (float)BUDGET[pid], WBEAM[pid],
               (int)ROOT[t], KMAXP1, scale, cell_size, CHUNK + tidx * Ntot * DH, CHUNK_AX + tidx * Ntot * NA,
               SPINE + t * Sp * NC,
               LEAF + tidx * Ltot, ADM + tidx * Mm, Ltot, Mm, CELLS + pid * (WM * R), NVIS + pid, q_bf16);
}

extern "C" __global__ void __launch_bounds__(32)
fsb_beam_pair_cuda(const BeamP B0, const BeamP B1,
                   const float* __restrict__ Q, const float* __restrict__ RC, const float rc_s,
                   const float* __restrict__ QPOS, const float* __restrict__ LAM, const float* __restrict__ SIG,
                   const i64 Bq, const i64 nq, const float scale, const float cell_size, const i64 qpos_s,
                   const int q_bf16) {
    __shared__ WarpS ws;
    const int lane = threadIdx.x & 31;
    const i64 pid = (i64)blockIdx.x;
    if (pid >= nq) return;
    const BeamP B = blockIdx.y ? B1 : B0;
    const i64 t = pid / Bq;
    const i64 tidx = B.TIDX[t];
#if RC_T
    const float rcq = RC[t];
#else
    const float rcq = rc_s;
#endif
    const float* qp_row = q_bf16 ? reinterpret_cast<const float*>(reinterpret_cast<const unsigned short*>(Q) + pid * DH)
                                 : (Q + pid * DH);
    warp_query(ws, lane, qp_row, rcq, QPOS[pid * qpos_s], LAM[t], f_mul(2.0f, SIG[t]), (float)B.BUDGET[pid],
               B.WBEAM[pid], (int)B.ROOT[t], B.KMAXP1, scale, cell_size, B.CHUNK + tidx * B.Ntot * DH,
               B.CHUNK_AX + tidx * B.Ntot * NA,
               B.SPINE + t * B.Sp * NC, B.LEAF + tidx * B.Ltot, B.ADM + tidx * B.Mm, B.Ltot, B.Mm,
               B.CELLS + pid * (WM * R), B.NVIS + pid, q_bf16);
}
// >>> SECTION: _ROWS_KERNEL <<<

// ---- row-cooperative prefill descent: one block per QB queries of a (beam, row), two lanes per query (lane h: columns
// 2 k + h)
#define NT (2 * QB)
#define NWARP_R (NT / 32)
#define SCS 36                                    // score row stride (floats): conflict-free float4 rows
#define CUS 34                                    // child-union row stride (shorts): 17 banks, conflict-free int loads
#define EMPTY_E ((int)0x80000000)
#define CAR0 (UMAX + 64)                          // carried frontier entry: CAR0 + node ref (spine refs >= -64)

__device__ float fsb_qzero[128];                  // zero rows for the padded tail of a strided stage view (NQ < Bq)

static_assert(NWARP_R * sizeof(WarpS) <= OFF_UM, "per-warp scratch of the fallback must fit the tile region");
static_assert(UMAX % 32 == 0 && UMAX <= NT, "one thread per union slot in the parent ranking");

#ifdef PROF                                       // phase cycle counters per (block, warp), ridden out on FBC + 256 (profiling builds only)
__device__ int g_fallback;
#define PMARK(k) \
    do { if (lane == 0) { const unsigned long long t_ = clock64(); pacc[k] += t_ - ptim; ptim = t_; } } while (0)
#else
#define PMARK(k) do { } while (0)
#endif

__device__ __forceinline__ int fchild(const float f) { return (int)(i64)f; }

__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src)
                 : "memory");
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src)
                 : "memory");
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }

__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

__device__ __forceinline__ void prefetch_node(const unsigned short* cmub, const float* cax, const float* sbase,
                                              const int ref) {         // the node's mu row and aux row
    if (ref != -1) {
        const char* m = reinterpret_cast<const char*>(node_mu_at(cmub, sbase, ref));
        prefetch_l2(m); prefetch_l2(m + 128);
        prefetch_l2(reinterpret_cast<const char*>(aux_row(cax, sbase, ref)));
    }
}

// r + (a > b) / r + (a >= b) as a compare and a predicated add (ordered compares: NaN never counts)
__device__ __forceinline__ int add_gt(int r, const float a, const float b) {
    asm("{\n\t.reg .pred p;\n\tsetp.gt.f32 p, %1, %2;\n\t@p add.s32 %0, %0, 1;\n\t}" : "+r"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ int add_ge(int r, const float a, const float b) {
    asm("{\n\t.reg .pred p;\n\tsetp.ge.f32 p, %1, %2;\n\t@p add.s32 %0, %0, 1;\n\t}" : "+r"(r) : "f"(a), "f"(b));
    return r;
}

// in-lane halving of x[4 m + j], where block m holds the lane's columns 2 k + h with k = 4 (m ^ rot) + j. The chunk
// steps pair blocks m and m + 4, m + 2, m + 1, i.e. chunks a and a ^ 4, a ^ 2, a ^ 1 for any rot: the chunks congruent
// mod 4, 2, 1, which are the chunk pairs of _halve_rows; then positions j <-> j + 2, j + 1. The same tree, each add
// commuted at most.
__device__ __forceinline__ float halve32(float* x) {
#pragma unroll
    for (int m = 0; m < 4; m++)
#pragma unroll
        for (int j = 0; j < 4; j++) x[4 * m + j] = f_add(x[4 * m + j], x[4 * m + 16 + j]);
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
        for (int j = 0; j < 4; j++) x[4 * m + j] = f_add(x[4 * m + j], x[4 * m + 8 + j]);
#pragma unroll
    for (int j = 0; j < 4; j++) x[j] = f_add(x[j], x[4 + j]);
    x[0] = f_add(x[0], x[2]);
    x[1] = f_add(x[1], x[3]);
    return f_add(x[0], x[1]);
}

__device__ __forceinline__ unsigned spread16(unsigned x) {     // bit i -> bit 2 i
    x = (x | (x << 8)) & 0x00FF00FFu;
    x = (x | (x << 4)) & 0x0F0F0F0Fu;
    x = (x | (x << 2)) & 0x33333333u;
    return (x | (x << 1)) & 0x55555555u;
}

__device__ __forceinline__ float4 lds128(const unsigned a) {
    float4 v;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
    return v;
}

__device__ __forceinline__ float no_nan(const float x) {
    float r;
    asm("max.f32 %0, %1, 0fFF800000;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void add_pair(int& rj, int& rk, const float sj, const float sk) {
    asm("{\n\t.reg .pred p;\n\tsetp.gt.f32 p, %2, %3;\n\t@p add.s32 %0, %0, 1;\n\t@!p add.s32 %1, %1, 1;\n\t}"
        : "+r"(rj), "+r"(rk) : "f"(sk), "f"(sj));
}

__device__ __forceinline__ float halve8(float* x) {
#pragma unroll
    for (int j = 0; j < 4; j++) x[j] = f_add(x[j], x[4 + j]);
    x[0] = f_add(x[0], x[2]);
    x[1] = f_add(x[1], x[3]);
    return f_add(x[0], x[1]);
}

// this lane's half of the dot of the query with the tile node at shared address b = node + h * 128 + rot * 16 (node
// 256-byte aligned): block m reads chunk m ^ rot at b ^ (16 m); the 8 lanes of a phase (rot = lane & 7) hit 8 distinct
// bank quads whatever their nodes
__device__ __forceinline__ float half_dot(const float* qr, const unsigned b) {
    float x[8];
#pragma unroll
    for (int m = 0; m < 2; m++) {
        const float4 v0 = lds128(b ^ (16u * m));
        const float4 v1 = lds128(b ^ (16u * (m + 4)));
        const float4 v2 = lds128(b ^ (16u * (m + 2)));
        const float4 v3 = lds128(b ^ (16u * (m + 6)));
        x[4 * m]     = f_add(f_add(f_mul(qr[4 * m], v0.x),     f_mul(qr[4 * m + 16], v1.x)),
                             f_add(f_mul(qr[4 * m + 8], v2.x), f_mul(qr[4 * m + 24], v3.x)));
        x[4 * m + 1] = f_add(f_add(f_mul(qr[4 * m + 1], v0.y), f_mul(qr[4 * m + 17], v1.y)),
                             f_add(f_mul(qr[4 * m + 9], v2.y), f_mul(qr[4 * m + 25], v3.y)));
        x[4 * m + 2] = f_add(f_add(f_mul(qr[4 * m + 2], v0.z), f_mul(qr[4 * m + 18], v1.z)),
                             f_add(f_mul(qr[4 * m + 10], v2.z), f_mul(qr[4 * m + 26], v3.z)));
        x[4 * m + 3] = f_add(f_add(f_mul(qr[4 * m + 3], v0.w), f_mul(qr[4 * m + 19], v1.w)),
                             f_add(f_mul(qr[4 * m + 11], v2.w), f_mul(qr[4 * m + 27], v3.w)));
    }
    return halve8(x);
}

__device__ __forceinline__ void half_dot2(const float* qr, const unsigned ba, const unsigned bb,
                                          float& sa, float& sb) {
    float xa[8], xb[8];
#pragma unroll
    for (int m = 0; m < 2; m++) {
        const float4 va0 = lds128(ba ^ (16u * m));
        const float4 vb0 = lds128(bb ^ (16u * m));
        const float4 va1 = lds128(ba ^ (16u * (m + 4)));
        const float4 vb1 = lds128(bb ^ (16u * (m + 4)));
        const float4 va2 = lds128(ba ^ (16u * (m + 2)));
        const float4 vb2 = lds128(bb ^ (16u * (m + 2)));
        const float4 va3 = lds128(ba ^ (16u * (m + 6)));
        const float4 vb3 = lds128(bb ^ (16u * (m + 6)));
        xa[4 * m]     = f_add(f_add(f_mul(qr[4 * m], va0.x),     f_mul(qr[4 * m + 16], va1.x)),
                              f_add(f_mul(qr[4 * m + 8], va2.x), f_mul(qr[4 * m + 24], va3.x)));
        xa[4 * m + 1] = f_add(f_add(f_mul(qr[4 * m + 1], va0.y), f_mul(qr[4 * m + 17], va1.y)),
                              f_add(f_mul(qr[4 * m + 9], va2.y), f_mul(qr[4 * m + 25], va3.y)));
        xa[4 * m + 2] = f_add(f_add(f_mul(qr[4 * m + 2], va0.z), f_mul(qr[4 * m + 18], va1.z)),
                              f_add(f_mul(qr[4 * m + 10], va2.z), f_mul(qr[4 * m + 26], va3.z)));
        xa[4 * m + 3] = f_add(f_add(f_mul(qr[4 * m + 3], va0.w), f_mul(qr[4 * m + 19], va1.w)),
                              f_add(f_mul(qr[4 * m + 11], va2.w), f_mul(qr[4 * m + 27], va3.w)));
        xb[4 * m]     = f_add(f_add(f_mul(qr[4 * m], vb0.x),     f_mul(qr[4 * m + 16], vb1.x)),
                              f_add(f_mul(qr[4 * m + 8], vb2.x), f_mul(qr[4 * m + 24], vb3.x)));
        xb[4 * m + 1] = f_add(f_add(f_mul(qr[4 * m + 1], vb0.y), f_mul(qr[4 * m + 17], vb1.y)),
                              f_add(f_mul(qr[4 * m + 9], vb2.y), f_mul(qr[4 * m + 25], vb3.y)));
        xb[4 * m + 2] = f_add(f_add(f_mul(qr[4 * m + 2], vb0.z), f_mul(qr[4 * m + 18], vb1.z)),
                              f_add(f_mul(qr[4 * m + 10], vb2.z), f_mul(qr[4 * m + 26], vb3.z)));
        xb[4 * m + 3] = f_add(f_add(f_mul(qr[4 * m + 3], vb0.w), f_mul(qr[4 * m + 19], vb1.w)),
                              f_add(f_mul(qr[4 * m + 11], vb2.w), f_mul(qr[4 * m + 27], vb3.w)));
    }
    sa = halve8(xa);
    sb = halve8(xb);
}

#if Q_BF16 && QB == 128 && CAP == 64 && UMAX == 128
__device__ __forceinline__ unsigned pack_bf16x2(float x, float y) {
    unsigned short rx, ry;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(rx) : "f"(x));
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(ry) : "f"(y));
    return (unsigned)rx | ((unsigned)ry << 16);
}
__device__ __forceinline__ float bf2f_s(unsigned short h) {
    return __uint_as_float(((unsigned)h) << 16);
}
__device__ __forceinline__ void mma16816_tc(float (&d)[4], const unsigned (&a)[4], const unsigned b0, const unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void stage_meta_only(const int nt, const int b, const int tid,
                                                float4* s_umeta, const float* cax, const float* sbase) {
    for (int u = tid; u < nt; u += NT) {
        const int ref = __float_as_int(s_umeta[b * UMAX + u].x);
        if (ref != -1) {
            const float* p = aux_row(cax, sbase, ref);
            s_umeta[b * UMAX + u] = make_float4(__int_as_float(ref), __ldg(p + 3), __ldg(p + 4), __ldg(p + 5));
        }
    }
}
__device__ __forceinline__ void stage_tile_tc2(const int u0, const int nt, const int b, const int lane, const int wid,
                                               float4* s_umeta, float4* s_um2, unsigned* s_mu_u32,
                                               const unsigned short* cmub, const float* cax, const float* sbase,
                                               const float qpos_blk, const float cell_size,
                                               const float lam, const float sig2, const float scale) {
    // chunk mu rows land by cp.async (lanes 0-7, one 16-byte piece each: the (ul & 7) << 2 swizzle permutes whole
    // 4-u32 pieces, so each piece's bytes stay contiguous), all rows of the warp's share in flight at once. The aux
    // metadata then stages one union entry per thread (the six fields' loads issue in parallel; the first loop's
    // readers see ref either way since .x is rewritten with its own value), and the caller's cp_async_wait_all +
    // __syncthreads publishes everything before the tile is read.
    for (int ul = wid; ul < nt; ul += NWARP_R) {
        const int u = u0 + ul;
        const int ref = __float_as_int(s_umeta[b * UMAX + u].x);
        if (ref != -1) {
            if (ref >= 0) {
                // a chunk row: cp.async moves the stored bf16 pairs unchanged
                if (lane < 8)
                    cp_async16(&s_mu_u32[ul * 32 + 4 * (lane ^ (ul & 7))],
                               reinterpret_cast<const unsigned*>(cmub + (i64)ref * DH) + 4 * lane);
            } else {
                const float* p = sbase + (i64)(-ref - 2) * NC;
                s_mu_u32[ul * 32 + (lane ^ ((ul & 7) << 2))] =
                    pack_bf16x2(__ldg(p + 2 * lane), __ldg(p + 2 * lane + 1));
            }
        } else {
            s_mu_u32[ul * 32 + (lane ^ ((ul & 7) << 2))] = 0u;
        }
    }
    for (int ua = wid * 32 + lane; ua < nt; ua += NT) {
        const int u = u0 + ua;
        const int ref = __float_as_int(s_umeta[b * UMAX + u].x);
        if (ref != -1) {
            const float* pa = aux_row(cax, sbase, ref);
            const float rad = __ldg(pa), amax = __ldg(pa + 1), pmin = __ldg(pa + 2);
            const float pmin_pos = f_mul(f_add(pmin, 0.5f), cell_size);
            const float dx = __int_as_float(__float_as_int(f_sub(qpos_blk, pmin_pos)) & 0x7fffffff);
            const float pos_b = f_mul(f_mul(lam, amax), erf_tl(f_div_full(dx, sig2)));
            *reinterpret_cast<float2*>(s_um2 + u) = make_float2(f_mul(rad, scale), pos_b);
            s_umeta[b * UMAX + u] = make_float4(__int_as_float(ref), __ldg(pa + 3), __ldg(pa + 4), __ldg(pa + 5));
        }
    }
    cp_async_commit();
}
template <int NB>
__device__ __forceinline__ void rw_direct_bf16(const unsigned short* q_dots, const float4* s_um2, const float qn,
                                               const short* curow, int* frow, const int n, const int h,
                                               const int ncar, const i64 wbeam, const int swz_q) {
    constexpr int H = NB / 2;
    const unsigned pmask = 3u << ((threadIdx.x) & 30);
    float my[H];
#pragma unroll
    for (int k = 0; k < H; k++) {
        const int c = 2 * k + h;
        if (c < n) {
            const int u = curow[c];
            const float2 tm = *reinterpret_cast<const float2*>(s_um2 + u);
#if BOUND
            my[k] = no_nan(f_add(f_add(bf2f_s(q_dots[u ^ swz_q]), f_mul(qn, tm.x)), tm.y));
#else
            my[k] = no_nan(f_add(bf2f_s(q_dots[u ^ swz_q]), tm.y));
#endif
        } else {
            my[k] = NEG_INF;
        }
    }
    int r[H];
#pragma unroll
    for (int k = 0; k < H; k++) {
        const float ok = __shfl_xor_sync(pmask, my[k], 1);
        r[k] = h ? (int)(ok >= my[k]) : (int)(ok > my[k]);
    }
#pragma unroll
    for (int j = 0; j < H; j++) {
        const float oj = __shfl_xor_sync(pmask, my[j], 1);
        int rj_p = 0;
#pragma unroll
        for (int k = j + 1; k < H; k++) {
            add_pair(r[j], r[k], my[j], my[k]);
            add_pair(rj_p, r[k], oj, my[k]);
        }
        r[j] += __shfl_xor_sync(pmask, rj_p, 1);
    }
#pragma unroll
    for (int k = 0; k < H; k++) {
        const int c = 2 * k + h;
        const int crank = ncar + r[k];
        if (c < n && my[k] > -5e8f && crank < wbeam && crank < WM) frow[crank] = (int)curow[c];
    }
}
template <int NB>
__device__ __forceinline__ void rw_reg(const float* my, const short* curow, int* frow, const int n, const int h,
                                       const int ncar, const i64 wbeam) {
    constexpr int H = NB / 2;
    const unsigned pmask = 3u << ((threadIdx.x) & 30);
    int r[H];
#pragma unroll
    for (int k = 0; k < H; k++) {
        const float ok = __shfl_xor_sync(pmask, my[k], 1);
        r[k] = h ? (int)(ok >= my[k]) : (int)(ok > my[k]);
    }
#pragma unroll
    for (int j = 0; j < H; j++) {
        const float oj = __shfl_xor_sync(pmask, my[j], 1);
        int rj_p = 0;
#pragma unroll
        for (int k = j + 1; k < H; k++) {
            add_pair(r[j], r[k], my[j], my[k]);
            add_pair(rj_p, r[k], oj, my[k]);
        }
        r[j] += __shfl_xor_sync(pmask, rj_p, 1);
    }
#pragma unroll
    for (int k = 0; k < H; k++) {
        const int c = 2 * k + h;
        const int crank = ncar + r[k];
        if (c < n && my[k] > -5e8f && crank < wbeam && crank < WM) frow[crank] = (int)curow[c];
    }
}
#endif

// stage union slots [u0, u0 + nt) of buffer b, one warp per node: mu split into even / odd columns (the tile), rad /
// amax / pmin, and cnt / children for the next level (slots without a node are skipped: no query has them as a
// child). Chunk mu rows come from the bf16 slab (converted on the way in), spine rows from the packed fp32 spines
__device__ __forceinline__ void stage_tile(const int u0, const int nt, const int b, const int lane, const int wid,
                                           float4* s_umeta, float4* s_um2, float* s_mu, const unsigned short* cmub,
                                           const float* cax, const float* sbase) {
    const int c0 = (lane & 1) * 32 + (lane >> 1);                      // tile column of node column `lane`
    for (int ul = wid; ul < nt; ul += NWARP_R) {
        const int u = u0 + ul;
        const int ref = __float_as_int(s_umeta[b * UMAX + u].x);
        if (ref != -1) {
            float* d = s_mu + ul * 64;
            if (ref >= 0) {
                const unsigned short* mp = cmub + (i64)ref * DH;
                d[c0] = __uint_as_float(((unsigned)__ldg(mp + lane)) << 16);
                d[c0 + 16] = __uint_as_float(((unsigned)__ldg(mp + lane + 32)) << 16);
            } else {
                const float* p = sbase + (i64)(-ref - 2) * NC;
                d[c0] = __ldg(p + lane); d[c0 + 16] = __ldg(p + lane + 32);
            }
            if (lane < 6) {
                const float* pa = aux_row(cax, sbase, ref);
                (lane < 3 ? reinterpret_cast<float*>(s_um2 + u) + lane
                          : reinterpret_cast<float*>(s_umeta + b * UMAX + u) + lane - 2)[0] = __ldg(pa + lane);
            }
        }
    }
}

// rank of this lane's children c = 2 k + h among the query's n children by (score desc, slot asc), i.e.
// #{m < c: s_m >= s_c} + #{m > c: s_m > s_c} (scores past n are -inf), and the frontier entry of each child kept
template <int NB>
__device__ __forceinline__ void rank_write(const float* srow, const short* curow, int* frow, const int n, const int h,
                                           const int ncar, const i64 wbeam) {
    constexpr int H = NB / 2;
    float my[H], ot[H];
#pragma unroll
    for (int m = 0; m < NB; m += 4) {
        const float4 v = *reinterpret_cast<const float4*>(srow + m);
        const float s0 = no_nan(v.x), s1 = no_nan(v.y), s2 = no_nan(v.z), s3 = no_nan(v.w);
        my[m >> 1] = h ? s1 : s0;       ot[m >> 1] = h ? s0 : s1;
        my[(m >> 1) + 1] = h ? s3 : s2; ot[(m >> 1) + 1] = h ? s2 : s3;
    }
    int r[H];
#pragma unroll
    for (int k = 0; k < H; k++) r[k] = h ? (int)(ot[k] >= my[k]) : (int)(ot[k] > my[k]);
#pragma unroll
    for (int j = 0; j < H; j++) {
#pragma unroll
        for (int k = j + 1; k < H; k++) add_pair(r[j], r[k], my[j], my[k]);
#pragma unroll
        for (int k = 0; k < H; k++) {
            if (j < k) r[k] = add_ge(r[k], ot[j], my[k]);
            else if (j > k) r[k] = add_gt(r[k], ot[j], my[k]);
        }
    }
#pragma unroll
    for (int k = 0; k < H; k++) {
        const int c = 2 * k + h;
        const int crank = ncar + r[k];
        if (c < n && my[k] > -5e8f && crank < wbeam && crank < WM) frow[crank] = (int)curow[c];
    }
}

// the fallback's continuation of one query from level lev0: its frontier entry for this lane's slot (a union slot of
// the previous level, whose staged node fields um_prev holds, a carried CAR0 + node ref, or EMPTY_E), then the warp
// descent. Out of line: inlined, its registers weigh on the cooperative descent (1-2% slower without any fallback)
__device__ __noinline__ void continue_query(WarpS& S, const int lane, const int ev, const float4* um_prev,
                                            const float* qrow, const float qn, const int nvis, const float qpos,
                                            const float lam, const float sig2, const float bud, const i64 wbeam,
                                            const int lev0, const int KMAXP1, const float scale, const float cell_size,
                                            const unsigned short* cmub, const float* cax, const float* sbase,
                                            const i64* leafb, const i64* admb,
                                            const i64 Ltot, const i64 Mm, i64* out, float* nvis_out) {
    int f_ref = -1;
    float f_cnt = 0.0f, f_chl = -1.0f, f_chr = -1.0f;
    if (ev >= 0 && ev < UMAX) {
        const float4 m = um_prev[ev];
        f_ref = __float_as_int(m.x); f_cnt = m.y; f_chl = m.z; f_chr = m.w;
    } else if (ev != EMPTY_E) {
        f_ref = ev - CAR0;
        const float* p = aux_row(cax, sbase, f_ref);
        f_cnt = p[3]; f_chl = p[4]; f_chr = p[5];
    }
    float qv[PER];                                                    // columns lane + 32 k (DH == DHP == 64)
#if Q_BF16
    const unsigned short* qrow_bf16 = reinterpret_cast<const unsigned short*>(qrow);
#pragma unroll
    for (int k = 0; k < PER; k++) qv[k] = __uint_as_float(((unsigned)qrow_bf16[lane + 32 * k]) << 16);
#else
#pragma unroll
    for (int k = 0; k < PER; k++) qv[k] = qrow[lane + 32 * k];
#endif
    warp_descend(S, lane, qv, qn, qpos, lam, sig2, bud, wbeam, f_ref, f_cnt, f_chl, f_chr, lev0, nvis, KMAXP1, scale,
                 cell_size, cmub, cax, sbase, leafb, admb, Ltot, Mm, out, nvis_out);
}

extern "C" __global__ void __launch_bounds__(NT, MINB)
fsb_beam_rows(const BeamP B0, const BeamP B1, const float* __restrict__ Q, const float* __restrict__ QPOS,
              const float* __restrict__ RC, const float rc_s, const float* __restrict__ LAM,
              const float* __restrict__ SIG, const int Bq, const float scale, const float cell_size,
              const i64 QST_T, const i64 QST_B, const int NQ, int* __restrict__ FBC) {
    extern __shared__ __align__(128) unsigned char smem[];
    float* s_mu = reinterpret_cast<float*>(smem + OFF_MU);          // [CAP][64] tile: even, then odd columns
    float4* s_umeta = reinterpret_cast<float4*>(smem + OFF_UM);     // [2][UMAX] union: ref (bits), cnt, children
    float4* s_um2 = reinterpret_cast<float4*>(smem + OFF_UM2);      // [UMAX] the level's union: rad, amax, pmin
    unsigned char* s_mark = smem + OFF_MARK;                        // [2][UMAX] parents expanded at a level
    short* s_rank = reinterpret_cast<short*>(smem + OFF_RANK);      // [UMAX] their rank
    int* s_front = reinterpret_cast<int*>(smem + OFF_FRONT);        // [QB][16] frontier entries
    float* s_sc = reinterpret_cast<float*>(smem + OFF_SC);          // [QB][SCS] child dots, then scores
    short* s_cu = reinterpret_cast<short*>(smem + OFF_CU);          // [QB][CUS] child union slots
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int qi = tid >> 1, h = tid & 1, rot = lane & 7;
#ifdef PROF
    unsigned long long ptim = clock64(), pacc[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#endif
    const unsigned pmask = 3u << (lane & 30);
    const int bpr = (Bq + QB - 1) / QB;                                 // blocks per row
    const bool sec = (i64)blockIdx.x >= B0.rows * bpr;
    const i64 blk = sec ? (i64)blockIdx.x - B0.rows * bpr : (i64)blockIdx.x;
    const i64 t = blk / bpr;
    const int q0 = (int)(blk - t * bpr) * QB;                          // the block's first query of the row
#define BSEL(f) (sec ? B1.f : B0.f)
    const i64* ROOT = BSEL(ROOT);
    i64* CELLS = BSEL(CELLS);
    float* NVIS = BSEL(NVIS);
    const i64* WBEAM = BSEL(WBEAM);
    const i64* BUDGET = BSEL(BUDGET);
    const i64* TIDX = BSEL(TIDX);
    const unsigned short* CHUNK = BSEL(CHUNK);
    const float* CHUNK_AX = BSEL(CHUNK_AX);
    const float* SPINE = BSEL(SPINE);
    const i64* LEAF = BSEL(LEAF);
    const i64* ADM = BSEL(ADM);
    const i64 Ntot = BSEL(Ntot), Sp = BSEL(Sp), Ltot = BSEL(Ltot), Mm = BSEL(Mm);
    const int KMAXP1 = BSEL(KMAXP1);
#undef BSEL
    const i64 tidx = TIDX[t];
    const unsigned short* cmub = CHUNK + tidx * Ntot * DH;
    const float* cax = CHUNK_AX + tidx * Ntot * NA;
    const float* sbase = SPINE + t * Sp * NC;
    const i64* leafb = LEAF + tidx * Ltot;
    const i64* admb = ADM + tidx * Mm;
#if RC_T
    const float rcq = RC[t];
#else
    const float rcq = rc_s;
#endif
    const float lam = LAM[t];
    const float sig2 = f_mul(2.0f, SIG[t]);
    const int root = (int)ROOT[t];
    const bool qok = q0 + qi < Bq;
    const bool qr_ok = qok && q0 + qi < NQ;                           // a strided stage view's pads read zeros
    const i64 pid = t * Bq + (qok ? q0 + qi : 0);
    const i64 qrow = t * QST_T + (qr_ok ? q0 + qi : 0) * QST_B;
#if Q_BF16 && QB == 128 && CAP == 64 && UMAX == 128
    float qn;
    {
        const unsigned short* qp = reinterpret_cast<const unsigned short*>(Q) + qrow;
        float qs = 0.0f;
        if (qr_ok) {
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const uint4 a = __ldg(reinterpret_cast<const uint4*>(qp + 32 * h + 8 * m));
                const float f0 = __uint_as_float(a.x << 16), f1 = __uint_as_float(a.x & 0xffff0000u);
                const float f2 = __uint_as_float(a.y << 16), f3 = __uint_as_float(a.y & 0xffff0000u);
                const float f4 = __uint_as_float(a.z << 16), f5 = __uint_as_float(a.z & 0xffff0000u);
                const float f6 = __uint_as_float(a.w << 16), f7 = __uint_as_float(a.w & 0xffff0000u);
                qs = f_add(qs, f_add(f_add(f_mul(f0, f0), f_mul(f1, f1)), f_add(f_mul(f2, f2), f_mul(f3, f3))));
                qs = f_add(qs, f_add(f_add(f_mul(f4, f4), f_mul(f5, f5)), f_add(f_mul(f6, f6), f_mul(f7, f7))));
            }
        }
        qn = f_mul(f_sqrt_ftz(f_add(qs, __shfl_xor_sync(FULL, qs, 1))), rcq);
    }
    const int g_tc = lane >> 2, t4_tc = lane & 3;
    unsigned qa[4][4];
    {
        const unsigned short* qbase_bf = reinterpret_cast<const unsigned short*>(Q) + t * QST_T + q0 * QST_B;
        const int r0 = 16 * wid + g_tc, r1 = r0 + 8;
        const bool ok0 = q0 + r0 < NQ, ok1 = q0 + r1 < NQ;
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            const int c0 = 16 * ks + 2 * t4_tc, c1 = c0 + 8;
            qa[ks][0] = ok0 ? __ldg(reinterpret_cast<const unsigned*>(qbase_bf + r0 * QST_B + c0)) : 0u;
            qa[ks][1] = ok1 ? __ldg(reinterpret_cast<const unsigned*>(qbase_bf + r1 * QST_B + c0)) : 0u;
            qa[ks][2] = ok0 ? __ldg(reinterpret_cast<const unsigned*>(qbase_bf + r0 * QST_B + c1)) : 0u;
            qa[ks][3] = ok1 ? __ldg(reinterpret_cast<const unsigned*>(qbase_bf + r1 * QST_B + c1)) : 0u;
        }
    }
    const float qpos_blk = QPOS[t * Bq + q0 + min(QB / 2, Bq - q0 - 1)];
    const float qpos = qok ? QPOS[pid] : 0.0f;
    const float bud = qok ? (float)BUDGET[pid] : 0.0f;
    const i64 wbeam = qok ? WBEAM[pid] : 0;
    int* frow = s_front + qi * 16;
    short* curow = s_cu + qi * CUS;
    float* srow = reinterpret_cast<float*>(s_cu) + 2 * qi;
    (void)s_sc;
    (void)rot;
#else
    // this lane's half of the query (chunk blocks in its rotated order) and the radius-scaled norm (same halving tree)
    float qr[32];
#if Q_BF16
    const unsigned short* qp = reinterpret_cast<const unsigned short*>(Q) + qrow;
#pragma unroll
    for (int m = 0; m < 8; m++) {
        const int mc = m ^ rot;
        const uint4 a = qr_ok ? __ldg(reinterpret_cast<const uint4*>(qp + 8 * mc)) : make_uint4(0, 0, 0, 0);
        qr[4 * m]     = __uint_as_float(h ? (a.x & 0xffff0000u) : (a.x << 16));
        qr[4 * m + 1] = __uint_as_float(h ? (a.y & 0xffff0000u) : (a.y << 16));
        qr[4 * m + 2] = __uint_as_float(h ? (a.z & 0xffff0000u) : (a.z << 16));
        qr[4 * m + 3] = __uint_as_float(h ? (a.w & 0xffff0000u) : (a.w << 16));
    }
#else
    const float4* qp = reinterpret_cast<const float4*>(Q + qrow);  // block m: columns 8 (m ^ rot) .. + 8
#pragma unroll
    for (int m = 0; m < 8; m++) {
        const int mc = m ^ rot;
        const float4 a = qr_ok ? __ldg(qp + 2 * mc) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4 b = qr_ok ? __ldg(qp + 2 * mc + 1) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        qr[4 * m] = h ? a.y : a.x; qr[4 * m + 1] = h ? a.w : a.z;
        qr[4 * m + 2] = h ? b.y : b.x; qr[4 * m + 3] = h ? b.w : b.z;
    }
#endif
    float qn;
    {
        float x2[32];
#pragma unroll
        for (int k = 0; k < 32; k++) x2[k] = f_mul(qr[k], qr[k]);
        const float hs = halve32(x2);
        qn = f_mul(f_sqrt_ftz(f_add(hs, __shfl_xor_sync(FULL, hs, 1))), rcq);
    }
    const float qpos = qok ? QPOS[pid] : 0.0f;
    const float bud = qok ? (float)BUDGET[pid] : 0.0f;
    const i64 wbeam = qok ? WBEAM[pid] : 0;
    int* frow = s_front + qi * 16;
    short* curow = s_cu + qi * CUS;
    float* srow = s_sc + qi * SCS;
#endif
    // the union before level 0 is the root (buffer 1); every query starts from it
    if (tid == 0 && root != -1) {
        const float* p = aux_row(cax, sbase, root);
        s_umeta[UMAX] = make_float4(__int_as_float(root), p[3], p[4], p[5]);
    }
    for (int v = tid; v < 2 * UMAX; v += NT) s_mark[v] = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) frow[8 * h + j] = (8 * h + j == 0 && root != -1) ? 0 : EMPTY_E;
    __syncthreads();
    int nvis = 0, lastbuf = 1, fb_lev = 0;
    bool done = !qok, coop = true;
    PMARK(0);
    for (int lev = 0; lev < KMAXP1; lev++) {
        const int pb = (lev + 1) & 1, cb = lev & 1;
        // ---- A: this lane's frontier slots 8 h .. 8 h + 7: expand (cnt > budget; marks the parent's slot in the previous
        // union), carried (cnt <= budget; stored as CAR0 + ref from now on), empty. A node whose count is neither (NaN)
        // is dropped, as by the warp kernels: it holds the level open (an expand bit without children or a mark) and
        // is not carried
        int e[8];
        unsigned pk = 0;                      // open | carried << 8 | left child << 16 | right child << 24 (open:
                                              // expanded or a NaN count)
        unsigned char* mk_w = s_mark + cb * UMAX;
        {
            const int4 f0 = *reinterpret_cast<const int4*>(frow + 8 * h);
            const int4 f1 = *reinterpret_cast<const int4*>(frow + 8 * h + 4);
            e[0] = f0.x; e[1] = f0.y; e[2] = f0.z; e[3] = f0.w; e[4] = f1.x; e[5] = f1.y; e[6] = f1.z; e[7] = f1.w;
        }
        float4 um[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const bool live = !done && e[j] >= 0 && e[j] < UMAX;
            um[j] = s_umeta[pb * UMAX + (live ? e[j] : 0)];
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int ev = e[j];
            const bool valid = !done && 8 * h + j < WM && ev != EMPTY_E;
            const bool live = valid && ev >= 0 && ev < UMAX;
            const bool ex = live && um[j].y > bud;
            const bool car = valid && (!live || um[j].y <= bud);
            pk |= ((unsigned)(valid && !car) << j) | ((unsigned)car << (8 + j)) |
                  ((unsigned)(ex && fchild(um[j].z) != -1) << (16 + j)) |
                  ((unsigned)(ex && fchild(um[j].w) != -1) << (24 + j));
            if (ex) mk_w[ev] = 1;
            if (live && car) {
                e[j] = CAR0 + __float_as_int(um[j].x);
                frow[8 * h + j] = e[j];
            }
        }
        const unsigned po = __shfl_xor_sync(FULL, pk, 1);
        const unsigned lo = h ? po : pk, hi = h ? pk : po;
        const unsigned E = (lo & 0xffu) | ((hi & 0xffu) << 8);
        const unsigned C = ((lo >> 8) & 0xffu) | (((hi >> 8) & 0xffu) << 8);
        const unsigned CM = spread16(((lo >> 16) & 0xffu) | (((hi >> 16) & 0xffu) << 8)) |
                            (spread16((lo >> 24) | ((hi >> 24) << 8)) << 1);
        const int ncar = __popc(C);
        if (!done && E == 0u && ncar <= wbeam) done = true;            // the frontier is a fixed point
        const int n = done ? 0 : __popc(CM);
        // ---- B1: the level's marks are in; stop once every query is done
        PMARK(1);
        if (__syncthreads_count(!done) == 0) break;
        PMARK(2);
        // ---- B: every warp ballots the level's marks (slots of the previous union expanded by some query); the thread of
        // each marked slot writes its rank and the union slots of its two children
        unsigned bal[UMAX / 32];
        int below = 0, tot = 0;
#pragma unroll
        for (int w = 0; w < UMAX / 32; w++) {
            bal[w] = __ballot_sync(FULL, mk_w[32 * w + lane] != 0);
            const int c = __popc(bal[w]);
            below += w < wid ? c : 0;
            tot += c;
        }
        const int Pc = 2 * tot;
        if (tid < UMAX && Pc <= UMAX) {
            unsigned bw = 0;
#pragma unroll
            for (int w = 0; w < UMAX / 32; w++) bw = w == wid ? bal[w] : bw;
            if ((bw >> lane) & 1u) {
                const int r = below + __popc(bw & ((1u << lane) - 1u));
                const float4 pm = s_umeta[pb * UMAX + tid];
                s_rank[tid] = (short)r;
                s_umeta[cb * UMAX + 2 * r].x = __int_as_float(fchild(pm.z));
                s_umeta[cb * UMAX + 2 * r + 1].x = __int_as_float(fchild(pm.w));
            }
        }
        __syncthreads();                                              // B2
        if (tid < UMAX) mk_w[tid] = 0;                                // every warp has read the marks
        PMARK(3);
        if (Pc > UMAX) { coop = false; fb_lev = lev; break; }
#if Q_BF16 && QB == 128 && CAP == 64 && UMAX == 128
        const int npass = (Pc + CAP - 1) / CAP;
        const int nq = n > 16 ? 32 : (n > 8 ? 16 : (n > 4 ? 8 : (n > 0 ? 4 : 0)));
        const bool ff = __syncthreads_and(done || (ncar + n <= wbeam && ncar + n <= WM));
        if (ff) {
            stage_meta_only(Pc, cb, tid, s_umeta, cax, sbase);
        } else {
            stage_tile_tc2(0, min(CAP, Pc), cb, lane, wid, s_umeta, s_um2, reinterpret_cast<unsigned*>(s_mu),
                           cmub, cax, sbase, qpos_blk, cell_size, lam, sig2, scale);
        }
        PMARK(8);
        if (CM == FULL) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int r2 = 2 * s_rank[e[j]];
                reinterpret_cast<int*>(curow)[8 * h + j] = (r2 & 0xffff) | ((r2 + 1) << 16);
            }
        } else if (n > 0) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int w = 8 * h + j;
                if ((E >> w) & 1u) {
                    const unsigned two = (CM >> (2 * w)) & 3u;
                    const int c0 = __popc(CM & ((1u << (2 * w)) - 1u));
                    const int r2 = 2 * s_rank[e[j]];
                    if (two & 1u) curow[c0] = (short)r2;
                    if (two & 2u) curow[c0 + (two & 1u)] = (short)(r2 + 1);
                }
            }
        }
        cp_async_wait_all();
        __syncthreads();
        PMARK(4);
        if (ff) {
            if (!done) {
                const int4 emp4 = make_int4(EMPTY_E, EMPTY_E, EMPTY_E, EMPTY_E);
                *reinterpret_cast<int4*>(frow + 8 * h) = emp4;
                *reinterpret_cast<int4*>(frow + 8 * h + 4) = emp4;
                __syncwarp(pmask);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int w = 8 * h + j;
                    if ((C >> w) & 1u) frow[__popc(C & ((1u << w) - 1u))] = e[j];
                }
                for (int c = h; c < n; c += 2) frow[ncar + c] = (int)curow[c];
                __syncwarp(pmask);
                nvis += n;
            }
        } else {
            unsigned* s_mu_u32 = reinterpret_cast<unsigned*>(s_mu);
            unsigned* s_dot_u32 = s_mu_u32 + (CAP * DH) / 2;
            const unsigned short* s_dot_bf = reinterpret_cast<const unsigned short*>(s_dot_u32);
            const int r0_tc = 16 * wid + g_tc, r1_tc = r0_tc + 8;
            const int swz_r0 = (g_tc & 7) << 1, swz_r1 = ((g_tc + 8) & 7) << 1;
            const int swz_q = (qi & 7) << 2;
            if (npass == 1) {
                const int nt_cur = min(CAP, Pc);
                const int ntiles = (nt_cur + 7) >> 3;
                for (int nt = 0; nt < ntiles; nt++) {
                    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    const int u_row = min(8 * nt + g_tc, nt_cur - 1);
                    const int swz = (u_row & 7) << 2;
                    const unsigned* mp = s_mu_u32 + u_row * 32;
#pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        const int col0 = (8 * ks + t4_tc) ^ swz;
                        const int col1 = (8 * ks + 4 + t4_tc) ^ swz;
                        mma16816_tc(acc, qa[ks], mp[col0], mp[col1]);
                    }
                    const int u_pair = 4 * nt + t4_tc;
                    s_dot_u32[r0_tc * (CAP / 2) + (u_pair ^ swz_r0)] = pack_bf16x2(f_mul(acc[0], scale), f_mul(acc[1], scale));
                    s_dot_u32[r1_tc * (CAP / 2) + (u_pair ^ swz_r1)] = pack_bf16x2(f_mul(acc[2], scale), f_mul(acc[3], scale));
                }
                __syncwarp();
                PMARK(9);
                if (!done) {
                    const int4 emp4 = make_int4(EMPTY_E, EMPTY_E, EMPTY_E, EMPTY_E);
                    *reinterpret_cast<int4*>(frow + 8 * h) = emp4;
                    *reinterpret_cast<int4*>(frow + 8 * h + 4) = emp4;
                    __syncwarp(pmask);
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int w = 8 * h + j;
                        if ((C >> w) & 1u) {
                            const int ncb = __popc(C & ((1u << w) - 1u));
                            if (ncb < wbeam && ncb < WM) frow[ncb] = e[j];
                        }
                    }
                    if (ncar < wbeam && ncar < WM) {
                        const unsigned short* q_dots = s_dot_bf + qi * CAP;
                        // one padded instantiation: c >= n scores stay -inf, which no valid rank counts, and the
                        // emit guard drops them; the nq-tailored chain serialized divergent warps
                        rw_direct_bf16<32>(q_dots, s_um2, qn, curow, frow, n, h, ncar, wbeam, swz_q);
                    }
                    __syncwarp(pmask);
                    nvis += n;
                }
            } else {
                float my_multi[16];
#pragma unroll
                for (int k = 0; k < 16; k++) my_multi[k] = NEG_INF;
                for (int p = 0; p < npass; p++) {
                    const int t0_p = p * CAP, t1_p = min(t0_p + CAP, Pc), nt_cur = t1_p - t0_p;
                    if (p > 0) {
                        __syncthreads();
                        stage_tile_tc2(t0_p, nt_cur, cb, lane, wid, s_umeta, s_um2, s_mu_u32,
                                       cmub, cax, sbase, qpos_blk, cell_size, lam, sig2, scale);
                        cp_async_wait_all();
                        __syncthreads();
                    }
                    const int ntiles = (nt_cur + 7) >> 3;
                    for (int nt = 0; nt < ntiles; nt++) {
                        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        const int u_row = min(8 * nt + g_tc, nt_cur - 1);
                        const int swz = (u_row & 7) << 2;
                        const unsigned* mp = s_mu_u32 + u_row * 32;
#pragma unroll
                        for (int ks = 0; ks < 4; ks++) {
                            const int col0 = (8 * ks + t4_tc) ^ swz;
                            const int col1 = (8 * ks + 4 + t4_tc) ^ swz;
                            mma16816_tc(acc, qa[ks], mp[col0], mp[col1]);
                        }
                        const int u_pair = 4 * nt + t4_tc;
                        s_dot_u32[r0_tc * (CAP / 2) + (u_pair ^ swz_r0)] = pack_bf16x2(f_mul(acc[0], scale), f_mul(acc[1], scale));
                        s_dot_u32[r1_tc * (CAP / 2) + (u_pair ^ swz_r1)] = pack_bf16x2(f_mul(acc[2], scale), f_mul(acc[3], scale));
                    }
                    __syncwarp();
                    if (!done && n > 0) {
                        const unsigned short* q_dots = s_dot_bf + qi * CAP;
#pragma unroll
                        for (int k = 0; k < 16; k++) {
                            const int c = 2 * k + h;
                            if (c < n) {
                                const int u = curow[c];
                                if (u >= t0_p && u < t1_p) {
                                    const float2 tm = *reinterpret_cast<const float2*>(s_um2 + u);
#if BOUND
                                    my_multi[k] = no_nan(f_add(f_add(bf2f_s(q_dots[(u - t0_p) ^ swz_q]), f_mul(qn, tm.x)), tm.y));
#else
                                    my_multi[k] = no_nan(f_add(bf2f_s(q_dots[(u - t0_p) ^ swz_q]), tm.y));
#endif
                                }
                            }
                        }
                    }
                }
                PMARK(10);
                if (!done) {
                    const int4 emp4 = make_int4(EMPTY_E, EMPTY_E, EMPTY_E, EMPTY_E);
                    *reinterpret_cast<int4*>(frow + 8 * h) = emp4;
                    *reinterpret_cast<int4*>(frow + 8 * h + 4) = emp4;
                    __syncwarp(pmask);
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int w = 8 * h + j;
                        if ((C >> w) & 1u) {
                            const int ncb = __popc(C & ((1u << w) - 1u));
                            if (ncb < wbeam && ncb < WM) frow[ncb] = e[j];
                        }
                    }
                    if (ncar < wbeam && ncar < WM) {
                        rw_reg<32>(my_multi, curow, frow, n, h, ncar, wbeam);
                    }
                    __syncwarp(pmask);
                    nvis += n;
                }
            }
        }
#else
        // ---- C: stage the first tile of the union (cp.async) while this query lists its children (slot order) as union
        // slots and fills -inf past n; then prefetch the next level's candidates
        const int npass = (Pc + CAP - 1) / CAP;
        const int nq = n > 16 ? 32 : (n > 8 ? 16 : (n > 4 ? 8 : (n > 0 ? 4 : 0)));
        stage_tile(0, min(CAP, Pc), cb, lane, wid, s_umeta, s_um2, s_mu, cmub, cax, sbase);
        PMARK(8);
        if (CM == FULL) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int r2 = 2 * s_rank[e[j]];
                reinterpret_cast<int*>(curow)[8 * h + j] = (r2 & 0xffff) | ((r2 + 1) << 16);
            }
        } else if (n > 0) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int w = 8 * h + j;
                if ((E >> w) & 1u) {
                    const unsigned two = (CM >> (2 * w)) & 3u;
                    const int c0 = __popc(CM & ((1u << (2 * w)) - 1u));
                    const int r2 = 2 * s_rank[e[j]];
                    if (two & 1u) curow[c0] = (short)r2;
                    if (two & 2u) curow[c0 + (two & 1u)] = (short)(r2 + 1);
                }
            }
            for (int c = n + h; c < nq; c += 2) srow[c] = NEG_INF;
        }
        PMARK(9);
        cp_async_wait_all();
        __syncthreads();                                              // B3: tile 0 staged, children listed
        PMARK(10);
        if (lev + 1 < KMAXP1) {
            for (int u = tid; u < min(CAP, Pc); u += NT) {           // candidates of the next union -> L2
                const float4 um = s_umeta[cb * UMAX + u];
                if (__float_as_int(um.x) != -1 && um.y > 1.0f) {
                    prefetch_node(cmub, cax, sbase, fchild(um.z));
                    prefetch_node(cmub, cax, sbase, fchild(um.w));
                }
            }
        }
        PMARK(4);
        // ---- D: child dots, two at a time: each lane sums its half of both, the halves are swapped and lane h keeps
        // child c + h's dot (and computes its score in place when npass == 1). A union larger than the tile takes
        // several passes.
        const unsigned tb = (unsigned)__cvta_generic_to_shared(s_mu) + 128u * h + 16u * rot;
        if (npass == 1) {
#pragma unroll 2
            for (int c = 0; c < n; c += 2) {
                const int uu = *reinterpret_cast<const int*>(curow + c);
                const bool hb = c + 1 < n;
                const int ua = (int)(short)(uu & 0xffff), ub = hb ? (uu >> 16) : ua;
                const float4 tm = s_um2[h ? ub : ua];
                const float pmin_pos = f_mul(f_add(tm.z, 0.5f), cell_size);
                const float dx = __int_as_float(__float_as_int(f_sub(qpos, pmin_pos)) & 0x7fffffff);
                const float div_x = f_div_full(dx, sig2);
                float sa, sb;
                half_dot2(qr, tb + 256u * ua, tb + 256u * ub, sa, sb);
                const float dot = f_add(h ? sb : sa, __shfl_xor_sync(pmask, h ? sa : sb, 1));
                if (h == 0 || hb) {
                    float content = f_mul(dot, scale);
#if BOUND
                    content = f_add(content, f_mul(f_mul(qn, tm.x), scale));
#endif
                    srow[c + h] = f_add(content, f_mul(f_mul(lam, tm.y), erf_tl(div_x)));
                }
            }
        } else {
            for (int p = 0; p < npass; p++) {
                const int t0 = p * CAP, t1 = min(t0 + CAP, Pc);
                if (p > 0) {
                    __syncthreads();
                    stage_tile(t0, t1 - t0, cb, lane, wid, s_umeta, s_um2, s_mu, cmub, cax, sbase);
                    cp_async_wait_all();
                    __syncthreads();
                }
                for (int c = 0; c < n; c += 2) {
                    const int uu = *reinterpret_cast<const int*>(curow + c);
                    const bool hb = c + 1 < n;
                    const int ua = (int)(short)(uu & 0xffff), ub = hb ? (uu >> 16) : ua;
                    const bool ina = ua >= t0 && ua < t1, inb = hb && ub >= t0 && ub < t1;
                    if (!ina && !inb) continue;
                    float sa, sb;
                    half_dot2(qr, tb + 256u * (ina ? ua - t0 : 0), tb + 256u * (inb ? ub - t0 : 0), sa, sb);
                    const float dot = f_add(h ? sb : sa, __shfl_xor_sync(pmask, h ? sa : sb, 1));
                    if (h ? inb : ina) srow[c + h] = dot;
                }
            }
            if (n > 0) {
#pragma unroll 2
                for (int c = h; c < n; c += 2) {
                    const float4 tm = s_um2[curow[c]];
                    float content = f_mul(srow[c], scale);
#if BOUND
                    content = f_add(content, f_mul(f_mul(qn, tm.x), scale));
#endif
                    const float pmin_pos = f_mul(f_add(tm.z, 0.5f), cell_size);
                    const float dx = __int_as_float(__float_as_int(f_sub(qpos, pmin_pos)) & 0x7fffffff);
                    srow[c] = f_add(content, f_mul(f_mul(lam, tm.y), erf_tl(f_div_full(dx, sig2))));
                }
            }
        }
        PMARK(5);
        // ---- E: the new frontier: carried slots keep their order (positions below the beam width), the children
        // follow at ncar + rank
        if (!done) {
            const int4 emp4 = make_int4(EMPTY_E, EMPTY_E, EMPTY_E, EMPTY_E);
            *reinterpret_cast<int4*>(frow + 8 * h) = emp4;
            *reinterpret_cast<int4*>(frow + 8 * h + 4) = emp4;
            __syncwarp(pmask);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int w = 8 * h + j;
                if ((C >> w) & 1u) {
                    const int ncb = __popc(C & ((1u << w) - 1u));
                    if (ncb < wbeam && ncb < WM) frow[ncb] = e[j];
                }
            }
            if (ncar < wbeam && ncar < WM) {
                if (nq == 32) {
                    rank_write<32>(srow, curow, frow, n, h, ncar, wbeam);
                } else if (nq == 16) {
                    rank_write<16>(srow, curow, frow, n, h, ncar, wbeam);
                } else if (nq == 8) {
                    rank_write<8>(srow, curow, frow, n, h, ncar, wbeam);
                } else if (nq == 4) {
                    rank_write<4>(srow, curow, frow, n, h, ncar, wbeam);
                }
            }
            __syncwarp(pmask);
            nvis += n;
        }
#endif
        lastbuf = cb;
        PMARK(6);
    }
    if (coop) {
        // cells of the final frontiers: each lane reads src / off / len of its slots' nodes and their cells, stages
        // them in the (now free) shared memory, and the block stores its queries' rows (contiguous in CELLS) with
        // 16-byte stores
        int ref[8];
        float src[8], off[8], ln[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int w = 8 * h + j;
            const int ev = (qok && w < WM) ? frow[w] : EMPTY_E;
            ref[j] = ev == EMPTY_E ? -1
                     : ((ev >= 0 && ev < UMAX) ? __float_as_int(s_umeta[lastbuf * UMAX + ev].x) : ev - CAR0);
            src[j] = 0.0f; off[j] = 0.0f; ln[j] = 0.0f;
            if (ref[j] != -1) {
                const float* p = aux_row(cax, sbase, ref[j]);
                src[j] = __ldg(p + 6); off[j] = __ldg(p + 7); ln[j] = __ldg(p + 8);
            }
        }
#if OSTAGE
        i64* ost = reinterpret_cast<i64*>(smem);                         // [QB][WM * R]
        __syncthreads();                                                 // every lane holds its frontier entries
#endif
        if (qok) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int w = 8 * h + j;
                if (w < WM) {
#pragma unroll
                    for (int tr = 0; tr < R; tr++) {
                        i64 cell = -1;
                        if (ref[j] != -1 && src[j] > 0.0f && (float)tr < ln[j]) {
                            const i64 pos = (i64)off[j] + tr;
                            cell = src[j] == 1.0f ? __ldg(leafb + min(max(pos, (i64)0), Ltot - 1))
                                                  : __ldg(admb + min(max(pos, (i64)0), Mm - 1));
                        }
#if OSTAGE
                        ost[qi * (WM * R) + w * R + tr] = cell;
#else
                        CELLS[pid * (WM * R) + w * R + tr] = cell;
#endif
                    }
                }
            }
            if (h == 0) NVIS[pid] = (float)nvis;
        }
#if OSTAGE
        __syncthreads();
        const int tot_c = min(QB, Bq - q0) * (WM * R);
        i64* gout = CELLS + (t * Bq + q0) * (WM * R);
        if ((tot_c & 1) == 0 && (reinterpret_cast<size_t>(gout) & 15) == 0) {
            for (int i = tid; i < tot_c / 2; i += NT)
                reinterpret_cast<longlong2*>(gout)[i] = reinterpret_cast<const longlong2*>(ost)[i];
        } else {
            for (int i = tid; i < tot_c; i += NT) gout[i] = ost[i];
        }
#endif
        PMARK(7);
#ifdef PROF
        if (lane == 0 && FBC != nullptr) {
            unsigned long long* pdst = reinterpret_cast<unsigned long long*>(FBC + 256);
            for (int k = 0; k < 12; k++) atomicAdd(pdst + wid * 12 + k, pacc[k]);
        }
#endif
    } else {
        // a level's union outgrew the shared tables: each warp continues the descents of QB / NWARP_R of the block's
        // queries, one at a time, from where the block left them at that level. Their frontiers there are the
        // warp-per-query descent's (union entries index the previous level's union, whose node fields are staged;
        // carried entries hold node refs), and so are their norms and visits so far, which each pair leaves in its
        // score row
#ifdef PROF
        if (tid == 0) atomicAdd(&g_fallback, 1);
#endif
        if (FBC != nullptr && tid == 0) {                             // test / diagnostic counters (null by default)
            atomicAdd(FBC, 1);
            atomicAdd(FBC + 1 + min(fb_lev, 63), 1);
        }
        if (h == 0) {
            reinterpret_cast<int*>(srow)[0] = nvis;
#if BOUND
            srow[1] = qn;                                             // (the mean score never reads the norm)
#endif
        }
        __syncthreads();
        const int pbf = (fb_lev + 1) & 1, slot = lane >> 1;
        WarpS* ws = reinterpret_cast<WarpS*>(smem + OFF_MU) + wid;
        for (int k = 0; k < QB / NWARP_R; k++) {
            const int ql = wid * (QB / NWARP_R) + k;
            if (q0 + ql >= Bq) break;
            const i64 p2 = t * Bq + q0 + ql;
#if Q_BF16 && QB == 128 && CAP == 64 && UMAX == 128
            const float* sq = reinterpret_cast<const float*>(s_cu) + 2 * ql;
#else
            const float* sq = s_sc + ql * SCS;
#endif
            const bool pq_ok = q0 + ql < NQ;
#if Q_BF16
            const float* qrow_p = pq_ok ? reinterpret_cast<const float*>(reinterpret_cast<const unsigned short*>(Q) + t * QST_T + (i64)(q0 + ql) * QST_B)
                                        : fsb_qzero;
#else
            const float* qrow_p = pq_ok ? Q + t * QST_T + (i64)(q0 + ql) * QST_B : fsb_qzero;
#endif
            continue_query(*ws, lane, slot < WM ? s_front[ql * 16 + slot] : EMPTY_E, s_umeta + pbf * UMAX,
                           qrow_p, BOUND ? sq[1] : 0.0f, reinterpret_cast<const int*>(sq)[0], QPOS[p2], lam, sig2,
                           (float)BUDGET[p2], WBEAM[p2], fb_lev, KMAXP1, scale, cell_size, cmub, cax, sbase, leafb, admb,
                           Ltot, Mm, CELLS + p2 * (WM * R), NVIS + p2);
        }
#ifdef PROF
        PMARK(11);
        if (lane == 0 && FBC != nullptr) {
            unsigned long long* pdst = reinterpret_cast<unsigned long long*>(FBC + 256);
            for (int k = 0; k < 12; k++) atomicAdd(pdst + wid * 12 + k, pacc[k]);
        }
#endif
    }
}

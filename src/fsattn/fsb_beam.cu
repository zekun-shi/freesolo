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
// >>> SECTION: _ROWS2_KERNEL <<<

// ---- lean row-cooperative prefill descent (FSB_BEAM_V2=1): fsb_beam_rows' tensor-core configuration (bf16 queries,
// 128 per block, head dim 64, frontier width 16, up to 4 cells per frontier node) with the same levels, unions, bf16
// dots and scores, restructured around four changes:
// - selection by sorting networks on packed keys, key = (order-preserving bits of the score, low 5 bits cleared) |
//   (31 - child index): each lane sorts the keys of its 8 frontier slots' children (16 keys, Batcher's network) and the
//   lane pair merges the two sorted halves (bitonic), so lane h ends with ranks 8 h .. 8 h + 7. The 32-bit order is
//   fsb_beam_rows' (score desc, child asc) except for two children whose scores agree in their top 27 bits (relative
//   2^-18), which it orders by child index; the EXACT build (FSB_BEAM_V2_EXACT=1) finds such pairs among a query's
//   ranks 0..k (k kept) and has the warp select that level again with 64-bit keys, so it selects exactly as
//   fsb_beam_rows (and so does the KEY64 test build, 64-bit keys at every level);
// - per-query state in registers: the open frontier (union slots in rank order, a byte each, with 4-bit codes of the
//   union's nodes staged for the block's budget); carried nodes, which never expand again, leave it for a
//   shared-memory prefix in carry order;
// - two block barriers per level: one after the expanded parents' marks (bytes over the union, read back by ballots),
//   after which the marked parents' threads (one per child, all warps) stage the next union and publish its slot
//   ranks, and one after that union is staged;
// - warp-local dots: each warp dots its own 16 queries (its mma rows) with the union, DCH slots at a time, and gathers
//   its queries' child scores after a __syncwarp.
// A union of more than UC2 slots hands the block's queries to the warp-per-query descent from that level, as in
// fsb_beam_rows.

// UC2: union slots of a level (a larger union falls back), DCH: union slots per warp dot chunk (fsb_beam_cuda)
#define MK2 (UC2 + 16)                            // bytes per mark buffer: one per union slot, then the level flags

static_assert(UC2 >= 32 && UC2 <= UMAX && UC2 <= 128 && (UC2 & (UC2 - 1)) == 0 && DCH == 64 && WM == 16 && R <= 4 &&
              DH == 64 && QB == 128, "fsb_beam_rows2's configuration");

// Selection keys: 32-bit (unsigned: the score's top 27 order-preserving bits, then 31 - child index) or 64-bit (u64:
// all 32 score bits above the child index, exactly fsb_beam_rows' (score desc, child asc) order). The levels select with
// 32-bit keys; the EXACT build redoes a warp's selection with 64-bit keys when two of a query's ranks 0..k agree in the
// 32-bit keys' score bits, the only case where the two orders can differ; the KEY64 test build uses 64-bit keys only.
typedef unsigned long long u64;

template <typename K> __device__ __forceinline__ K kmax(const K a, const K b) { return a > b ? a : b; }
template <typename K> __device__ __forceinline__ K kmin(const K a, const K b) { return a > b ? b : a; }

template <typename K> __device__ __forceinline__ void ce_desc(K& a, K& b) {   // a <- max, b <- min
    const K x = kmax(a, b);
    b = kmin(a, b);
    a = x;
}

// Batcher's odd-even merge sort of 16 keys, descending (63 compare-exchanges)
template <typename K> __device__ __forceinline__ void sort16_desc(K (&k)[16]) {
#define CE(a, b) ce_desc(k[a], k[b])
    CE(0,1); CE(2,3); CE(4,5); CE(6,7); CE(8,9); CE(10,11); CE(12,13); CE(14,15); CE(0,2); CE(1,3); CE(4,6); CE(5,7);
    CE(8,10); CE(9,11); CE(12,14); CE(13,15); CE(1,2); CE(5,6); CE(9,10); CE(13,14); CE(0,4); CE(1,5); CE(2,6);
    CE(3,7); CE(8,12); CE(9,13); CE(10,14); CE(11,15); CE(2,4); CE(3,5); CE(10,12); CE(11,13); CE(1,2); CE(3,4);
    CE(5,6); CE(9,10); CE(11,12); CE(13,14); CE(0,8); CE(1,9); CE(2,10); CE(3,11); CE(4,12); CE(5,13); CE(6,14);
    CE(7,15); CE(4,8); CE(5,9); CE(6,10); CE(7,11); CE(2,4); CE(3,5); CE(6,8); CE(7,9); CE(10,12); CE(11,13); CE(1,2);
    CE(3,4); CE(5,6); CE(7,8); CE(9,10); CE(11,12); CE(13,14);
#undef CE
}

// the top 16 of a lane pair's 32 keys, ranked: each lane sorts its 16 (lane 0: A, lane 1: B, descending), the first
// stage of the bitonic merge of A with B reversed keeps c[j] = max(A[j], B[15 - j]) (lane 0 forms j < 8, lane 1 j >= 8:
// one shuffle each), the second splits c: lane 0 the larger 8 (ranks 0-7), lane 1 the rest (ranks 8-15), each sorted
// by a bitonic 8-sorter. Returns rank 16 (the largest of the first stage's discarded min(A[j], B[15 - j])).
template <typename K> __device__ __forceinline__ K pair_top16(K (&key)[16], K (&r8)[8], const int h) {
    sort16_desc(key);
    K c[8], d = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const K mine = h ? key[7 - i] : key[i];                     // lane 0: A[i], lane 1: B[7 - i] (j = 8 + i)
        const K other = __shfl_xor_sync(FULL, h ? key[15 - i] : key[8 + i], 1);   // B[15 - i], A[8 + i]
        c[i] = kmax(mine, other);
        d = kmax(d, kmin(mine, other));
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const K other = __shfl_xor_sync(FULL, c[i], 1);             // lane 0 gets c[8 + i], lane 1 c[i]
        r8[i] = h ? kmin(c[i], other) : kmax(c[i], other);
    }
    ce_desc(r8[0], r8[4]); ce_desc(r8[1], r8[5]); ce_desc(r8[2], r8[6]); ce_desc(r8[3], r8[7]);
    ce_desc(r8[0], r8[2]); ce_desc(r8[1], r8[3]); ce_desc(r8[4], r8[6]); ce_desc(r8[5], r8[7]);
    ce_desc(r8[0], r8[1]); ce_desc(r8[2], r8[3]); ce_desc(r8[4], r8[5]); ce_desc(r8[6], r8[7]);
    return kmax(d, __shfl_xor_sync(FULL, d, 1));
}

// slot of the r-th byte of a lane's 8 packed bytes (lo: bytes 0-3, hi: 4-7)
__device__ __forceinline__ int byte8(const unsigned lo, const unsigned hi, const int j) {
    return (int)(((j < 4 ? lo : hi) >> (8 * (j & 3))) & 0xffu);
}

// a node's 4-bit code for budget bud: expanded (cnt > bud) | carried (cnt <= bud) << 1 | left child << 2 | right child
// << 3 (children only when expanded; a NaN count is neither expanded nor carried)
__device__ __forceinline__ unsigned node_code(const float cnt, const float chl, const float chr, const float bud) {
    const bool ex = cnt > bud, ca = cnt <= bud;
    return (unsigned)ex | ((unsigned)ca << 1) | ((unsigned)(ex && chl != -1.0f) << 2) |
           ((unsigned)(ex && chr != -1.0f) << 3);
}

// union-slot pair index of a chunk dot row (DCH / 2 = 32 pairs), swizzled so one n-tile's fragment stores (rows g,
// pairs t4) hit 32 banks
__device__ __forceinline__ int dswz(const int row) { return (row & 7) << 2; }

__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// mma16816_tc without the volatile ordering (registers only), so independent n-tiles' chains interleave
__device__ __forceinline__ void mma_bf16(float (&d)[4], const unsigned (&a)[4], const unsigned b0, const unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// pack_bf16x2 in one instruction: x rounded to bf16 in the low half, y in the high half (round to nearest even)
__device__ __forceinline__ unsigned pack2_bf16(const float x, const float y) {
    unsigned r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(y), "f"(x));
    return r;
}

// the selection key of a child with score s and child index c, or 0 when it has no child (exists false) or its score is
// not above the reference's -5e8 (NaN included): the order-preserving bits of s (sign bit set for s >= 0, every bit
// flipped below) with the low 5 bits replaced by 31 - c (32-bit keys) or followed by 31 - c (64-bit keys), so equal
// score bits rank the earlier child first. A mean score can be -0 (a -0 dot and a -0 prior), which counts as +0; a
// bound score never is (with a radius coefficient >= 0 its radius term is +0 or positive). A real child's 32-bit key
// is at least 32 (its score bits are not all zero).
template <typename K> __device__ __forceinline__ K child_key(const float s, const int c, const bool exists) {
#if BOUND
    const unsigned u = __float_as_uint(s);
#else
    const unsigned u = __float_as_uint(f_add(s, 0.0f));
#endif
    const unsigned o = u ^ ((unsigned)((int)u >> 31) | 0x80000000u);
    if (sizeof(K) == 8) return (exists && s > -5e8f) ? (K)(((u64)o << 32) | (u64)(31 - c)) : (K)0;
    return (exists && s > -5e8f) ? (K)((o & ~31u) | (unsigned)(31 - c)) : (K)0;
}

// one dot chunk of a warp (union slots [cd, cd + DCH), all of the union when !CHUNKED): the mma of its 16 queries with
// the chunk's tile rows (B fragments by ldmatrix: thread i addresses row 8 nt + (i & 7), piece i >> 3, + 4 for the
// second load; rows past the union repeat its last row), bf16 after the scale as fsb_beam_rows, then this lane's
// children's keys from their dot pair and score-term pair (a frontier slot's two children sit at union slots 2 pi,
// 2 pi + 1, pi = base / 2 of the slot's base in bv; pk holds the child bits)
template <bool CHUNKED, typename K>
__device__ __forceinline__ void chunk_keys(K (&key)[16], const unsigned* s_mu, unsigned* s_dw, const float4* s_ux2,
                                           const unsigned (&qa)[4][4], const unsigned bv_lo, const unsigned bv_hi,
                                           const unsigned pk, const float qn, const float scale, const int Pc,
                                           const int cd, const int lane, const int h) {
    const int g = lane >> 2, t4 = lane & 3, ql = lane >> 1;
    const unsigned pm = CHUNKED ? 0x3f3f3f3fu : 0x1f1f1f1fu;              // pair indices below UC2 / 2 (DCH / 2)
    const unsigned pv_lo = (bv_lo >> 1) & pm, pv_hi = (bv_hi >> 1) & pm;
#pragma unroll
    for (int grp = 0; grp < DCH / 16; grp++) {
        if (cd + 16 * grp < Pc) {
            // two n-tiles at once, so their mma chains interleave
            unsigned b[2][8];
#pragma unroll
            for (int n2 = 0; n2 < 2; n2++) {
                const int u_row = min(cd + 16 * grp + 8 * n2 + (lane & 7), Pc - 1);
                const unsigned ra = (unsigned)__cvta_generic_to_shared(s_mu + u_row * 32);
                ldsm_x4(b[n2][0], b[n2][1], b[n2][2], b[n2][3],
                        ra + 16u * ((unsigned)(lane >> 3) ^ (unsigned)(u_row & 7)));
                ldsm_x4(b[n2][4], b[n2][5], b[n2][6], b[n2][7],
                        ra + 16u * ((unsigned)((lane >> 3) + 4) ^ (unsigned)(u_row & 7)));
            }
            float acc[2][4];
#pragma unroll
            for (int n2 = 0; n2 < 2; n2++) acc[n2][0] = acc[n2][1] = acc[n2][2] = acc[n2][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < 4; ks++)
#pragma unroll
                for (int n2 = 0; n2 < 2; n2++) mma_bf16(acc[n2], qa[ks], b[n2][2 * ks], b[n2][2 * ks + 1]);
#pragma unroll
            for (int n2 = 0; n2 < 2; n2++) {
                const int pr = 4 * (2 * grp + n2) + t4;
                s_dw[g * (DCH / 2) + (pr ^ dswz(g))] = pack2_bf16(f_mul(acc[n2][0], scale), f_mul(acc[n2][1], scale));
                s_dw[(g + 8) * (DCH / 2) + (pr ^ dswz(g + 8))] =
                    pack2_bf16(f_mul(acc[n2][2], scale), f_mul(acc[n2][3], scale));
            }
        }
    }
    __syncwarp();
    const unsigned* drow = s_dw + ql * (DCH / 2);
#pragma unroll
    for (int jj = 0; jj < 8; jj++) {
        const int pi = byte8(pv_lo, pv_hi, jj);
        int pl = pi;
        unsigned two = (pk >> (4 * jj + 2)) & 3u;
        if (CHUNKED) {
            pl = pi - (cd >> 1);
            if ((unsigned)pl >= (unsigned)(DCH / 2)) { pl = 0; two = 0u; }
        }
        const unsigned dd = drow[pl ^ dswz(ql)];
        const float4 tm = s_ux2[pi];
        const float dl = __uint_as_float(dd << 16), dr = __uint_as_float(dd & 0xffff0000u);
#if BOUND
        const float sl = f_add(f_add(dl, f_mul(qn, tm.x)), tm.y);
        const float sr = f_add(f_add(dr, f_mul(qn, tm.z)), tm.w);
#else
        const float sl = f_add(dl, tm.y);
        const float sr = f_add(dr, tm.w);
#endif
        const int c = 2 * (8 * h + jj);
        if (!CHUNKED || (two & 1u))
            key[2 * jj] = CHUNKED ? child_key<K>(sl, c, true) : child_key<K>(sl, c, (two & 1u) != 0u);
        if (!CHUNKED || (two & 2u))
            key[2 * jj + 1] = CHUNKED ? child_key<K>(sr, c + 1, true) : child_key<K>(sr, c + 1, (two & 2u) != 0u);
    }
    __syncwarp();
}

// a warp's children's keys (all of the union, or DCH slots at a time) and the lane pair's ranked top 16 (pair_top16);
// final queries hold no keys. Returns rank 16.
template <typename K>
__device__ __forceinline__ K warp_select(K (&r8)[8], const unsigned* s_mu, unsigned* s_dw, const float4* s_ux2,
                                         const unsigned (&qa)[4][4], const unsigned bv_lo, const unsigned bv_hi,
                                         const unsigned pk, const float qn, const float scale, const int Pc,
                                         const int lane, const int h) {
    K key[16];
#pragma unroll
    for (int i = 0; i < 16; i++) key[i] = 0;
    if (Pc <= DCH) {
        chunk_keys<false>(key, s_mu, s_dw, s_ux2, qa, bv_lo, bv_hi, pk, qn, scale, Pc, 0, lane, h);
    } else {
        for (int cd = 0; cd < Pc; cd += DCH)
            chunk_keys<true>(key, s_mu, s_dw, s_ux2, qa, bv_lo, bv_hi, pk, qn, scale, Pc, cd, lane, h);
    }
    return pair_top16(key, r8, h);
}

// the new open entries from a lane's ranks 8 h .. 8 h + 7: the kept ranks (below k, the free slots after the carried
// prefix), their union slots from the child index in the key (s_cs: the query's open ranks' left child slots)
template <typename K>
__device__ __forceinline__ int open_slots(const K (&r8)[8], const unsigned char* cs, const int k, const int h,
                                          unsigned& nlo, unsigned& nhi) {
    nlo = 0u; nhi = 0u;
    int nv = 0;
#pragma unroll
    for (int jj = 0; jj < 8; jj++) {
        const K kk = r8[jj];
        const int cix = 31 - (int)(kk & 31u);
        const unsigned slot = (unsigned)(cs[cix >> 1] + (cix & 1));
        if (8 * h + jj < k && kk != 0) {
            if (jj < 4) nlo |= slot << (8 * jj);
            else nhi |= slot << (8 * (jj - 4));
            nv++;
        }
    }
    return nv;
}

extern "C" __global__ void __launch_bounds__(NT, MINB2)
fsb_beam_rows2(const BeamP B0, const BeamP B1, const float* __restrict__ Q, const float* __restrict__ QPOS,
               const float* __restrict__ RC, const float rc_s, const float* __restrict__ LAM,
               const float* __restrict__ SIG, const int Bq, const float scale, const float cell_size,
               const i64 QST_T, const i64 QST_B, const int NQ, int* __restrict__ FBC) {
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned* s_mu = reinterpret_cast<unsigned*>(smem + OFF2_MU);     // [UC2][32] union tile: bf16 pairs, row u's
                                                                       // 16-byte pieces XOR (u & 7)
    unsigned* s_dot = reinterpret_cast<unsigned*>(smem + OFF2_DOT);   // [8][16][DCH / 2] each warp's chunk dots
    float4* s_um = reinterpret_cast<float4*>(smem + OFF2_UM);         // [2][UC2] union nodes: ref (bits), cnt, children
    float2* s_ux = reinterpret_cast<float2*>(smem + OFF2_UX);         // [UC2] the staged union: rad * scale, prior
    int* s_car = reinterpret_cast<int*>(smem + OFF2_CAR);             // [QB][16] carried node refs, in carry order
    unsigned char* s_cs = smem + OFF2_CS;                              // [QB][16] open rank r's left child slot
    unsigned char* s_mk = smem + OFF2_MK;                              // [2][MK2] parent marks, then the level flags
    unsigned char* s_rank = smem + OFF2_RANK;                          // [UC2] a marked parent's children's union slot
    unsigned char* s_code = smem + OFF2_CODE;                          // [2][UC2] the union nodes' 4-bit codes
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int qi = tid >> 1, h = tid & 1, ql = lane >> 1;
#ifdef PROF
    unsigned long long ptim = clock64(), pacc[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#endif
    const int bpr = (Bq + QB - 1) / QB;
    const bool sec = (i64)blockIdx.x >= B0.rows * bpr;
    const i64 blk = sec ? (i64)blockIdx.x - B0.rows * bpr : (i64)blockIdx.x;
    const i64 t = blk / bpr;
    const int q0 = (int)(blk - t * bpr) * QB;
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
    // the query's radius-scaled norm (lane h: columns 32 h .. 32 h + 31) and the warp's mma A fragments (rows 16 wid +
    // g and + 8), as fsb_beam_rows reads them
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
    const int g = lane >> 2, t4 = lane & 3;
    unsigned qa[4][4];
    {
        const unsigned short* qbase_bf = reinterpret_cast<const unsigned short*>(Q) + t * QST_T + q0 * QST_B;
        const int r0 = 16 * wid + g, r1 = r0 + 8;
        const bool ok0 = q0 + r0 < NQ, ok1 = q0 + r1 < NQ;
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            const int c0 = 16 * ks + 2 * t4, c1 = c0 + 8;
            qa[ks][0] = ok0 ? __ldg(reinterpret_cast<const unsigned*>(qbase_bf + r0 * QST_B + c0)) : 0u;
            qa[ks][1] = ok1 ? __ldg(reinterpret_cast<const unsigned*>(qbase_bf + r1 * QST_B + c0)) : 0u;
            qa[ks][2] = ok0 ? __ldg(reinterpret_cast<const unsigned*>(qbase_bf + r0 * QST_B + c1)) : 0u;
            qa[ks][3] = ok1 ? __ldg(reinterpret_cast<const unsigned*>(qbase_bf + r1 * QST_B + c1)) : 0u;
        }
    }
    const float qpos_blk = QPOS[t * Bq + q0 + min(QB / 2, Bq - q0 - 1)];
    const float bud = qok ? (float)BUDGET[pid] : 0.0f;
    const i64 wbeam = qok ? WBEAM[pid] : 0;
    const int wl = (int)min(max(wbeam, (i64)0), (i64)WM);             // frontier slots that can be kept
    // the budget the staged codes use: the block's first query's (every query of a prefill row shares its row's)
    const float bud_blk = (float)BUDGET[t * Bq + q0];
    const bool ubud = !qok || bud == bud_blk;
    for (int v = tid; v < 2 * MK2 / 4; v += NT) reinterpret_cast<unsigned*>(s_mk)[v] = 0u;
    unsigned o_lo = 0u, o_hi = 0u;            // open entries: union slots in rank order, lane h ranks 8 h .. 8 h + 7
    int nopen = (qok && root != -1) ? 1 : 0, ncar = 0, nvis = 0;
    bool done = !qok;
    int ub = 0, fb_lev = -1;                  // the union buffer the open entries index; the fallback level
    int ncar0 = 0, nvis0 = 0, nopen0 = 0;     // the level's starting state (the fallback restarts the level)
    // ---- the levels that keep every child are the same for every query of a block with one width and budget: warp 0
    // walks them once (lane j: frontier node j; carried nodes in slot order, children in child order), up to the
    // first level fsb_beam_rows' vote makes a selecting one, the fixed point or the level limit, and stages that
    // level's frontier as its union (slot j); the queries start there with the walk's carried prefix and visits.
    // Otherwise level 0's union is the root (slot 0), every query's one open entry.
    int lev0 = 0;
    const int w_blk = (int)min(max(WBEAM[t * Bq + q0], (i64)0), (i64)WM);
    const bool uni = __syncthreads_and(!qok || (wl == w_blk && bud == bud_blk));
    int* ff = reinterpret_cast<int*>(s_dot);  // the walk's frontier refs [0, 16), carried refs [16, 32), its state
    if (wid == 0) {
        int fref = lane == 0 ? root : -1, nf = root != -1 ? 1 : 0, ncf = 0, nvf = 0, lv = 0;
        float fcnt = 0.0f, fchl = -1.0f, fchr = -1.0f;
        if (fref != -1) {
            const float* p = aux_row(cax, sbase, fref);
            fcnt = p[3]; fchl = p[4]; fchr = p[5];
        }
        const unsigned lt = (1u << lane) - 1u;
        while (uni && lv < KMAXP1 && nf > 0) {
            const unsigned cd = lane < nf ? node_code(fcnt, fchl, fchr, bud_blk) : 0u;
            const unsigned car = __ballot_sync(FULL, cd & 2u), lc = __ballot_sync(FULL, cd & 4u);
            const unsigned rc = __ballot_sync(FULL, cd & 8u);
            const int ncarry = __popc(car), n = __popc(lc) + __popc(rc);
            if (__ballot_sync(FULL, cd & 1u) == 0u) break;
            if (ncf + ncarry + n > w_blk && (ncarry < nf || ncf + ncarry > w_blk)) break;
            if ((cd & 2u) && ncf + __popc(car & lt) < w_blk) ff[16 + ncf + __popc(car & lt)] = fref;
            const int cpos = __popc(lc & lt) + __popc(rc & lt);
            if (cd & 4u) ff[cpos] = (int)fchl;
            if (cd & 8u) ff[cpos + ((cd & 4u) ? 1 : 0)] = (int)fchr;
            __syncwarp();
            ncf = min(ncf + ncarry, w_blk);
            nvf += n;
            nf = n;
            lv++;
            fref = lane < nf ? ff[lane] : -1;
            __syncwarp();
            if (fref != -1) {
                const float* p = aux_row(cax, sbase, fref);
                fcnt = __ldg(p + 3); fchl = __ldg(p + 4); fchr = __ldg(p + 5);
            }
        }
        if (lane < nf) {
            s_um[lane] = make_float4(__int_as_float(fref), fcnt, fchl, fchr);
            s_code[lane] = (unsigned char)node_code(fcnt, fchl, fchr, bud_blk);
        }
        if (lane == 0) { ff[32] = nf; ff[33] = ncf; ff[34] = nvf; ff[35] = lv; }
    }
    __syncthreads();
    if (uni) {
        lev0 = ff[35];
        if (qok) {
            ncar = ff[33];
            nvis = ff[34];
            nopen = ff[32];
            o_lo = h ? 0x0b0a0908u : 0x03020100u;
            o_hi = h ? 0x0f0e0d0cu : 0x07060504u;
#pragma unroll
            for (int jj = 0; jj < 8; jj++)
                if (8 * h + jj < ncar) s_car[qi * 16 + 8 * h + jj] = ff[16 + 8 * h + jj];
        }
    }
    bool done0 = done;
    PMARK(0);
    for (int lev = lev0; lev < KMAXP1; lev++) {
        unsigned char* mk = s_mk + ub * MK2;
        const float4* um_cur = s_um + ub * UC2;
        ncar0 = ncar; nvis0 = nvis; nopen0 = nopen; done0 = done;
        // ---- A: the open entries as 4-bit codes: expanded (cnt > budget: their children are the candidates, their
        // slots are marked), carried (cnt <= budget: appended to the carried prefix in slot order, positions past the
        // width drop), left / right child present; a NaN count is neither expanded nor carried: dropped, as in every
        // beam kernel. The union stages the codes for the block's budget; a query with a budget of its own derives them
        // from the node fields (children are stored as exact integers, -1 for none). Nibble j of pk is open rank
        // 8 h + j (0 past the open count). A query with nothing to expand, or no free slot for a child, is final.
        const int nmine = done ? 0 : min(max(nopen - 8 * h, 0), 8);
        const unsigned char* code_cur = s_code + ub * UC2;
        unsigned pk = 0u;
        if (ubud) {
#pragma unroll
            for (int jj = 0; jj < 8; jj++)
                pk |= (jj < nmine ? (unsigned)code_cur[byte8(o_lo, o_hi, jj) & (UC2 - 1)] : 0u) << (4 * jj);
        } else {
#pragma unroll
            for (int jj = 0; jj < 8; jj++) {
                const float4 m = um_cur[byte8(o_lo, o_hi, jj) & (UC2 - 1)];
                pk |= (jj < nmine ? node_code(m.y, m.z, m.w, bud) : 0u) << (4 * jj);
            }
        }
        const unsigned po = __shfl_xor_sync(FULL, pk, 1);
        const unsigned lo = h ? po : pk, hi = h ? pk : po;
        int k = 0, n = 0;
        if (!done) {
            n = __popc(lo & 0xccccccccu) + __popc(hi & 0xccccccccu);
            const int ncarry = __popc(lo & 0x22222222u) + __popc(hi & 0x22222222u);
            const int cbase = ncar + (h ? __popc(lo & 0x22222222u) : 0);
            for (unsigned cm = pk & 0x22222222u; cm != 0u; cm &= cm - 1u) {      // this lane's carried entries
                const int b = __ffs(cm) - 1, jj = b >> 2;
                const int pos = cbase + __popc(pk & 0x22222222u & ((1u << b) - 1u));
                if (pos < wl) s_car[qi * 16 + pos] = __float_as_int(um_cur[byte8(o_lo, o_hi, jj) & (UC2 - 1)].x);
            }
            // the level selects (ranks children by score) when a query has more children than free slots, the vote of
            // fsb_beam_rows, which also counts a query whose frontier is not yet at its fixed point (open entries,
            // or more carried nodes than slots: a width-0 root) though it keeps no child
            if (ncar + ncarry + n > wl && (ncarry < nopen || ncar + ncarry > wl)) {
                mk[UC2] = 1;
                mk[UC2 + 1] = 1;
            }
            ncar = min(ncar + ncarry, wl);
            nvis += n;
            k = wl - ncar;
            if (((lo | hi) & 0x11111111u) == 0u || k <= 0) {
                done = true;
                nopen = 0;
                n = 0;
            } else {
#pragma unroll
                for (int jj = 0; jj < 8; jj++)
                    if ((pk >> (4 * jj)) & 1u) mk[byte8(o_lo, o_hi, jj)] = 1;
                mk[UC2] = 1;                                           // a query descends further
            }
        }
        PMARK(1);
        __syncthreads();                                               // B1: the level's marks and flags
        PMARK(2);
        const unsigned fl = *reinterpret_cast<const unsigned short*>(mk + UC2);
        if ((fl & 0xffu) == 0u) break;                                 // every query is final
        const bool any_sel = (fl >> 8) != 0u;
        unsigned bal[4];
#pragma unroll
        for (int w = 0; w < 4; w++) bal[w] = w < UC2 / 32 ? __ballot_sync(FULL, mk[32 * w + lane] != 0) : 0u;
        const int c0 = __popc(bal[0]), c01 = c0 + __popc(bal[1]), c012 = c01 + __popc(bal[2]);
        const int Pc = 2 * (c012 + __popc(bal[3]));
#ifdef PROF
        if (tid == 0 && FBC != nullptr) {                             // union sizes (8-slot bins) of select / keep-all
            atomicAdd(FBC + (any_sel ? 128 : 192) + min(Pc >> 3, 63), 1);
        }
#endif
        if (Pc > UC2) { fb_lev = lev; break; }
        // the other buffer's marks were last read before this barrier: clear them for the next level
        if (tid < MK2 / 4) reinterpret_cast<unsigned*>(s_mk + (ub ^ 1) * MK2)[tid] = 0u;
        // ---- B: the next union: the marked parents' children in slot order, slot 2 r + side for the parent of rank r.
        // Thread p + 128 side stages child side of parent slot p (all 8 warps take part): its tile row by cp.async (a
        // spine row rounds its fp32 mean to bf16 as the warp kernels' tile does) when some query selects, its node
        // fields and code; the left child's thread records the parent's child base 2 r in s_rank
        float4* um_nxt = s_um + (ub ^ 1) * UC2;
        unsigned char* code_nxt = s_code + (ub ^ 1) * UC2;
        {
            const int p = tid & 127, sd = tid >> 7, pw = p >> 5;
            const unsigned bw = pw == 0 ? bal[0] : (pw == 1 ? bal[1] : (pw == 2 ? bal[2] : bal[3]));
            if (p < UC2 && ((bw >> lane) & 1u)) {
                const int pre = pw == 0 ? 0 : (pw == 1 ? c0 : (pw == 2 ? c01 : c012));
                const int base = 2 * (pre + __popc(bw & ((1u << lane) - 1u)));
                if (sd == 0) s_rank[p] = (unsigned char)base;
                const float4 pm = um_cur[p];
                const int ref = (int)(sd ? pm.w : pm.z);
                const int u = base + sd;
                unsigned* dst = s_mu + u * 32;
                if (ref != -1) {
                    const float* pa = aux_row(cax, sbase, ref);
                    const float cnt = __ldg(pa + 3), chl = __ldg(pa + 4), chr = __ldg(pa + 5);
                    if (any_sel) {
                        if (ref >= 0) {
                            const unsigned* src = reinterpret_cast<const unsigned*>(cmub + (i64)ref * DH);
#pragma unroll
                            for (int pc = 0; pc < 8; pc++) cp_async16(dst + 4 * (pc ^ (u & 7)), src + 4 * pc);
                        } else {
                            const float* sp = sbase + (i64)(-ref - 2) * NC;
#pragma unroll
                            for (int w = 0; w < 32; w++)
                                dst[w ^ ((u & 7) << 2)] = pack_bf16x2(__ldg(sp + 2 * w), __ldg(sp + 2 * w + 1));
                        }
                        const float rad = __ldg(pa), amax = __ldg(pa + 1), pmin = __ldg(pa + 2);
                        const float pmin_pos = f_mul(f_add(pmin, 0.5f), cell_size);
                        const float dx = __int_as_float(__float_as_int(f_sub(qpos_blk, pmin_pos)) & 0x7fffffff);
                        s_ux[u] = make_float2(f_mul(rad, scale), f_mul(f_mul(lam, amax), erf_tl(f_div_full(dx, sig2))));
                    }
                    um_nxt[u] = make_float4(__int_as_float(ref), cnt, chl, chr);
                    code_nxt[u] = (unsigned char)node_code(cnt, chl, chr, bud_blk);
                } else if (any_sel) {
#pragma unroll
                    for (int pc = 0; pc < 8; pc++) *reinterpret_cast<uint4*>(dst + 4 * pc) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        PMARK(3);
        __syncthreads();                                               // B2: the union is staged
        PMARK(4);
        // each expanded entry's children sit at union slots base, base + 1 (its parent's s_rank)
        unsigned bv_lo = 0u, bv_hi = 0u;
        if (!done) {
#pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                bv_lo |= (unsigned)s_rank[byte8(o_lo, o_hi, jj) & (UC2 - 1)] << (8 * jj);
                bv_hi |= (unsigned)s_rank[byte8(o_lo, o_hi, jj + 4) & (UC2 - 1)] << (8 * jj);
            }
        }
        if (!any_sel) {
            // ---- C0: no query selects: every child is kept, in child order (fsb_beam_rows' block-wide keep-all
            // level), scattered through the warp's dot rows
            unsigned char* no = reinterpret_cast<unsigned char*>(s_dot + wid * (16 * (DCH / 2))) + ql * 16;
            if (!done) {
                const int pbase = h ? __popc(lo & 0xccccccccu) : 0;
#pragma unroll
                for (int jj = 0; jj < 8; jj++) {
                    const unsigned two = (pk >> (4 * jj + 2)) & 3u;
                    const int pos = pbase + __popc(pk & 0xccccccccu & ((1u << (4 * jj + 2)) - 1u));
                    const int base = byte8(bv_lo, bv_hi, jj);
                    if (two & 1u) no[pos] = (unsigned char)base;
                    if (two & 2u) no[pos + (int)(two & 1u)] = (unsigned char)(base + 1);
                }
            }
            __syncwarp();
            if (!done) {
                const uint2 v = *reinterpret_cast<const uint2*>(no + 8 * h);
                o_lo = v.x; o_hi = v.y;
                nopen = n;
            }
            PMARK(7);
        } else {
            // ---- C: the warp's dots with the union, DCH slots at a time (bf16 after the scale, as fsb_beam_rows), and
            // the keys of this lane's children (frontier slots 8 h + jj, left then right; a slot's two children are
            // one dot pair and one pair of score terms); D: the pair's ranks 0-15 (pair_top16) and the new open entries
            *reinterpret_cast<uint2*>(s_cs + qi * 16 + 8 * h) = make_uint2(bv_lo, bv_hi);
            unsigned* s_dw = s_dot + wid * (16 * (DCH / 2));
            const float4* s_ux2 = reinterpret_cast<const float4*>(s_ux);
            if (__any_sync(FULL, !done)) {
                unsigned nlo, nhi;
                int nv;
#if KEY64
                u64 r8[8];
                warp_select(r8, s_mu, s_dw, s_ux2, qa, bv_lo, bv_hi, pk, qn, scale, Pc, lane, h);
                PMARK(5);
                nv = open_slots(r8, s_cs + qi * 16, k, h, nlo, nhi);
#else
                unsigned r8[8];
                const unsigned r16 = warp_select(r8, s_mu, s_dw, s_ux2, qa, bv_lo, bv_hi, pk, qn, scale, Pc, lane, h);
                PMARK(5);
                bool redo = false;
#if EXACT
                // the 32-bit ranks 0..k are fsb_beam_rows' unless two of them share their score bits (ranks r, r + 1
                // of real children, keys >= 32): then the warp selects again with 64-bit keys. Ranks 7 / 8 and
                // 15 / 16 are checked by lane 1.
                const unsigned r7 = __shfl_xor_sync(FULL, r8[7], 1);
                bool tie = false;
#pragma unroll
                for (int j = 0; j < 7; j++) tie |= 8 * h + j + 1 <= k && r8[j] >= 32u && (r8[j] ^ r8[j + 1]) < 32u;
                if (h) {
                    tie |= k >= 8 && r7 >= 32u && (r7 ^ r8[0]) < 32u;
                    tie |= k >= 16 && r8[7] >= 32u && (r8[7] ^ r16) < 32u;
                }
                redo = __any_sync(FULL, tie);
#ifdef PROF
                if (lane == 0 && FBC != nullptr) {                     // selecting warp-levels, and those redone
                    atomicAdd(FBC + 121, 1);
                    if (redo) atomicAdd(FBC + 120, 1);
                }
#endif
                if (redo) {
                    u64 x8[8];
                    warp_select(x8, s_mu, s_dw, s_ux2, qa, bv_lo, bv_hi, pk, qn, scale, Pc, lane, h);
                    nv = open_slots(x8, s_cs + qi * 16, k, h, nlo, nhi);
                }
#endif
                if (!redo) nv = open_slots(r8, s_cs + qi * 16, k, h, nlo, nhi);
                (void)r16;
#endif
                const int nvo = __shfl_xor_sync(FULL, nv, 1);
                if (!done) {
                    o_lo = nlo; o_hi = nhi;
                    nopen = nv + nvo;
                }
            }
            PMARK(6);
        }
        ub ^= 1;
    }
    if (fb_lev < 0) {
        // ---- E: cells of the final frontiers: the carried prefix, then any open entries the level limit left; each
        // lane reads src / off / len of its slots 8 h .. 8 h + 7 and their cells, stored as 16-byte pairs
        if (!done) {
#pragma unroll
            for (int jj = 0; jj < 8; jj++)
                if (8 * h + jj < nopen)
                    s_car[qi * 16 + ncar + 8 * h + jj] = __float_as_int(s_um[ub * UC2 + byte8(o_lo, o_hi, jj)].x);
        }
        __syncwarp();
        const int nfr = ncar + nopen;
        int ref[8];
        float src[8], off[8], ln[8];
#pragma unroll
        for (int jj = 0; jj < 8; jj++) {
            const int w = 8 * h + jj;
            ref[jj] = (qok && w < nfr) ? s_car[qi * 16 + w] : -1;
            src[jj] = 0.0f; off[jj] = 0.0f; ln[jj] = 0.0f;
            if (ref[jj] != -1) {
                const float* p = aux_row(cax, sbase, ref[jj]);
                src[jj] = __ldg(p + 6); off[jj] = __ldg(p + 7); ln[jj] = __ldg(p + 8);
            }
        }
        if (qok) {
            // cells 8 R h .. 8 R h + 8 R - 1 of the query's row: slot 8 h + e / R, cell e % R of the node
            longlong2* dst = reinterpret_cast<longlong2*>(CELLS + (pid * 16 + 8 * h) * R);
#pragma unroll
            for (int i = 0; i < 4 * R; i++) {
                i64 cv[2];
#pragma unroll
                for (int z = 0; z < 2; z++) {
                    const int jj = (2 * i + z) / R, tr = (2 * i + z) % R;
                    cv[z] = -1;
                    if (ref[jj] != -1 && src[jj] > 0.0f && (float)tr < ln[jj]) {
                        const i64 pos = (i64)off[jj] + tr;
                        cv[z] = src[jj] == 1.0f ? __ldg(leafb + min(max(pos, (i64)0), Ltot - 1))
                                                : __ldg(admb + min(max(pos, (i64)0), Mm - 1));
                    }
                }
                dst[i] = make_longlong2(cv[0], cv[1]);
            }
            if (h == 0) NVIS[pid] = (float)nvis;
        }
        PMARK(8);
    } else {
        // a level's union outgrew UC2 slots: every query restarts that level in the warp-per-query descent, from its
        // frontier at the level's start (the carried prefix as CAR0 + ref, then the open entries as slots of the
        // union in um_cur, whose node fields are staged), with its norm and visits so far
        if (FBC != nullptr && tid == 0) {                             // test / diagnostic counters (null by default)
            atomicAdd(FBC, 1);
            atomicAdd(FBC + 1 + min(fb_lev, 63), 1);
        }
        int* fr = s_car + qi * 16;
        if (!done0) {
#pragma unroll
            for (int jj = 0; jj < 8; jj++)
                if (8 * h + jj < nopen0) fr[ncar0 + 8 * h + jj] = byte8(o_lo, o_hi, jj);
        }
        __syncwarp();
#pragma unroll
        for (int jj = 0; jj < 8; jj++) {
            const int w = 8 * h + jj;
            if (w < ncar0) fr[w] = CAR0 + fr[w];
            else if (done0 || w >= ncar0 + nopen0) fr[w] = EMPTY_E;
        }
        if (h == 0) {
            reinterpret_cast<int*>(s_cs + qi * 16)[0] = nvis0;
            reinterpret_cast<float*>(s_cs + qi * 16)[1] = qn;
        }
        __syncthreads();
        WarpS* ws = reinterpret_cast<WarpS*>(smem + OFF2_MU) + wid;
        const float4* um_prev = s_um + ub * UC2;
        for (int kq = 0; kq < QB / NWARP_R; kq++) {
            const int q2 = wid * (QB / NWARP_R) + kq;
            if (q0 + q2 >= Bq) break;
            const i64 p2 = t * Bq + q0 + q2;
            const bool pq_ok = q0 + q2 < NQ;
            const unsigned short* q2p = reinterpret_cast<const unsigned short*>(Q) + t * QST_T + (i64)(q0 + q2) * QST_B;
            const float* qrow_p = pq_ok ? reinterpret_cast<const float*>(q2p) : fsb_qzero;
            const int slot = lane >> 1;
            continue_query(*ws, lane, slot < WM ? s_car[q2 * 16 + slot] : EMPTY_E, um_prev, qrow_p,
                           BOUND ? reinterpret_cast<const float*>(s_cs + q2 * 16)[1] : 0.0f,
                           reinterpret_cast<const int*>(s_cs + q2 * 16)[0], QPOS[p2], lam, sig2, (float)BUDGET[p2],
                           WBEAM[p2], fb_lev, KMAXP1, scale, cell_size, cmub, cax, sbase, leafb, admb, Ltot, Mm,
                           CELLS + p2 * (WM * R), NVIS + p2);
        }
        PMARK(9);
    }
#ifdef PROF
    if (lane == 0 && FBC != nullptr) {
        unsigned long long* pdst = reinterpret_cast<unsigned long long*>(FBC + 256);
        for (int kb = 0; kb < 16; kb++) atomicAdd(pdst + wid * 16 + kb, pacc[kb]);
    }
#endif
}
// >>> SECTION: _SPINES_KERNEL <<<

// ---- the fast prefill stage's spines and beam row tables in one launch (FSB_BEAM_V2=1): build_spines of each beam's
// split forest (fsb_build_fast._spine_kernel) and fsb_prefill._row_tables_kernel, bitwise. One block of two warps per
// stage row, warp f building forest f's spine; both then write the row's tables.

__device__ __forceinline__ float f_div_rn_ftz(float a, float b) {
    float r; asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_max(float a, float b) {
    float r; asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_min(float a, float b) {
    float r; asm("min.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }

// the row's spine, _spine_kernel instruction for instruction (mul.rn / add.rn / sub.rn, div.rn.ftz, the norms'
// pairwise halving over the 64 columns, sqrt.rn.ftz, max / min): the chunk roots selected by the bits of m merge low
// bit first; row k of sp (NC floats) is the merge at height k, or zeros with children -1 where nothing merges. One
// warp, lane l holding columns l and l + 32; lane k < S loads chunk root k's fields and stages its mean in stg (S rows
// of 32 words), so the merges wait on no load. -> the root ref (-1 for m = 0) and the height
__device__ void row_spine(float* sp, unsigned* stg, const i64 m, const unsigned short* cmub, const float* cax,
                          const i64 M, const i64 Ntot, const int S, const int lane, i64& root, i64& height) {
    i64 croot = 0, off = 0;
    bool present = false, bit = false;
    float c_rad = 0.0f, c_am = 0.0f, c_pm = 0.0f, c_cnt = 0.0f;
    if (lane < S) {
        i64 base = 0;                                                   // the nodes of the chunks below height lane
        for (int k = 0; k < lane; k++) base += (M >> k) * ((2LL << k) - 1);
        off = m & ~((2LL << lane) - 1);
        croot = base + (off >> lane) * ((2LL << lane) - 1);
        bit = ((m >> lane) & 1) != 0;
        present = bit && (M >> lane) > 0;
        const i64 gi = min(max(croot, (i64)0), Ntot - 1);
        const float* pa = cax + gi * NA;
        c_rad = __ldg(pa); c_am = __ldg(pa + 1); c_pm = __ldg(pa + 2); c_cnt = __ldg(pa + 3);
        const uint4* src = reinterpret_cast<const uint4*>(cmub + gi * DH);
#pragma unroll
        for (int pc = 0; pc < 8; pc++) reinterpret_cast<uint4*>(stg + lane * 32)[pc] = __ldg(src + pc);
    }
    __syncwarp();
    float r0 = 0.0f, r1 = 0.0f, r_rad = 0.0f, r_am = 0.0f, r_pm = 1073741824.0f, r_cnt = 0.0f;
    i64 r_ref = -1;
    int pop = 0, hb = 0;
    for (int k = 0; k < S; k++) {
        const float k_rad = __shfl_sync(FULL, c_rad, k), k_am = __shfl_sync(FULL, c_am, k);
        const float k_pm = __shfl_sync(FULL, c_pm, k), k_cnt = __shfl_sync(FULL, c_cnt, k);
        const i64 k_croot = __shfl_sync(FULL, croot, k), k_off = __shfl_sync(FULL, off, k);
        const bool k_bit = __shfl_sync(FULL, (int)bit, k) != 0, k_present = __shfl_sync(FULL, (int)present, k) != 0;
        const unsigned short* row16 = reinterpret_cast<const unsigned short*>(stg + k * 32);
        const float c0 = __uint_as_float((unsigned)row16[lane] << 16);
        const float c1 = __uint_as_float((unsigned)row16[lane + 32] << 16);
        const bool mk = k_present && r_cnt > 0.0f;
        const float tot = f_max(f_add(k_cnt, r_cnt), 1.0f);
        const float n0 = f_div_rn_ftz(f_add(f_mul(k_cnt, c0), f_mul(r_cnt, r0)), tot);
        const float n1 = f_div_rn_ftz(f_add(f_mul(k_cnt, c1), f_mul(r_cnt, r1)), tot);
        const float a0 = f_sub(c0, n0), a1 = f_sub(c1, n1), b0 = f_sub(r0, n0), b1 = f_sub(r1, n1);
        float sa = f_add(f_mul(a0, a0), f_mul(a1, a1)), sb = f_add(f_mul(b0, b0), f_mul(b1, b1));
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            sa = f_add(sa, __shfl_xor_sync(FULL, sa, o));
            sb = f_add(sb, __shfl_xor_sync(FULL, sb, o));
        }
        const float n_rad = f_max(f_add(k_rad, f_sqrt_ftz(sa)), f_add(r_rad, f_sqrt_ftz(sb)));
        const float n_am = f_max(k_am, r_am), n_pm = f_min(k_pm, r_pm), n_cnt = f_add(k_cnt, r_cnt);
        float* row = sp + k * NC;
        row[lane] = mk ? n0 : 0.0f;
        row[lane + 32] = mk ? n1 : 0.0f;
        if (lane < 9) {
            const float v = lane == 0 ? n_rad : lane == 1 ? n_am : lane == 2 ? n_pm : lane == 3 ? n_cnt
                          : lane == 4 ? (float)k_croot : lane == 5 ? (float)r_ref : lane == 6 ? 2.0f
                          : lane == 7 ? (float)k_off : n_cnt;
            row[DH + lane] = mk ? v : ((lane == 4 || lane == 5) ? -1.0f : 0.0f);
        }
        r0 = mk ? n0 : (k_present ? c0 : r0);
        r1 = mk ? n1 : (k_present ? c1 : r1);
        r_rad = mk ? n_rad : (k_present ? k_rad : r_rad);
        r_am = mk ? n_am : (k_present ? k_am : r_am);
        r_pm = mk ? n_pm : (k_present ? k_pm : r_pm);
        r_cnt = mk ? n_cnt : (k_present ? k_cnt : r_cnt);
        r_ref = mk ? (i64)(-(k + 2)) : (k_present ? k_croot : r_ref);
        pop += k_present ? 1 : 0;
        hb = k_bit ? k : hb;
    }
    root = m > 0 ? r_ref : -1;
    height = m > 1 ? (i64)(hb + (pop >= 2 ? 1 : 0)) : 0;
}

// one forest's stage spines: m_b and head index per row, the split node tables, the outputs (spine rows (rows, S, NC),
// root, height); on = 0: no such forest
struct SpinesP {
    const i64* MB; const i64* TIDX; const unsigned short* CHUNK; const float* CHUNK_AX; float* SPN; i64* ROOT;
    i64* HGT; i64 Ntot; i64 M; int S; int on;
};

extern "C" __global__ void __launch_bounds__(64)
fsb_spines2(const SpinesP P0, const SpinesP P1, const i64* __restrict__ HEAD, const float* __restrict__ SELS,
            const float* __restrict__ SIGMA, i64* __restrict__ W1, i64* __restrict__ B1, i64* __restrict__ W2,
            i64* __restrict__ B2, float* __restrict__ LAM, float* __restrict__ SIG, const int Bq, const int delta,
            const int width, const int uwidth, const int has_u) {
    __shared__ __align__(16) unsigned stg[2][32 * 32];
    __shared__ int hts[2];
    const i64 r = blockIdx.x;
    const int lane = threadIdx.x & 31, f = threadIdx.x >> 5;
    const SpinesP P = f ? P1 : P0;
    if (P.on) {
        const i64 tidx = P.TIDX[r];
        i64 root, height;
        row_spine(P.SPN + r * (i64)P.S * NC, stg[f], P.MB[r], P.CHUNK + tidx * P.Ntot * DH,
                  P.CHUNK_AX + tidx * P.Ntot * NA, P.M, P.Ntot, P.S, lane, root, height);
        if (lane == 0) {
            P.ROOT[r] = root;
            P.HGT[r] = height;
            hts[f] = (int)height;
        }
    }
    __syncthreads();
    // _row_tables_kernel's integer formulas: the main beam D = max(h - delta, 0), width min(width, 2^D), budget
    // 2^max(h - D, 0); the second beam with its own forest's height hu (du = hu - max(hu - delta, 0)) or the main
    // beam's (du = its exponent), Du = max(hu - du, 0), width min(uwidth, 2^Du), budget 2^du
    const int h1 = hts[0];
    const int D = max(h1 - delta, 0), dl = max(h1 - D, 0);
    const i64 w1 = min((i64)width, (i64)1 << D), b1 = (i64)1 << dl;
    i64 w2 = 0, b2 = 0;
    if (has_u) {
        const int hu = P1.on ? hts[1] : h1;
        const int du = P1.on ? hu - max(hu - delta, 0) : dl;
        const int Du = max(hu - du, 0);
        w2 = min((i64)uwidth, (i64)1 << Du);
        b2 = (i64)1 << du;
    }
    for (int qq = threadIdx.x; qq < Bq; qq += 64) {
        W1[r * Bq + qq] = w1;
        B1[r * Bq + qq] = b1;
        if (has_u) {
            W2[r * Bq + qq] = w2;
            B2[r * Bq + qq] = b2;
        }
    }
    if (threadIdx.x == 0) {
        const i64 hd = HEAD[r];
        LAM[r] = SELS[hd];
        SIG[r] = SIGMA[hd];
    }
}

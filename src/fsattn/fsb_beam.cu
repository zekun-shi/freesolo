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

__device__ __forceinline__ const float* node_at(const float* cbase, const float* sbase, const int r) {
    return r >= 0 ? cbase + (i64)r * NC : sbase + (i64)(-r - 2) * NC;
}

// onebeam table descriptor for the paired kernels (passed by value)
struct BeamP {
    const i64* ROOT; i64* CELLS; float* NVIS; const i64* WBEAM; const i64* BUDGET; const i64* TIDX;
    const float* CHUNK; const float* SPINE; const i64* LEAF; const i64* ADM;
    i64 Ntot; i64 Sp; i64 Ltot; i64 Mm; i64 rows; int KMAXP1; int pad_;
};

// per-warp scratch of the warp-per-query descent
struct __align__(16) WarpS {
    float sc[32];
    float cnt[32], chl[32], chr[32];
    const float* ptr[32];
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
    const float* pc = S.ptr[c];
    const float rad = __ldg(pc + DH), amax = __ldg(pc + DH + 1), pmin = __ldg(pc + DH + 2);
    const float cnt = __ldg(pc + DH + 3), chl = __ldg(pc + DH + 4), chr = __ldg(pc + DH + 5);
    float v[CH];
#pragma unroll
    for (int i = 0; i < CH; i++) {
        const float* p = S.ptr[i];
        float x[PER];
#pragma unroll
        for (int k = 0; k < PER; k++) {
            const int col = lane + 32 * k;
            x[k] = f_mul(qv[k], (col < DH) ? __ldg(p + col) : 0.0f);
        }
        v[i] = lane_halve(x);
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
                                             const float* cbase, const float* sbase, const i64* leafb,
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
            S.ptr[pos] = node_at(cbase, sbase, child);
            S.cref[pos] = child;
        }
        if (lane >= n) S.ptr[lane] = g_zero;
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
            const float* p = node_at(cbase, sbase, ref);
            const float src = p[DH + 6], off = p[DH + 7], ln = p[DH + 8];
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
                                           const float* cbase, const float* sbase, const i64* leafb, const i64* admb,
                                           const i64 Ltot, const i64 Mm, i64* out, float* nvis_out) {
    float qv[PER], x2[PER];
#pragma unroll
    for (int k = 0; k < PER; k++) {
        const int col = lane + 32 * k;
        qv[k] = (col < DH) ? qp[col] : 0.0f;
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
            const float* p = node_at(cbase, sbase, f_ref);
            f_cnt = p[DH + 3]; f_chl = p[DH + 4]; f_chr = p[DH + 5];
        }
    }
    warp_descend(S, lane, qv, qn, qpos, lam, sig2, bud, wbeam, f_ref, f_cnt, f_chl, f_chr, 0, 0, KMAXP1, scale,
                 cell_size, cbase, sbase, leafb, admb, Ltot, Mm, out, nvis_out);
}
// >>> SECTION: _WARP_KERNEL <<<

extern "C" __global__ void __launch_bounds__(NWARP * 32)
fsb_beam_cuda(const i64* __restrict__ ROOT, i64* __restrict__ CELLS, float* __restrict__ NVIS,
              const float* __restrict__ Q, const float* __restrict__ RC, const float rc_s,
              const float* __restrict__ QPOS, const float* __restrict__ LAM, const float* __restrict__ SIG,
              const i64* __restrict__ WBEAM, const i64* __restrict__ BUDGET, const i64* __restrict__ TIDX,
              const float* __restrict__ CHUNK, const float* __restrict__ SPINE,
              const i64* __restrict__ LEAF, const i64* __restrict__ ADM,
              const i64 Ntot, const i64 Sp, const i64 Ltot, const i64 Mm, const i64 Bq, const i64 nq,
              const float scale, const float cell_size, const int KMAXP1) {
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
    warp_query(ws[wid], lane, Q + pid * DH, rcq, QPOS[pid], LAM[t], f_mul(2.0f, SIG[t]), (float)BUDGET[pid], WBEAM[pid],
               (int)ROOT[t], KMAXP1, scale, cell_size, CHUNK + tidx * Ntot * NC, SPINE + t * Sp * NC,
               LEAF + tidx * Ltot, ADM + tidx * Mm, Ltot, Mm, CELLS + pid * (WM * R), NVIS + pid);
}

extern "C" __global__ void __launch_bounds__(32)
fsb_beam_pair_cuda(const BeamP B0, const BeamP B1,
                   const float* __restrict__ Q, const float* __restrict__ RC, const float rc_s,
                   const float* __restrict__ QPOS, const float* __restrict__ LAM, const float* __restrict__ SIG,
                   const i64 Bq, const i64 nq, const float scale, const float cell_size) {
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
    warp_query(ws, lane, Q + pid * DH, rcq, QPOS[pid], LAM[t], f_mul(2.0f, SIG[t]), (float)B.BUDGET[pid], B.WBEAM[pid],
               (int)B.ROOT[t], B.KMAXP1, scale, cell_size, B.CHUNK + tidx * B.Ntot * NC, B.SPINE + t * B.Sp * NC,
               B.LEAF + tidx * B.Ltot, B.ADM + tidx * B.Mm, B.Ltot, B.Mm, B.CELLS + pid * (WM * R), B.NVIS + pid);
}
// >>> SECTION: _ROWS_KERNEL <<<

// ---- row-cooperative prefill descent: one block per QB queries of a (beam, row), two lanes per query (lane h: columns
// 2 k + h)
#define NT (2 * QB)
#define NWARP_R (NT / 32)
#define SCS 36                                    // score row stride (floats): conflict-free float4 rows
#define EMPTY_E ((int)0x80000000)
#define CAR0 (UMAX + 64)                          // carried frontier entry: CAR0 + node ref (spine refs >= -64)

static_assert(NWARP_R * sizeof(WarpS) <= OFF_UM, "per-warp scratch of the fallback must fit the tile region");
static_assert(UMAX % 32 == 0 && UMAX <= NT, "one thread per union slot in the parent ranking");

#ifdef PROF                                       // phase cycle counters per (block, warp) (profiling builds only)
__device__ unsigned long long g_prof[PROF][8][12];
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

__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }

__device__ __forceinline__ void prefetch_node(const float* p, const int ref) {    // the node's 292 bytes
    if (ref != -1) {
        const char* c = reinterpret_cast<const char*>(p);
        prefetch_l2(c); prefetch_l2(c + 128); prefetch_l2(c + 256);
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

// this lane's half of the dot of the query with the tile node at shared address b = node + h * 128 + rot * 16 (node
// 256-byte aligned): block m reads chunk m ^ rot at b ^ (16 m); the 8 lanes of a phase (rot = lane & 7) hit 8 distinct
// bank quads whatever their nodes
__device__ __forceinline__ float half_dot(const float* qr, const unsigned b) {
    float x[32];
#pragma unroll
    for (int m = 0; m < 8; m++) {
        const float4 v = lds128(b ^ (16u * m));
        x[4 * m] = f_mul(qr[4 * m], v.x); x[4 * m + 1] = f_mul(qr[4 * m + 1], v.y);
        x[4 * m + 2] = f_mul(qr[4 * m + 2], v.z); x[4 * m + 3] = f_mul(qr[4 * m + 3], v.w);
    }
    return halve32(x);
}

// stage union slots [u0, u0 + nt) of buffer b, one warp per node: mu split into even / odd columns (the tile), rad /
// amax / pmin, and cnt / children for the next level (cp.async, 4 bytes each; slots without a node are skipped: no
// query has them as a child)
__device__ __forceinline__ void stage_tile(const int u0, const int nt, const int b, const int lane, const int wid,
                                           float4* s_umeta, float4* s_um2, float* s_mu, const float* cbase,
                                           const float* sbase) {
    const int c0 = (lane & 1) * 32 + (lane >> 1);                      // tile column of node column `lane`
    for (int ul = wid; ul < nt; ul += NWARP_R) {
        const int u = u0 + ul;
        const int ref = __float_as_int(s_umeta[b * UMAX + u].x);
        if (ref != -1) {
            const float* p = node_at(cbase, sbase, ref);
            float* d = s_mu + ul * 64;
            cp_async4(d + c0, p + lane);
            cp_async4(d + c0 + 16, p + lane + 32);
            if (lane < 6)
                cp_async4(lane < 3 ? reinterpret_cast<float*>(s_um2 + u) + lane
                                   : reinterpret_cast<float*>(s_umeta + b * UMAX + u) + lane - 2, p + DH + lane);
        }
    }
}

// rank of this lane's children c = 2 k + h among the query's n children by (score desc, slot asc), i.e.
// #{m < c: s_m >= s_c} + #{m > c: s_m > s_c} (scores past n are -inf), and the frontier entry of each child kept
template <int NB>
__device__ __forceinline__ void rank_write(const float* srow, const short* curow, int* frow, const int n, const int h,
                                           const int ncar, const i64 wbeam) {
    float s[NB];
#pragma unroll
    for (int m = 0; m < NB; m += 4) {
        const float4 v = *reinterpret_cast<const float4*>(srow + m);
        s[m] = v.x; s[m + 1] = v.y; s[m + 2] = v.z; s[m + 3] = v.w;
    }
#pragma unroll
    for (int k = 0; k < NB / 2; k++) {
        const int c = 2 * k + h;
        if (c < n) {
            const float sc = h ? s[2 * k + 1] : s[2 * k];
            int r = h ? (int)(s[2 * k] >= sc) : (int)(s[2 * k + 1] > sc);
#pragma unroll
            for (int m = 0; m < NB; m++) {
                if (m < 2 * k) r = add_ge(r, s[m], sc);
                else if (m > 2 * k + 1) r = add_gt(r, s[m], sc);
            }
            const int crank = ncar + r;
            if (sc > -5e8f && crank < wbeam && crank < WM) frow[crank] = (int)curow[c];
        }
    }
}

// the fallback's continuation of one query from level lev0: its frontier entry for this lane's slot (a union slot of
// the previous level, whose staged node fields um_prev holds, a carried CAR0 + node ref, or EMPTY_E), then the warp
// descent. Out of line: inlined, its registers weigh on the cooperative descent (1-2% slower without any fallback)
__device__ __noinline__ void continue_query(WarpS& S, const int lane, const int ev, const float4* um_prev,
                                            const float* qrow, const float qn, const int nvis, const float qpos,
                                            const float lam, const float sig2, const float bud, const i64 wbeam,
                                            const int lev0, const int KMAXP1, const float scale, const float cell_size,
                                            const float* cbase, const float* sbase, const i64* leafb, const i64* admb,
                                            const i64 Ltot, const i64 Mm, i64* out, float* nvis_out) {
    int f_ref = -1;
    float f_cnt = 0.0f, f_chl = -1.0f, f_chr = -1.0f;
    if (ev >= 0 && ev < UMAX) {
        const float4 m = um_prev[ev];
        f_ref = __float_as_int(m.x); f_cnt = m.y; f_chl = m.z; f_chr = m.w;
    } else if (ev != EMPTY_E) {
        f_ref = ev - CAR0;
        const float* p = node_at(cbase, sbase, f_ref);
        f_cnt = p[DH + 3]; f_chl = p[DH + 4]; f_chr = p[DH + 5];
    }
    float qv[PER];                                                    // columns lane + 32 k (DH == DHP == 64)
#pragma unroll
    for (int k = 0; k < PER; k++) qv[k] = qrow[lane + 32 * k];
    warp_descend(S, lane, qv, qn, qpos, lam, sig2, bud, wbeam, f_ref, f_cnt, f_chl, f_chr, lev0, nvis, KMAXP1, scale,
                 cell_size, cbase, sbase, leafb, admb, Ltot, Mm, out, nvis_out);
}

extern "C" __global__ void __launch_bounds__(NT, MINB)
fsb_beam_rows(const BeamP B0, const BeamP B1, const float* __restrict__ Q, const float* __restrict__ QPOS,
              const float* __restrict__ RC, const float rc_s, const float* __restrict__ LAM,
              const float* __restrict__ SIG, const int Bq, const float scale, const float cell_size,
              int* __restrict__ FBC) {
    extern __shared__ __align__(128) unsigned char smem[];
    float* s_mu = reinterpret_cast<float*>(smem + OFF_MU);          // [CAP][64] tile: even, then odd columns
    float4* s_umeta = reinterpret_cast<float4*>(smem + OFF_UM);     // [2][UMAX] union: ref (bits), cnt, children
    float4* s_um2 = reinterpret_cast<float4*>(smem + OFF_UM2);      // [UMAX] the level's union: rad, amax, pmin
    unsigned char* s_mark = smem + OFF_MARK;                        // [2][UMAX] parents expanded at a level
    short* s_rank = reinterpret_cast<short*>(smem + OFF_RANK);      // [UMAX] their rank
    int* s_front = reinterpret_cast<int*>(smem + OFF_FRONT);        // [QB][16] frontier entries
    float* s_sc = reinterpret_cast<float*>(smem + OFF_SC);          // [QB][SCS] child dots, then scores
    short* s_cu = reinterpret_cast<short*>(smem + OFF_CU);          // [QB][32] child union slots
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
    const float* CHUNK = BSEL(CHUNK);
    const float* SPINE = BSEL(SPINE);
    const i64* LEAF = BSEL(LEAF);
    const i64* ADM = BSEL(ADM);
    const i64 Ntot = BSEL(Ntot), Sp = BSEL(Sp), Ltot = BSEL(Ltot), Mm = BSEL(Mm);
    const int KMAXP1 = BSEL(KMAXP1);
#undef BSEL
    const i64 tidx = TIDX[t];
    const float* cbase = CHUNK + tidx * Ntot * NC;
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
    const i64 pid = t * Bq + (qok ? q0 + qi : 0);
    // this lane's half of the query (chunk blocks in its rotated order) and the radius-scaled norm (same halving tree)
    float qr[32];
    {
        const float4* qp = reinterpret_cast<const float4*>(Q + pid * DH);  // block m: columns 8 (m ^ rot) .. + 8
#pragma unroll
        for (int m = 0; m < 8; m++) {
            const int mc = m ^ rot;
            const float4 a = qok ? __ldg(qp + 2 * mc) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 b = qok ? __ldg(qp + 2 * mc + 1) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            qr[4 * m] = h ? a.y : a.x; qr[4 * m + 1] = h ? a.w : a.z;
            qr[4 * m + 2] = h ? b.y : b.x; qr[4 * m + 3] = h ? b.w : b.z;
        }
    }
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
    short* curow = s_cu + qi * 32;
    float* srow = s_sc + qi * SCS;
    // the union before level 0 is the root (buffer 1); every query starts from it
    if (tid == 0 && root != -1) {
        const float* p = node_at(cbase, sbase, root);
        s_umeta[UMAX] = make_float4(__int_as_float(root), p[DH + 3], p[DH + 4], p[DH + 5]);
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
        // ---- C: stage the first tile of the union (cp.async) while this query lists its children (slot order) as union
        // slots and fills -inf past n; then prefetch the next level's candidates
        const int npass = (Pc + CAP - 1) / CAP;
        stage_tile(0, min(CAP, Pc), cb, lane, wid, s_umeta, s_um2, s_mu, cbase, sbase);
        PMARK(8);
        if (n > 0) {
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
            for (int c = n + h; c < 32; c += 2) srow[c] = NEG_INF;
        }
        PMARK(9);
        cp_async_wait_all();
        __syncthreads();                                              // B3: tile 0 staged, children listed
        PMARK(10);
        for (int u = tid; u < min(CAP, Pc); u += NT) {               // candidates of the next union -> L2
            const float4 um = s_umeta[cb * UMAX + u];
            if (__float_as_int(um.x) != -1 && um.y > 1.0f) {
                prefetch_node(node_at(cbase, sbase, fchild(um.z)), fchild(um.z));
                prefetch_node(node_at(cbase, sbase, fchild(um.w)), fchild(um.w));
            }
        }
        PMARK(4);
        // ---- D: child dots, two at a time: each lane sums its half of both, the halves are swapped and lane h keeps
        // child c + h's dot. A union larger than the tile takes several passes.
        const unsigned tb = (unsigned)__cvta_generic_to_shared(s_mu) + 128u * h + 16u * rot;
        if (npass == 1) {
            for (int c = 0; c < n; c += 2) {
                const int uu = *reinterpret_cast<const int*>(curow + c);
                const int ua = (int)(short)(uu & 0xffff), ub = c + 1 < n ? (uu >> 16) : ua;
                const float sa = half_dot(qr, tb + 256u * ua);
                const float sb = half_dot(qr, tb + 256u * ub);
                const float dot = f_add(h ? sb : sa, __shfl_xor_sync(pmask, h ? sa : sb, 1));
                if (h == 0 || c + 1 < n) srow[c + h] = dot;
            }
        } else {
            for (int p = 0; p < npass; p++) {
                const int t0 = p * CAP, t1 = min(t0 + CAP, Pc);
                if (p > 0) {
                    __syncthreads();
                    stage_tile(t0, t1 - t0, cb, lane, wid, s_umeta, s_um2, s_mu, cbase, sbase);
                    cp_async_wait_all();
                    __syncthreads();
                }
                for (int c = 0; c < n; c += 2) {
                    const int ua = curow[c];
                    const bool hb = c + 1 < n;
                    const int ub = hb ? (int)curow[c + 1] : ua;
                    const bool ina = ua >= t0 && ua < t1, inb = hb && ub >= t0 && ub < t1;
                    if (!ina && !inb) continue;
                    const float sa = half_dot(qr, tb + 256u * (ina ? ua - t0 : 0));
                    const float sb = half_dot(qr, tb + 256u * (inb ? ub - t0 : 0));
                    const float dot = f_add(h ? sb : sa, __shfl_xor_sync(pmask, h ? sa : sb, 1));
                    if (h ? inb : ina) srow[c + h] = dot;
                }
            }
        }
        // ---- D2: scores of this lane's children from their dots
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
        PMARK(5);
        // ---- E: the new frontier: carried slots keep their order (positions below the beam width), the children
        // follow at ncar + rank
        if (!done) {
            __syncwarp(pmask);
#pragma unroll
            for (int j = 0; j < 8; j++) frow[8 * h + j] = EMPTY_E;
            __syncwarp(pmask);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int w = 8 * h + j;
                if ((C >> w) & 1u) {
                    const int ncb = __popc(C & ((1u << w) - 1u));
                    if (ncb < wbeam && ncb < WM) frow[ncb] = e[j];
                }
            }
            if (n > 16) rank_write<32>(srow, curow, frow, n, h, ncar, wbeam);
            else if (n > 8) rank_write<16>(srow, curow, frow, n, h, ncar, wbeam);
            else if (n > 0) rank_write<8>(srow, curow, frow, n, h, ncar, wbeam);
            __syncwarp(pmask);
            nvis += n;
        }
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
                const float* p = node_at(cbase, sbase, ref[j]);
                src[j] = __ldg(p + DH + 6); off[j] = __ldg(p + DH + 7); ln[j] = __ldg(p + DH + 8);
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
        if (lane == 0)
            for (int k = 0; k < 12; k++) g_prof[blockIdx.x][wid][k] = pacc[k];
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
            const float* sq = s_sc + ql * SCS;
            continue_query(*ws, lane, slot < WM ? s_front[ql * 16 + slot] : EMPTY_E, s_umeta + pbf * UMAX,
                           Q + p2 * DH, BOUND ? sq[1] : 0.0f, reinterpret_cast<const int*>(sq)[0], QPOS[p2], lam, sig2,
                           (float)BUDGET[p2], WBEAM[p2], fb_lev, KMAXP1, scale, cell_size, cbase, sbase, leafb, admb,
                           Ltot, Mm, CELLS + p2 * (WM * R), NVIS + p2);
        }
    }
}
// >>> SECTION: _LEGACY_KERNEL <<<

#define PER (DHP / 32)
__device__ float g_zero[DHP + 16];                // node-sized zeros: the address idle child slots read

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

// scores of the n compacted children (n <= CH): q.mu as a reduce-scatter over the 32 lanes (offsets 16, 8, ...: the
// remaining halving steps, in order), then the butterfly within each group of 32 / CH lanes; lane l ends with child
// l / (32 / CH), whose fields it also loads
template <int CH>
__device__ __forceinline__ void score_children(const int lane, const int n, const float* qv, const float qn,
                                               const float qpos, const float lam, const float sig2, const float scale,
                                               const float cell_size, const float* const* sptr, float* sc, float* scnt,
                                               float* schl, float* schr) {
    constexpr int G = 32 / CH;
    const int c = lane / G;
    const bool cv = c < n;
    const float* pc = cv ? sptr[c] : g_zero;
    const float rad = __ldg(pc + DH), amax = __ldg(pc + DH + 1), pmin = __ldg(pc + DH + 2);
    const float cnt = __ldg(pc + DH + 3), chl = __ldg(pc + DH + 4), chr = __ldg(pc + DH + 5);
    float v[CH];
#pragma unroll
    for (int i = 0; i < CH; i++) {
        const float* p = (i < n) ? sptr[i] : g_zero;
        float x[PER];
#pragma unroll
        for (int k = 0; k < PER; k++) {
            const int col = lane + 32 * k;
            x[k] = f_mul(qv[k], (col < DH) ? __ldg(p + col) : 0.0f);
        }
        v[i] = lane_halve(x);
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
    if (cv && (lane % G) == 0) { sc[c] = score; scnt[c] = cnt; schl[c] = chl; schr[c] = chr; }
}

extern "C" __global__ void __launch_bounds__(NWARP * 32)
fsb_beam_cuda(const i64* __restrict__ ROOT, i64* __restrict__ CELLS, float* __restrict__ NVIS,
              const float* __restrict__ Q, const float* __restrict__ RC, const float rc_s,
              const float* __restrict__ QPOS, const float* __restrict__ LAM, const float* __restrict__ SIG,
              const i64* __restrict__ WBEAM, const i64* __restrict__ BUDGET, const i64* __restrict__ TIDX,
              const float* __restrict__ CHUNK, const float* __restrict__ SPINE,
              const i64* __restrict__ LEAF, const i64* __restrict__ ADM,
              const i64 Ntot, const i64 Sp, const i64 Ltot, const i64 Mm, const i64 Bq, const i64 nq,
              const float scale, const float cell_size, const int KMAXP1) {
    __shared__ const float* s_ptr[NWARP][32];
    __shared__ int s_cref[NWARP][32];
    __shared__ float s_sc[NWARP][32], s_cnt[NWARP][32], s_chl[NWARP][32], s_chr[NWARP][32];
    __shared__ int s_nref[NWARP][WM];
    __shared__ float s_ncnt[NWARP][WM], s_nchl[NWARP][WM], s_nchr[NWARP][WM];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const i64 pid = (i64)blockIdx.x * NWARP + wid;
    if (pid >= nq) return;
    const i64 t = pid / Bq;
    const i64 tidx = TIDX[t];
    const float* cbase = CHUNK + tidx * Ntot * NC;
    const float* sbase = SPINE + t * Sp * NC;
#define NODE(r) ((r) >= 0 ? cbase + (i64)(r) * NC : sbase + (i64)(-(r) - 2) * NC)
    const float* qp = Q + pid * DH;
    float qv[PER], x2[PER];
#pragma unroll
    for (int k = 0; k < PER; k++) {
        const int col = lane + 32 * k;
        qv[k] = (col < DH) ? qp[col] : 0.0f;
        x2[k] = f_mul(qv[k], qv[k]);
    }
    float qs = lane_halve(x2);
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) qs = f_add(qs, __shfl_xor_sync(FULL, qs, o));
#if RC_T
    const float qn = f_mul(f_sqrt_ftz(qs), RC[t]);
#else
    const float qn = f_mul(f_sqrt_ftz(qs), rc_s);
#endif
    const float qpos = QPOS[pid], lam = LAM[t];
    const float sig2 = f_mul(2.0f, SIG[t]);
    const float bud = (float)BUDGET[pid];
    const i64 wbeam = WBEAM[pid];
    const int side = lane & 1, slot = lane >> 1;
    const bool in_w = lane < 2 * WM;
    const unsigned lt = (1u << lane) - 1u;
    int f_ref = -1;
    float f_cnt = 0.0f, f_chl = -1.0f, f_chr = -1.0f;
    if (slot == 0) {
        f_ref = (int)ROOT[t];
        if (f_ref != -1) { const float* p = NODE(f_ref); f_cnt = p[DH + 3]; f_chl = p[DH + 4]; f_chr = p[DH + 5]; }
    }
    int nvis = 0;
    for (int lev = 0; lev < KMAXP1; lev++) {
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
            s_ptr[wid][pos] = NODE(child);
            s_cref[wid][pos] = child;
        }
        __syncwarp();
        if (n <= 8)
            score_children<8>(lane, n, qv, qn, qpos, lam, sig2, scale, cell_size, s_ptr[wid], s_sc[wid], s_cnt[wid],
                              s_chl[wid], s_chr[wid]);
        else if (n <= 16)
            score_children<16>(lane, n, qv, qn, qpos, lam, sig2, scale, cell_size, s_ptr[wid], s_sc[wid], s_cnt[wid],
                               s_chl[wid], s_chr[wid]);
        else
            score_children<32>(lane, n, qv, qn, qpos, lam, sig2, scale, cell_size, s_ptr[wid], s_sc[wid], s_cnt[wid],
                               s_chl[wid], s_chr[wid]);
        __syncwarp();
        // carried slots keep their order ahead of every child; children rank by (score desc, slot asc)
        int crank = ncar;
        float sc = 0.0f;
        if (lane < n) {
            sc = s_sc[wid][lane];
            for (int m = 0; m < n; m++) {
                const float sm = s_sc[wid][m];
                crank += (sm > sc) || (sm == sc && m < lane);
            }
        }
        const bool c_ok = lane < n && sc > -5e8f && crank < wbeam && crank < WM;
        const int ncb = __popc(car & lt);
        const bool f_ok = carried && side == 0 && ncb < wbeam && ncb < WM;
        if (lane < WM) s_nref[wid][lane] = -1;
        __syncwarp();
        if (f_ok) { s_nref[wid][ncb] = f_ref; s_ncnt[wid][ncb] = f_cnt; s_nchl[wid][ncb] = f_chl; s_nchr[wid][ncb] = f_chr; }
        if (c_ok) {
            s_nref[wid][crank] = s_cref[wid][lane];
            s_ncnt[wid][crank] = s_cnt[wid][lane]; s_nchl[wid][crank] = s_chl[wid][lane]; s_nchr[wid][crank] = s_chr[wid][lane];
        }
        __syncwarp();
        if (in_w) { f_ref = s_nref[wid][slot]; f_cnt = s_ncnt[wid][slot]; f_chl = s_nchl[wid][slot]; f_chr = s_nchr[wid][slot]; }
        __syncwarp();
    }
    // cells epilogue: src/off/len of each frontier node, then the cell ids from the leaf (src == 1) or admission table
    if (in_w && side == 0) s_nref[wid][slot] = f_ref;
    __syncwarp();
    i64* out = CELLS + pid * (WM * R);
    for (int idx = lane; idx < WM * R; idx += 32) {
        const int w = idx / R, tr = idx - w * R;
        const int ref = s_nref[wid][w];
        i64 cell = -1;
        if (ref != -1) {
            const float* p = NODE(ref);
            const float src = p[DH + 6], off = p[DH + 7], ln = p[DH + 8];
            if (src > 0.0f && (float)tr < ln) {
                const i64 pos = (i64)off + tr;
                if (src == 1.0f) cell = LEAF[tidx * Ltot + min(max(pos, (i64)0), Ltot - 1)];
                else cell = ADM[tidx * Mm + min(max(pos, (i64)0), Mm - 1)];
            }
        }
        out[idx] = cell;
    }
    if (lane == 0) NVIS[pid] = (float)nvis;
}

// >>> SECTION: _COMMON <<<

// ---- fsb_build_cuda: common helpers (appended after fsb_beam_cuda._HELPERS) ----
__device__ __forceinline__ float f_mul_ftz(float a, float b) { float r; asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_div_ftz(float a, float b) { float r; asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_max(float a, float b) { float r; asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_min(float a, float b) { float r; asm("min.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_pow2neg(int p) { return __int_as_float((127 - p) << 23); }      // 2^-p, p <= 126
__device__ __forceinline__ float f_sz(unsigned neg) { return neg ? -0.0f : 0.0f; }
__device__ __forceinline__ unsigned sgn(float v) { return __float_as_uint(v) >> 31; }
typedef unsigned long long u64;
// fp32 <-> bf16 in an unsigned short: cvt.rn.bf16 (nearest, ties to even) on the way in, a 16-bit shift on the way
// out
__device__ __forceinline__ unsigned short f2bf(float x) {
    unsigned short r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ float bf2f(unsigned short h) { return __uint_as_float(((unsigned)h) << 16); }
// stable-sort key of a projection: value order with -0 == +0, ties by the slot (torch's stable argsort)
__device__ __forceinline__ u64 sort_key(float p, unsigned slot) {
    unsigned b = __float_as_uint(p == 0.0f ? 0.0f : p);
    b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((u64)b << 32) | (u64)slot;
}
__device__ __forceinline__ u64 shfl_u64(u64 v, int src) {
    unsigned lo = __shfl_sync(FULL, (unsigned)v, src), hi = __shfl_sync(FULL, (unsigned)(v >> 32), src);
    return ((u64)hi << 32) | lo;
}
__device__ __forceinline__ u64 shfl_xor_u64(u64 v, int o) {
    unsigned lo = __shfl_xor_sync(FULL, (unsigned)v, o), hi = __shfl_xor_sync(FULL, (unsigned)(v >> 32), o);
    return ((u64)hi << 32) | lo;
}
// one reduce-scatter round of _halve_rows (fsb_beam_cuda.rs_round): lanes with bit o keep the upper M values
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
// pairwise halving over 2^Q values f(m), m = 0..2^Q-1 (pairs m, m + 2^(Q-1) first): depth-first over the low bits,
// so the leaves are visited in bit-reversed order and at most Q partials are live
template <int J, int Q>
struct HR {
    template <class F> __device__ __forceinline__ static float run(F& f, const int fixed) {
        const float a = HR<J + 1, Q>::run(f, fixed);
        const float b = HR<J + 1, Q>::run(f, fixed | (1 << J));
        return f_add(a, b);
    }
};
template <int Q>
struct HR<Q, Q> {
    template <class F> __device__ __forceinline__ static float run(F& f, const int fixed) { return f(fixed); }
};
// the same over PER-wide values (f fills a[PER])
template <int J, int Q>
struct HRV {
    template <class F> __device__ __forceinline__ static void run(F& f, const int fixed, float* out) {
        float a[PER], b[PER];
        HRV<J + 1, Q>::run(f, fixed, a);
        HRV<J + 1, Q>::run(f, fixed | (1 << J), b);
#pragma unroll
        for (int i = 0; i < PER; i++) out[i] = f_add(a[i], b[i]);
    }
};
template <int Q>
struct HRV<Q, Q> {
    template <class F> __device__ __forceinline__ static void run(F& f, const int fixed, float* out) { f(fixed, out); }
};
// reduce-scatter of 32 row sums whose in-lane partials prod(r) are produced on demand (first round pairs r, r + 16)
template <class F>
__device__ __forceinline__ float rs_lazy(F& prod, const int lane) {
    float v[16];
    const bool up = (lane & 16) != 0;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const float a = prod(i), b = prod(i + 16);
        v[i] = f_add(up ? b : a, __shfl_xor_sync(FULL, up ? a : b, 16));
    }
    rs_round<8>(v, lane, 8); rs_round<4>(v, lane, 4); rs_round<2>(v, lane, 2); rs_round<1>(v, lane, 1);
    return v[0];
}
// 32 row sums (v[r] = row r's partial over this lane's columns after the in-lane step) -> lane s holds row s
__device__ __forceinline__ float rs_all(float* v, const int lane) {
    rs_round<16>(v, lane, 16); rs_round<8>(v, lane, 8); rs_round<4>(v, lane, 4); rs_round<2>(v, lane, 2);
    rs_round<1>(v, lane, 1);
    return v[0];
}
// sum over the head dimension of a per-lane pair (dims lane, lane + 32), every lane gets the total
__device__ __forceinline__ float dim_allsum(const float* a) {
    float s = a[0];
#if PER == 2
    s = f_add(a[0], a[1]);
#endif
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) s = f_add(s, __shfl_xor_sync(FULL, s, o));
    return s;
}
// _unit on a lane's dims: v / max(sqrt_rn(sum v*v), 1e-6) with tl.sqrt_rn / tl.math.div_rn (ftz flavours)
__device__ __forceinline__ void unit_inplace(float* v) {
    float sq[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) sq[i] = f_mul(v[i], v[i]);
    const float den = f_max(f_sqrt_ftz(dim_allsum(sq)), 1e-6f);
#pragma unroll
    for (int i = 0; i < PER; i++) v[i] = f_div_ftz(v[i], den);
}

// _unit of the NN node vectors a warp holds (lane: dims lane, lane + 32 of every node): the head-dimension sums run
// as one reduce-scatter over the nodes (offsets 16, 8, ... in _halve_rows' order, then butterflies), one sqrt per lane,
// the divisor broadcast back; bitwise the same as NN butterfly sums
template <int NN>
__device__ __forceinline__ void unit_nodes(float (&v)[NN][PER], const int lane) {
    if constexpr (NN == 1) {
        unit_inplace(v[0]);
    } else {
        constexpr int A = NN == 2 ? 1 : NN == 4 ? 2 : NN == 8 ? 3 : NN == 16 ? 4 : 5;
        float s[NN];
#pragma unroll
        for (int n = 0; n < NN; n++) {
            s[n] = f_mul(v[n][0], v[n][0]);
#if PER == 2
            s[n] = f_add(s[n], f_mul(v[n][1], v[n][1]));
#endif
        }
        if constexpr (A >= 1) rs_round<NN / 2>(s, lane, 16);
        if constexpr (A >= 2) rs_round<(NN >= 4 ? NN / 4 : 1)>(s, lane, 8);
        if constexpr (A >= 3) rs_round<(NN >= 8 ? NN / 8 : 1)>(s, lane, 4);
        if constexpr (A >= 4) rs_round<(NN >= 16 ? NN / 16 : 1)>(s, lane, 2);
        if constexpr (A >= 5) rs_round<1>(s, lane, 1);
#pragma unroll
        for (int o = 16 >> A; o >= 1; o >>= 1) s[0] = f_add(s[0], __shfl_xor_sync(FULL, s[0], o));
        const float den = f_max(f_sqrt_ftz(s[0]), 1e-6f);             // node lane >> (5 - A)
#pragma unroll
        for (int n = 0; n < NN; n++) {
            const float dn = __shfl_sync(FULL, den, n << (5 - A));
#pragma unroll
            for (int i = 0; i < PER; i++) v[n][i] = f_div_ftz(v[n][i], dn);
        }
    }
}

// FSB_BUILD_UNIFORM=1 (fsb_build_cuda.uniform_on) compiles the split-table build kernels with BUILD_UNIFORM 1: every
// value that a branch around shuffle code tests is one ptxas can see is warp-uniform. The task fields, the warp index,
// the fold flag and the admitted counts are broadcast from lane 0 (UNI), and a forest's tables are picked by a select
// on the uniform forest index (FSEL) instead of indexing the Args pointer arrays. On a value it cannot prove uniform,
// ptxas guards each group of shuffles with a divergence branch to an out-of-line second copy of the code that runs
// every shuffle through a WARPSYNC helper call, and the register allocation that serves both copies spills. The
// arithmetic, its operands and its order are unchanged, so every output keeps its bits.
#ifndef BUILD_UNIFORM
#define BUILD_UNIFORM 0
#endif
#if BUILD_UNIFORM
#define UNI(v) __shfl_sync(FULL, (v), 0)
#define FSEL(a, f) ((f) ? (a)[1] : (a)[0])
#else
#define UNI(v) (v)
#define FSEL(a, f) (a)[f]
#endif

// kernel arguments of fsb_forest_build
struct Args {
    const int* tasks; const int* lay;
    const float* kb; const float* ce; const float* spr;
    const i64* adm[2]; const i64* mt[2];
    float* node[2]; i64* leaf[2];
    unsigned short* node_mu[2];   // bf16 means of the split table (lean16; null while node holds a packed row)
    float* node_aux[2];           // its fp32 aggregates [rad | amax | pmin | cnt | chl | chr | src | off | len]
    int* scr;                  // slot lists of chunks of height >= 10: [f][k - 10][t][M] (int32)
    int* ctl;                  // this call's control words: the completion flags of its G nodes
    int M, Ntot, Ltot, has_sp, nbig, T;
};
// dependency flags inside one launch: a task waits only for tasks earlier in the table, i.e. of lower blockIdx; the
// producer publishes its global writes before raising the flag. The control words belong to the call (allocated with
// its outputs, so a captured graph owns its own) and are cleared by the call's prologue, the launch before the build
// kernel on the same stream.
// This relies on the hardware starting the blocks of the 1-D grid in increasing blockIdx order, so that the producer
// a block waits for is resident or finished. NVIDIA GPUs do; the CUDA programming model does not promise it, and in
// another order every resident block could end up waiting for one not yet started. Claiming the tasks in table order
// from an atomic counter would remove the assumption, but the task index then comes through shared memory and the
// compiler can no longer recompute the task's fields from blockIdx instead of holding them: at the 128-register cap
// the spills grow from 56 to 368 bytes (Dh 64) and the build kernel takes 5% longer on 131k-token inputs, so
// the block index stays the task index.
__device__ __forceinline__ void wait_flag(const int* fl) {
    if (threadIdx.x == 0) {
        int v;
        while (true) {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(fl) : "memory");
            if (v != 0) break;
            __nanosleep(500);
        }
    }
    __syncthreads();
}
__device__ __forceinline__ void set_flag(int* fl) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"(fl), "r"(1) : "memory");
}
// >>> SECTION: _PROLOGUE <<<

// (the uniform module, BUILD_UNIFORM 1, holds only the split-table build kernels fsb_forest_build16 and _lean16)
#if !BUILD_UNIFORM
// ---- K0: admission tables of both forests, one CTA (1024 threads) per (forest, head) ----
// adm[t] = admitted cells ascending then -1 (argsort(where(admit, id, id + M)) masked), m_tot[t] = #admitted,
// mb[t, b] = searchsorted(adm with -1 -> M, E[b]) (left): #admitted < E[b] for E[b] <= M, M above.
// CTL[0, nctl): the build kernel's control words, cleared here (the build kernel is the next launch on the stream).
extern "C" __global__ void __launch_bounds__(1024)
fsb_forest_prologue(const float* __restrict__ CE, const float* __restrict__ TAU0, const float* __restrict__ TAU1,
                    const i64* __restrict__ E, i64* __restrict__ ADM0, i64* __restrict__ ADM1,
                    i64* __restrict__ MT0, i64* __restrict__ MT1, i64* __restrict__ MB0, i64* __restrict__ MB1,
                    int* __restrict__ CTL, const int nctl, const int T, const int M, const int H, const int nb) {
    for (int i = blockIdx.x * 1024 + threadIdx.x; i < nctl; i += gridDim.x * 1024) CTL[i] = 0;
    __shared__ int s_warp[32];
    __shared__ int s_tot;
    const int f = blockIdx.x / T, t = blockIdx.x - f * T;
    const float tau = (f == 0 ? TAU0 : TAU1)[t % H];
    i64* adm = (f == 0 ? ADM0 : ADM1) + (i64)t * M;
    const float* ce = CE + (i64)t * M;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int per = (M + 1023) / 1024;
    const int c0 = min(tid * per, M), c1 = min(c0 + per, M);
    int cnt = 0;
    for (int c = c0; c < c1; c++) cnt += ce[c] >= tau;
    // block exclusive scan of cnt
    int incl = cnt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(FULL, incl, o);
        if (lane >= o) incl += y;
    }
    if (lane == 31) s_warp[wid] = incl;
    __syncthreads();
    if (wid == 0) {
        int w = s_warp[lane], wi = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(FULL, wi, o);
            if (lane >= o) wi += y;
        }
        s_warp[lane] = wi - w;
        if (lane == 31) s_tot = wi;
    }
    __syncthreads();
    int pos = s_warp[wid] + incl - cnt;
    const int tot = s_tot;
    for (int c = c0; c < c1; c++)
        if (ce[c] >= tau) adm[pos++] = c;
    for (int c = tot + tid; c < M; c += 1024) adm[c] = -1;
    if (tid == 0) (f == 0 ? MT0 : MT1)[t] = tot;
    __syncthreads();
    i64* mb = (f == 0 ? MB0 : MB1) + (i64)t * nb;
    for (int b = tid; b < nb; b += 1024) {
        const i64 e = E[b];
        i64 r;
        if (e > M) r = M;
        else {
            int lo = 0, hi = tot;                  // first admitted index with adm >= e
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (adm[mid] < e) lo = mid + 1; else hi = mid;
            }
            r = lo;
        }
        mb[b] = r;
    }
}
#endif
// >>> SECTION: _CPHASE <<<

// ---- C phase: nodes of 64..512 rows of one 512-slot batch, CTA-wide, rows in shared memory ----
#define LOGDH (DH == 64 ? 6 : 5)
struct SM {                    // shared-memory carve-up of one CTA (end: one past the last float, SMEM_FLOATS)
    float* Xs; int* pw; int* cell; float* ce; float* sp; u64* keys; u64* keys2;
    float* mu; float* v; float* vr; float* pv; float* ws; int* far; float* pvp; u64* zb; float* end;
};
__device__ __forceinline__ SM carve(float* smem) {
    SM S;
    float* p = smem;
    S.Xs = p; p += 512 * XS;
    S.keys = (u64*)p; p += 2 * 512;
    S.keys2 = (u64*)p; p += 2 * 512;
    S.zb = (u64*)p; p += 2 * 16;
    S.pw = (int*)p; p += 512;
    S.cell = (int*)p; p += 512;
    S.ce = p; p += 512;
    S.sp = p; p += 512;
    S.mu = p; p += 8 * DH;
    S.v = p; p += 8 * DH;
    S.vr = p; p += 8 * DH;
    S.pv = p; p += 512;
    S.ws = p; p += 16 * 5;
    S.far = (int*)p; p += 16;
    S.pvp = p; p += 16;
    S.end = p;
    return S;
}
#define SMEM_FLOATS (512 * XS + 4 * 512 + 2 * 16 + 4 * 512 + 3 * 8 * DH + 512 + 16 * 5 + 32)   // _smem_bytes mirrors it

// barrier of the NS threads (one per slot) that build one group of slots: the CTA for the build kernel's 512-slot
// batches, named barrier bid for a smaller group of the small-chunk kernel (fsb_forest_build_small)
template <int NS>
__device__ __forceinline__ void group_sync(const int bid) {
    if constexpr (NS == 512) __syncthreads();
    else asm volatile("bar.sync %0, %1;" :: "r"(bid), "n"(NS) : "memory");
}

// column halving of node n (thread role (g, d, n) with G = V / DH lanes per (n, d)): sum over the node's slots of
// val(slot, d), pad term folded in first, then the cross-lane steps; result valid in lane g == 0
template <int LV, class F>
__device__ __forceinline__ float col_sum(F& val, const int g, const int n) {
    constexpr int V = 1 << LV, G = V / DH;
    auto f = [&](const int m) { return val(n * V + g + G * m); };
    float s = HR<0, LOGDH>::run(f, 0);
#pragma unroll
    for (int o = G / 2; o >= 1; o >>= 1) s = f_add(s, __shfl_xor_sync(FULL, s, o));
    return s;
}

// One C level (nodes of V = 2^LV rows, NN = NS / V of them, NS slots and threads: 512 in the build kernel). S.Xs and x
// hold the raw rows of the slots at level start;
// after the mean each warp centres its rows into S.Xs in place (the row passes re-read them, so the 32-row tile is not
// live across the CTA-wide column sums) and the split gathers the next order's raw rows from Kbar (kt: the head's rows,
// cells from S.cell). The pad zero is folded once into each column sum (see the W phase note).
template <int LV, int NS = 512>
__device__ __forceinline__ void c_level(float (&x)[32][PER], const SM& S, const Rec& R, const float* __restrict__ kt,
                                         const int tid, const int lane, const int wid, const bool fold,
                                         const bool has_sp, const int bid = 0) {
    constexpr int V = 1 << LV, NN = NS / V, G = V / DH, LG = LV - LOGDH;
    const int g = tid & (G - 1), d = (tid >> LG) & (DH - 1), n = tid >> (LG + LOGDH);       // column role
    const int nw = (32 * wid) >> LV;                                                     // node of this warp's rows
    const int row = 32 * wid + lane;
    const float zd = f_sz((unsigned)(S.zb[(n * V) >> R.ks] >> d) & 1u);
    const bool m16 = R.node_mu != nullptr;
    // ---- mean over the raw rows (pad zero folded once)
    {
        auto val = [&](const int s) { return S.Xs[s * XS + d]; };
        float s = col_sum<LV>(val, g, n);
        if (fold) s = f_add(s, zd);
        const float mu = f_mul_ftz(s, f_pow2neg(LV));
        if (g == 0) S.mu[n * DH + d] = mu;
    }
    group_sync<NS>(bid);
    // ---- centre the warp's rows (registers and S.Xs), squared distances, warp partial summaries
    {
        const float* muw = S.mu + nw * DH;
        float mul[PER];
#pragma unroll
        for (int i = 0; i < PER; i++) mul[i] = muw[lane + 32 * i];
#pragma unroll
        for (int r = 0; r < 32; r++)
#pragma unroll
            for (int i = 0; i < PER; i++) {
                x[r][i] = f_sub(x[r][i], mul[i]);
                S.Xs[(32 * wid + r) * XS + lane + 32 * i] = x[r][i];
            }
        auto sq = [&](const int r) {
            float a = f_mul(x[r][0], x[r][0]);
#if PER == 2
            a = f_add(a, f_mul(x[r][1], x[r][1]));
#endif
            return a;
        };
        const float d2 = rs_lazy(sq, lane);
        float rq = has_sp ? f_add(f_sqrt_ftz(d2), S.sp[row]) : d2;
        float am = S.ce[row], bd = d2;
        int pm = S.cell[row], far = row - nw * V;
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            const float rq2 = __shfl_xor_sync(FULL, rq, o), am2 = __shfl_xor_sync(FULL, am, o);
            const float bd2 = __shfl_xor_sync(FULL, bd, o);
            const int pm2 = __shfl_xor_sync(FULL, pm, o), far2 = __shfl_xor_sync(FULL, far, o);
            rq = f_max(rq, rq2); am = f_max(am, am2); pm = min(pm, pm2);
            if (bd2 > bd || (bd2 == bd && far2 < far)) { bd = bd2; far = far2; }
        }
        if (lane == 0) {
            float* w5 = S.ws + 5 * wid;
            w5[0] = bd; w5[1] = __int_as_float(far); w5[2] = rq; w5[3] = am; w5[4] = __int_as_float(pm);
        }
    }
    group_sync<NS>(bid);
    // ---- node summaries and records; v0 = Xc[far] / max(sqrt(d2[far]), 1e-6) (one warp per node)
    if (tid < NN) {
        const float* w5 = S.ws + 5 * (tid * (V / 32));
        float bd = w5[0], rq = w5[2], am = w5[3];
        int far = __float_as_int(w5[1]), pm = __float_as_int(w5[4]);
        for (int w = 1; w < V / 32; w++) {
            w5 += 5;
            const float bd2 = w5[0];
            const int far2 = __float_as_int(w5[1]);
            rq = f_max(rq, w5[2]); am = f_max(am, w5[3]); pm = min(pm, __float_as_int(w5[4]));
            if (bd2 > bd || (bd2 == bd && far2 < far)) { bd = bd2; far = far2; }
        }
        S.far[tid] = far;
        S.pvp[tid] = bd;                                   // d2 of the farthest row (its |Xc|^2), read below
        // lean16: slack covering the bf16 mean stored below (the build itself centres on the fp32 mean)
        float esp = 0.0f;
        if (m16) {
            const float* mrow = S.mu + tid * DH;
#pragma unroll 8
            for (int dd = 0; dd < DH; dd++) {
                const float df = f_sub(mrow[dd], bf2f(f2bf(mrow[dd])));
                esp = f_add(esp, f_mul(df, df));
            }
            esp = f_sqrt_ftz(esp);
        }
        if (node_full(R, tid * V)) {
            float rad = has_sp ? rq : f_sqrt_ftz(rq);
            if (has_sp && fold) rad = f_max(rad, 0.0f);
            write_scalars(R, tid * V, LV, rad, f_max(am, 0.0f), pm, esp);
        }
    }
    for (int e = tid; e < NN * DH; e += NS)
        if (node_full(R, (e / DH) * V)) {
            if (R.node_mu != nullptr) mu16_ptr(R, (e / DH) * V, LV)[e % DH] = f2bf(S.mu[e]);
            else mu_ptr(R, (e / DH) * V, LV)[e % DH] = S.mu[e];
        }
    group_sync<NS>(bid);
    if (wid < NN) {
        const int nn = wid, far = S.far[nn];
        const float den = f_max(f_sqrt_ftz(S.pvp[nn]), 1e-6f);
        bool alln = true;
        float vv[PER];
#pragma unroll
        for (int i = 0; i < PER; i++) vv[i] = f_div_ftz(S.Xs[(nn * V + far) * XS + lane + 32 * i], den);
#pragma unroll
        for (int i = 0; i < PER; i++) {
            const float zi = f_sz((unsigned)(S.zb[(nn * V) >> R.ks] >> (lane + 32 * i)) & 1u);
            const float xc = f_mul(f_sub(zi, S.mu[nn * DH + lane + 32 * i]), 0.0f);
            alln = alln && (sgn(xc) != sgn(vv[i]));
            S.v[nn * DH + lane + 32 * i] = vv[i];
        }
        const bool an = __all_sync(FULL, alln);
        __syncwarp();
        if (lane == 0) S.pvp[nn] = f_sz(an);
    }
    group_sync<NS>(bid);
    const float xcp_d = f_mul(f_sub(zd, S.mu[n * DH + d]), 0.0f);
    // ---- two power iterations
#pragma unroll 1
    for (int it = 0; it < 2; it++) {
        {
            const float* vw = S.v + nw * DH;
            float vl[PER];
#pragma unroll
            for (int i = 0; i < PER; i++) vl[i] = vw[lane + 32 * i];
            auto dv = [&](const int r) {
                const float* xr = S.Xs + (32 * wid + r) * XS + lane;
                float a = f_mul(xr[0], vl[0]);
#if PER == 2
                a = f_add(a, f_mul(xr[32], vl[1]));
#endif
                return a;
            };
            S.pv[row] = rs_lazy(dv, lane);
        }
        group_sync<NS>(bid);
        {
            auto val = [&](const int s) { return f_mul(S.Xs[s * XS + d], S.pv[s]); };
            float s = col_sum<LV>(val, g, n);
            if (fold) s = f_add(s, f_mul(xcp_d, S.pvp[n]));
            if (g == 0) S.vr[n * DH + d] = s;
        }
        group_sync<NS>(bid);
        if (wid < NN) {
            const int nn = wid;
            float vv[PER];
            bool alln = true;
#pragma unroll
            for (int i = 0; i < PER; i++) vv[i] = S.vr[nn * DH + lane + 32 * i];
            unit_inplace(vv);
#pragma unroll
            for (int i = 0; i < PER; i++) {
                const float zi = f_sz((unsigned)(S.zb[(nn * V) >> R.ks] >> (lane + 32 * i)) & 1u);
                const float xc = f_mul(f_sub(zi, S.mu[nn * DH + lane + 32 * i]), 0.0f);
                alln = alln && (sgn(xc) != sgn(vv[i]));
                S.v[nn * DH + lane + 32 * i] = vv[i];
            }
            const bool an = __all_sync(FULL, alln);
            if (lane == 0) S.pvp[nn] = f_sz(an);
        }
        group_sync<NS>(bid);
    }
    // ---- projections -> sort keys (value order, -0 == +0, ties by slot in node)
    {
        const float* vw = S.v + nw * DH;
        float vl[PER];
#pragma unroll
        for (int i = 0; i < PER; i++) vl[i] = vw[lane + 32 * i];
        auto dv = [&](const int r) {
            const float* xr = S.Xs + (32 * wid + r) * XS + lane;
            float a = f_mul(xr[0], vl[0]);
#if PER == 2
            a = f_add(a, f_mul(xr[32], vl[1]));
#endif
            return a;
        };
        const float proj = rs_lazy(dv, lane);
        // ---- split order: (value with -0 == +0, slot in node) sorted within the warp by a shuffle bitonic network
        unsigned kk = __float_as_uint(proj == 0.0f ? 0.0f : proj);
        kk = (kk & 0x80000000u) ? ~kk : (kk | 0x80000000u);
        int sl = row - nw * V;
#pragma unroll
        for (int size = 2; size <= 32; size <<= 1) {
#pragma unroll
            for (int j = size >> 1; j > 0; j >>= 1) {
                const unsigned k2 = __shfl_xor_sync(FULL, kk, j);
                const int s2 = __shfl_xor_sync(FULL, sl, j);
                const bool asc = (lane & size) == 0, lower = (lane & j) == 0;
                const bool less2 = k2 < kk || (k2 == kk && s2 < sl);
                if ((lower == asc) ? less2 : !less2) { kk = k2; sl = s2; }
            }
        }
        S.keys[row] = ((u64)kk << 32) | (u64)(unsigned)sl;
    }
    group_sync<NS>(bid);
    // ---- merge the node's sorted 32-runs pairwise (runs of 64, 128, ... V): an element's place in the merged run is
    // its place in its own run plus the keys below it in the partner run (keys are unique); ping-pong buffers
    u64* kin = S.keys;
    u64* kout = S.keys2;
#pragma unroll
    for (int L = 32; L < V; L <<= 1) {
        const u64 key = kin[row];
        const int b0 = row & ~(2 * L - 1), own = row & (L - 1);
        const u64* part = kin + b0 + ((row & L) ? 0 : L);
        int pos = 0;
#pragma unroll
        for (int step = L; step >= 1; step >>= 1)
            if (pos + step <= L && part[pos + step - 1] < key) pos += step;
        kout[b0 + own + pos] = key;
        group_sync<NS>(bid);
        u64* tmp = kin; kin = kout; kout = tmp;
    }
    const u64* ksort = kin;
    // ---- permutation into the sorted order: the raw rows of the new slots from Kbar, cells, ceilings, spreads
    {
        const int src_l = nw * V + (int)(unsigned)ksort[row];
        const int c = S.cell[src_l];
        const float e = S.ce[src_l], p = S.sp[src_l];
        const float* kl = kt + lane;
#pragma unroll
        for (int r = 0; r < 32; r++) {
            const float* pr = kl + (i64)max(__shfl_sync(FULL, c, r), 0) * DH;
#pragma unroll
            for (int i = 0; i < PER; i++) x[r][i] = __ldg(pr + 32 * i);
        }
        group_sync<NS>(bid);
#pragma unroll
        for (int r = 0; r < 32; r++)
#pragma unroll
            for (int i = 0; i < PER; i++) S.Xs[(32 * wid + r) * XS + lane + 32 * i] = x[r][i];
        S.cell[row] = c; S.ce[row] = e; S.sp[row] = p;
    }
    group_sync<NS>(bid);
}

// C levels lv0 .. 6 of a group of NS slots (levels above log2(NS) do not exist in it)
template <int NS = 512>
__device__ __forceinline__ void c_phase(const int lv0, float (&x)[32][PER], const SM& S, const Rec& R, const float* kt,
                                         const int tid, const int lane, const int wid, const bool fold,
                                         const bool has_sp, const int bid = 0) {
    if constexpr (NS >= 512)
        if (lv0 >= 9) { c_level<9, NS>(x, S, R, kt, tid, lane, wid, fold, has_sp, bid); DBG_MARK(9); }
    if constexpr (NS >= 256)
        if (lv0 >= 8) { c_level<8, NS>(x, S, R, kt, tid, lane, wid, fold, has_sp, bid); DBG_MARK(8); }
    if constexpr (NS >= 128)
        if (lv0 >= 7) { c_level<7, NS>(x, S, R, kt, tid, lane, wid, fold, has_sp, bid); DBG_MARK(7); }
    c_level<6, NS>(x, S, R, kt, tid, lane, wid, fold, has_sp, bid); DBG_MARK(6);
}
// >>> SECTION: _WPHASE <<<

#ifdef DBG_TIME
extern "C" __device__ u64 g_dbg[8192 * 16];
#define DBG_MARK(slot) do { if ((threadIdx.x & 511) == 0) g_dbg[blockIdx.x * 16 + (slot)] = clock64(); } while (0)
__device__ __forceinline__ u64 gtimer() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
#define DBG_START() do { if (threadIdx.x == 0) { g_dbg[blockIdx.x * 16 + 3] = smid(); g_dbg[blockIdx.x * 16 + 4] = gtimer(); } } while (0)
#define DBG_END() do { __syncthreads(); if (threadIdx.x == 0) g_dbg[blockIdx.x * 16 + 5] = gtimer(); } while (0)
#else
#define DBG_START() do { } while (0)
#define DBG_END() do { } while (0)
#define DBG_MARK(slot) do { } while (0)
#endif
// ---- W phase: one warp, 32 rows (slots) in registers; nodes of V <= 32 rows are aligned lane groups ----
// lane l holds dims l (+ 32) of every row in x[r][i]; per-row scalars (cell, ceil, spread) live in lane r.
struct Rec {                  // where a warp's records go
    float* node;              // forest node table of head t: (Ntot, NC)
    unsigned short* node_mu;  // its bf16 means (lean16; null while node is the packed table)
    float* node_aux;          // its fp32 aggregates (lean16)
    i64* leaf;                // leaf table of head t: (Ltot,)
    int k, ks, l0, n0, jb;    // chunk height, subtree height, subtree root level/index, chunk of subtree 0
    int jsh;                  // log2 of the chunk stride: subtree i belongs to chunk jb + (i << jsh)
    int base, leafbase;       // bases[k], leafbases[k]
    int nfull;                // subtrees (relative to jb) that belong to full chunks: index < nfull
    int zsh;                  // log2 of the rows per chunk inside a warp (min(k, 5)): chunk of a warp row = r >> zsh
};
// record of the node of V rows that starts at batch slot gs (V = 2^lv)
__device__ __forceinline__ void rec_pos(const Rec& R, const int gs, const int lv, int& row, int& h, int& idx, int& j) {
    const int i = gs >> R.ks, local = gs & ((1 << R.ks) - 1), d = R.ks - lv;
    j = R.jb + (i << R.jsh);
    const int l = R.l0 + d;
    idx = (R.n0 << d) + (local >> lv);
    h = (1 << l) - 1 + idx;
    row = R.base + j * ((2 << R.k) - 1) + h;
}
__device__ __forceinline__ void write_scalars(const Rec& R, const int gs, const int lv, const float rad, const float amax,
                                              const int pmin, const float eps) {
    int row, h, idx, j;
    rec_pos(R, gs, lv, row, h, idx, j);
    float* p = R.node_aux != nullptr ? R.node_aux + (i64)row * NA : R.node + (i64)row * NC + DH;
    const int V = 1 << lv;
    const int rb = R.base + j * ((2 << R.k) - 1);
    const bool inner = lv > 0;
    // split table: the mean is stored in bf16, so the radius carries the rounding slack (packed path keeps rad bit-exact)
    p[0] = R.node_aux != nullptr ? f_add(rad, eps) : rad; p[1] = amax; p[2] = (float)pmin; p[3] = (float)V;
    p[4] = inner ? (float)(rb + 2 * h + 1) : -1.0f; p[5] = inner ? (float)(rb + 2 * h + 2) : -1.0f;
    p[6] = 1.0f; p[7] = (float)(R.leafbase + j * (1 << R.k) + idx * V); p[8] = (float)V;
}
__device__ __forceinline__ float* mu_ptr(const Rec& R, const int gs, const int lv) {
    int row, h, idx, j;
    rec_pos(R, gs, lv, row, h, idx, j);
    return R.node + (i64)row * NC;
}
__device__ __forceinline__ unsigned short* mu16_ptr(const Rec& R, const int gs, const int lv) {
    int row, h, idx, j;
    rec_pos(R, gs, lv, row, h, idx, j);
    return R.node_mu + (i64)row * DH;
}
__device__ __forceinline__ bool node_full(const Rec& R, const int gs) { return (gs >> R.ks) < R.nfull; }

// Rows: Xw (the warp's 32-row slice of S.Xs) holds the raw rows in their load order and is never rewritten; org is
// the load-order index of the row in slot `lane`, cell/ce/sp its cell, ceiling and spread. The pad zero z of the
// padded group build is folded once into each node sum: for z = +-0, halving(t_r + z) == halving(t_r) + z (a nonzero
// partial is unchanged by a zero, a zero partial ends as a sum of zeros plus z either way), so it is bitwise the
// per-entry fold of the Triton builder. Level 5 runs on the full 32-row tile; nodes of <= 16 rows run on 16-row halves
// (rows in lanes 2r, 2r + 1) so the tile, the node vectors and the reduce-scatter fit in registers.
__device__ __forceinline__ float zsel(const u64 zb, const int bit) { return f_sz((unsigned)(zb >> bit) & 1u); }

template <int LV, bool RELOAD = true>
__device__ __forceinline__ void w_level(float (&x)[32][PER], int& cell, float& ce, float& sp, int& org, const u64 zb,
                                         const float* Xw, int* pw, int* po, const Rec& R, const int lane,
                                         const int gs0, const bool fold, const bool has_sp) {
    constexpr int V = 1 << LV, NN = 32 / V;
    constexpr bool KEEPMU = LV >= 3;
    float mus[KEEPMU ? NN : 1][PER];
    float epsw[LV > 0 ? NN : 1];
    unsigned xcpb = 0;
    float* const mp0 = R.node_mu != nullptr ? nullptr : mu_ptr(R, gs0, LV) + lane;
    const bool m16 = R.node_mu != nullptr;
    unsigned short* const mp0b = m16 ? mu16_ptr(R, gs0, LV) + lane : nullptr;
    const int nodes = (2 << R.k) - 1, npc = 1 << max(R.ks - LV, 0);
    // ---- mean (pad zero folded into the node sum once), record, centring; the fp32 mean drives all build math and
    // the stored bf16 mean (epsw: its rounding slack, a full-warp butterfly over the node's DH dims split per lane)
#pragma unroll
    for (int n = 0; n < NN; n++) {
        const bool wr = node_full(R, gs0 + n * V);
        const i64 moff = n + ((n * V) >> R.ks) * ((nodes << R.jsh) - npc);
        float* mp = m16 ? nullptr : mp0 + moff * NC;
        unsigned short* mpb = m16 ? mp0b + moff * DH : nullptr;
        float e2 = 0.0f;
#pragma unroll
        for (int i = 0; i < PER; i++) {
            auto ld = [&](const int q) { return x[n * V + q][i]; };
            float s = HR<0, LV>::run(ld, 0);
            const float z = zsel(zb, n * V * PER + i);
            if (fold) s = f_add(s, z);
            const float mu = f_mul_ftz(s, f_pow2neg(LV));
            const unsigned short mw = m16 ? f2bf(mu) : 0;
            if constexpr (LV > 0)
                if (m16) {
                    const float df = f_sub(mu, bf2f(mw));
                    e2 = f_add(e2, f_mul(df, df));
                }
            if (wr) { if (m16) mpb[32 * i] = mw; else mp[32 * i] = mu; }
            if constexpr (LV > 0) {
                xcpb |= sgn(f_mul(f_sub(z, mu), 0.0f)) << ((n * PER + i) & 31);
#pragma unroll
                for (int q = 0; q < V; q++) x[n * V + q][i] = f_sub(x[n * V + q][i], mu);
                if constexpr (KEEPMU) mus[n][i] = mu;
            }
        }
        if constexpr (LV > 0)
            if (m16) {
#pragma unroll
                for (int o = 16; o >= 1; o >>= 1) e2 = f_add(e2, __shfl_xor_sync(FULL, e2, o));
                epsw[n] = f_sqrt_ftz(e2);
            }
    }
    if constexpr (LV == 0) {
        // a leaf's centred row is zero (or a flushed subnormal whose square underflows): d2 = +0, rad = sqrt(+0) or
        // +0 + spread; the split table stores the row itself in bf16, so its radius carries the rounding slack
        auto sqe = [&](const int r) {
            const float d0 = f_sub(x[r][0], bf2f(f2bf(x[r][0])));
            float a = f_mul(d0, d0);
#if PER == 2
            const float d1 = f_sub(x[r][1], bf2f(f2bf(x[r][1])));
            a = f_add(a, f_mul(d1, d1));
#endif
            return a;
        };
        float le = 0.0f;
        if (m16) le = f_sqrt_ftz(rs_lazy(sqe, lane));
        float rad = has_sp ? f_add(0.0f, sp) : 0.0f;
        if (has_sp && fold) rad = f_max(rad, 0.0f);
        if (node_full(R, gs0 + lane)) {
            write_scalars(R, gs0 + lane, 0, rad, f_max(ce, 0.0f), cell, le);
            int row, h, idx, j;
            rec_pos(R, gs0 + lane, 0, row, h, idx, j);
            R.leaf[R.leafbase + j * (1 << R.k) + idx] = (i64)cell;
        }
        return;
    } else {
        // ---- squared distances (lane s: row s), node summaries over aligned groups of V lanes
        auto sq = [&](const int r) {
            float a = f_mul(x[r][0], x[r][0]);
#if PER == 2
            a = f_add(a, f_mul(x[r][1], x[r][1]));
#endif
            return a;
        };
        const float d2 = rs_lazy(sq, lane);
        float rq = has_sp ? f_add(f_sqrt_ftz(d2), sp) : d2;
        float am = ce, bd = d2;
        int pm = cell, far = lane;
#pragma unroll
        for (int o = V / 2; o >= 1; o >>= 1) {
            const float rq2 = __shfl_xor_sync(FULL, rq, o), am2 = __shfl_xor_sync(FULL, am, o);
            const float bd2 = __shfl_xor_sync(FULL, bd, o);
            const int pm2 = __shfl_xor_sync(FULL, pm, o), far2 = __shfl_xor_sync(FULL, far, o);
            rq = f_max(rq, rq2); am = f_max(am, am2); pm = min(pm, pm2);
            if (bd2 > bd || (bd2 == bd && far2 < far)) { bd = bd2; far = far2; }
        }
        if ((lane & (V - 1)) == 0 && node_full(R, gs0 + lane)) {
            float eps = 0.0f;
            if (m16) {
#pragma unroll
                for (int un = 0; un < NN; un++) eps = (un == (lane >> LV)) ? epsw[un] : eps;
            }
            float rad = has_sp ? rq : f_sqrt_ftz(rq);
            if (has_sp && fold) rad = f_max(rad, 0.0f);
            write_scalars(R, gs0 + lane, LV, rad, f_max(am, 0.0f), pm, eps);
        }
        // ---- v0 = Xc[far] / max(sqrt(|Xc[far]|^2), 1e-6); |Xc[far]|^2 is d2[far] = bd (same pairwise sum)
        float v[NN][PER];
        const float den0 = f_max(f_sqrt_ftz(bd), 1e-6f);
#pragma unroll
        for (int n = 0; n < NN; n++) {
            const float dn = __shfl_sync(FULL, den0, n * V);
            if constexpr (KEEPMU) {
                const int of = __shfl_sync(FULL, org, __shfl_sync(FULL, far, n * V));
#pragma unroll
                for (int i = 0; i < PER; i++) v[n][i] = f_div_ftz(f_sub(Xw[of * XS + lane + 32 * i], mus[n][i]), dn);
            } else {
                const int fq = __shfl_sync(FULL, far, n * V) - n * V;
#pragma unroll
                for (int i = 0; i < PER; i++) {
                    float s = x[n * V][i];
#pragma unroll
                    for (int q = 1; q < V; q++) s = (fq == q) ? x[n * V + q][i] : s;
                    v[n][i] = f_div_ftz(s, dn);
                }
            }
        }
        auto dotv = [&](const int r) {
            float a = f_mul(x[r][0], v[r / V][0]);
#if PER == 2
            a = f_add(a, f_mul(x[r][1], v[r / V][1]));
#endif
            return a;
        };
        // ---- two power iterations: v <- unit(sum_r Xc_r (Xc_r . v) + pad term)
#pragma unroll
        for (int it = 0; it < 2; it++) {
            const float pv = rs_lazy(dotv, lane);
            unsigned negm = 0;                      // bit n * PER: node n's pad term Xc_pad . v is -0 (every product -0)
            if (fold) {
                unsigned vs = 0;
#pragma unroll
                for (int n = 0; n < NN; n++)
#pragma unroll
                    for (int i = 0; i < PER; i++) vs |= sgn(v[n][i]) << (n * PER + i);
                unsigned dd = xcpb ^ vs;            // bit set: that dim's product (z - mu) * 0 * v is -0
#if PER == 2
                dd &= dd >> 1;
#endif
                negm = __reduce_and_sync(FULL, dd);
            }
#pragma unroll
            for (int n = 0; n < NN; n++) {
                auto term = [&](const int q, float* o) {
                    const float pr = __shfl_sync(FULL, pv, n * V + q);
#pragma unroll
                    for (int i = 0; i < PER; i++) o[i] = f_mul(x[n * V + q][i], pr);
                };
                HRV<0, LV>::run(term, 0, v[n]);
                if (fold) {
#pragma unroll
                    for (int i = 0; i < PER; i++)
                        v[n][i] = f_add(v[n][i], __uint_as_float((((xcpb >> (n * PER + i)) ^ (negm >> (n * PER))) & 1u) << 31));
                }
            }
            unit_nodes<NN>(v, lane);
        }
        const float proj = rs_lazy(dotv, lane);
        // ---- stable split: rank by (proj with -0 == +0, slot) inside the node
        unsigned kb = __float_as_uint(proj == 0.0f ? 0.0f : proj);
        kb = (kb & 0x80000000u) ? ~kb : (kb | 0x80000000u);
        int rank = 0;
#pragma unroll
        for (int o = 1; o < V; o++) {
            const int pl = (lane & ~(V - 1)) | ((lane + o) & (V - 1));
            const unsigned k2 = __shfl_sync(FULL, kb, pl);
            rank += (k2 < kb) || (k2 == kb && pl < lane);
        }
        pw[(lane & ~(V - 1)) + rank] = lane;
        __syncwarp();
        const int src = pw[lane];
        cell = __shfl_sync(FULL, cell, src); ce = __shfl_sync(FULL, ce, src); sp = __shfl_sync(FULL, sp, src);
        org = __shfl_sync(FULL, org, src);
        if constexpr (RELOAD) {
            po[lane] = org;
            __syncwarp();
#pragma unroll
            for (int q = 0; q < 32; q++) {
                const int oq = po[q];
#pragma unroll
                for (int i = 0; i < PER; i++) x[q][i] = Xw[oq * XS + lane + 32 * i];
            }
        }
        __syncwarp();
    }
}


// ---- half-tile W levels (nodes of V <= 8 rows): 16 rows per pass, row r of the half in lanes 2r and 2r + 1
// 16 row sums from in-lane partials prod(r) (pairs d, d + 32 already summed): offsets 16, 8, 4, 2 as reduce-scatter
// rounds, then the (d, d + 1) step as a butterfly; lanes 2r, 2r + 1 end with row r's total (same pairwise tree)
template <class F>
__device__ __forceinline__ float rs16_lazy(F& prod, const int lane) {
    float v[8];
    const bool up = (lane & 16) != 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float a = prod(i), b = prod(i + 8);
        v[i] = f_add(up ? b : a, __shfl_xor_sync(FULL, up ? a : b, 16));
    }
    rs_round<4>(v, lane, 8); rs_round<2>(v, lane, 4); rs_round<1>(v, lane, 2);
    return f_add(v[0], __shfl_xor_sync(FULL, v[0], 1));
}

template <int LV>
__device__ __forceinline__ void h_level(float (&x)[16][PER], int& cell, float& ce, float& sp, int& org, const unsigned zh,
                                        const float* Xw, int* pw, int* po, const Rec& R, const int lane, const int gsh,
                                        const bool fold, const bool has_sp) {
    constexpr int V = 1 << LV, NN = 16 / V;
    constexpr bool KEEPMU = LV >= 3;
    float mus[KEEPMU ? NN : 1][PER];
    float epsw[LV > 0 ? NN : 1];
    unsigned xcpb = 0;
    const int row = lane >> 1;
    float* const mp0 = R.node_mu != nullptr ? nullptr : mu_ptr(R, gsh, LV) + lane;
    const bool m16 = R.node_mu != nullptr;
    unsigned short* const mp0b = m16 ? mu16_ptr(R, gsh, LV) + lane : nullptr;
    const int nodes = (2 << R.k) - 1, npc = 1 << max(R.ks - LV, 0);
#pragma unroll
    for (int n = 0; n < NN; n++) {
        const bool wr = node_full(R, gsh + n * V);
        const i64 moff = n + ((n * V) >> R.ks) * ((nodes << R.jsh) - npc);
        float* mp = m16 ? nullptr : mp0 + moff * NC;
        unsigned short* mpb = m16 ? mp0b + moff * DH : nullptr;
        float e2 = 0.0f;
#pragma unroll
        for (int i = 0; i < PER; i++) {
            auto ld = [&](const int q) { return x[n * V + q][i]; };
            float s = HR<0, LV>::run(ld, 0);
            const float z = f_sz((zh >> (n * V * PER + i)) & 1u);
            if (fold) s = f_add(s, z);
            const float mu = f_mul_ftz(s, f_pow2neg(LV));
            const unsigned short mw = m16 ? f2bf(mu) : 0;
            if constexpr (LV > 0)
                if (m16) {
                    const float df = f_sub(mu, bf2f(mw));
                    e2 = f_add(e2, f_mul(df, df));
                }
            if (wr) { if (m16) mpb[32 * i] = mw; else mp[32 * i] = mu; }
            if constexpr (LV > 0) {
                xcpb |= sgn(f_mul(f_sub(z, mu), 0.0f)) << ((n * PER + i) & 31);
#pragma unroll
                for (int q = 0; q < V; q++) x[n * V + q][i] = f_sub(x[n * V + q][i], mu);
                if constexpr (KEEPMU) mus[n][i] = mu;
            }
        }
        if constexpr (LV > 0)
            if (m16) {
#pragma unroll
                for (int o = 16; o >= 1; o >>= 1) e2 = f_add(e2, __shfl_xor_sync(FULL, e2, o));
                epsw[n] = f_sqrt_ftz(e2);
            }
    }
    if constexpr (LV == 0) {
        // split tables store the leaf row itself in bf16: its radius picks up the rounding slack
        auto sqe = [&](const int r) {
            const float d0 = f_sub(x[r][0], bf2f(f2bf(x[r][0])));
            float a = f_mul(d0, d0);
#if PER == 2
            const float d1 = f_sub(x[r][1], bf2f(f2bf(x[r][1])));
            a = f_add(a, f_mul(d1, d1));
#endif
            return a;
        };
        float le = 0.0f;
        if (m16) le = f_sqrt_ftz(rs16_lazy(sqe, lane));
        float rad = has_sp ? f_add(0.0f, sp) : 0.0f;
        if (has_sp && fold) rad = f_max(rad, 0.0f);
        if ((lane & 1) == 0 && node_full(R, gsh + row)) {
            write_scalars(R, gsh + row, 0, rad, f_max(ce, 0.0f), cell, le);
            int rr, h, idx, j;
            rec_pos(R, gsh + row, 0, rr, h, idx, j);
            R.leaf[R.leafbase + j * (1 << R.k) + idx] = (i64)cell;
        }
        return;
    } else {
        auto sq = [&](const int r) {
            float a = f_mul(x[r][0], x[r][0]);
#if PER == 2
            a = f_add(a, f_mul(x[r][1], x[r][1]));
#endif
            return a;
        };
        const float d2 = rs16_lazy(sq, lane);
        float rq = has_sp ? f_add(f_sqrt_ftz(d2), sp) : d2;
        float am = ce, bd = d2;
        int pm = cell, far = row;
#pragma unroll
        for (int o = V; o >= 2; o >>= 1) {
            const float rq2 = __shfl_xor_sync(FULL, rq, o), am2 = __shfl_xor_sync(FULL, am, o);
            const float bd2 = __shfl_xor_sync(FULL, bd, o);
            const int pm2 = __shfl_xor_sync(FULL, pm, o), far2 = __shfl_xor_sync(FULL, far, o);
            rq = f_max(rq, rq2); am = f_max(am, am2); pm = min(pm, pm2);
            if (bd2 > bd || (bd2 == bd && far2 < far)) { bd = bd2; far = far2; }
        }
        if ((lane & (2 * V - 1)) == 0 && node_full(R, gsh + row)) {
            float eps = 0.0f;
            if (m16) {
#pragma unroll
                for (int un = 0; un < NN; un++) eps = (un == (row >> LV)) ? epsw[un] : eps;
            }
            float rad = has_sp ? rq : f_sqrt_ftz(rq);
            if (has_sp && fold) rad = f_max(rad, 0.0f);
            write_scalars(R, gsh + row, LV, rad, f_max(am, 0.0f), pm, eps);
        }
        float v[NN][PER];
        const float den0 = f_max(f_sqrt_ftz(bd), 1e-6f);
#pragma unroll
        for (int n = 0; n < NN; n++) {
            const float dn = __shfl_sync(FULL, den0, 2 * n * V);
            if constexpr (KEEPMU) {
                const int of = __shfl_sync(FULL, org, 2 * __shfl_sync(FULL, far, 2 * n * V));
#pragma unroll
                for (int i = 0; i < PER; i++) v[n][i] = f_div_ftz(f_sub(Xw[of * XS + lane + 32 * i], mus[n][i]), dn);
            } else {
                const int fq = __shfl_sync(FULL, far, 2 * n * V) - n * V;
#pragma unroll
                for (int i = 0; i < PER; i++) {
                    float s = x[n * V][i];
#pragma unroll
                    for (int q = 1; q < V; q++) s = (fq == q) ? x[n * V + q][i] : s;
                    v[n][i] = f_div_ftz(s, dn);
                }
            }
        }
        auto dotv = [&](const int r) {
            float a = f_mul(x[r][0], v[r / V][0]);
#if PER == 2
            a = f_add(a, f_mul(x[r][1], v[r / V][1]));
#endif
            return a;
        };
#pragma unroll
        for (int it = 0; it < 2; it++) {
            const float pv = rs16_lazy(dotv, lane);
            unsigned negm = 0;                      // bit n * PER: node n's pad term Xc_pad . v is -0 (every product -0)
            if (fold) {
                unsigned vs = 0;
#pragma unroll
                for (int n = 0; n < NN; n++)
#pragma unroll
                    for (int i = 0; i < PER; i++) vs |= sgn(v[n][i]) << (n * PER + i);
                unsigned dd = xcpb ^ vs;            // bit set: that dim's product (z - mu) * 0 * v is -0
#if PER == 2
                dd &= dd >> 1;
#endif
                negm = __reduce_and_sync(FULL, dd);
            }
#pragma unroll
            for (int n = 0; n < NN; n++) {
                auto term = [&](const int q, float* o) {
                    const float pr = __shfl_sync(FULL, pv, 2 * (n * V + q));
#pragma unroll
                    for (int i = 0; i < PER; i++) o[i] = f_mul(x[n * V + q][i], pr);
                };
                HRV<0, LV>::run(term, 0, v[n]);
                if (fold) {
#pragma unroll
                    for (int i = 0; i < PER; i++)
                        v[n][i] = f_add(v[n][i], __uint_as_float((((xcpb >> (n * PER + i)) ^ (negm >> (n * PER))) & 1u) << 31));
                }
            }
            unit_nodes<NN>(v, lane);
        }
        const float proj = rs16_lazy(dotv, lane);
        unsigned kb = __float_as_uint(proj == 0.0f ? 0.0f : proj);
        kb = (kb & 0x80000000u) ? ~kb : (kb | 0x80000000u);
        int rank = 0;
#pragma unroll
        for (int o = 1; o < V; o++) {
            const int pr = (row & ~(V - 1)) | ((row + o) & (V - 1));
            const unsigned k2 = __shfl_sync(FULL, kb, 2 * pr);
            rank += (k2 < kb) || (k2 == kb && pr < row);
        }
        if ((lane & 1) == 0) pw[(row & ~(V - 1)) + rank] = row;
        __syncwarp();
        const int src = 2 * pw[row];
        cell = __shfl_sync(FULL, cell, src); ce = __shfl_sync(FULL, ce, src); sp = __shfl_sync(FULL, sp, src);
        org = __shfl_sync(FULL, org, src);
        if ((lane & 1) == 0) po[row] = org;
        __syncwarp();
#pragma unroll
        for (int q = 0; q < 16; q++) {
            const int oq = po[q];
#pragma unroll
            for (int i = 0; i < PER; i++) x[q][i] = Xw[oq * XS + lane + 32 * i];
        }
        __syncwarp();
    }
}

// levels lv0 (<= 4) .. 0 of the 16-slot half h of a warp's 32 slots (lane l: slot l's cell, ceiling, spread and load
// order org); rows from Xw via org
__device__ __forceinline__ void h_half(const int lv0, const int h, const int cell32, const float ce32, const float sp32,
                                       const int org32, const u64 zb, const float* Xw, int* pw, int* po, const Rec& R,
                                       const int lane, const int gs0, const bool fold, const bool has_sp) {
    const int src = 16 * h + (lane >> 1);
    int cell = __shfl_sync(FULL, cell32, src), org = __shfl_sync(FULL, org32, src);
    float ce = __shfl_sync(FULL, ce32, src), sp = __shfl_sync(FULL, sp32, src);
    const unsigned zh = (unsigned)(zb >> (16 * PER * h));
    float x[16][PER];
#pragma unroll
    for (int q = 0; q < 16; q++) {
        const int oq = __shfl_sync(FULL, org, 2 * q);
#pragma unroll
        for (int i = 0; i < PER; i++) x[q][i] = Xw[oq * XS + lane + 32 * i];
    }
    const int gsh = gs0 + 16 * h;
    if (lv0 >= 4) { h_level<4>(x, cell, ce, sp, org, zh, Xw, pw, po, R, lane, gsh, fold, has_sp); DBG_MARK(10 + 4); }
    if (lv0 >= 3) { h_level<3>(x, cell, ce, sp, org, zh, Xw, pw, po, R, lane, gsh, fold, has_sp); DBG_MARK(10 + 3); }
    if (lv0 >= 2) { h_level<2>(x, cell, ce, sp, org, zh, Xw, pw, po, R, lane, gsh, fold, has_sp); DBG_MARK(10 + 2); }
    if (lv0 >= 1) { h_level<1>(x, cell, ce, sp, org, zh, Xw, pw, po, R, lane, gsh, fold, has_sp); DBG_MARK(10 + 1); }
    h_level<0>(x, cell, ce, sp, org, zh, Xw, pw, po, R, lane, gsh, fold, has_sp); DBG_MARK(10);
}

// levels lv0 (<= 4) .. 0 of both 16-slot halves of a warp's 32 slots, one half at a time
__device__ __forceinline__ void h_phase(const int lv0, const int cell32, const float ce32, const float sp32, const int org32,
                                        const u64 zb, const float* Xw, int* pw, int* po, const Rec& R, const int lane,
                                        const int gs0, const bool fold, const bool has_sp) {
#pragma unroll 1
    for (int h = 0; h < 2; h++) h_half(lv0, h, cell32, ce32, sp32, org32, zb, Xw, pw, po, R, lane, gs0, fold, has_sp);
}

// W phase from nodes of 2^lv0 rows: the full-tile level 5 (x holds the rows), then the half-tile levels
__device__ __forceinline__ void w_phase(const int lv0, float (&x)[32][PER], int& cell, float& ce, float& sp, int& org,
                                         const u64 zb, const float* Xw, int* pw, int* po, const Rec& R, const int lane,
                                         const int gs0, const bool fold, const bool has_sp) {
    if (lv0 >= 5) { w_level<5, false>(x, cell, ce, sp, org, zb, Xw, pw, po, R, lane, gs0, fold, has_sp); DBG_MARK(10 + 5); }
    h_phase(min(lv0, 4), cell, ce, sp, org, zb, Xw, pw, po, R, lane, gs0, fold, has_sp);
}
// >>> SECTION: _GPHASE <<<

// ---- G phase: one node of V >= 1024 rows per CTA, rows streamed from L2 through the node's slot -> cell list ----
// warp w owns slots w + 16 m (m = 0 .. V/16 - 1), visited in batches of 32 in bit-reversed order of m: a batch is a
// complete height-5 subtree of the slot halving (reduced in registers), batches combine pairwise (register stack),
// then the four cross-warp steps (w with w + 8, + 4, + 2, + 1) run in shared memory.
struct GS {                    // shared-memory carve-up of a G CTA (end: one past the last float, GSMEM_FLOATS)
    u64* keys; int* cell; float* wpart; float* mu; float* v; float* ws; int* misc; float* pvp; float* end;
};
__device__ __forceinline__ GS gcarve(float* smem) {
    GS S;
    float* p = smem;
    S.keys = (u64*)p; p += 2 * 8192;
    S.cell = (int*)p; p += 8192;
    S.wpart = p; p += 16 * DH;
    S.mu = p; p += DH;
    S.v = p; p += DH;
    S.ws = p; p += 16 * 5;
    S.misc = (int*)p; p += 16;
    S.pvp = p; p += 16;
    S.end = p;
    return S;
}
#define GSMEM_FLOATS (2 * 8192 + 8192 + 16 * DH + 2 * DH + 16 * 5 + 32)   // more than SMEM_FLOATS for DH = 32
// rows of batch b of warp w: x[i] = Kbar row of slot w + 16 * bitrev_q(32 b + i)
__device__ __forceinline__ void g_load(float (&x)[32][PER], const GS& S, const float* kt, const int w, const int b,
                                       const int q, const int lane) {
#pragma unroll
    for (int i = 0; i < 32; i++) {
        const int m = (int)(__brev((unsigned)(32 * b + i)) >> (32 - q));
        const float* p = kt + (i64)S.cell[w + 16 * m] * DH + lane;
#pragma unroll
        for (int ii = 0; ii < PER; ii++) x[i][ii] = __ldg(p + 32 * ii);
    }
}
// pairwise tree over the 32 batch rows in visit order ((i, i^1) first), per dim
template <class F>
__device__ __forceinline__ void batch_tree(F& val, float* out) {
    auto f = [&](const int i, float* o) { val(i, o); };
    // HRV pairs m with m + 2^(Q-1) first; visit order i = bitrev5(m) turns that into (i, i ^ 1) first
    auto g = [&](const int m, float* o) { f((int)(__brev((unsigned)m) >> 27), o); };
    HRV<0, 5>::run(g, 0, out);
}
// register stack over batches: pending[j] holds a finished subtree of 2^j batches (pairs b, b ^ 1 first)
struct BStack {
    float s[5][PER];
    unsigned pend;
};
__device__ __forceinline__ void bstack_push(BStack& st, const float* cur_in) {
    float cur[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) cur[i] = cur_in[i];
    bool carry = true;
#pragma unroll
    for (int j = 0; j < 5; j++) {
        if (carry) {
            if ((st.pend >> j) & 1u) {
#pragma unroll
                for (int i = 0; i < PER; i++) cur[i] = f_add(st.s[j][i], cur[i]);
                st.pend &= ~(1u << j);
            } else {
#pragma unroll
                for (int i = 0; i < PER; i++) st.s[j][i] = cur[i];
                st.pend |= 1u << j;
                carry = false;
            }
        }
    }
}
// after nb (power of two) pushes exactly one subtree is pending: the warp's halving over its slots
__device__ __forceinline__ void bstack_top(const BStack& st, const int nb, float* out) {
    const int j = 31 - __clz(nb);
#pragma unroll
    for (int jj = 0; jj < 5; jj++)
        if (jj == j)
#pragma unroll
            for (int i = 0; i < PER; i++) out[i] = st.s[jj][i];
}
// cross-warp halving of the 16 warp partials (w with w + 8, then + 4, + 2, + 1); thread d < DH gets dim d
__device__ __forceinline__ float g_cross(const GS& S, const int d) {
    float a[16];
#pragma unroll
    for (int w = 0; w < 16; w++) a[w] = S.wpart[w * DH + d];
#pragma unroll
    for (int h = 8; h >= 1; h >>= 1)
#pragma unroll
        for (int w = 0; w < h; w++) a[w] = f_add(a[w], a[w + h]);
    return a[0];
}

template <bool LEAN = false>
__device__ void g_node(const Args& A, const GS& S, const int f, const int t, const int k, const int j, const int l,
                       const int n, const i64* SRC64, const int* SRC, int* DST, const bool fold, const bool has_sp) {
    const int tid = threadIdx.x, lane = tid & 31, wid = UNI(tid >> 5);
    const int lvn = k - l, V = 1 << lvn, q = lvn - 4, nb = V >> 9;
    const float* kt = A.kb + (i64)t * A.M * DH;
    const float* cet = A.ce + (i64)t * A.M;
    const float* spt = A.spr + (i64)t * A.M;
    for (int s = tid; s < V; s += 512) S.cell[s] = SRC64 != nullptr ? (int)SRC64[s] : __ldcg(SRC + s);
    __syncthreads();
    float zl[PER];
    {
        const int c0 = (int)FSEL(A.adm, f)[(i64)t * A.M + ((i64)j << k)];
#pragma unroll
        for (int i = 0; i < PER; i++) zl[i] = f_sz(sgn(kt[(i64)c0 * DH + lane + 32 * i]));
    }
    float x[32][PER];
    // ---- pass 1: mean
    {
        BStack st; st.pend = 0;
        for (int b = 0; b < nb; b++) {
            g_load(x, S, kt, wid, b, q, lane);
            auto val = [&](const int i, float* o) {
#pragma unroll
                for (int ii = 0; ii < PER; ii++) o[ii] = fold ? f_add(x[i][ii], zl[ii]) : x[i][ii];
            };
            float bs[PER];
            batch_tree(val, bs);
            bstack_push(st, bs);
        }
        float ws_[PER];
        bstack_top(st, nb, ws_);
#pragma unroll
        for (int i = 0; i < PER; i++) S.wpart[wid * DH + lane + 32 * i] = ws_[i];
        __syncthreads();
        if (tid < DH) {
            const float mu = f_mul_ftz(g_cross(S, tid), f_pow2neg(lvn));
            S.mu[tid] = mu;
        }
        __syncthreads();
    }
    float mul[PER], xcp[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) {
        mul[i] = S.mu[lane + 32 * i];
        xcp[i] = f_mul(f_sub(zl[i], mul[i]), 0.0f);
    }
    // ---- pass 2: squared distances, summaries, farthest slot
    {
        float rq = 0.0f, am = 0.0f, bd = -1.0f;
        int pm = 0x7fffffff, far = 0x7fffffff;
        bool first = true;
        for (int b = 0; b < nb; b++) {
            g_load(x, S, kt, wid, b, q, lane);
#pragma unroll
            for (int i = 0; i < 32; i++)
#pragma unroll
                for (int ii = 0; ii < PER; ii++) x[i][ii] = f_sub(x[i][ii], mul[ii]);
            auto sq = [&](const int r) {
                float a = f_mul(x[r][0], x[r][0]);
#if PER == 2
                a = f_add(a, f_mul(x[r][1], x[r][1]));
#endif
                return a;
            };
            const float d2 = rs_lazy(sq, lane);
            const int m = (int)(__brev((unsigned)(32 * b + lane)) >> (32 - q));
            const int slot = wid + 16 * m, c = S.cell[slot];
            const float r2 = has_sp ? f_add(f_sqrt_ftz(d2), spt[c]) : d2;
            const float a2 = cet[c];
            if (first) { rq = r2; am = a2; bd = d2; pm = c; far = slot; first = false; }
            else {
                rq = f_max(rq, r2); am = f_max(am, a2); pm = min(pm, c);
                if (d2 > bd || (d2 == bd && slot < far)) { bd = d2; far = slot; }
            }
        }
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            const float rq2 = __shfl_xor_sync(FULL, rq, o), am2 = __shfl_xor_sync(FULL, am, o);
            const float bd2 = __shfl_xor_sync(FULL, bd, o);
            const int pm2 = __shfl_xor_sync(FULL, pm, o), far2 = __shfl_xor_sync(FULL, far, o);
            rq = f_max(rq, rq2); am = f_max(am, am2); pm = min(pm, pm2);
            if (bd2 > bd || (bd2 == bd && far2 < far)) { bd = bd2; far = far2; }
        }
        if (lane == 0) {
            float* w5 = S.ws + 5 * wid;
            w5[0] = bd; w5[1] = __int_as_float(far); w5[2] = rq; w5[3] = am; w5[4] = __int_as_float(pm);
        }
        __syncthreads();
        if (tid == 0) {
            const float* w5 = S.ws;
            float bd0 = w5[0], rq0 = w5[2], am0 = w5[3];
            int far0 = __float_as_int(w5[1]), pm0 = __float_as_int(w5[4]);
            for (int w = 1; w < 16; w++) {
                w5 += 5;
                const float bd2 = w5[0];
                const int far2 = __float_as_int(w5[1]);
                rq0 = f_max(rq0, w5[2]); am0 = f_max(am0, w5[3]); pm0 = min(pm0, __float_as_int(w5[4]));
                if (bd2 > bd0 || (bd2 == bd0 && far2 < far0)) { bd0 = bd2; far0 = far2; }
            }
            S.misc[0] = far0;
            float rad = has_sp ? rq0 : f_sqrt_ftz(rq0);
            if (has_sp && fold) rad = f_max(rad, 0.0f);
            // record scalars (split table: radius carries the bf16 mean-rounding slack; the fp32 mean built the node)
            const int jo = LEAN ? 0 : j;
            const int nodes = (2 << k) - 1, rb = A.lay[k] + jo * nodes, h = (1 << l) - 1 + n;
            float* p = FSEL(A.node_mu, f) != nullptr ? FSEL(A.node_aux, f) + ((i64)t * A.Ntot + rb + h) * NA
                                                     : FSEL(A.node, f) + ((i64)t * A.Ntot + rb + h) * NC + DH;
            float eps = 0.0f;
            if (FSEL(A.node_mu, f) != nullptr) {
                float s2 = 0.0f;
#pragma unroll 8
                for (int dd = 0; dd < DH; dd++) {
                    const float df = f_sub(S.mu[dd], bf2f(f2bf(S.mu[dd])));
                    s2 = f_add(s2, f_mul(df, df));
                }
                eps = f_sqrt_ftz(s2);
            }
            p[0] = FSEL(A.node_mu, f) != nullptr ? f_add(rad, eps) : rad; p[1] = f_max(am0, 0.0f); p[2] = (float)pm0; p[3] = (float)V;
            p[4] = (float)(rb + 2 * h + 1); p[5] = (float)(rb + 2 * h + 2); p[6] = 1.0f;
            p[7] = (float)(A.lay[16 + k] + jo * (1 << k) + n * V); p[8] = (float)V;
        }
        if (tid < DH) {
            const int jo = LEAN ? 0 : j;
            const int nodes = (2 << k) - 1, rb = A.lay[k] + jo * nodes, h = (1 << l) - 1 + n;
            if (FSEL(A.node_mu, f) != nullptr)
                FSEL(A.node_mu, f)[((i64)t * A.Ntot + rb + h) * DH + tid] = f2bf(S.mu[tid]);
            else
                FSEL(A.node, f)[((i64)t * A.Ntot + rb + h) * NC + tid] = S.mu[tid];
        }
        __syncthreads();
    }
    // ---- v0 = unit(Xc[far]); pad-term sign for the first v-sum
    if (wid == 0) {
        const int c = S.cell[S.misc[0]];
        float vv[PER];
        bool alln = true;
#pragma unroll
        for (int i = 0; i < PER; i++) vv[i] = f_sub(kt[(i64)c * DH + lane + 32 * i], mul[i]);
        unit_inplace(vv);
#pragma unroll
        for (int i = 0; i < PER; i++) { S.v[lane + 32 * i] = vv[i]; alln = alln && (sgn(xcp[i]) != sgn(vv[i])); }
        const bool an = __all_sync(FULL, alln);
        if (lane == 0) S.pvp[0] = f_sz(an);
    }
    __syncthreads();
    // ---- passes 3, 4: v <- unit(sum_s Xc_s (Xc_s . v)), pad term folded first
#pragma unroll 1
    for (int it = 0; it < 2; it++) {
        float vl[PER], u[PER];
#pragma unroll
        for (int i = 0; i < PER; i++) { vl[i] = S.v[lane + 32 * i]; u[i] = f_mul(xcp[i], S.pvp[0]); }
        BStack st; st.pend = 0;
        for (int b = 0; b < nb; b++) {
            g_load(x, S, kt, wid, b, q, lane);
#pragma unroll
            for (int i = 0; i < 32; i++)
#pragma unroll
                for (int ii = 0; ii < PER; ii++) x[i][ii] = f_sub(x[i][ii], mul[ii]);
            auto dv = [&](const int r) {
                float a = f_mul(x[r][0], vl[0]);
#if PER == 2
                a = f_add(a, f_mul(x[r][1], vl[1]));
#endif
                return a;
            };
            const float pv = rs_lazy(dv, lane);
            auto val = [&](const int i, float* o) {
                const float pr = __shfl_sync(FULL, pv, i);
#pragma unroll
                for (int ii = 0; ii < PER; ii++) {
                    const float tt = f_mul(x[i][ii], pr);
                    o[ii] = fold ? f_add(tt, u[ii]) : tt;
                }
            };
            float bs[PER];
            batch_tree(val, bs);
            bstack_push(st, bs);
        }
        float ws_[PER];
        bstack_top(st, nb, ws_);
#pragma unroll
        for (int i = 0; i < PER; i++) S.wpart[wid * DH + lane + 32 * i] = ws_[i];
        __syncthreads();
        if (tid < DH) S.v[tid] = g_cross(S, tid);
        __syncthreads();
        if (wid == 0) {
            float vv[PER];
            bool alln = true;
#pragma unroll
            for (int i = 0; i < PER; i++) vv[i] = S.v[lane + 32 * i];
            unit_inplace(vv);
#pragma unroll
            for (int i = 0; i < PER; i++) { S.v[lane + 32 * i] = vv[i]; alln = alln && (sgn(xcp[i]) != sgn(vv[i])); }
            const bool an = __all_sync(FULL, alln);
            if (lane == 0) S.pvp[0] = f_sz(an);
        }
        __syncthreads();
    }
    // ---- pass 5: projections -> keys; bitonic sort; sorted cells -> DST
    {
        float vl[PER];
#pragma unroll
        for (int i = 0; i < PER; i++) vl[i] = S.v[lane + 32 * i];
        for (int b = 0; b < nb; b++) {
            g_load(x, S, kt, wid, b, q, lane);
#pragma unroll
            for (int i = 0; i < 32; i++)
#pragma unroll
                for (int ii = 0; ii < PER; ii++) x[i][ii] = f_sub(x[i][ii], mul[ii]);
            auto dv = [&](const int r) {
                float a = f_mul(x[r][0], vl[0]);
#if PER == 2
                a = f_add(a, f_mul(x[r][1], vl[1]));
#endif
                return a;
            };
            const float proj = rs_lazy(dv, lane);
            const int m = (int)(__brev((unsigned)(32 * b + lane)) >> (32 - q));
            const int slot = wid + 16 * m;
            S.keys[slot] = sort_key(proj, (unsigned)slot);
        }
        __syncthreads();
        for (int size = 2; size <= V; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int pp = tid; pp < V / 2; pp += 512) {
                    const int i = 2 * pp - (pp & (stride - 1)), jx = i + stride;
                    const bool asc = size == V || (i & size) == 0;
                    const u64 a = S.keys[i], bk = S.keys[jx];
                    if ((a > bk) == asc) { S.keys[i] = bk; S.keys[jx] = a; }
                }
                __syncthreads();
            }
        }
        for (int p = tid; p < V; p += 512) DST[p] = S.cell[(int)(unsigned)S.keys[p]];
        __syncthreads();
    }
}
// >>> SECTION: _BHELPERS <<<

// ---- batch helpers of the build kernels (fsb_forest_build*, fsb_forest_build_small) ----
// records of the n chunks ja + (c << jsh) of height k that hold no admitted cell: mu = signed zeros of Kbar[t, 0] (the
// cell an empty slot reads), rad = amax = 0, pmin = 2^30, static fields; leaves -1. One warp per node record, warps
// wid of nw (the CTA's warps by default)
__device__ void empty_chunks(const Args& A, const int f, const int t, const int k, const int ja, const int n,
                             const int jsh, const int wid = threadIdx.x >> 5, const int nw = blockDim.x >> 5) {
    if (n <= 0) return;
    const int lane = threadIdx.x & 31;
    const int nodes = (2 << k) - 1, base = A.lay[k], lbase = A.lay[16 + k];
    const bool m16 = FSEL(A.node_mu, f) != nullptr;
    float* nt = m16 ? nullptr : FSEL(A.node, f) + (i64)t * A.Ntot * NC;
    unsigned short* ntm = m16 ? FSEL(A.node_mu, f) + (i64)t * A.Ntot * DH : nullptr;
    float* nta = m16 ? FSEL(A.node_aux, f) + (i64)t * A.Ntot * NA : nullptr;
    float z[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) z[i] = f_sz(sgn(A.kb[(i64)t * A.M * DH + lane + 32 * i]));
    const int tot = n * nodes;
    for (int it = wid; it < tot; it += nw) {
        {
            const int jj = ja + ((it / nodes) << jsh), h = it - (it / nodes) * nodes;
            const int rb = base + jj * nodes;
            const int l = 31 - __clz(h + 1), idx = h + 1 - (1 << l), V = 1 << (k - l);
            if (m16) {
                unsigned short* pm = ntm + (i64)(rb + h) * DH;
#pragma unroll
                for (int i = 0; i < PER; i++) pm[lane + 32 * i] = f2bf(z[i]);
                if (lane < 9) {
                    float val;
                    switch (lane) {
                        case 0: case 1: val = 0.0f; break;
                        case 2: val = 1073741824.0f; break;
                        case 3: case 8: val = (float)V; break;
                        case 4: val = l < k ? (float)(rb + 2 * h + 1) : -1.0f; break;
                        case 5: val = l < k ? (float)(rb + 2 * h + 2) : -1.0f; break;
                        case 6: val = 1.0f; break;
                        default: val = (float)(lbase + jj * (1 << k) + idx * V); break;
                    }
                    nta[(i64)(rb + h) * NA + lane] = val;
                }
            } else {
                float* p = nt + (i64)(rb + h) * NC;
#pragma unroll
                for (int i = 0; i < PER; i++) p[lane + 32 * i] = z[i];
                if (lane < 9) {
                    float val;
                    switch (lane) {
                        case 0: case 1: val = 0.0f; break;
                        case 2: val = 1073741824.0f; break;
                        case 3: case 8: val = (float)V; break;
                        case 4: val = l < k ? (float)(rb + 2 * h + 1) : -1.0f; break;
                        case 5: val = l < k ? (float)(rb + 2 * h + 2) : -1.0f; break;
                        case 6: val = 1.0f; break;
                        default: val = (float)(lbase + jj * (1 << k) + idx * V); break;
                    }
                    p[DH + lane] = val;
                }
            }
        }
    }
    i64* lt = FSEL(A.leaf, f) + (i64)t * A.Ltot + lbase;
    for (int e2 = 32 * wid + lane; e2 < n << k; e2 += 32 * nw)
        lt[((ja + ((e2 >> k) << jsh)) << k) + (e2 & ((1 << k) - 1))] = -1;
}

// adm position of batch slot s: the slot's chunk i = s >> k of the batch is chunk j0 + (i << jsh)
__device__ __forceinline__ int slot_pos(const int j0, const int k, const int jsh, const int s) {
    return ((j0 + ((s >> k) << jsh)) << k) + (s & ((1 << k) - 1));
}
// the batch's chunks that are full (j0 + (i << jsh) < mt >> k, a prefix of the batch)
__device__ __forceinline__ int batch_full(const int mt, const int k, const int j0, const int nsub, const int jsh) {
    return max(0, min(((mt >> k) - j0 + (1 << jsh) - 1) >> jsh, nsub));
}

// rows of the 512 slots from S.cell (cell 0's row for -1 slots: finite filler, never recorded); all loads first
__device__ __forceinline__ void load_rows(const SM& S, const float* kt, const int tid) {
    constexpr int Q = DH / 4, U = 512 * Q / 512;
    float4 buf[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
        const int e = tid + 512 * u, s = e / Q, q = e - s * Q;
        buf[u] = __ldg(reinterpret_cast<const float4*>(kt + (i64)max(S.cell[s], 0) * DH) + q);
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
        const int e = tid + 512 * u, s = e / Q, q = e - s * Q;
        *reinterpret_cast<float4*>(S.Xs + s * XS + 4 * q) = buf[u];
    }
}
// sign bits (bit d) of row `row` of S.Xs into S.zb[ci]: the pad zero of the chunk whose first cell sits there
__device__ __forceinline__ void zb_of_row(const SM& S, const int ci, const int row, const int lane) {
    const unsigned lo = __ballot_sync(FULL, sgn(S.Xs[row * XS + lane]));
    const unsigned hi = DH == 64 ? __ballot_sync(FULL, sgn(S.Xs[row * XS + lane + (DH == 64 ? 32 : 0)])) : 0u;
    if (lane == 0) S.zb[ci] = (u64)lo | ((u64)hi << 32);
}

// >>> SECTION: _BUILD <<<

// ---- K1: forest construction tasks (512 threads; dynamic shared memory) ----
// task record (8 ints): type, f, t, k, j0, nsub, l0, n0.  type 0: batch of nsub chunks j0 + (i << l0) of height
// k <= 9 (nsub << k <= 512 slots; l0 = 0: consecutive chunks, 1: every other chunk), for k <= 5 of the n0 heads
// t .. t + n0 - 1 (0: one head); LAY: bases[16] | leafbases[16] | fold mask (bit k: Kpad(k) > k)

// type 0, k <= 5: a batch of nsub chunks of height k, one warp per 32 slots. np > 1 packs the same chunks of the np
// heads t0 .. t0 + np - 1 into one block, 16 / 2^ceil(log2 np) warps per head (the table keeps each head's slots
// within them); warps are independent here, so each one is a warp of that head's own batch
template <bool LEAN = false>
__device__ __forceinline__ void task_wbatch(const Args& A, const int f, const int t0, const int k, const int j0,
                                         const int nsub, const int jsh, const int np, const bool fold) {
    extern __shared__ float smem[];
    const SM S = carve(smem);
    const int tid = threadIdx.x, lane = tid & 31, wid = UNI(tid >> 5);
    const int M = A.M, has_sp = A.has_sp;
    const int jo = LEAN ? 0 : j0;
    for (int p = 0; p < np; p++) {
        const int nf_p = batch_full((int)FSEL(A.mt, f)[t0 + p], k, j0, nsub, jsh);
        empty_chunks(A, f, t0 + p, k, jo + (nf_p << jsh), nsub - nf_p, jsh);
    }
    const int pb = np > 1 ? 32 - __clz(np - 1) : 0;       // the head of warp wid: wid >> (4 - pb)
    const int t = t0 + (wid >> (4 - pb));
    if (t >= t0 + np) return;
    const int mt = UNI((int)FSEL(A.mt, f)[t]);
    const int nfull = batch_full(mt, k, j0, nsub, jsh);
    const int gs0 = 32 * (wid & ((16 >> pb) - 1));       // the warp's first slot in its head's batch
    if (gs0 >= (nfull << k)) return;
    Rec R;
    R.node = FSEL(A.node_mu, f) != nullptr ? nullptr : FSEL(A.node, f) + (i64)t * A.Ntot * NC;
    R.node_mu = FSEL(A.node_mu, f) != nullptr ? FSEL(A.node_mu, f) + (i64)t * A.Ntot * DH : nullptr;
    R.node_aux = FSEL(A.node_aux, f) != nullptr ? FSEL(A.node_aux, f) + (i64)t * A.Ntot * NA : nullptr;
    R.leaf = FSEL(A.leaf, f) + (i64)t * A.Ltot;
    R.k = k; R.ks = k; R.l0 = 0; R.n0 = 0; R.jb = jo; R.jsh = jsh;
    R.base = A.lay[k]; R.leafbase = A.lay[16 + k];
    R.nfull = nfull; R.zsh = min(k, 5);
    float* Xw = S.Xs + wid * 32 * XS;
    int* pw = S.pw + wid * 32;
    float x[32][PER];
    u64 zb = 0;
    const int s = gs0 + lane;
    const bool ok = s < (nfull << k);
    int cell = ok ? (int)FSEL(A.adm, f)[(i64)t * M + slot_pos(j0, k, jsh, s)] : -1;
    float ce = ok ? A.ce[(i64)t * M + cell] : 0.0f;
    float sp = (ok && has_sp) ? A.spr[(i64)t * M + cell] : 0.0f;
    const float* kt = A.kb + (i64)t * M * DH + lane;
#pragma unroll
    for (int r = 0; r < 32; r++) {
        const float* p = kt + (i64)max(__shfl_sync(FULL, cell, r), 0) * DH;
#pragma unroll
        for (int i = 0; i < PER; i++) x[r][i] = __ldg(p + 32 * i);
    }
    unsigned zc[PER];                        // pad sign of the current chunk (its first row), per dim
#pragma unroll
    for (int r = 0; r < 32; r++)
#pragma unroll
        for (int i = 0; i < PER; i++) {
            Xw[r * XS + lane + 32 * i] = x[r][i];
            if ((r & ((1 << k) - 1)) == 0) zc[i] = sgn(x[r][i]);
            zb |= (u64)zc[i] << (r * PER + i);
        }
    __syncwarp();
    DBG_MARK(1);
    int org = lane;
    w_phase(k, x, cell, ce, sp, org, zb, Xw, pw, S.cell + 32 * wid, R, lane, gs0, fold, has_sp != 0);
}

// the C phase from nodes of 2^lv0 rows (rows already in S.Xs / x), then each warp's W phase; zrow(i): the S.zb entry
// of the chunk of warp row block i
__device__ __forceinline__ void cw_phases(const int lv0, float (&x)[32][PER], const SM& S, const Rec& R, const float* kt,
                                          const int tid,
                                          const int lane, const int wid, const int nrow, const int zshift,
                                          const bool fold, const bool has_sp) {
    c_phase(lv0, x, S, R, kt, tid, lane, wid, fold, has_sp);
    const int gs0 = 32 * wid;
    if (gs0 >= nrow) return;
    int cell = S.cell[gs0 + lane];
    float ce = S.ce[gs0 + lane], sp = S.sp[gs0 + lane];
    u64 zb = 0;
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const u64 bit = (S.zb[gs0 >> zshift] >> (lane + 32 * i)) & 1ull;
#pragma unroll
        for (int r = 0; r < 32; r++) zb |= bit << (r * PER + i);
    }
    DBG_MARK(1);
    int org = lane;                                       // S.cell's warp slice is free now: the org list
    w_phase(5, x, cell, ce, sp, org, zb, S.Xs + wid * 32 * XS, S.pw + wid * 32, S.cell + gs0, R, lane, gs0, fold,
            has_sp);
}

// type 0, 6 <= k <= 9: a batch of nsub chunks of height k (512 slots), C phase then W phase
template <bool LEAN = false>
__device__ __forceinline__ void task_cbatch(const Args& A, const int f, const int t, const int k, const int j0, const int nsub,
                                         const int jsh, const bool fold) {
    extern __shared__ float smem[];
    const SM S = carve(smem);
    const int tid = threadIdx.x, lane = tid & 31, wid = UNI(tid >> 5);
    const int M = A.M, has_sp = A.has_sp;
    const int mt = UNI((int)FSEL(A.mt, f)[t]);
    const int nfull = batch_full(mt, k, j0, nsub, jsh);
    const int jo = LEAN ? 0 : j0;
    empty_chunks(A, f, t, k, jo + (nfull << jsh), nsub - nfull, jsh);
    if (nfull == 0) return;
    Rec R;
    R.node = FSEL(A.node_mu, f) != nullptr ? nullptr : FSEL(A.node, f) + (i64)t * A.Ntot * NC;
    R.node_mu = FSEL(A.node_mu, f) != nullptr ? FSEL(A.node_mu, f) + (i64)t * A.Ntot * DH : nullptr;
    R.node_aux = FSEL(A.node_aux, f) != nullptr ? FSEL(A.node_aux, f) + (i64)t * A.Ntot * NA : nullptr;
    R.leaf = FSEL(A.leaf, f) + (i64)t * A.Ltot;
    R.k = k; R.ks = k; R.l0 = 0; R.n0 = 0; R.jb = jo; R.jsh = jsh;
    R.base = A.lay[k]; R.leafbase = A.lay[16 + k];
    R.nfull = nfull; R.zsh = 5;
    const int nrow = nfull << k;
    {
        const int c = tid < nrow ? (int)FSEL(A.adm, f)[(i64)t * M + slot_pos(j0, k, jsh, tid)] : -1;
        S.cell[tid] = c;
        S.ce[tid] = c >= 0 ? A.ce[(i64)t * M + c] : 0.0f;
        S.sp[tid] = (c >= 0 && has_sp) ? A.spr[(i64)t * M + c] : 0.0f;
    }
    __syncthreads();
    load_rows(S, A.kb + (i64)t * M * DH, tid);
    __syncthreads();
    DBG_MARK(2);
    if (wid < nsub) zb_of_row(S, wid, wid << k, lane);
    __syncthreads();
    float x[32][PER];
#pragma unroll
    for (int r = 0; r < 32; r++)
#pragma unroll
        for (int i = 0; i < PER; i++) x[r][i] = S.Xs[(32 * wid + r) * XS + lane + 32 * i];
    cw_phases(k, x, S, R, A.kb + (i64)t * M * DH, tid, lane, wid, nrow, k, fold, has_sp != 0);
}

// type 1: G node of chunk j0 (height k >= 10), level l0, index n0
template <bool LEAN = false>
__device__ __forceinline__ void task_gnode(const Args& A, const int f, const int t, const int k, const int j0, const int l0,
                                        const int n0, const bool fold, const int flag, const int dep) {
    extern __shared__ float smem[];
    const int M = A.M;
    const int mt = UNI((int)FSEL(A.mt, f)[t]);
    const bool full = ((j0 + 1) << k) <= mt;
    if (!full) {
        if (l0 == 0) {
            const int jo = LEAN ? 0 : j0;
            empty_chunks(A, f, t, k, jo, 1, 0);
        }
        return;
    }
    if (dep >= 0) wait_flag(A.ctl + dep);
    int* scr = A.scr + (((i64)f * A.nbig + (k - 10)) * A.T + t) * M + ((i64)j0 << k);
    const GS G = gcarve(smem);
    const int V = 1 << (k - l0);
    g_node<LEAN>(A, G, f, t, k, j0, l0, n0, l0 == 0 ? FSEL(A.adm, f) + (i64)t * M + ((i64)j0 << k) : nullptr, scr + n0 * V,
                 scr + n0 * V, fold, A.has_sp != 0);
    set_flag(A.ctl + flag);
}

// type 2: the 512-row subtree at level k - 9, index n0, of chunk j0 (height k >= 10)
template <bool LEAN = false>
__device__ __forceinline__ void task_sub(const Args& A, const int f, const int t, const int k, const int j0, const int n0,
                                      const bool fold, const int dep) {
    extern __shared__ float smem[];
    const SM S = carve(smem);
    const int tid = threadIdx.x, lane = tid & 31, wid = UNI(tid >> 5);
    const int M = A.M, has_sp = A.has_sp;
    const int mt = UNI((int)FSEL(A.mt, f)[t]);
    if (((j0 + 1) << k) > mt) return;
    wait_flag(A.ctl + dep);
    const int* scr = A.scr + (((i64)f * A.nbig + (k - 10)) * A.T + t) * M + ((i64)j0 << k) + n0 * 512;
    Rec R;
    R.node = FSEL(A.node_mu, f) != nullptr ? nullptr : FSEL(A.node, f) + (i64)t * A.Ntot * NC;
    R.node_mu = FSEL(A.node_mu, f) != nullptr ? FSEL(A.node_mu, f) + (i64)t * A.Ntot * DH : nullptr;
    R.node_aux = FSEL(A.node_aux, f) != nullptr ? FSEL(A.node_aux, f) + (i64)t * A.Ntot * NA : nullptr;
    R.leaf = FSEL(A.leaf, f) + (i64)t * A.Ltot;
    R.k = k; R.ks = 9; R.l0 = k - 9; R.n0 = n0; R.jb = LEAN ? 0 : j0; R.jsh = 0;
    R.base = A.lay[k]; R.leafbase = A.lay[16 + k];
    R.nfull = 1; R.zsh = 5;
    {
        const int c = __ldcg(scr + tid);
        S.cell[tid] = c;
        S.ce[tid] = A.ce[(i64)t * M + c];
        S.sp[tid] = has_sp ? A.spr[(i64)t * M + c] : 0.0f;
    }
    if (wid == 0) {
        const int c0 = (int)FSEL(A.adm, f)[(i64)t * M + ((i64)j0 << k)];
        const float* zr = A.kb + ((i64)t * M + c0) * DH;
        const unsigned lo = __ballot_sync(FULL, sgn(zr[lane]));
        const unsigned hi = DH == 64 ? __ballot_sync(FULL, sgn(zr[lane + (DH == 64 ? 32 : 0)])) : 0u;
        if (lane == 0) S.zb[0] = (u64)lo | ((u64)hi << 32);
    }
    __syncthreads();
    load_rows(S, A.kb + (i64)t * M * DH, tid);
    __syncthreads();
    float x[32][PER];
#pragma unroll
    for (int r = 0; r < 32; r++)
#pragma unroll
        for (int i = 0; i < PER; i++) x[r][i] = S.Xs[(32 * wid + r) * XS + lane + 32 * i];
    cw_phases(9, x, S, R, A.kb + (i64)t * M * DH, tid, lane, wid, 512, 10, fold, has_sp != 0);
}

template <bool LEAN = false>
__device__ __forceinline__ void build_task(const int* __restrict__ TASKS, const int* __restrict__ LAY,
                 const float* __restrict__ KB, const float* __restrict__ CE, const float* __restrict__ SPR,
                 const i64* __restrict__ ADM0, const i64* __restrict__ ADM1,
                 const i64* __restrict__ MT0, const i64* __restrict__ MT1,
                 float* __restrict__ NODE0, float* __restrict__ NODE1,
                 unsigned short* __restrict__ MU0, unsigned short* __restrict__ MU1,
                 float* __restrict__ AX0, float* __restrict__ AX1,
                 i64* __restrict__ LEAF0, i64* __restrict__ LEAF1,
                 int* __restrict__ SCR, int* __restrict__ CTL, const int M, const int Ntot, const int Ltot,
                 const int has_sp, const int nbig, const int T) {
    Args A;
    A.tasks = TASKS; A.lay = LAY; A.kb = KB; A.ce = CE; A.spr = SPR;
    A.adm[0] = ADM0; A.adm[1] = ADM1; A.mt[0] = MT0; A.mt[1] = MT1;
    A.node[0] = NODE0; A.node[1] = NODE1; A.leaf[0] = LEAF0; A.leaf[1] = LEAF1;
    A.node_mu[0] = MU0; A.node_mu[1] = MU1; A.node_aux[0] = AX0; A.node_aux[1] = AX1;
    A.M = M; A.Ntot = Ntot; A.Ltot = Ltot; A.has_sp = has_sp; A.scr = SCR; A.ctl = CTL; A.nbig = nbig; A.T = T;
    const int* tk = TASKS + 10 * blockIdx.x;
    const int type = UNI(tk[0]), f = UNI(tk[1]), t = UNI(tk[2]), k = UNI(tk[3]), j0 = UNI(tk[4]), nsub = UNI(tk[5]);
    const int l0 = UNI(tk[6]), n0 = UNI(tk[7]), flag = UNI(tk[8]), dep = UNI(tk[9]);
    if constexpr (LEAN) {
        A.adm[0] += (i64)t * (i64)(LAY[34] - M);
        if (ADM1 != nullptr) A.adm[1] += (i64)t * (i64)(LAY[35] - M);
        A.kb += (i64)t * (i64)(LAY[36] - M) * DH;
        A.ce += (i64)t * (i64)(LAY[37] - M);
        if (has_sp) A.spr += (i64)t * (i64)(LAY[38] - M);
    }
    const bool fold = UNI((LAY[32] >> k) & 1);
    DBG_MARK(0);
    if (type == 0) {
        if (k <= 5) task_wbatch<LEAN>(A, f, t, k, j0, nsub, l0, max(n0, 1), fold);
        else task_cbatch<LEAN>(A, f, t, k, j0, nsub, l0, fold);
    } else if (type == 1) {
        task_gnode<LEAN>(A, f, t, k, j0, l0, n0, fold, flag, dep);
    } else {
        task_sub<LEAN>(A, f, t, k, j0, n0, fold, dep);
    }
}

#if BUILD_UNIFORM
// ---- the uniform module's task body (BUILD_UNIFORM) ----
// build_task inlines the C levels into task_cbatch and task_sub, and the per-warp levels into task_wbatch, task_cbatch
// and task_sub; ptxas merges and register-allocates those copies differently (one of them spilling inside the
// half-tile levels). build_task_u reaches c_phase and w_phase from one call site each, so the level code exists once.
// Its per-kind preambles are task_wbatch's, task_cbatch's and task_sub's, load for load and store for store; G nodes
// run task_gnode.
// the record target of task kind (k, ks, l0, n0, jb, jsh, nfull, zsh) on head t of forest f
__device__ __forceinline__ Rec task_rec(const Args& A, const int f, const int t, const int k, const int ks, const int l0,
                                        const int n0, const int jb, const int jsh, const int nfull, const int zsh) {
    Rec R;
    R.node = FSEL(A.node_mu, f) != nullptr ? nullptr : FSEL(A.node, f) + (i64)t * A.Ntot * NC;
    R.node_mu = FSEL(A.node_mu, f) != nullptr ? FSEL(A.node_mu, f) + (i64)t * A.Ntot * DH : nullptr;
    R.node_aux = FSEL(A.node_aux, f) != nullptr ? FSEL(A.node_aux, f) + (i64)t * A.Ntot * NA : nullptr;
    R.leaf = FSEL(A.leaf, f) + (i64)t * A.Ltot;
    R.k = k; R.ks = ks; R.l0 = l0; R.n0 = n0; R.jb = jb; R.jsh = jsh;
    R.base = A.lay[k]; R.leafbase = A.lay[16 + k];
    R.nfull = nfull; R.zsh = zsh;
    return R;
}

template <bool LEAN = false>
__device__ __forceinline__ void build_task_u(const int* __restrict__ TASKS, const int* __restrict__ LAY,
                 const float* __restrict__ KB, const float* __restrict__ CE, const float* __restrict__ SPR,
                 const i64* __restrict__ ADM0, const i64* __restrict__ ADM1,
                 const i64* __restrict__ MT0, const i64* __restrict__ MT1,
                 float* __restrict__ NODE0, float* __restrict__ NODE1,
                 unsigned short* __restrict__ MU0, unsigned short* __restrict__ MU1,
                 float* __restrict__ AX0, float* __restrict__ AX1,
                 i64* __restrict__ LEAF0, i64* __restrict__ LEAF1,
                 int* __restrict__ SCR, int* __restrict__ CTL, const int M, const int Ntot, const int Ltot,
                 const int has_sp, const int nbig, const int T) {
    Args A;
    A.tasks = TASKS; A.lay = LAY; A.kb = KB; A.ce = CE; A.spr = SPR;
    A.adm[0] = ADM0; A.adm[1] = ADM1; A.mt[0] = MT0; A.mt[1] = MT1;
    A.node[0] = NODE0; A.node[1] = NODE1; A.leaf[0] = LEAF0; A.leaf[1] = LEAF1;
    A.node_mu[0] = MU0; A.node_mu[1] = MU1; A.node_aux[0] = AX0; A.node_aux[1] = AX1;
    A.M = M; A.Ntot = Ntot; A.Ltot = Ltot; A.has_sp = has_sp; A.scr = SCR; A.ctl = CTL; A.nbig = nbig; A.T = T;
    const int* tk = TASKS + 10 * blockIdx.x;
    const int type = UNI(tk[0]), f = UNI(tk[1]), t = UNI(tk[2]), k = UNI(tk[3]), j0 = UNI(tk[4]), nsub = UNI(tk[5]);
    const int l0 = UNI(tk[6]), n0 = UNI(tk[7]), flag = UNI(tk[8]), dep = UNI(tk[9]);
    if constexpr (LEAN) {
        A.adm[0] += (i64)t * (i64)(LAY[34] - M);
        if (ADM1 != nullptr) A.adm[1] += (i64)t * (i64)(LAY[35] - M);
        A.kb += (i64)t * (i64)(LAY[36] - M) * DH;
        A.ce += (i64)t * (i64)(LAY[37] - M);
        if (has_sp) A.spr += (i64)t * (i64)(LAY[38] - M);
    }
    const bool fold = UNI((LAY[32] >> k) & 1), hsp = has_sp != 0;
    DBG_MARK(0);
    if (type == 1) {
        task_gnode<LEAN>(A, f, t, k, j0, l0, n0, fold, flag, dep);
        return;
    }
    extern __shared__ float smem[];
    const SM S = carve(smem);
    const int tid = threadIdx.x, lane = tid & 31, wid = UNI(tid >> 5);
    Rec R;
    float x[32][PER];
    int cell, gs0, lv0;
    float ce, sp;
    u64 zb = 0;
    if (type == 0 && k <= 5) {
        // task_wbatch: nsub chunks j0 + (i << l0) of height k <= 5, one warp per 32 slots; np heads per block
        const int np = max(n0, 1), jsh = l0, jo = LEAN ? 0 : j0;
        for (int p = 0; p < np; p++) {
            const int nf_p = batch_full((int)FSEL(A.mt, f)[t + p], k, j0, nsub, jsh);
            empty_chunks(A, f, t + p, k, jo + (nf_p << jsh), nsub - nf_p, jsh);
        }
        const int pb = np > 1 ? 32 - __clz(np - 1) : 0;    // the head of warp wid: wid >> (4 - pb)
        const int tw = t + (wid >> (4 - pb));
        if (tw >= t + np) return;
        const int mt = UNI((int)FSEL(A.mt, f)[tw]);
        const int nfull = batch_full(mt, k, j0, nsub, jsh);
        gs0 = 32 * (wid & ((16 >> pb) - 1));                // the warp's first slot in its head's batch
        if (gs0 >= (nfull << k)) return;
        R = task_rec(A, f, tw, k, k, 0, 0, jo, jsh, nfull, min(k, 5));
        float* Xw = S.Xs + wid * 32 * XS;
        const int s = gs0 + lane;
        const bool ok = s < (nfull << k);
        cell = ok ? (int)FSEL(A.adm, f)[(i64)tw * M + slot_pos(j0, k, jsh, s)] : -1;
        ce = ok ? A.ce[(i64)tw * M + cell] : 0.0f;
        sp = (ok && has_sp) ? A.spr[(i64)tw * M + cell] : 0.0f;
        const float* kl = A.kb + (i64)tw * M * DH + lane;
#pragma unroll
        for (int r = 0; r < 32; r++) {
            const float* p = kl + (i64)max(__shfl_sync(FULL, cell, r), 0) * DH;
#pragma unroll
            for (int i = 0; i < PER; i++) x[r][i] = __ldg(p + 32 * i);
        }
        unsigned zc[PER];                                    // pad sign of the current chunk (its first row), per dim
#pragma unroll
        for (int r = 0; r < 32; r++)
#pragma unroll
            for (int i = 0; i < PER; i++) {
                Xw[r * XS + lane + 32 * i] = x[r][i];
                if ((r & ((1 << k) - 1)) == 0) zc[i] = sgn(x[r][i]);
                zb |= (u64)zc[i] << (r * PER + i);
            }
        __syncwarp();
        lv0 = k;
    } else {
        // task_cbatch (type 0: nsub chunks of height 6 <= k <= 9) and task_sub (type 2: the 512-row subtree n0 at
        // level k - 9 of chunk j0, height k >= 10): the slots and their rows, the C levels from 2^lc rows, then each
        // warp's 32 slots (zb: the pad signs of their chunk's first row, S.zb[gs0 >> zshift])
        const float* kt = A.kb + (i64)t * M * DH;
        const int mt = UNI((int)FSEL(A.mt, f)[t]);
        int nrow, lc, zshift;
        if (type == 0) {
            const int jsh = l0, jo = LEAN ? 0 : j0;
            const int nfull = batch_full(mt, k, j0, nsub, jsh);
            empty_chunks(A, f, t, k, jo + (nfull << jsh), nsub - nfull, jsh);
            if (nfull == 0) return;
            R = task_rec(A, f, t, k, k, 0, 0, jo, jsh, nfull, 5);
            nrow = nfull << k; lc = k; zshift = k;
            const int c = tid < nrow ? (int)FSEL(A.adm, f)[(i64)t * M + slot_pos(j0, k, jsh, tid)] : -1;
            S.cell[tid] = c;
            S.ce[tid] = c >= 0 ? A.ce[(i64)t * M + c] : 0.0f;
            S.sp[tid] = (c >= 0 && has_sp) ? A.spr[(i64)t * M + c] : 0.0f;
            __syncthreads();
            load_rows(S, kt, tid);
            __syncthreads();
            DBG_MARK(2);
            if (wid < nsub) zb_of_row(S, wid, wid << k, lane);
            __syncthreads();
        } else {
            if (((j0 + 1) << k) > mt) return;
            wait_flag(A.ctl + dep);
            const int* scr = A.scr + (((i64)f * A.nbig + (k - 10)) * A.T + t) * M + ((i64)j0 << k) + n0 * 512;
            R = task_rec(A, f, t, k, 9, k - 9, n0, LEAN ? 0 : j0, 0, 1, 5);
            nrow = 512; lc = 9; zshift = 10;
            {
                const int c = __ldcg(scr + tid);
                S.cell[tid] = c;
                S.ce[tid] = A.ce[(i64)t * M + c];
                S.sp[tid] = has_sp ? A.spr[(i64)t * M + c] : 0.0f;
            }
            if (wid == 0) {
                const int c0 = (int)FSEL(A.adm, f)[(i64)t * M + ((i64)j0 << k)];
                const float* zr = A.kb + ((i64)t * M + c0) * DH;
                const unsigned lo = __ballot_sync(FULL, sgn(zr[lane]));
                const unsigned hi = DH == 64 ? __ballot_sync(FULL, sgn(zr[lane + (DH == 64 ? 32 : 0)])) : 0u;
                if (lane == 0) S.zb[0] = (u64)lo | ((u64)hi << 32);
            }
            __syncthreads();
            load_rows(S, kt, tid);
            __syncthreads();
        }
#pragma unroll
        for (int r = 0; r < 32; r++)
#pragma unroll
            for (int i = 0; i < PER; i++) x[r][i] = S.Xs[(32 * wid + r) * XS + lane + 32 * i];
        c_phase(lc, x, S, R, kt, tid, lane, wid, fold, hsp);
        gs0 = 32 * wid;
        if (gs0 >= nrow) return;
        cell = S.cell[gs0 + lane];
        ce = S.ce[gs0 + lane]; sp = S.sp[gs0 + lane];
#pragma unroll
        for (int i = 0; i < PER; i++) {
            const u64 bit = (S.zb[gs0 >> zshift] >> (lane + 32 * i)) & 1ull;
#pragma unroll
            for (int r = 0; r < 32; r++) zb |= bit << (r * PER + i);
        }
        lv0 = 5;
    }
    DBG_MARK(1);
    int org = lane;                                         // S.cell's warp slice is free now: the org list
    w_phase(lv0, x, cell, ce, sp, org, zb, S.Xs + wid * 32 * XS, S.pw + wid * 32, S.cell + 32 * wid, R, lane, gs0, fold,
            hsp);
}
#define BUILD_TASK build_task_u
#else
#define BUILD_TASK build_task
#endif

#if !BUILD_UNIFORM
extern "C" __global__ void __launch_bounds__(256)
fsb_prep_lean_tasks(int* __restrict__ TASKS_OUT, int* __restrict__ LAY_OUT,
                    i64* __restrict__ JB0, const i64* __restrict__ MT0,
                    const u64 k_packed, const int n_k, const int T,
                    const int fold_mask, const int Kmax_lean,
                    const int M_adm0, const int M_kb, const int M_ce, const int M_spr) {
    const int tid = threadIdx.x;
    const int nrec = n_k * T;
    for (int r = tid; r < nrec; r += 256) {
        const int idx = r / T, t = r - idx * T;
        const int k = (int)((k_packed >> (4 * idx)) & 0xFull);
        const i64 mi = MT0[t];
        const int j0 = ((mi >> k) & 1ll) ? (int)((mi >> (k + 1)) << 1) : ((int)(mi >> k) + 1);
        int* tk = TASKS_OUT + 10 * r;
        tk[0] = 0; tk[1] = 0; tk[2] = t; tk[3] = k; tk[4] = j0;
        tk[5] = 1; tk[6] = 0; tk[7] = 0; tk[8] = -1; tk[9] = -1;
    }
    if (tid < 16) {
        LAY_OUT[tid] = (2 << tid) - tid - 2;
        LAY_OUT[16 + tid] = (1 << tid) - 1;
    } else if (tid == 32) {
        LAY_OUT[32] = fold_mask;
        LAY_OUT[33] = Kmax_lean;
        LAY_OUT[34] = M_adm0;
        LAY_OUT[35] = 0;
        LAY_OUT[36] = M_kb;
        LAY_OUT[37] = M_ce;
        LAY_OUT[38] = M_spr;
        LAY_OUT[39] = 0;
    }
    if (JB0 != nullptr) {
        const int K1 = Kmax_lean + 1;
        for (int e = tid; e < T * K1; e += 256) {
            const int tt = e / K1, kk = e - tt * K1;
            const i64 mi = MT0[tt];
            JB0[e] = ((mi >> kk) & 1ll) ? ((mi >> (kk + 1)) << 1) : 0ll;
        }
    }
}

// fsb_prep_lean_tasks with the dirty heights decided on the device (a decode ingest whose admitted counts the host has
// not read): the records of heights 0 .. NK - 1 for every head, where a height is dirty exactly when the host's ingest
// would rebuild it (_ingest_lean: some head holds a chunk of that height after the append, and the height held no
// chunk in any head before it or some head's chunk index or presence changed), from MT0 (m_tot after the append) and
// NN (the admitted counts the append added, so m_tot before it is MT0 - NN). Clean heights get nsub = 0 records,
// which the build kernel leaves without a write; dirty ones get fsb_prep_lean_tasks' records, so the build writes the
// same rows with the same bits, except that a head whose chunk at the height is the one it held before (same index)
// skips its rebuild where SKIP_MASK has the height's bit: the rows hold that chunk's build already, and a build is a
// pure function of the chunk's cells (append-only admissions, completed cells) and of the height's fold flag
// (fold_mask), so the caller sets the bit only where the rows were built with this launch's flag. NK <= 16; one CTA.
// WORK (the batched launch, fsb_forest_build_lean16_many): every record that builds is also listed there as
// (forest F << 16 | record), after the count WORK[0].
__device__ __forceinline__ void prep_lean_dirty(int* __restrict__ TASKS_OUT, int* __restrict__ LAY_OUT,
                                                i64* __restrict__ JB0, const i64* __restrict__ MT0,
                                                const i64* __restrict__ NN, const int NK, const int T,
                                                const int fold_mask, const int Kmax_lean, const int M_adm0,
                                                const int M_kb, const int M_ce, const int M_spr, const int skip_mask,
                                                int* __restrict__ WORK, const int F) {
    __shared__ int s_now[16], s_was[16], s_moved[16];
    const int tid = threadIdx.x;
    if (tid < 16) { s_now[tid] = 0; s_was[tid] = 0; s_moved[tid] = 0; }
    __syncthreads();
    const int nrec = NK * T;
    for (int r = tid; r < nrec; r += 256) {
        const int k = r / T, t = r - k * T;
        const i64 m1 = MT0[t], m0 = m1 - NN[t];
        const bool h1 = ((m1 >> k) & 1ll) != 0, h0 = ((m0 >> k) & 1ll) != 0;
        const i64 j1 = h1 ? ((m1 >> (k + 1)) << 1) : 0ll, j0 = h0 ? ((m0 >> (k + 1)) << 1) : 0ll;
        if (h1) atomicOr(&s_now[k], 1);
        if (h0) atomicOr(&s_was[k], 1);
        if (h1 != h0 || j1 != j0) atomicOr(&s_moved[k], 1);
    }
    __syncthreads();
    for (int r = tid; r < nrec; r += 256) {
        const int k = r / T, t = r - k * T;
        const i64 mi = MT0[t], m0 = mi - NN[t];
        const bool kept = ((skip_mask >> k) & 1) != 0 && ((mi >> k) & 1ll) != 0 && ((m0 >> k) & 1ll) != 0
                          && (mi >> (k + 1)) == (m0 >> (k + 1));
        const bool dirty = s_now[k] != 0 && (s_was[k] == 0 || s_moved[k] != 0) && !kept;
        const int j0 = ((mi >> k) & 1ll) ? (int)((mi >> (k + 1)) << 1) : ((int)(mi >> k) + 1);
        int* tk = TASKS_OUT + 10 * r;
        tk[0] = 0; tk[1] = 0; tk[2] = t; tk[3] = k; tk[4] = j0;
        tk[5] = dirty ? 1 : 0; tk[6] = 0; tk[7] = 0; tk[8] = -1; tk[9] = -1;
        if (dirty && WORK != nullptr) WORK[2 + atomicAdd(&WORK[0], 1)] = (F << 16) | r;
    }
    if (tid < 16) {
        LAY_OUT[tid] = (2 << tid) - tid - 2;
        LAY_OUT[16 + tid] = (1 << tid) - 1;
    } else if (tid == 32) {
        LAY_OUT[32] = fold_mask;
        LAY_OUT[33] = Kmax_lean;
        LAY_OUT[34] = M_adm0;
        LAY_OUT[35] = 0;
        LAY_OUT[36] = M_kb;
        LAY_OUT[37] = M_ce;
        LAY_OUT[38] = M_spr;
        LAY_OUT[39] = 0;
    }
    const int K1 = Kmax_lean + 1;
    for (int e = tid; e < T * K1; e += 256) {
        const int tt = e / K1, kk = e - tt * K1;
        const i64 mi = MT0[tt];
        JB0[e] = ((mi >> kk) & 1ll) ? ((mi >> (kk + 1)) << 1) : 0ll;
    }
}

extern "C" __global__ void __launch_bounds__(256)
fsb_prep_lean_dirty(int* __restrict__ TASKS_OUT, int* __restrict__ LAY_OUT,
                    i64* __restrict__ JB0, const i64* __restrict__ MT0, const i64* __restrict__ NN,
                    const int NK, const int T, const int fold_mask, const int Kmax_lean,
                    const int M_adm0, const int M_kb, const int M_ce, const int M_spr, const int skip_mask) {
    prep_lean_dirty(TASKS_OUT, LAY_OUT, JB0, MT0, NN, NK, T, fold_mask, Kmax_lean, M_adm0, M_kb, M_ce, M_spr, skip_mask,
                    nullptr, 0);
}

// fsb_prep_lean_dirty's counterpart for the chunks of height KLO .. KHI (>= 10) of such an ingest: TASKS holds NREC
// _chunk_rows records, from the host, of the one chunk each head completes if its count crosses a multiple of 2^KLO
// (the most cells the append can admit is short of 2^KLO, so a head completes at most that chunk); the records of a
// chunk the append did complete (the head holds it after the append and did not before) stay, the others become
// type-0 records of no chunk, which write nothing. Then NKEPT records of chunks the host found built with another fold
// flag than this launch's: such a chunk is rebuilt where its height is dirty and the head still holds it (the
// host-decided ingest rebuilds every chunk of a dirty height with the current flag), else nothing. After them come
// T (KHI - KLO + 1) records, one per head and height: the empty chunk where the height is dirty (as in
// fsb_prep_lean_dirty) and the head holds no chunk of it, which is what the host-decided ingest writes there, else
// nothing. Every other chunk that stays in place keeps its rows (built with this layout's fold flags). Clears the
// NFLAG completion flags of CTL the G-node records signal. One CTA.
extern "C" __global__ void __launch_bounds__(256)
fsb_prep_lean_tall(int* __restrict__ TASKS, int* __restrict__ CTL, const i64* __restrict__ MT0,
                   const i64* __restrict__ NN, const int NREC, const int NKEPT, const int NFLAG, const int T,
                   const int KLO, const int KHI) {
    __shared__ int s_now[16], s_was[16], s_moved[16];
    const int tid = threadIdx.x;
    if (tid < 16) { s_now[tid] = 0; s_was[tid] = 0; s_moved[tid] = 0; }
    for (int e = tid; e < NFLAG; e += 256) CTL[e] = 0;
    __syncthreads();
    const int nk = KHI - KLO + 1;
    for (int r = tid; r < nk * T; r += 256) {
        const int k = KLO + r / T, t = r - (r / T) * T;
        const i64 m1 = MT0[t], m0 = m1 - NN[t];
        const bool h1 = ((m1 >> k) & 1ll) != 0, h0 = ((m0 >> k) & 1ll) != 0;
        const i64 j1 = h1 ? ((m1 >> (k + 1)) << 1) : 0ll, j0 = h0 ? ((m0 >> (k + 1)) << 1) : 0ll;
        if (h1) atomicOr(&s_now[k], 1);
        if (h0) atomicOr(&s_was[k], 1);
        if (h1 != h0 || j1 != j0) atomicOr(&s_moved[k], 1);
    }
    __syncthreads();
    for (int r = tid; r < NREC + NKEPT; r += 256) {
        int* tk = TASKS + 10 * r;
        const int t = tk[2], k = tk[3];
        const i64 j = (i64)tk[4], m1 = MT0[t], m0 = m1 - NN[t];
        const bool now = ((m1 >> k) & 1ll) != 0 && ((m1 >> (k + 1)) << 1) == j;
        const bool was = ((m0 >> k) & 1ll) != 0 && ((m0 >> (k + 1)) << 1) == j;
        const bool dirty = s_now[k] != 0 && (s_was[k] == 0 || s_moved[k] != 0);
        if (r < NREC ? (!now || was) : !(dirty && now && was)) {
            tk[0] = 0; tk[5] = 0; tk[6] = 0; tk[7] = 0; tk[8] = -1; tk[9] = -1;
        }
    }
    for (int r = tid; r < nk * T; r += 256) {
        const int k = KLO + r / T, t = r - (r / T) * T;
        const i64 m1 = MT0[t];
        const bool dirty = s_now[k] != 0 && (s_was[k] == 0 || s_moved[k] != 0);
        int* tk = TASKS + 10 * (NREC + NKEPT + r);
        tk[0] = 0; tk[1] = 0; tk[2] = t; tk[3] = k; tk[4] = (int)(m1 >> k) + 1;
        tk[5] = (dirty && ((m1 >> k) & 1ll) == 0) ? 1 : 0; tk[6] = 0; tk[7] = 0; tk[8] = -1; tk[9] = -1;
    }
}

extern "C" __global__ void __launch_bounds__(512)
fsb_spine_lean(const float* __restrict__ NODE0, const i64* __restrict__ MT0,
               float* __restrict__ SP_NODE, i64* __restrict__ SP_ROOT,
               i64* __restrict__ SP_HGT, const int T, const int Ntot, const int Kmax_lean) {
    const int t = blockIdx.x * 16 + (threadIdx.x >> 5);
    if (t >= T) return;
    const int lane = threadIdx.x & 31;
    const i64 m = MT0[t];
    const int S = Kmax_lean + 1;
    const float* nt = NODE0 + (i64)t * Ntot * NC;
    float r_mu[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) r_mu[i] = 0.0f;
    float r_rad = 0.0f, r_am = 0.0f, r_pm = 1073741824.0f, r_cnt = 0.0f;
    i64 r_ref = -1, pop = 0, hb = 0;
    for (int k = 0; k < S; k++) {
        const i64 kk = (i64)k;
        const i64 off = m & ~((2ll << kk) - 1ll);
        const i64 croot = (i64)((2 << k) - k - 2);
        const bool bit = ((m >> kk) & 1ll) != 0;
        const bool present = bit;
        const bool mk = present && (r_cnt > 0.0f);
        float* row = SP_NODE + ((i64)t * S + kk) * NC;
        if (present) {
            const float* np = nt + min(max(croot, 0ll), (i64)(Ntot - 1)) * NC;
            float c_mu[PER];
#pragma unroll
            for (int i = 0; i < PER; i++) c_mu[i] = np[lane + 32 * i];
            const float c_rad = np[DH];
            const float c_am = np[DH + 1];
            const float c_pm = np[DH + 2];
            const float c_cnt = np[DH + 3];
            if (mk) {
                const float tot = f_max(f_add(c_cnt, r_cnt), 1.0f);
                float n_mu[PER], sq0[PER], sq1[PER];
#pragma unroll
                for (int i = 0; i < PER; i++) {
                    n_mu[i] = f_div_ftz(f_add(f_mul(c_cnt, c_mu[i]), f_mul(r_cnt, r_mu[i])), tot);
                    const float a = f_sub(c_mu[i], n_mu[i]);
                    const float b = f_sub(r_mu[i], n_mu[i]);
                    sq0[i] = f_mul(a, a);
                    sq1[i] = f_mul(b, b);
                }
                const float nrm0 = f_sqrt_ftz(dim_allsum(sq0));
                const float nrm1 = f_sqrt_ftz(dim_allsum(sq1));
                const float n_rad = f_max(f_add(c_rad, nrm0), f_add(r_rad, nrm1));
                const float n_am = f_max(c_am, r_am);
                const float n_pm = f_min(c_pm, r_pm);
                const float n_cnt = f_add(c_cnt, r_cnt);
#pragma unroll
                for (int i = 0; i < PER; i++) row[lane + 32 * i] = n_mu[i];
                if (lane == 0) {
                    row[DH] = n_rad;
                    row[DH + 1] = n_am;
                    row[DH + 2] = n_pm;
                    row[DH + 3] = n_cnt;
                    row[DH + 4] = (float)croot;
                    row[DH + 5] = (float)r_ref;
                    row[DH + 6] = 2.0f;
                    row[DH + 7] = (float)off;
                    row[DH + 8] = n_cnt;
                }
#pragma unroll
                for (int i = 0; i < PER; i++) r_mu[i] = n_mu[i];
                r_rad = n_rad; r_am = n_am; r_pm = n_pm; r_cnt = n_cnt;
                r_ref = -(kk + 2);
            } else {
#pragma unroll
                for (int i = 0; i < PER; i++) row[lane + 32 * i] = 0.0f;
                if (lane < 9) row[DH + lane] = (lane == 4 || lane == 5) ? -1.0f : 0.0f;
#pragma unroll
                for (int i = 0; i < PER; i++) r_mu[i] = c_mu[i];
                r_rad = c_rad; r_am = c_am; r_pm = c_pm; r_cnt = c_cnt;
                r_ref = croot;
            }
            pop += 1;
            hb = kk;
        } else {
#pragma unroll
            for (int i = 0; i < PER; i++) row[lane + 32 * i] = 0.0f;
            if (lane < 9) row[DH + lane] = (lane == 4 || lane == 5) ? -1.0f : 0.0f;
        }
    }
    if (lane == 0) {
        SP_ROOT[t] = (m > 0) ? r_ref : -1ll;
        SP_HGT[t] = (m > 1) ? (hb + (pop >= 2 ? 1ll : 0ll)) : 0ll;
    }
}

extern "C" __global__ void __launch_bounds__(512, 1)
fsb_forest_build(const int* __restrict__ TASKS, const int* __restrict__ LAY,
                 const float* __restrict__ KB, const float* __restrict__ CE, const float* __restrict__ SPR,
                 const i64* __restrict__ ADM0, const i64* __restrict__ ADM1,
                 const i64* __restrict__ MT0, const i64* __restrict__ MT1,
                 float* __restrict__ NODE0, float* __restrict__ NODE1, i64* __restrict__ LEAF0, i64* __restrict__ LEAF1,
                 int* __restrict__ SCR, int* __restrict__ CTL, const int M, const int Ntot, const int Ltot,
                 const int has_sp, const int nbig, const int T) {
    extern __shared__ float smem[];
    DBG_START();
    build_task<false>(TASKS, LAY, KB, CE, SPR, ADM0, ADM1, MT0, MT1, NODE0, NODE1, nullptr, nullptr, nullptr, nullptr,
                      LEAF0, LEAF1, SCR, CTL, M, Ntot, Ltot, has_sp, nbig, T);
    DBG_END();
}
#endif

// fsb_forest_build with the split destination table instead of the packed rows: the same task records and input
// strides, with means landing in an unsigned-short bf16 slab and aggregates in a 9-wide fp32 slab; every store keying
// off the destination pointers covers the mean's bf16 rounding error with the radius it records
extern "C" __global__ void __launch_bounds__(512, 1)
fsb_forest_build16(const int* __restrict__ TASKS, const int* __restrict__ LAY,
                   const float* __restrict__ KB, const float* __restrict__ CE, const float* __restrict__ SPR,
                   const i64* __restrict__ ADM0, const i64* __restrict__ ADM1,
                   const i64* __restrict__ MT0, const i64* __restrict__ MT1,
                   unsigned short* __restrict__ MU0, unsigned short* __restrict__ MU1,
                   float* __restrict__ AX0, float* __restrict__ AX1,
                   i64* __restrict__ LEAF0, i64* __restrict__ LEAF1,
                   int* __restrict__ SCR, int* __restrict__ CTL, const int M, const int Ntot, const int Ltot,
                   const int has_sp, const int nbig, const int T) {
    extern __shared__ float smem[];
    DBG_START();
    BUILD_TASK<false>(TASKS, LAY, KB, CE, SPR, ADM0, ADM1, MT0, MT1, nullptr, nullptr, MU0, MU1, AX0, AX1,
                      LEAF0, LEAF1, SCR, CTL, M, Ntot, Ltot, has_sp, nbig, T);
    DBG_END();
}

#if !BUILD_UNIFORM
extern "C" __global__ void __launch_bounds__(512, 1)
fsb_forest_build_lean(const int* __restrict__ TASKS, const int* __restrict__ LAY,
                      const float* __restrict__ KB, const float* __restrict__ CE, const float* __restrict__ SPR,
                      const i64* __restrict__ ADM0, const i64* __restrict__ ADM1,
                      const i64* __restrict__ MT0, const i64* __restrict__ MT1,
                      float* __restrict__ NODE0, float* __restrict__ NODE1, i64* __restrict__ LEAF0, i64* __restrict__ LEAF1,
                      int* __restrict__ SCR, int* __restrict__ CTL, const int M, const int Ntot, const int Ltot,
                      const int has_sp, const int nbig, const int T,
                      i64* __restrict__ JB0, i64* __restrict__ JB1) {
    extern __shared__ float smem[];
    if (blockIdx.x == 0 && JB0 != nullptr) {
        const int K1 = LAY[33] + 1;
        for (int e = threadIdx.x; e < T * K1; e += 512) {
            const int t = e / K1, k = e - t * K1;
            const i64 mi = MT0[t];
            JB0[e] = ((mi >> k) & 1) ? ((mi >> (k + 1)) << 1) : 0;
            if (JB1 != nullptr) {
                const i64 mi1 = MT1[t];
                JB1[e] = ((mi1 >> k) & 1) ? ((mi1 >> (k + 1)) << 1) : 0;
            }
        }
    }
    build_task<true>(TASKS, LAY, KB, CE, SPR, ADM0, ADM1, MT0, MT1, NODE0, NODE1, nullptr, nullptr, nullptr, nullptr,
                     LEAF0, LEAF1, SCR, CTL, M, Ntot, Ltot, has_sp, nbig, T);
}
#endif

// fsb_forest_build_lean with the split destination table: means land in an unsigned-short bf16 slab (rows centred on
// the stored mean, so the radius a writer records covers exactly what a beam reads), aggregates in a 9-wide fp32 slab
extern "C" __global__ void __launch_bounds__(512, 1)
fsb_forest_build_lean16(const int* __restrict__ TASKS, const int* __restrict__ LAY,
                        const float* __restrict__ KB, const float* __restrict__ CE, const float* __restrict__ SPR,
                        const i64* __restrict__ ADM0, const i64* __restrict__ ADM1,
                        const i64* __restrict__ MT0, const i64* __restrict__ MT1,
                        unsigned short* __restrict__ MU0, unsigned short* __restrict__ MU1,
                        float* __restrict__ AX0, float* __restrict__ AX1,
                        i64* __restrict__ LEAF0, i64* __restrict__ LEAF1,
                        int* __restrict__ SCR, int* __restrict__ CTL, const int M, const int Ntot, const int Ltot,
                        const int has_sp, const int nbig, const int T,
                        i64* __restrict__ JB0, i64* __restrict__ JB1) {
    extern __shared__ float smem[];
    if (blockIdx.x == 0 && JB0 != nullptr) {
        const int K1 = LAY[33] + 1;
        for (int e = threadIdx.x; e < T * K1; e += 512) {
            const int t = e / K1, k = e - t * K1;
            const i64 mi = MT0[t];
            JB0[e] = ((mi >> k) & 1) ? ((mi >> (k + 1)) << 1) : 0;
            if (JB1 != nullptr) {
                const i64 mi1 = MT1[t];
                JB1[e] = ((mi1 >> k) & 1) ? ((mi1 >> (k + 1)) << 1) : 0;
            }
        }
    }
    BUILD_TASK<true>(TASKS, LAY, KB, CE, SPR, ADM0, ADM1, MT0, MT1, nullptr, nullptr, MU0, MU1, AX0, AX1,
                     LEAF0, LEAF1, SCR, CTL, M, Ntot, Ltot, has_sp, nbig, T);
}

#if !BUILD_UNIFORM
// fsb_spine_lean over the split table (bf16 means upcast, fp32 aggregates) for head t, one warp; the spine rows stay
// packed fp32
__device__ __forceinline__ void spine_lean16_row(const unsigned short* __restrict__ NMU0, const float* __restrict__ NAX0,
                                                 const i64* __restrict__ MT0, float* __restrict__ SP_NODE,
                                                 i64* __restrict__ SP_ROOT, i64* __restrict__ SP_HGT, const int Ntot,
                                                 const int Kmax_lean, const int t) {
    const int lane = threadIdx.x & 31;
    const i64 m = MT0[t];
    const int S = Kmax_lean + 1;
    const unsigned short* ntm = NMU0 + (i64)t * Ntot * DH;
    const float* nta = NAX0 + (i64)t * Ntot * NA;
    float r_mu[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) r_mu[i] = 0.0f;
    float r_rad = 0.0f, r_am = 0.0f, r_pm = 1073741824.0f, r_cnt = 0.0f;
    i64 r_ref = -1, pop = 0, hb = 0;
    for (int k = 0; k < S; k++) {
        const i64 kk = (i64)k;
        const i64 off = m & ~((2ll << kk) - 1ll);
        const i64 croot = (i64)((2 << k) - k - 2);
        const bool bit = ((m >> kk) & 1ll) != 0;
        const bool present = bit;
        const bool mk = present && (r_cnt > 0.0f);
        float* row = SP_NODE + ((i64)t * S + kk) * NC;
        if (present) {
            const i64 crow = min(max(croot, 0ll), (i64)(Ntot - 1));
            const unsigned short* npm = ntm + crow * DH;
            const float* npa = nta + crow * NA;
            float c_mu[PER];
#pragma unroll
            for (int i = 0; i < PER; i++) c_mu[i] = bf2f(npm[lane + 32 * i]);
            const float c_rad = npa[0];
            const float c_am = npa[1];
            const float c_pm = npa[2];
            const float c_cnt = npa[3];
            if (mk) {
                const float tot = f_max(f_add(c_cnt, r_cnt), 1.0f);
                float n_mu[PER], sq0[PER], sq1[PER];
#pragma unroll
                for (int i = 0; i < PER; i++) {
                    n_mu[i] = f_div_ftz(f_add(f_mul(c_cnt, c_mu[i]), f_mul(r_cnt, r_mu[i])), tot);
                    const float a = f_sub(c_mu[i], n_mu[i]);
                    const float b = f_sub(r_mu[i], n_mu[i]);
                    sq0[i] = f_mul(a, a);
                    sq1[i] = f_mul(b, b);
                }
                const float nrm0 = f_sqrt_ftz(dim_allsum(sq0));
                const float nrm1 = f_sqrt_ftz(dim_allsum(sq1));
                const float n_rad = f_max(f_add(c_rad, nrm0), f_add(r_rad, nrm1));
                const float n_am = f_max(c_am, r_am);
                const float n_pm = f_min(c_pm, r_pm);
                const float n_cnt = f_add(c_cnt, r_cnt);
#pragma unroll
                for (int i = 0; i < PER; i++) row[lane + 32 * i] = n_mu[i];
                if (lane == 0) {
                    row[DH] = n_rad;
                    row[DH + 1] = n_am;
                    row[DH + 2] = n_pm;
                    row[DH + 3] = n_cnt;
                    row[DH + 4] = (float)croot;
                    row[DH + 5] = (float)r_ref;
                    row[DH + 6] = 2.0f;
                    row[DH + 7] = (float)off;
                    row[DH + 8] = n_cnt;
                }
#pragma unroll
                for (int i = 0; i < PER; i++) r_mu[i] = n_mu[i];
                r_rad = n_rad; r_am = n_am; r_pm = n_pm; r_cnt = n_cnt;
                r_ref = -(kk + 2);
            } else {
#pragma unroll
                for (int i = 0; i < PER; i++) row[lane + 32 * i] = 0.0f;
                if (lane < 9) row[DH + lane] = (lane == 4 || lane == 5) ? -1.0f : 0.0f;
#pragma unroll
                for (int i = 0; i < PER; i++) r_mu[i] = c_mu[i];
                r_rad = c_rad; r_am = c_am; r_pm = c_pm; r_cnt = c_cnt;
                r_ref = croot;
            }
            pop += 1;
            hb = kk;
        } else {
#pragma unroll
            for (int i = 0; i < PER; i++) row[lane + 32 * i] = 0.0f;
            if (lane < 9) row[DH + lane] = (lane == 4 || lane == 5) ? -1.0f : 0.0f;
        }
    }
    if (lane == 0) {
        SP_ROOT[t] = (m > 0) ? r_ref : -1ll;
        SP_HGT[t] = (m > 1) ? (hb + (pop >= 2 ? 1ll : 0ll)) : 0ll;
    }
}

extern "C" __global__ void __launch_bounds__(512)
fsb_spine_lean16(const unsigned short* __restrict__ NMU0, const float* __restrict__ NAX0, const i64* __restrict__ MT0,
                 float* __restrict__ SP_NODE, i64* __restrict__ SP_ROOT, i64* __restrict__ SP_HGT,
                 const int T, const int Ntot, const int Kmax_lean) {
    const int t = blockIdx.x * 16 + (threadIdx.x >> 5);
    if (t >= T) return;
    spine_lean16_row(NMU0, NAX0, MT0, SP_NODE, SP_ROOT, SP_HGT, Ntot, Kmax_lean, t);
}

// ---- one device-decided block change of many decode forests in four launches (FSB_DECODE_ASYNC: fsb_tree's
// _ingest_async_many, fsb_build_cuda.lean_dirty_build_many): the forests' tables and strides in a descriptor array ----
struct LeanDesc {
    const float* ce; i64 ce_sB, ce_sH, ce_sM;      // the layer's ceilings: cell c of head (b, h) at ce + b sB + h sH + c sM
    const float* tau;                               // admission thresholds per head (fp32)
    i64* adm; i64 adm_s0;                           // admitted cells per head, row stride adm_s0
    i64* mt; i64* mc;                               // m_tot per head; the append's counts (admitted, then m_tot after it)
    int* tasks; i64* jb;                            // the forest's task buffer (records, then the layout vector); chunk index
    const float* kb; const float* ce_rows; const float* spr;  // the build's inputs (row strides in the layout vector)
    unsigned short* mu; float* ax; i64* leaf;       // the lean arena
    float* sp_node; i64* sp_root; i64* sp_hgt;      // the spines
    int M_adm0, M_kb, M_ce, M_spr, Ntot, Ltot, kmax, nk, skip, has_sp;
};

// _append_adm_kernel (fsb_build_fast) for forest blockIdx.y, head blockIdx.x: the cells n_seen .. n_seen + nn - 1 whose
// ceiling reaches the head's threshold appended in cell order after its m_tot, the rest of those slots -1, then m_tot
// and the counts (the same values: integer bookkeeping and one fp32 comparison per cell). Block (0, 0) also clears
// the batched build's work count and claim counter. One warp per (forest, head).
extern "C" __global__ void __launch_bounds__(32)
fsb_append_many(const LeanDesc* __restrict__ D, int* __restrict__ WORK, const i64 n_seen, const int nn, const int H,
                const int T) {
    const LeanDesc& d = D[blockIdx.y];
    const int t = blockIdx.x, lane = threadIdx.x;
    if (blockIdx.x == 0 && blockIdx.y == 0 && lane == 0) { WORK[0] = 0; WORK[1] = 0; }
    const int b = t / H, h = t - b * H;
    const float tau = d.tau[t];
    const i64 m0 = d.mt[t];
    const float* ce = d.ce + b * d.ce_sB + h * d.ce_sH;
    i64* adm = d.adm + (i64)t * d.adm_s0;
    i64 tot = 0;
    for (int off = 0; off < nn; off += 32) {
        const int idx = off + lane;
        const i64 cid = n_seen + idx;
        const bool in = idx < nn;
        const float c = in ? ce[cid * d.ce_sM] : __int_as_float(0xff800000);       // -inf
        const bool a = in && c >= tau;
        const unsigned bal = __ballot_sync(0xffffffffu, a);
        if (a) adm[m0 + tot + __popc(bal & ((1u << lane) - 1u))] = cid;
        tot += __popc(bal);
    }
    for (int off = 0; off < nn; off += 32) {
        const int idx = off + lane;
        if (idx < nn - tot) adm[m0 + tot + idx] = -1;
    }
    if (lane == 0) {
        d.mt[t] = m0 + tot;
        d.mc[t] = tot;
        d.mc[T + t] = m0 + tot;
    }
}

// fsb_prep_lean_dirty for forest blockIdx.x, listing its records that build in WORK
extern "C" __global__ void __launch_bounds__(256)
fsb_prep_lean_dirty_many(const LeanDesc* __restrict__ D, int* __restrict__ WORK, const int T, const int fold_mask) {
    const LeanDesc& d = D[blockIdx.x];
    prep_lean_dirty(d.tasks, d.tasks + 10 * d.nk * T, d.jb, d.mt, d.mc, d.nk, T, fold_mask, d.kmax, d.M_adm0, d.M_kb,
                    d.M_ce, d.M_spr, d.skip, WORK, blockIdx.x);
}

// fsb_forest_build_lean16 over the records WORK lists (any forests, one CTA per record), with persistent CTAs claiming
// them in list order: every record builds a chunk of height <= 9, which waits for no other record
extern "C" __global__ void __launch_bounds__(512, 1)
fsb_forest_build_lean16_many(const LeanDesc* __restrict__ D, int* __restrict__ WORK, const int M, const int nbig,
                             const int T) {
    __shared__ int s_job;
    for (;;) {
        if (threadIdx.x == 0) s_job = atomicAdd(&WORK[1], 1);
        __syncthreads();
        const int i = s_job;
        if (i >= WORK[0]) return;
        const int e = WORK[2 + i];
        const LeanDesc& d = D[e >> 16];
        const int r = e & 0xffff;
        // build_task reads the record at TASKS + 10 blockIdx.x: shift the base so that is record r
        const int* tasks = reinterpret_cast<const int*>(reinterpret_cast<unsigned long long>(d.tasks)
                                                        + 40ull * (unsigned long long)(long long)(r - (int)blockIdx.x));
        build_task<true>(tasks, d.tasks + 10 * d.nk * T, d.kb, d.ce_rows, d.spr, d.adm, nullptr, d.mt, d.mt, nullptr,
                         nullptr, d.mu, d.mu, d.ax, d.ax, d.leaf, d.leaf, nullptr, nullptr, M, d.Ntot, d.Ltot, d.has_sp,
                         nbig, T);
        __syncthreads();
    }
}

// fsb_spine_lean16 for every forest: block f * nb + i holds forest f's heads 16 i .. 16 i + 15
extern "C" __global__ void __launch_bounds__(512)
fsb_spine_lean16_many(const LeanDesc* __restrict__ D, const int T, const int nb) {
    const int f = blockIdx.x / nb;
    const LeanDesc& d = D[f];
    const int t = (blockIdx.x - f * nb) * 16 + (threadIdx.x >> 5);
    if (t >= T) return;
    spine_lean16_row(d.mu, d.ax, d.mt, d.sp_node, d.sp_root, d.sp_hgt, d.Ntot, d.kmax, t);
}
#endif

// >>> SECTION: _SMALL <<<

// ---- K1s (FSB_BUILD_SMALL=1): the chunks (height k <= 8) of a short or mid prefill's trimmed build ----
// fsb_forest_build_small builds the chunks that the build kernel batches 512 slots per CTA (task type 0) in units that
// spread a short prefill's few chunks over the GPU. A 512-thread CTA hosts eight 32-slot tile slots; each warp reads
// its unit record (8 ints: kind, f, t, k, j0, nsub, jsh, aux), 16 records per CTA. Kind 0: an idle warp. Kind 1
// (k <= 4): one warp builds the chunks j0 + (i << jsh), i < nsub (at most 16 slots) of head t, levels k .. 0 on one
// half tile (task_wbatch's h-levels). Kind 2 (k >= 5): tile ti (aux bits 1-3) of the 2^(k - 5) tiles of chunk j0, on
// tile slot s (aux bits 8-10). Its A warp (aux bit 0 clear) loads the tile, runs the C levels k .. 6 together with the
// chunk's other A warps (c_level on the chunk's 2^k slots, named barrier 9 + s0 / 2 over those threads, s0 the
// chunk's first tile slot) and the level-5 split of its tile (w_level<5>); with aux bit 12 (split) it then hands the
// tile's slot order to its B warp (aux bit 0 set) over named barrier 1 + s and builds half 0 while B builds half 1,
// else it builds both halves itself. The table (fsb_build_cuda._small_units) puts one height in each CTA and the A
// and B warps of a tile on different schedulers. Every record and bit is the build kernel's: the same level functions
// on the same slots in the same order, only the warps that run them differ. Every branch around shuffles tests a value
// that ptxas can see is warp-uniform (a record field or count broadcast from lane 0, uni(), a kernel parameter): on a
// value it cannot prove uniform it compiles the code twice, once more with each shuffle behind a WARPSYNC call, and the
// copies spill.
#ifdef DBG_TIME
extern "C" __device__ u64 g_dbs[65536];                // per warp: 8 globaltimer stamps of its unit (0 start, 7 end)
#define DBS(slot) \
    do { if ((threadIdx.x & 31) == 0) g_dbs[8 * (16 * blockIdx.x + (threadIdx.x >> 5)) + (slot)] = gtimer(); } while (0)
#else
#define DBS(slot) do { } while (0)
#endif
struct SSM {                   // shared memory of a small-chunk CTA: eight tile slots of 32 rows (end: SSMEM_FLOATS)
    float* Xs; u64* keys; u64* keys2; u64* hzb; u64* zb; int* cell; float* ce; float* sp; float* pv;
    float* mu; float* v; float* vr; float* ws; int* far; float* pvp; int* pw; int* po;
    int* hcell; float* hce; float* hsp; int* horg; float* end;
};
__device__ __forceinline__ SSM carve_small(float* smem) {
    SSM S;
    float* p = smem;
    S.Xs = p; p += 256 * XS;
    S.keys = (u64*)p; p += 2 * 256;
    S.keys2 = (u64*)p; p += 2 * 256;
    S.hzb = (u64*)p; p += 2 * 256;
    S.zb = (u64*)p; p += 2 * 8;
    S.cell = (int*)p; p += 256;
    S.ce = p; p += 256;
    S.sp = p; p += 256;
    S.pv = p; p += 256;
    S.mu = p; p += 4 * DH;
    S.v = p; p += 4 * DH;
    S.vr = p; p += 4 * DH;
    S.ws = p; p += 8 * 5;
    S.far = (int*)p; p += 4;
    S.pvp = p; p += 4;
    S.pw = (int*)p; p += 16 * 32;
    S.po = (int*)p; p += 16 * 32;
    S.hcell = (int*)p; p += 256;
    S.hce = p; p += 256;
    S.hsp = p; p += 256;
    S.horg = (int*)p; p += 256;
    S.end = p;
    return S;
}
#define SSMEM_FLOATS (256 * XS + 6 * 256 + 16 + 4 * 256 + 12 * DH + 48 + 2 * 512 + 4 * 256)   // _small_smem_bytes

__device__ __forceinline__ int uni(const int v) { return __shfl_sync(FULL, v, 0); }
struct SArgs {                 // the inputs and one forest's tables (selected by f), uniform scalars, no arrays
    const int* lay; const float* kb; const float* ce; const float* spr; const i64* adm; const i64* mt;
    unsigned short* node_mu; float* node_aux; i64* leaf;
    int M, Ntot, Ltot, has_sp;
};

// empty_chunks for one forest's split tables, warps wr of nw
__device__ __forceinline__ void small_empty(const SArgs& A, const int t, const int k, const int ja, const int n,
                                            const int jsh, const int wr, const int nw) {
    if (n <= 0) return;
    const int lane = threadIdx.x & 31;
    const int nodes = (2 << k) - 1, base = A.lay[k], lbase = A.lay[16 + k];
    unsigned short* ntm = A.node_mu + (i64)t * A.Ntot * DH;
    float* nta = A.node_aux + (i64)t * A.Ntot * NA;
    float z[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) z[i] = f_sz(sgn(A.kb[(i64)t * A.M * DH + lane + 32 * i]));
    const int tot = n * nodes;
    for (int it = wr; it < tot; it += nw) {
        const int jj = ja + ((it / nodes) << jsh), h = it - (it / nodes) * nodes;
        const int rb = base + jj * nodes;
        const int l = 31 - __clz(h + 1), idx = h + 1 - (1 << l), V = 1 << (k - l);
        unsigned short* pm = ntm + (i64)(rb + h) * DH;
#pragma unroll
        for (int i = 0; i < PER; i++) pm[lane + 32 * i] = f2bf(z[i]);
        if (lane < 9) {
            float val;
            switch (lane) {
                case 0: case 1: val = 0.0f; break;
                case 2: val = 1073741824.0f; break;
                case 3: case 8: val = (float)V; break;
                case 4: val = l < k ? (float)(rb + 2 * h + 1) : -1.0f; break;
                case 5: val = l < k ? (float)(rb + 2 * h + 2) : -1.0f; break;
                case 6: val = 1.0f; break;
                default: val = (float)(lbase + jj * (1 << k) + idx * V); break;
            }
            nta[(i64)(rb + h) * NA + lane] = val;
        }
    }
    i64* lt = A.leaf + (i64)t * A.Ltot + lbase;
    for (int e2 = 32 * wr + lane; e2 < n << k; e2 += 32 * nw)
        lt[((ja + ((e2 >> k) << jsh)) << k) + (e2 & ((1 << k) - 1))] = -1;
}

// the build kernel's carve-up of the chunk on tile slots s0 .. s0 + 2^(k - 5) - 1 (s0 a multiple of the tile count):
// its rows and slot arrays, and node slots that no other chunk of the CTA uses
__device__ __forceinline__ SM group_view(const SSM& T, const int s0) {
    SM S;
    S.Xs = T.Xs + 32 * s0 * XS;
    S.keys = T.keys + 32 * s0;
    S.keys2 = T.keys2 + 32 * s0;
    S.zb = T.zb + s0;
    S.pw = nullptr;                                    // the C levels do not use it
    S.cell = T.cell + 32 * s0;
    S.ce = T.ce + 32 * s0;
    S.sp = T.sp + 32 * s0;
    S.pv = T.pv + 32 * s0;
    S.mu = T.mu + (s0 >> 1) * DH;
    S.v = T.v + (s0 >> 1) * DH;
    S.vr = T.vr + (s0 >> 1) * DH;
    S.ws = T.ws + 5 * s0;
    S.far = T.far + (s0 >> 1);
    S.pvp = T.pvp + (s0 >> 1);
    S.end = T.end;
    return S;
}

// load_rows over a group of NS slots and threads (every slot admitted)
template <int NS>
__device__ __forceinline__ void load_rows_g(const SM& S, const float* kt, const int tid) {
    constexpr int Q = DH / 4;
    float4 buf[Q];
#pragma unroll
    for (int u = 0; u < Q; u++) {
        const int e = tid + NS * u, s = e / Q, q = e - s * Q;
        buf[u] = __ldg(reinterpret_cast<const float4*>(kt + (i64)S.cell[s] * DH) + q);
    }
#pragma unroll
    for (int u = 0; u < Q; u++) {
        const int e = tid + NS * u, s = e / Q, q = e - s * Q;
        *reinterpret_cast<float4*>(S.Xs + s * XS + 4 * q) = buf[u];
    }
}

__device__ __forceinline__ Rec small_rec(const SArgs& A, const int t, const int k, const int j0, const int jsh,
                                         const int nfull) {
    Rec R;
    R.node = nullptr;
    R.node_mu = A.node_mu + (i64)t * A.Ntot * DH;
    R.node_aux = A.node_aux + (i64)t * A.Ntot * NA;
    R.leaf = A.leaf + (i64)t * A.Ltot;
    R.k = k; R.ks = k; R.l0 = 0; R.n0 = 0; R.jb = j0; R.jsh = jsh;
    R.base = A.lay[k]; R.leafbase = A.lay[16 + k];
    R.nfull = nfull; R.zsh = min(k, 5);
    return R;
}

// kind 2, A warp of a chunk of height 5 (one tile): task_wbatch's warp on the one-chunk batch, its rows into Xw
__device__ __forceinline__ void small_tile_load(const SArgs& A, const int t, const int j0, float* Xw, int& cell,
                                                float& ce, float& sp, u64& zb) {
    const int lane = threadIdx.x & 31, M = A.M;
    cell = (int)A.adm[(i64)t * M + ((i64)j0 << 5) + lane];
    ce = A.ce[(i64)t * M + cell];
    sp = A.has_sp ? A.spr[(i64)t * M + cell] : 0.0f;
    const float* kt = A.kb + (i64)t * M * DH + lane;
    float x[32][PER];
#pragma unroll
    for (int r = 0; r < 32; r++) {
        const float* p = kt + (i64)__shfl_sync(FULL, cell, r) * DH;
#pragma unroll
        for (int i = 0; i < PER; i++) x[r][i] = __ldg(p + 32 * i);
    }
    unsigned zc[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) zc[i] = sgn(x[0][i]);
    zb = 0;
#pragma unroll
    for (int r = 0; r < 32; r++)
#pragma unroll
        for (int i = 0; i < PER; i++) {
            Xw[r * XS + lane + 32 * i] = x[r][i];
            zb |= (u64)zc[i] << (r * PER + i);
        }
    __syncwarp();
}

// kind 2, A warp ti of a chunk of NS = 2^k slots on tile slots s0 = s - ti ..: task_cbatch on the chunk (loads, C
// levels k .. 6 with the chunk's other A warps); leaves the tile's slots in level-5 order in Xw (its rows) and in
// cell / ce / sp, the pad signs in zb
template <int NS>
__device__ __forceinline__ void small_c(const SArgs& A, const SSM& T, const Rec& R, const int t, const int k,
                                        const int j0, const int ti, const int s, const bool fold, int& cell, float& ce,
                                        float& sp, u64& zb) {
    const int lane = threadIdx.x & 31, M = A.M;
    const int s0 = s - ti, bid = 9 + (s0 >> 1), gt = 32 * ti + lane;
    const SM S = group_view(T, s0);
    const float* kt = A.kb + (i64)t * M * DH;
    {
        const int c = (int)A.adm[(i64)t * M + ((i64)j0 << k) + gt];
        S.cell[gt] = c;
        S.ce[gt] = A.ce[(i64)t * M + c];
        S.sp[gt] = A.has_sp ? A.spr[(i64)t * M + c] : 0.0f;
    }
    group_sync<NS>(bid);
    load_rows_g<NS>(S, kt, gt);
    group_sync<NS>(bid);
    if (ti == 0) zb_of_row(S, 0, 0, lane);
    group_sync<NS>(bid);
    DBS(1);
    float x[32][PER];
#pragma unroll
    for (int r = 0; r < 32; r++)
#pragma unroll
        for (int i = 0; i < PER; i++) x[r][i] = S.Xs[(32 * ti + r) * XS + lane + 32 * i];
    c_phase<NS>(k, x, S, R, kt, gt, lane, ti, fold, A.has_sp != 0, bid);
    DBS(2);
    cell = S.cell[gt];
    ce = S.ce[gt];
    sp = S.sp[gt];
    zb = 0;
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const u64 bit = (S.zb[0] >> (lane + 32 * i)) & 1ull;
#pragma unroll
        for (int r = 0; r < 32; r++) zb |= bit << (r * PER + i);
    }
}

// Every unit ends in the h-levels of one or two half tiles, and every unit of height >= 5 passes through the level-5
// split: the kernel reaches each of them from one place (one inlined copy each, with every branch to it warp-uniform),
// so the instruction cache holds one copy of the long straight-line level code and ptxas compiles it once, converged.
extern "C" __global__ void __launch_bounds__(512, 1)
fsb_forest_build_small(const int* __restrict__ UNITS, const int* __restrict__ LAY,
                       const float* __restrict__ KB, const float* __restrict__ CE, const float* __restrict__ SPR,
                       const i64* __restrict__ ADM0, const i64* __restrict__ ADM1,
                       const i64* __restrict__ MT0, const i64* __restrict__ MT1,
                       unsigned short* __restrict__ MU0, unsigned short* __restrict__ MU1,
                       float* __restrict__ AX0, float* __restrict__ AX1,
                       i64* __restrict__ LEAF0, i64* __restrict__ LEAF1,
                       const int M, const int Ntot, const int Ltot, const int has_sp, const int T) {
    extern __shared__ float smem[];
    const int lane = threadIdx.x & 31, wid = uni(threadIdx.x >> 5);
    const int* u = UNITS + 8 * (16 * blockIdx.x + wid);
    const int kind = uni(u[0]);
    if (kind == 0) return;
    DBS(0);
    DBG_MARK(1);
    const int f = uni(u[1]), t = uni(u[2]), k = uni(u[3]), j0 = uni(u[4]), nsub = uni(u[5]), jsh = uni(u[6]);
    const int aux = uni(u[7]);
    SArgs A;                                 // forest f's tables: selects on the uniform f, no indexed pointer arrays
    A.lay = LAY; A.kb = KB; A.ce = CE; A.spr = SPR;
    A.adm = f ? ADM1 : ADM0; A.mt = f ? MT1 : MT0;
    A.node_mu = f ? MU1 : MU0; A.node_aux = f ? AX1 : AX0; A.leaf = f ? LEAF1 : LEAF0;
    A.M = M; A.Ntot = Ntot; A.Ltot = Ltot; A.has_sp = has_sp;
    const SSM S = carve_small(smem);
    const bool fold = uni((LAY[32] >> k) & 1), hsp = has_sp != 0;
    const int s = (aux >> 8) & 7, ti = (aux >> 1) & 7;
    int* pw = S.pw + 32 * wid;
    int* po = S.po + 32 * wid;
    Rec R;
    float* Xw;
    int cell, org = lane, lv0 = 4, h = 0, gs0 = 0;
    float ce, sp;
    u64 zb = 0;
    bool w5 = false, split = false;
    if (kind == 1) {                         // the chunks j0 + (i << jsh), i < nsub, of height <= 4 (16 slots)
        const int nfull = uni(batch_full((int)A.mt[t], k, j0, nsub, jsh));
        small_empty(A, t, k, j0 + (nfull << jsh), nsub - nfull, jsh, 0, 1);
        if (nfull == 0) return;
        R = small_rec(A, t, k, j0, jsh, nfull);
        Xw = S.Xs + 16 * wid * XS;
        const bool ok = lane < (nfull << k);
        cell = ok ? (int)A.adm[(i64)t * M + slot_pos(j0, k, jsh, lane)] : -1;
        ce = ok ? A.ce[(i64)t * M + cell] : 0.0f;
        sp = (ok && hsp) ? A.spr[(i64)t * M + cell] : 0.0f;
        const float* kt = A.kb + (i64)t * M * DH + lane;
        float x[16][PER];
#pragma unroll
        for (int r = 0; r < 16; r++) {
            const float* p = kt + (i64)max(__shfl_sync(FULL, cell, r), 0) * DH;
#pragma unroll
            for (int i = 0; i < PER; i++) x[r][i] = __ldg(p + 32 * i);
        }
        unsigned zc[PER];                    // pad sign of the current chunk (its first row), per dim
#pragma unroll
        for (int r = 0; r < 16; r++)
#pragma unroll
            for (int i = 0; i < PER; i++) {
                Xw[r * XS + lane + 32 * i] = x[r][i];
                if ((r & ((1 << k) - 1)) == 0) zc[i] = sgn(x[r][i]);
                zb |= (u64)zc[i] << (r * PER + i);
            }
        __syncwarp();
        lv0 = k;
    } else {                                 // tile ti of chunk j0 of height k >= 5
        if (((j0 + 1) << k) > uni((int)A.mt[t])) {
            // the admitted cells are a prefix: a chunk with an unadmitted cell holds none; its A warps write the
            // empty records
            if (!(aux & 1)) small_empty(A, t, k, j0, 1, 0, ti, 1 << (k - 5));
            return;
        }
        R = small_rec(A, t, k, j0, 0, 1);
        gs0 = 32 * ti;
        Xw = S.Xs + 32 * s * XS;
        split = (aux >> 12) & 1;
        if (aux & 1) {                       // B: half 1 of the tile once A has split it
            asm volatile("bar.sync %0, 64;" :: "r"(1 + s) : "memory");
            DBS(4);
            const int e = 32 * s + lane;
            cell = S.hcell[e]; ce = S.hce[e]; sp = S.hsp[e]; org = S.horg[e]; zb = S.hzb[e];
            h = 1;
        } else {
            if (k == 5) {
                small_tile_load(A, t, j0, Xw, cell, ce, sp, zb);
                DBS(1);
            } else if (k == 6) {
                small_c<64>(A, S, R, t, k, j0, ti, s, fold, cell, ce, sp, zb);
            } else if (k == 7) {
                small_c<128>(A, S, R, t, k, j0, ti, s, fold, cell, ce, sp, zb);
            } else {
                small_c<256>(A, S, R, t, k, j0, ti, s, fold, cell, ce, sp, zb);
            }
            w5 = true;
        }
    }
    if (w5) {                                // the level-5 split of the tile (rows in slot order in Xw)
        float x[32][PER];
#pragma unroll
        for (int r = 0; r < 32; r++)
#pragma unroll
            for (int i = 0; i < PER; i++) x[r][i] = Xw[r * XS + lane + 32 * i];
        w_level<5, false>(x, cell, ce, sp, org, zb, Xw, pw, po, R, lane, gs0, fold, hsp);
        DBG_MARK(10 + 5);
        DBS(3);
        if (split) {
            const int e = 32 * s + lane;
            S.hcell[e] = cell; S.hce[e] = ce; S.hsp[e] = sp; S.horg[e] = org; S.hzb[e] = zb;
            asm volatile("bar.arrive %0, 64;" :: "r"(1 + s) : "memory");
        }
    }
    const int h1 = w5 && !split ? 2 : h + 1;
#pragma unroll 1
    for (int hh = h; hh < h1; hh++) {
        DBG_MARK(9);
        h_half(lv0, hh, cell, ce, sp, org, zb, Xw, pw, po, R, lane, gs0, fold, hsp);
    }
    __syncwarp();
    DBS(7);
}

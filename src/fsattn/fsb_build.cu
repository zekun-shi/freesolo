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

// One C level (nodes of V = 2^LV rows, NN = 512 / V of them). S.Xs and x hold the raw rows of the slots at level start;
// after the mean each warp centres its rows into S.Xs in place (the row passes re-read them, so the 32-row tile is not
// live across the CTA-wide column sums) and the split gathers the next order's raw rows from Kbar (kt: the head's rows,
// cells from S.cell). The pad zero is folded once into each column sum (see the W phase note).
template <int LV>
__device__ __forceinline__ void c_level(float (&x)[32][PER], const SM& S, const Rec& R, const float* __restrict__ kt,
                                         const int tid, const int lane, const int wid, const bool fold, const bool has_sp) {
    constexpr int V = 1 << LV, NN = 512 / V, G = V / DH, LG = LV - LOGDH;
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
    __syncthreads();
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
    __syncthreads();
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
    for (int e = tid; e < NN * DH; e += 512)
        if (node_full(R, (e / DH) * V)) {
            if (R.node_mu != nullptr) mu16_ptr(R, (e / DH) * V, LV)[e % DH] = f2bf(S.mu[e]);
            else mu_ptr(R, (e / DH) * V, LV)[e % DH] = S.mu[e];
        }
    __syncthreads();
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
    __syncthreads();
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
        __syncthreads();
        {
            auto val = [&](const int s) { return f_mul(S.Xs[s * XS + d], S.pv[s]); };
            float s = col_sum<LV>(val, g, n);
            if (fold) s = f_add(s, f_mul(xcp_d, S.pvp[n]));
            if (g == 0) S.vr[n * DH + d] = s;
        }
        __syncthreads();
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
        __syncthreads();
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
    __syncthreads();
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
        __syncthreads();
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
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 32; r++)
#pragma unroll
            for (int i = 0; i < PER; i++) S.Xs[(32 * wid + r) * XS + lane + 32 * i] = x[r][i];
        S.cell[row] = c; S.ce[row] = e; S.sp[row] = p;
    }
    __syncthreads();
}

__device__ __forceinline__ void c_phase(const int lv0, float (&x)[32][PER], const SM& S, const Rec& R, const float* kt,
                                         const int tid, const int lane, const int wid, const bool fold, const bool has_sp) {
    if (lv0 >= 9) { c_level<9>(x, S, R, kt, tid, lane, wid, fold, has_sp); DBG_MARK(9); }
    if (lv0 >= 8) { c_level<8>(x, S, R, kt, tid, lane, wid, fold, has_sp); DBG_MARK(8); }
    if (lv0 >= 7) { c_level<7>(x, S, R, kt, tid, lane, wid, fold, has_sp); DBG_MARK(7); }
    c_level<6>(x, S, R, kt, tid, lane, wid, fold, has_sp); DBG_MARK(6);
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

// levels lv0 (<= 4) .. 0 of both 16-slot halves of a warp's 32 slots, one half at a time; rows from Xw via org
__device__ __forceinline__ void h_phase(const int lv0, const int cell32, const float ce32, const float sp32, const int org32,
                                        const u64 zb, const float* Xw, int* pw, int* po, const Rec& R, const int lane,
                                        const int gs0, const bool fold, const bool has_sp) {
#pragma unroll 1
    for (int h = 0; h < 2; h++) {
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
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int lvn = k - l, V = 1 << lvn, q = lvn - 4, nb = V >> 9;
    const float* kt = A.kb + (i64)t * A.M * DH;
    const float* cet = A.ce + (i64)t * A.M;
    const float* spt = A.spr + (i64)t * A.M;
    for (int s = tid; s < V; s += 512) S.cell[s] = SRC64 != nullptr ? (int)SRC64[s] : __ldcg(SRC + s);
    __syncthreads();
    float zl[PER];
    {
        const int c0 = (int)A.adm[f][(i64)t * A.M + ((i64)j << k)];
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
            float* p = A.node_mu[f] != nullptr ? A.node_aux[f] + ((i64)t * A.Ntot + rb + h) * NA
                                               : A.node[f] + ((i64)t * A.Ntot + rb + h) * NC + DH;
            float eps = 0.0f;
            if (A.node_mu[f] != nullptr) {
                float s2 = 0.0f;
#pragma unroll 8
                for (int dd = 0; dd < DH; dd++) {
                    const float df = f_sub(S.mu[dd], bf2f(f2bf(S.mu[dd])));
                    s2 = f_add(s2, f_mul(df, df));
                }
                eps = f_sqrt_ftz(s2);
            }
            p[0] = A.node_mu[f] != nullptr ? f_add(rad, eps) : rad; p[1] = f_max(am0, 0.0f); p[2] = (float)pm0; p[3] = (float)V;
            p[4] = (float)(rb + 2 * h + 1); p[5] = (float)(rb + 2 * h + 2); p[6] = 1.0f;
            p[7] = (float)(A.lay[16 + k] + jo * (1 << k) + n * V); p[8] = (float)V;
        }
        if (tid < DH) {
            const int jo = LEAN ? 0 : j;
            const int nodes = (2 << k) - 1, rb = A.lay[k] + jo * nodes, h = (1 << l) - 1 + n;
            if (A.node_mu[f] != nullptr)
                A.node_mu[f][((i64)t * A.Ntot + rb + h) * DH + tid] = f2bf(S.mu[tid]);
            else
                A.node[f][((i64)t * A.Ntot + rb + h) * NC + tid] = S.mu[tid];
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
// >>> SECTION: _BUILD <<<

// ---- K1: forest construction tasks (512 threads; dynamic shared memory) ----
// task record (8 ints): type, f, t, k, j0, nsub, l0, n0.  type 0: batch of nsub chunks j0 + (i << l0) of height
// k <= 9 (nsub << k <= 512 slots; l0 = 0: consecutive chunks, 1: every other chunk); LAY: bases[16] | leafbases[16] |
// fold mask (bit k: Kpad(k) > k)

// records of the n chunks ja + (c << jsh) of height k that hold no admitted cell: mu = signed zeros of Kbar[t, 0] (the
// cell an empty slot reads), rad = amax = 0, pmin = 2^30, static fields; leaves -1. One warp per node record.
__device__ void empty_chunks(const Args& A, const int f, const int t, const int k, const int ja, const int n,
                             const int jsh) {
    if (n <= 0) return;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int nodes = (2 << k) - 1, base = A.lay[k], lbase = A.lay[16 + k];
    const bool m16 = A.node_mu[f] != nullptr;
    float* nt = m16 ? nullptr : A.node[f] + (i64)t * A.Ntot * NC;
    unsigned short* ntm = m16 ? A.node_mu[f] + (i64)t * A.Ntot * DH : nullptr;
    float* nta = m16 ? A.node_aux[f] + (i64)t * A.Ntot * NA : nullptr;
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
    i64* lt = A.leaf[f] + (i64)t * A.Ltot + lbase;
    for (int e2 = threadIdx.x; e2 < n << k; e2 += blockDim.x)
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

// type 0, k <= 5: a batch of nsub chunks of height k, one warp per 32 slots
template <bool LEAN = false>
__device__ __forceinline__ void task_wbatch(const Args& A, const int f, const int t, const int k, const int j0, const int nsub,
                                         const int jsh, const bool fold) {
    extern __shared__ float smem[];
    const SM S = carve(smem);
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int M = A.M, has_sp = A.has_sp;
    const int mt = (int)A.mt[f][t];
    const int nfull = batch_full(mt, k, j0, nsub, jsh);
    const int jo = LEAN ? 0 : j0;
    empty_chunks(A, f, t, k, jo + (nfull << jsh), nsub - nfull, jsh);
    const int gs0 = 32 * wid;
    if (gs0 >= (nfull << k)) return;
    Rec R;
    R.node = A.node_mu[f] != nullptr ? nullptr : A.node[f] + (i64)t * A.Ntot * NC;
    R.node_mu = A.node_mu[f] != nullptr ? A.node_mu[f] + (i64)t * A.Ntot * DH : nullptr;
    R.node_aux = A.node_aux[f] != nullptr ? A.node_aux[f] + (i64)t * A.Ntot * NA : nullptr;
    R.leaf = A.leaf[f] + (i64)t * A.Ltot;
    R.k = k; R.ks = k; R.l0 = 0; R.n0 = 0; R.jb = jo; R.jsh = jsh;
    R.base = A.lay[k]; R.leafbase = A.lay[16 + k];
    R.nfull = nfull; R.zsh = min(k, 5);
    float* Xw = S.Xs + wid * 32 * XS;
    int* pw = S.pw + wid * 32;
    float x[32][PER];
    u64 zb = 0;
    const int s = gs0 + lane;
    const bool ok = s < (nfull << k);
    int cell = ok ? (int)A.adm[f][(i64)t * M + slot_pos(j0, k, jsh, s)] : -1;
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
    w_phase(k, x, cell, ce, sp, org, zb, Xw, pw, S.cell + gs0, R, lane, gs0, fold, has_sp != 0);
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
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int M = A.M, has_sp = A.has_sp;
    const int mt = (int)A.mt[f][t];
    const int nfull = batch_full(mt, k, j0, nsub, jsh);
    const int jo = LEAN ? 0 : j0;
    empty_chunks(A, f, t, k, jo + (nfull << jsh), nsub - nfull, jsh);
    if (nfull == 0) return;
    Rec R;
    R.node = A.node_mu[f] != nullptr ? nullptr : A.node[f] + (i64)t * A.Ntot * NC;
    R.node_mu = A.node_mu[f] != nullptr ? A.node_mu[f] + (i64)t * A.Ntot * DH : nullptr;
    R.node_aux = A.node_aux[f] != nullptr ? A.node_aux[f] + (i64)t * A.Ntot * NA : nullptr;
    R.leaf = A.leaf[f] + (i64)t * A.Ltot;
    R.k = k; R.ks = k; R.l0 = 0; R.n0 = 0; R.jb = jo; R.jsh = jsh;
    R.base = A.lay[k]; R.leafbase = A.lay[16 + k];
    R.nfull = nfull; R.zsh = 5;
    const int nrow = nfull << k;
    {
        const int c = tid < nrow ? (int)A.adm[f][(i64)t * M + slot_pos(j0, k, jsh, tid)] : -1;
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
    const int mt = (int)A.mt[f][t];
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
    g_node<LEAN>(A, G, f, t, k, j0, l0, n0, l0 == 0 ? A.adm[f] + (i64)t * M + ((i64)j0 << k) : nullptr, scr + n0 * V,
                 scr + n0 * V, fold, A.has_sp != 0);
    set_flag(A.ctl + flag);
}

// type 2: the 512-row subtree at level k - 9, index n0, of chunk j0 (height k >= 10)
template <bool LEAN = false>
__device__ __forceinline__ void task_sub(const Args& A, const int f, const int t, const int k, const int j0, const int n0,
                                      const bool fold, const int dep) {
    extern __shared__ float smem[];
    const SM S = carve(smem);
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int M = A.M, has_sp = A.has_sp;
    const int mt = (int)A.mt[f][t];
    if (((j0 + 1) << k) > mt) return;
    wait_flag(A.ctl + dep);
    const int* scr = A.scr + (((i64)f * A.nbig + (k - 10)) * A.T + t) * M + ((i64)j0 << k) + n0 * 512;
    Rec R;
    R.node = A.node_mu[f] != nullptr ? nullptr : A.node[f] + (i64)t * A.Ntot * NC;
    R.node_mu = A.node_mu[f] != nullptr ? A.node_mu[f] + (i64)t * A.Ntot * DH : nullptr;
    R.node_aux = A.node_aux[f] != nullptr ? A.node_aux[f] + (i64)t * A.Ntot * NA : nullptr;
    R.leaf = A.leaf[f] + (i64)t * A.Ltot;
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
        const int c0 = (int)A.adm[f][(i64)t * M + ((i64)j0 << k)];
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
    const int type = tk[0], f = tk[1], t = tk[2], k = tk[3], j0 = tk[4], nsub = tk[5], l0 = tk[6], n0 = tk[7];
    const int flag = tk[8], dep = tk[9];
    if constexpr (LEAN) {
        A.adm[0] += (i64)t * (i64)(LAY[34] - M);
        if (ADM1 != nullptr) A.adm[1] += (i64)t * (i64)(LAY[35] - M);
        A.kb += (i64)t * (i64)(LAY[36] - M) * DH;
        A.ce += (i64)t * (i64)(LAY[37] - M);
        if (has_sp) A.spr += (i64)t * (i64)(LAY[38] - M);
    }
    const bool fold = (LAY[32] >> k) & 1;
    DBG_MARK(0);
    if (type == 0) {
        if (k <= 5) task_wbatch<LEAN>(A, f, t, k, j0, nsub, l0, fold);
        else task_cbatch<LEAN>(A, f, t, k, j0, nsub, l0, fold);
    } else if (type == 1) {
        task_gnode<LEAN>(A, f, t, k, j0, l0, n0, fold, flag, dep);
    } else {
        task_sub<LEAN>(A, f, t, k, j0, n0, fold, dep);
    }
}

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
    build_task<false>(TASKS, LAY, KB, CE, SPR, ADM0, ADM1, MT0, MT1, nullptr, nullptr, MU0, MU1, AX0, AX1,
                      LEAF0, LEAF1, SCR, CTL, M, Ntot, Ltot, has_sp, nbig, T);
    DBG_END();
}

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
    build_task<true>(TASKS, LAY, KB, CE, SPR, ADM0, ADM1, MT0, MT1, nullptr, nullptr, MU0, MU1, AX0, AX1,
                     LEAF0, LEAF1, SCR, CTL, M, Ntot, Ltot, has_sp, nbig, T);
}

// fsb_spine_lean over the split table (bf16 means upcast, fp32 aggregates); the spine rows stay packed fp32
extern "C" __global__ void __launch_bounds__(512)
fsb_spine_lean16(const unsigned short* __restrict__ NMU0, const float* __restrict__ NAX0, const i64* __restrict__ MT0,
                 float* __restrict__ SP_NODE, i64* __restrict__ SP_ROOT, i64* __restrict__ SP_HGT,
                 const int T, const int Ntot, const int Kmax_lean) {
    const int t = blockIdx.x * 16 + (threadIdx.x >> 5);
    if (t >= T) return;
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


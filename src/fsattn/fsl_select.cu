
#define NEG_SCORE (-1e9f)
// XS_SAFE (= XS_SAFE_BITS, set by _source): tl.erf(x) == 1.0f for every x >= 0x407ad8e6 (exhaustive); 42 ulps margin
__device__ __forceinline__ float f_fma(float a, float b, float c) {
    float r; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float f_sqrt(float a) { float r; asm("sqrt.rn.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float bf2f(unsigned h) { return __uint_as_float((h & 0xffffu) << 16); }
__device__ __forceinline__ unsigned f2bf(float f) {
    unsigned short r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r; }
__device__ __forceinline__ float max_nan(float a, float b) { return (a != a || a > b) ? a : b; }   // MaxNanFunctor
// a per-head parameter in fp32 or bf16 (bf16 -> fp32 is exact, as torch's .to(float32))
__device__ __forceinline__ float ld_par(const void* p, const int bf16, const long long h) {
    return bf16 ? bf2f(reinterpret_cast<const unsigned short*>(p)[h]) : reinterpret_cast<const float*>(p)[h]; }
// torch.topk's float radix key (TopKTypeConfig<float>::convert): larger key = larger value, NaN largest
__device__ __forceinline__ unsigned radix_key(float v) {
    const unsigned x = __float_as_uint(v);
    const unsigned mask = (x & 0x80000000u) ? 0xffffffffu : 0x80000000u;
    return (v == v) ? (x ^ mask) : 0xffffffffu;
}
__device__ __forceinline__ float key_float(unsigned k) {          // radix_key^-1 (exact for every non-NaN float)
    return __uint_as_float((k & 0x80000000u) ? (k ^ 0x80000000u) : ~k);
}
__device__ __forceinline__ bool gt_nan(float a, float b) { return (a != a && b == b) || a > b; }   // GTOp<float,true>
// one compare-exchange stage of torch's bitonicSortKVInPlace on (key, index, valid) held one slot per lane
__device__ __forceinline__ void bitonic_step(float& k, int& i, bool& v, const int lane, const int stride,
                                             const bool dir) {
    const float ko = __shfl_xor_sync(FULL, k, stride);
    const int io = __shfl_xor_sync(FULL, i, stride);
    const bool vo = __shfl_xor_sync(FULL, (int)v, stride) != 0;
    const bool lower = (lane & stride) == 0;
    const float ka = lower ? k : ko, kb = lower ? ko : k;
    const bool va = lower ? v : vo, vb = lower ? vo : v;
    const bool swap = (gt_nan(ka, kb) && va) || !vb;          // bitonicSwap: invalid slots sort last
    if (swap == dir) { k = ko; i = io; v = vo; }
}

// ---- cell ceilings: one warp per (b,h,cell), lane t takes tokens t, t+32, .. of the cell (head_dim 64) ----
extern "C" __global__ void __launch_bounds__(128) fsl_ceil(
    const unsigned short* __restrict__ K, float* __restrict__ ceil, const long long H, const long long M,
    const long long s, const long long sB, const long long sH, const long long sN, const long long ncells)
{
    const long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (w >= ncells) return;
    const long long bh = w / M, c = w - bh * M;
    const long long b = bh / H, h = bh - b * H;
    const unsigned short* kb = K + b * sB + h * sH;
    float mx = __int_as_float(0xff800000);
    for (long long t = lane; t < s; t += 32) {
        const uint4* row = reinterpret_cast<const uint4*>(kb + (c * s + t) * sN);
        unsigned u[32];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint4 r = row[k];
            u[4 * k] = r.x; u[4 * k + 1] = r.y; u[4 * k + 2] = r.z; u[4 * k + 3] = r.w;
        }
        float v[32];
#pragma unroll
        for (int l = 0; l < 32; l++) {             // element e is half (e & 1) of word e >> 1
            const float a = bf2f(u[l >> 1] >> (16 * (l & 1)));
            const float d = bf2f(u[(l + 32) >> 1] >> (16 * (l & 1)));
            v[l] = f_add(f_mul(a, a), f_mul(d, d));
        }
#pragma unroll
        for (int st = 1; st < 32; st <<= 1) {
#pragma unroll
            for (int l = 0; l < 32; l += 2 * st) v[l] = f_add(v[l], v[l + st]);
        }
        mx = max_nan(mx, f_sqrt(v[0]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = max_nan(mx, __shfl_xor_sync(FULL, mx, o));
    if (lane == 0) ceil[w] = mx;
}

// ---- cell ceilings and means in one pass (cell size 32, head_dim 64): one warp per (b,h,cell), lane t = token t ----
// mean: torch's reduce for mean(dim=cell) on these views has each thread sum all 32 tokens of 4 outputs into 4
// accumulators (token t into t % 4, in token order), then ((a0 + a1) + a2) + a3, times 1/32, rounded to bf16
extern "C" __global__ void __launch_bounds__(128) fsl_cellstats(
    const unsigned short* __restrict__ K, float* __restrict__ ceil, unsigned* __restrict__ kbar, const long long H,
    const long long M, const long long sB, const long long sH, const long long sN, const long long ncells)
{
    __shared__ uint4 tile[4][32 * 9];                 // per warp: 32 tokens x 144 B (conflict-free 16-byte stores)
    const long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31, wl = threadIdx.x >> 5;
    if (w >= ncells) return;
    const long long bh = w / M, c = w - bh * M;
    const long long b = bh / H, h = bh - b * H;
    const uint4* row = reinterpret_cast<const uint4*>(K + b * sB + h * sH + (c * 32 + lane) * sN);
    unsigned u[32];
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const uint4 r = row[k];
        u[4 * k] = r.x; u[4 * k + 1] = r.y; u[4 * k + 2] = r.z; u[4 * k + 3] = r.w;
        tile[wl][lane * 9 + k] = r;
    }
    float v[32];
#pragma unroll
    for (int l = 0; l < 32; l++) {
        const float a = bf2f(u[l >> 1] >> (16 * (l & 1)));
        const float d = bf2f(u[(l + 32) >> 1] >> (16 * (l & 1)));
        v[l] = f_add(f_mul(a, a), f_mul(d, d));
    }
#pragma unroll
    for (int st = 1; st < 32; st <<= 1) {
#pragma unroll
        for (int l = 0; l < 32; l += 2 * st) v[l] = f_add(v[l], v[l + st]);
    }
    float mx = f_sqrt(v[0]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = max_nan(mx, __shfl_xor_sync(FULL, mx, o));
    if (lane == 0) ceil[w] = mx;
    __syncwarp();
    const unsigned* tw = reinterpret_cast<const unsigned*>(tile[wl]);
    float a0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, a1[4] = {0.0f, 0.0f, 0.0f, 0.0f};     // dims 2*lane, 2*lane + 1
#pragma unroll
    for (int t = 0; t < 32; t++) {
        const unsigned x = tw[t * 36 + lane];
        a0[t & 3] = f_add(a0[t & 3], bf2f(x));
        a1[t & 3] = f_add(a1[t & 3], bf2f(x >> 16));
    }
    const float m0 = f_mul(f_add(f_add(f_add(a0[0], a0[1]), a0[2]), a0[3]), 0.03125f);
    const float m1 = f_mul(f_add(f_add(f_add(a1[0], a1[1]), a1[2]), a1[3]), 0.03125f);
    kbar[w * 32 + lane] = f2bf(m0) | (f2bf(m1) << 16);
}

// ---- top-P candidate pool of every query block ----
// A list holds the best cells seen so far at positions p = r*32 + lane, sorted by key descending, ties by cell
// ascending (cells arrive in index order, so a new cell goes after equal keys); block n is written once the cells
// c < vis(n) = min(M, n*Bq/s) are in.
#if PER == 1
// k <= 32: topk sorts its gathered picks (cells above the k-th value, then cells at it, each in index order) with the
// unstable SmallBitonicSort; replay that network so tied ceilings come out in topk's order
__device__ __forceinline__ void emit_small(int* __restrict__ o, const unsigned key0, const int idx0, const int cnt,
                                           const int Pk, const int lane, float* sv, int* si)
{
    const bool valid = lane < Pk;
    const unsigned kp = lane < cnt ? key0 : radix_key(NEG_SCORE);      // positions >= cnt: masked cells lane, ...
    const int ip = lane < cnt ? idx0 : lane;
    const unsigned kth = __shfl_sync(FULL, kp, Pk - 1);
    const bool ing = valid && kp > kth;
    const unsigned gm = __ballot_sync(FULL, ing);
    int rank = ing ? 0 : __popc(gm);
    for (int q = 0; q < 32; q++) {
        const int iq = __shfl_sync(FULL, ip, q);
        rank += (q < Pk) && ((((gm >> q) & 1u) != 0) == ing) && iq < ip;
    }
    if (valid) { sv[rank] = key_float(kp); si[rank] = ip; }
    __syncwarp();
    float k = valid ? sv[lane] : 0.0f;
    int i = valid ? si[lane] : 0;
    bool v = valid;
    __syncwarp();
    for (int size = 2; size < 32; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) bitonic_step(k, i, v, lane, stride, (lane & size) != 0);
    for (int stride = 16; stride > 0; stride >>= 1) bitonic_step(k, i, v, lane, stride, false);
    if (valid) o[lane] = i;
}
#endif

__device__ __forceinline__ void list_emit(int* __restrict__ o, const unsigned (&key)[PER], const int (&idx)[PER],
                                          const int cnt, const int Pk, const int lane, float* sv, int* si)
{
#if PER == 1
    emit_small(o, key[0], idx[0], cnt, Pk, lane, sv, si);
#else
    // k > 32: topk's stable radix sort keeps the gathered (index) order of tied keys: the list order as kept here
#pragma unroll
    for (int r = 0; r < PER; r++) {
        const int p = r * 32 + lane;
        if (p < Pk) o[p] = p < cnt ? idx[r] : p;
    }
#endif
}

__device__ __forceinline__ unsigned list_key_at(const unsigned (&key)[PER], const int p) {
    unsigned km = 0u;
#pragma unroll
    for (int r = 0; r < PER; r++) {
        const unsigned t = __shfl_sync(FULL, key[r], p & 31);
        if ((p >> 5) == r) km = t;
    }
    return km;
}

// insert cell c with key kk (c larger than every cell in the list; the caller checked it can enter)
__device__ __forceinline__ void list_insert(unsigned (&key)[PER], int (&idx)[PER], int& cnt, unsigned& kmin,
                                            const unsigned kk, const int c, const int Pk, const int lane)
{
    int pos = 0;
#pragma unroll
    for (int r = 0; r < PER; r++) pos += __popc(__ballot_sync(FULL, (r * 32 + lane < cnt) && key[r] >= kk));
    unsigned pk[PER];
    int pi[PER];
#pragma unroll
    for (int r = 0; r < PER; r++) {
        const unsigned ku = __shfl_up_sync(FULL, key[r], 1);
        const int iu = __shfl_up_sync(FULL, idx[r], 1);
        unsigned kw = 0u;
        int iw = 0;
        if (r > 0) { kw = __shfl_sync(FULL, key[r - 1], 31); iw = __shfl_sync(FULL, idx[r - 1], 31); }
        pk[r] = lane == 0 ? kw : ku;
        pi[r] = lane == 0 ? iw : iu;
    }
#pragma unroll
    for (int r = 0; r < PER; r++) {
        const int p = r * 32 + lane;
        if (p == pos) { key[r] = kk; idx[r] = c; }
        else if (p > pos) { key[r] = pk[r]; idx[r] = pi[r]; }
    }
    if (cnt < Pk) cnt++;
    kmin = list_key_at(key, cnt - 1);
}

// one chunk of 32 cells sb.. (lanes with lv, keys kl): insert the cells that enter, in order; with emit, write before
// each of them the blocks whose visible prefix ends at or before it, and when fin, the blocks up to cell cfin
__device__ __forceinline__ void list_chunk(unsigned (&key)[PER], int (&idx)[PER], int& cnt, unsigned& kmin,
                                           long long& n, long long& vis, const long long sb, const bool lv,
                                           const unsigned kl, const bool emit, const bool fin, const long long cfin,
                                           const long long nb, const long long M, const long long Bq,
                                           const long long s, const int Pk, const int lane, int* __restrict__ out,
                                           float* sv, int* si)
{
    // cells that can enter the list (kmin only grows, so this is a superset of the cells inserted below)
    unsigned cm = __ballot_sync(FULL, lv && (cnt < Pk || kl > kmin));
    while (true) {
        const int i = cm ? __ffs(cm) - 1 : 32;
        if (i == 32 && !fin) break;                   // nothing changes the list before the next chunk
        const long long c = i < 32 ? sb + i : cfin;
        if (emit) {
            while (n < nb && vis <= c) {              // blocks whose visible prefix ends before cell c
                list_emit(out + n * Pk, key, idx, cnt, Pk, lane, sv, si);
                n++;
                vis = n * Bq / s;
                if (vis > M) vis = M;
            }
        }
        if (i == 32) break;
        cm &= cm - 1;
        const unsigned kk = __shfl_sync(FULL, kl, i);
        if (cnt == Pk && kk <= kmin) continue;
        list_insert(key, idx, cnt, kmin, kk, (int)c, Pk, lane);
    }
}

// sequential: one warp per (b,h) walks every cell (keys prefetched 128 cells ahead)
extern "C" __global__ void __launch_bounds__(32) fsl_topp(
    const float* __restrict__ ceil, int* __restrict__ topp, const long long M, const long long nb,
    const long long Bq, const long long s, const int Pk)
{
    __shared__ float sv[32];
    __shared__ int si[32];
    const int lane = threadIdx.x;
    const long long bh = blockIdx.x;
    const float* cr = ceil + bh * M;
    int* out = topp + bh * nb * (long long)Pk;
    unsigned key[PER];
    int idx[PER];
#pragma unroll
    for (int r = 0; r < PER; r++) { key[r] = 0u; idx[r] = 0; }
    int cnt = 0;
    unsigned kmin = 0u;
    long long n = 0, vis = 0;
    unsigned nx[4];
#pragma unroll
    for (int k = 0; k < 4; k++) nx[k] = 32 * k + lane < M ? radix_key(cr[32 * k + lane]) : 0u;
    for (long long base = 0; base < M; base += 128) {
        unsigned cur[4];
#pragma unroll
        for (int k = 0; k < 4; k++) cur[k] = nx[k];
        if (base + 128 < M) {
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const long long cc = base + 128 + 32 * k + lane;
                nx[k] = cc < M ? radix_key(cr[cc]) : 0u;
            }
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const long long sb = base + 32 * k;
            if (sb >= M) break;
            list_chunk(key, idx, cnt, kmin, n, vis, sb, sb + lane < M, cur[k], true, sb + 32 >= M, M, nb, M, Bq, s,
                       Pk, lane, out, sv, si);
        }
    }
    while (n < nb) {
        list_emit(out + n * Pk, key, idx, cnt, Pk, lane, sv, si);
        n++;
    }
}

// parallel: one CTA per (b,h), warp w owns cells [w*S, (w+1)*S) (S a multiple of 32, every warp has cells):
// (1) each warp keeps the list of its own cells; (2) warp 0 merges them into the list before every segment;
// (3) each warp replays its cells from that list and writes the blocks whose visible prefix ends in its segment
extern "C" __global__ void __launch_bounds__(1024) fsl_topp_par(
    const float* __restrict__ ceil, int* __restrict__ topp, const long long M, const long long nb,
    const long long Bq, const long long s, const int Pk, const long long S)
{
    extern __shared__ unsigned dyn[];
    const int G = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int PS = PER * 32;
    unsigned* segk = dyn;
    int* segi = reinterpret_cast<int*>(segk + G * PS);
    unsigned* prek = reinterpret_cast<unsigned*>(segi + G * PS);
    int* prei = reinterpret_cast<int*>(prek + G * PS);
    unsigned* tmpk = reinterpret_cast<unsigned*>(prei + G * PS);
    int* tmpi = reinterpret_cast<int*>(tmpk + PS);
    int* segc = tmpi + PS;
    int* prec = segc + G;
    float* sv = reinterpret_cast<float*>(prec + G) + warp * 32;
    int* si = reinterpret_cast<int*>(reinterpret_cast<float*>(prec + G) + G * 32) + warp * 32;
    const long long bh = blockIdx.x;
    const float* cr = ceil + bh * M;
    int* out = topp + bh * nb * (long long)Pk;
    const long long c0 = (long long)warp * S;
    const long long c1 = c0 + S < M ? c0 + S : M;
    unsigned key[PER];
    int idx[PER];
#pragma unroll
    for (int r = 0; r < PER; r++) { key[r] = 0u; idx[r] = 0; }
    int cnt = 0;
    unsigned kmin = 0u;
    long long n = 0, vis = 0;
    for (long long sb = c0; sb < c1; sb += 32) {
        const bool lv = sb + lane < c1;
        list_chunk(key, idx, cnt, kmin, n, vis, sb, lv, lv ? radix_key(cr[sb + lane]) : 0u, false, false, 0, nb, M,
                   Bq, s, Pk, lane, out, sv, si);
    }
#pragma unroll
    for (int r = 0; r < PER; r++) { segk[warp * PS + r * 32 + lane] = key[r]; segi[warp * PS + r * 32 + lane] = idx[r]; }
    if (lane == 0) segc[warp] = cnt;
    __syncthreads();
    if (warp == 0) {
        cnt = 0;
        for (int g = 0; g < G; g++) {
#pragma unroll
            for (int r = 0; r < PER; r++) { prek[g * PS + r * 32 + lane] = key[r]; prei[g * PS + r * 32 + lane] = idx[r]; }
            if (lane == 0) prec[g] = cnt;
            __syncwarp();
            if (g == G - 1) break;
            // merge the list of segment g (all its cells come after the list's) and keep the first Pk
            const int cb = segc[g];
            const unsigned* bk = segk + g * PS;
            const int* bi = segi + g * PS;
            const unsigned* ak = prek + g * PS;
#pragma unroll
            for (int r = 0; r < PER; r++) {
                const int p = r * 32 + lane;
                if (p < cnt) {                         // entries of segment g with a larger key go first
                    const unsigned x = key[r];
                    int lo = 0, hi = cb;
                    while (lo < hi) { const int mid = (lo + hi) >> 1; if (bk[mid] > x) lo = mid + 1; else hi = mid; }
                    if (p + lo < Pk) { tmpk[p + lo] = x; tmpi[p + lo] = idx[r]; }
                }
                if (p < cb) {                          // list entries with a larger or equal key go first
                    const unsigned x = bk[p];
                    int lo = 0, hi = cnt;
                    while (lo < hi) { const int mid = (lo + hi) >> 1; if (ak[mid] >= x) lo = mid + 1; else hi = mid; }
                    if (p + lo < Pk) { tmpk[p + lo] = x; tmpi[p + lo] = bi[p]; }
                }
            }
            __syncwarp();
            cnt = cnt + cb < Pk ? cnt + cb : Pk;
#pragma unroll
            for (int r = 0; r < PER; r++) { key[r] = tmpk[r * 32 + lane]; idx[r] = tmpi[r * 32 + lane]; }
            __syncwarp();
        }
    }
    __syncthreads();
    cnt = prec[warp];
#pragma unroll
    for (int r = 0; r < PER; r++) { key[r] = prek[warp * PS + r * 32 + lane]; idx[r] = prei[warp * PS + r * 32 + lane]; }
    kmin = cnt > 0 ? list_key_at(key, cnt - 1) : 0u;
    n = (c0 * s + Bq - 1) / Bq;                       // first block with vis(n) >= c0
    vis = n * Bq / s;
    if (vis > M) vis = M;
    const bool last = c1 == M;
    for (long long sb = c0; sb < c1; sb += 32) {
        const bool lv = sb + lane < c1;
        const bool fin = sb + 32 >= c1;
        list_chunk(key, idx, cnt, kmin, n, vis, sb, lv, lv ? radix_key(cr[sb + lane]) : 0u, true, fin,
                   last ? M : c1 - 1, nb, M, Bq, s, Pk, lane, out, sv, si);
    }
    if (last) {
        while (n < nb) {
            list_emit(out + n * Pk, key, idx, cnt, Pk, lane, sv, si);
            n++;
        }
    }
}

// ---- candidate slots of a chunk of blocks: topP then the Wc nearest cells; ok = visible and first occurrence ----
// cmeta[slot] = cell if ok else -1 - cell (cell = candidate clamped at 0); kc = bf16 cell means of the candidates;
// q16 = the block's Q rows (zeros past N) when Q is given
extern "C" __global__ void __launch_bounds__(128) fsl_prep(
    const int* __restrict__ topp, const long long* __restrict__ hi, const unsigned short* __restrict__ kbar,
    const unsigned short* __restrict__ Q, int* __restrict__ cmeta, unsigned short* __restrict__ kc,
    unsigned short* __restrict__ q16, const long long nb, const long long M, const int nbc, const long long b0,
    const long long Bq, const long long s, const int Pk, const int Wc, const int D, const long long H,
    const long long N, const long long qsB, const long long qsH, const long long qsN)
{
    __shared__ int cc[CMAX];
    __shared__ unsigned char vv[CMAX];
    const int C = Pk + Wc;
    const long long blk = blockIdx.x;
    const long long bh = blk / nbc;
    const int nl = (int)(blk - bh * nbc);
    const long long n = b0 + nl;
    const long long hin = hi[n];
    const float bst = f_mul((float)n, (float)Bq);
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        const long long cand = j < Pk ? (long long)topp[(bh * nb + n) * Pk + j] : hin - (j - Pk);
        const bool valid = cand >= 0;
        const long long c = valid ? cand : 0;
        const float cend = f_mul(f_add((float)c, 1.0f), (float)s);
        cc[j] = (int)c;
        vv[j] = valid && cend <= bst;
    }
    __syncthreads();
    int* cm = cmeta + (bh * nbc + nl) * (long long)C;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        const int c = cc[j];
        bool dup = false;
        for (int i = 0; i < j; i++) dup |= cc[i] == c;
        cm[j] = (vv[j] && !dup) ? c : -1 - c;
    }
    const int vpr = D / 8;
    const uint4* src = reinterpret_cast<const uint4*>(kbar + bh * M * D);
    uint4* dst = reinterpret_cast<uint4*>(kc + (bh * nbc + nl) * (long long)C * D);
    for (int v = threadIdx.x; v < C * vpr; v += blockDim.x) {
        const int j = v / vpr;
        dst[v] = src[(long long)cc[j] * vpr + (v - j * vpr)];
    }
    if (Q != nullptr) {
        const long long b = bh / H, h = bh - b * H;
        const unsigned short* qb = Q + b * qsB + h * qsH;
        uint4* qd = reinterpret_cast<uint4*>(q16 + (bh * nbc + nl) * Bq * D);
        for (int v = threadIdx.x; v < Bq * vpr; v += blockDim.x) {
            const int r = v / vpr;
            const long long tok = n * Bq + r;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (tok < N) val = reinterpret_cast<const uint4*>(qb + tok * qsN)[v - r * vpr];
            qd[v] = val;
        }
    }
}

// ---- per-query top-KK of the candidate scores, written into sel_full ----
// dynamic shared memory: [content tile Bq x C bf16, rows of ru4 uint4 when TILE] [ts, cs: float C] [cl, nr: int C]
__device__ __forceinline__ void slot_table(const int* __restrict__ cm, const float* __restrict__ cr, const float lm,
                                           const long long s, const float bst, const float den, const int C,
                                           float* ts, float* cs, int* cl, int* nr)
{
    const float thr = f_mul(__uint_as_float(XS_SAFE), den);
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        const int v = cm[j];
        const bool ok = v >= 0;
        const int c = ok ? v : -1 - v;
        const float cen = f_mul(f_add((float)c, 0.5f), (float)s);
        cl[j] = ok ? c : -1;
        ts[j] = f_mul(lm, cr[c]);
        cs[j] = cen;
        // every query of the block is at distance >= bst - cen from a visible cell: erf == 1.0f exactly when that
        // distance is >= XS_SAFE * 2 sig (div.full's error is <= 2 ulp)
        nr[j] = !(f_sub(bst, cen) >= thr);
    }
}

template <bool TILE>
__device__ __forceinline__ void select_rows(
    const unsigned short* __restrict__ content, const int* __restrict__ cmeta, const float* __restrict__ ceil,
    const void* __restrict__ lam, const void* __restrict__ sigv, int* __restrict__ sel, const int nbc,
    const long long b0, const int Bq, const int C, const int ru4, const int H, const long long M, const long long s,
    const float scale, const long long Npad, const int content_only, const int pflags)
{
    extern __shared__ uint4 dsm[];
    const long long blk = blockIdx.x;
    const long long bh = blk / nbc;
    const int nl = (int)(blk - bh * nbc);
    const long long n = b0 + nl;
    const int h = (int)(bh % H);
    uint4* tile = dsm;
    float* ts = reinterpret_cast<float*>(dsm + (TILE ? Bq * ru4 : 0));
    float* cs = ts + C;
    int* cl = reinterpret_cast<int*>(cs + C);
    int* nr = cl + C;
    const float bst = f_mul((float)n, (float)Bq);
    const float sg = ld_par(sigv, pflags & 2, h);
    const float den = f_add(sg, sg);
    const unsigned short* cb = content + (bh * nbc + nl) * (long long)Bq * C;
    slot_table(cmeta + (bh * nbc + nl) * (long long)C, ceil + bh * M, ld_par(lam, pflags & 1, h), s, bst, den, C, ts,
               cs, cl, nr);
    if (TILE) {
        const int cu4 = C / 8;
        const uint4* src = reinterpret_cast<const uint4*>(cb);
        for (int v = threadIdx.x; v < Bq * cu4; v += blockDim.x) {
            const int r = v / cu4;
            tile[r * ru4 + (v - r * cu4)] = src[v];
        }
    }
    __syncthreads();
    const int q = threadIdx.x;
    if (q >= Bq) return;
    const float qpos = f_add((float)q, bst);
    float bv[KK];
    int bj[KK];
#pragma unroll
    for (int i = 0; i < KK; i++) { bv[i] = __int_as_float(0xff800000); bj[i] = 0; }
    const uint4* row = tile + q * ru4;
    const unsigned short* crow = cb + (long long)q * C;
    for (int j8 = 0; j8 < C; j8 += 8) {
        unsigned w[4];
        if (TILE) {
            const uint4 t4 = row[j8 >> 3];
            w[0] = t4.x; w[1] = t4.y; w[2] = t4.z; w[3] = t4.w;
        }
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int j = j8 + e;
            if (!TILE && j >= C) break;
            const unsigned raw = TILE ? (w[e >> 1] >> (16 * (e & 1))) : (unsigned)crow[j];
            const float x = bf2f(f2bf(f_mul(bf2f(raw), scale)));       // torch: bf16(content * scale)
            float sc;
            const int cj = cl[j];
            if (cj < 0) sc = NEG_SCORE;
            else if (content_only) sc = x;
            else if (nr[j]) {
                const float d = __uint_as_float(__float_as_uint(f_sub(qpos, cs[j])) & 0x7fffffffu);
                sc = f_fma(ts[j], erf_tl(f_div_full(d, den)), x);
            } else sc = f_fma(ts[j], 1.0f, x);
            if (sc > bv[KK - 1]) {    // strictly greater: equal scores keep the lower slot first (the Triton argmax)
#pragma unroll
                for (int i = KK - 1; i > 0; i--) {
                    if (sc > bv[i - 1]) { bv[i] = bv[i - 1]; bj[i] = bj[i - 1]; }
                    else if (sc > bv[i]) { bv[i] = sc; bj[i] = j; }
                }
                if (sc > bv[0]) { bv[0] = sc; bj[0] = j; }
            }
        }
    }
    int* o = sel + (bh * Npad + n * Bq + q) * (long long)KK;
#pragma unroll
    for (int i = 0; i < KK; i++) o[i] = cl[bj[i]];
}

extern "C" __global__ void __launch_bounds__(128) fsl_select(
    const unsigned short* __restrict__ content, const int* __restrict__ cmeta, const float* __restrict__ ceil,
    const void* __restrict__ lam, const void* __restrict__ sigv, int* __restrict__ sel, const int nbc,
    const long long b0, const int Bq, const int C, const int ru4, const int H, const long long M, const long long s,
    const float scale, const long long Npad, const int content_only, const int pflags)
{
    select_rows<true>(content, cmeta, ceil, lam, sigv, sel, nbc, b0, Bq, C, ru4, H, M, s, scale, Npad, content_only,
                    pflags);
}

extern "C" __global__ void __launch_bounds__(128) fsl_select_g(
    const unsigned short* __restrict__ content, const int* __restrict__ cmeta, const float* __restrict__ ceil,
    const void* __restrict__ lam, const void* __restrict__ sigv, int* __restrict__ sel, const int nbc,
    const long long b0, const int Bq, const int C, const int ru4, const int H, const long long M, const long long s,
    const float scale, const long long Npad, const int content_only, const int pflags)
{
    select_rows<false>(content, cmeta, ceil, lam, sigv, sel, nbc, b0, Bq, C, ru4, H, M, s, scale, Npad, content_only,
                    pflags);
}

#ifdef NTL
// ---- fused: candidate slots + content GEMM on tensor cores + top-KK, one CTA of 8 warps per query block ----
// content(q, j) = bf16 of the fp32 sum over d of Q[q,d]*Kbar[cell_j,d], four mma.m16n8k16 chained in k order from
// zero: the bits of cuBLAS's s16816 and s1688 kernels and of tl.dot for this GEMM (tests/test_fsl_fast_select.py
// compares them). Warp w owns query rows 16w..16w+15; lane (g = lane/4, t = lane%4) holds rows g, g+8 and columns
// 8nt + 2t, +1; the scaled content goes through shared memory to one thread per query row.
#define FROW 72                                   // smem row stride in bf16 (144 B: conflict-free ldmatrix)
__device__ __forceinline__ unsigned smem_addr(const void* p) {
    unsigned a; asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p)); return a; }
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void ldsm_x2(unsigned& r0, unsigned& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void cp16(void* dst, const void* src, const unsigned bytes) {     // zero-fills past bytes
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(smem_addr(dst)), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], const unsigned b0, const unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

extern "C" __global__ void __launch_bounds__(256, 4) fsl_fused(
    const int* __restrict__ topp, const long long* __restrict__ hi, const unsigned short* __restrict__ kbar,
    const unsigned short* __restrict__ Q, const float* __restrict__ ceil, const void* __restrict__ lam,
    const void* __restrict__ sigv, int* __restrict__ sel, unsigned short* __restrict__ dbg, const long long nb,
    const long long M, const int nbc, const long long b0, const int Bq, const long long s, const int Pk, const int Wc,
    const long long H, const long long N, const long long qsB, const long long qsH, const long long qsN,
    const float scale, const long long Npad, const int content_only, const int pflags)
{
    constexpr int C = 8 * NTL;
    static_assert(C <= 256, "one slot per thread");
    extern __shared__ uint4 dsm[];
    unsigned short* qs = reinterpret_cast<unsigned short*>(dsm);             // 128 x FROW
    unsigned short* ks = qs + 128 * FROW;                                    // C x FROW
    constexpr int SXH = C + 2;                                               // bf16 content row: odd word stride
    constexpr int FRONT = ((128 + C) * FROW * 2 > 128 * SXH * 2 ? (128 + C) * FROW * 2 : 128 * SXH * 2);
    float* ts = reinterpret_cast<float*>(reinterpret_cast<char*>(dsm) + ((FRONT + 15) / 16) * 16);
    float* cs = ts + C;
    int* cl = reinterpret_cast<int*>(cs + C);
    int* nr = cl + C;
    int* cc = nr + C;
    int* nearl = cc + C;                                                     // ok slots that need erf, any order
    int* nnear = nearl + C;
    const long long blk = blockIdx.x;
    const long long bh = blk / nbc;
    const int nl = (int)(blk - bh * nbc);
    const long long n = b0 + nl;
    const long long b = bh / H, h = bh - b * H;
    const int tid = threadIdx.x;
    const float bst = f_mul((float)n, (float)Bq);
    const float sg = ld_par(sigv, pflags & 2, h);
    const float den = f_add(sg, sg);
    // the block's Q rows (zeros past N and Bq) start loading at once
    {
        const unsigned short* qb = Q + b * qsB + h * qsH;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int v = tid + 256 * k, r = v >> 3, w = v & 7;
            const long long tok = n * Bq + r;
            const bool in = r < Bq && tok < N;
            cp16(qs + r * FROW + w * 8, in ? qb + tok * qsN + w * 8 : qb, in ? 16u : 0u);
        }
        cp_commit();
    }
    // slot j (one per thread): topP then the Wc nearest cells; clamped cell, visibility, its ceiling
    const long long hin = hi[n];
    int c = 0;
    bool vis = false;
    float cv = 0.0f;
    if (tid < C) {
        const long long cand = tid < Pk ? (long long)topp[(bh * nb + n) * Pk + tid] : hin - (tid - Pk);
        c = cand >= 0 ? (int)cand : 0;
        vis = cand >= 0 && f_mul(f_add((float)c, 1.0f), (float)s) <= bst;
        cv = ceil[bh * M + c];
        cc[tid] = c;
        nr[tid] = 0;
    }
    if (tid == 0) nnear[0] = 0;
    __syncthreads();
    {
        const uint4* src = reinterpret_cast<const uint4*>(kbar + bh * M * 64);
        for (int v = tid; v < C * 8; v += 256) {
            const int j = v >> 3, w = v & 7;
            cp16(ks + j * FROW + w * 8, src + (long long)cc[j] * 8 + w, 16u);
        }
        cp_commit();
        // first occurrence: topP cells are distinct and the near cells hin, hin-1, .. are too, so a valid slot repeats
        // an earlier one only when it is a near cell that topP holds (invalid slots are never ok either way)
        if (tid < Pk && cc[tid] <= hin && cc[tid] > hin - Wc) nr[Pk + (int)(hin - cc[tid])] = 1;
    }
    __syncthreads();
    if (tid < C) {
        const float cen = f_mul(f_add((float)c, 0.5f), (float)s);
        const bool ok = vis && !nr[tid];
        const bool near = !(f_sub(bst, cen) >= f_mul(__uint_as_float(XS_SAFE), den));
        cl[tid] = ok ? c : -1;
        ts[tid] = f_mul(ld_par(lam, pflags & 1, h), cv);
        cs[tid] = cen;
        nr[tid] = ok ? (near && !content_only ? 2 : 1) : 0;         // slot kind: 0 not ok, 1 x + t, 2 erf
        if (ok && near && !content_only) nearl[atomicAdd(nnear, 1)] = tid;
    }
    cp_wait_all();
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    float acc[NTL][4];
#pragma unroll
    for (int nt = 0; nt < NTL; nt++) { acc[nt][0] = 0.0f; acc[nt][1] = 0.0f; acc[nt][2] = 0.0f; acc[nt][3] = 0.0f; }
#pragma unroll
    for (int k0 = 0; k0 < 64; k0 += 16) {
        unsigned a[4];
        ldsm_x4(a, qs + (16 * warp + (lane & 15)) * FROW + k0 + 8 * (lane >> 4));
#pragma unroll
        for (int nt = 0; nt < NTL; nt++) {
            unsigned b0r, b1r;
            ldsm_x2(b0r, b1r, ks + (8 * nt + (lane & 7)) * FROW + k0 + 8 * ((lane >> 3) & 1));
            mma16816(acc[nt], a, b0r, b1r);
        }
    }
    // scaled content into shared memory (over the dead Q / mean tiles), then one thread per query row scores its
    // slots in order and keeps the top KK, as fsl_select does
    __syncthreads();
    unsigned short* sx = reinterpret_cast<unsigned short*>(dsm);             // 128 x SXH, bf16 (exact: see below)
    const int row0 = 16 * warp + g;
#pragma unroll
    for (int nt = 0; nt < NTL; nt++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = row0 + 8 * (e >> 1);
            const int j = 8 * nt + 2 * t4 + (e & 1);
            const unsigned cb = f2bf(acc[nt][e]);                          // the GEMM's bf16 content
            if (dbg != nullptr && row < Bq) dbg[((bh * nbc + nl) * (long long)Bq + row) * C + j] = (unsigned short)cb;
            sx[row * SXH + j] = (unsigned short)f2bf(f_mul(bf2f(cb), scale));   // torch: bf16(content * scale)
        }
    }
    __syncthreads();
    if (tid >= Bq) return;
    const unsigned short* xr = sx + tid * SXH;
    float bv[KK];
    int bj[KK];
#pragma unroll
    for (int i = 0; i < KK; i++) { bv[i] = __int_as_float(0xff800000); bj[i] = 0; }
    // slots without erf in slot order, branch-free (equal scores keep the lower slot: strict >); erf slots are
    // skipped (-inf never enters) and inserted below with the (score, slot) order, which gives the same top KK
#pragma unroll 8
    for (int j = 0; j < C; j++) {
        const float x = bf2f(xr[j]);
        const int kd = nr[j];
        float sc = content_only ? x : f_add(x, ts[j]);                 // == fma(lam*ceil, 1.0f, x): erf == 1
        sc = kd == 0 ? NEG_SCORE : sc;
        sc = kd == 2 ? __int_as_float(0xff800000) : sc;
        bool gt[KK];
#pragma unroll
        for (int i = 0; i < KK; i++) gt[i] = sc > bv[i];
#pragma unroll
        for (int i = KK - 1; i > 0; i--) {
            bv[i] = gt[i - 1] ? bv[i - 1] : (gt[i] ? sc : bv[i]);
            bj[i] = gt[i - 1] ? bj[i - 1] : (gt[i] ? j : bj[i]);
        }
        bv[0] = gt[0] ? sc : bv[0];
        bj[0] = gt[0] ? j : bj[0];
    }
    const int nn = nnear[0];
    const float qpos = f_add((float)tid, bst);
    for (int u = 0; u < nn; u++) {
        const int j = nearl[u];
        const float d = __uint_as_float(__float_as_uint(f_sub(qpos, cs[j])) & 0x7fffffffu);
        const float sc = f_fma(ts[j], erf_tl(f_div_full(d, den)), bf2f(xr[j]));
        bool gt[KK];
#pragma unroll
        for (int i = 0; i < KK; i++) gt[i] = sc > bv[i] || (sc == bv[i] && j < bj[i]);
#pragma unroll
        for (int i = KK - 1; i > 0; i--) {
            bv[i] = gt[i - 1] ? bv[i - 1] : (gt[i] ? sc : bv[i]);
            bj[i] = gt[i - 1] ? bj[i - 1] : (gt[i] ? j : bj[i]);
        }
        bv[0] = gt[0] ? sc : bv[0];
        bj[0] = gt[0] ? j : bj[0];
    }
    int* o = sel + (bh * Npad + n * Bq + tid) * (long long)KK;
#pragma unroll
    for (int i = 0; i < KK; i++) o[i] = cl[bj[i]];
}

// ---- dense exact cell selection: tiles all M_vis = min(M, (n*Bq)/s) visible cells in 64-cell tensor-core passes ----
extern "C" __global__ void __launch_bounds__(256, 4) fsl_dense_fused(
    const unsigned short* __restrict__ kbar, const unsigned short* __restrict__ Q,
    const float* __restrict__ ceil, const void* __restrict__ lam, const void* __restrict__ sigv,
    int* __restrict__ sel, const long long nb, const long long M, const int Bq, const long long s,
    const long long H, const long long N, const long long qsB, const long long qsH, const long long qsN,
    const float scale, const long long Npad, const int content_only, const int pflags)
{
    constexpr int CD = 64;
    constexpr int SXH = CD + 2;
    extern __shared__ uint4 dsm[];
    unsigned short* qs = reinterpret_cast<unsigned short*>(dsm);
    unsigned short* ks = reinterpret_cast<unsigned short*>(dsm);
    unsigned short* sx = ks + CD * FROW;
    float* ts = reinterpret_cast<float*>(reinterpret_cast<char*>(dsm) + (((CD * FROW + 128 * SXH) * 2 + 15) / 16) * 16);
    float* cs = ts + CD;
    int* nr = reinterpret_cast<int*>(cs + CD);
    int* nearl = nr + CD;
    int* nnear = nearl + CD;

    const long long blk = blockIdx.x;
    const long long BH = (long long)gridDim.x / nb;
    const long long nl = blk / BH;
    const long long bh = blk - nl * BH;
    const long long n = (nb - 1) - nl;
    const long long b = bh / H, h = bh - b * H;
    const int tid = threadIdx.x;
    const float bst = f_mul((float)n, (float)Bq);
    const long long mvis_raw = (n * Bq) / s;
    const long long mvis = mvis_raw < M ? mvis_raw : M;
    if (mvis <= 0) {
        if (tid < Bq) {
            int* o = sel + (bh * Npad + n * Bq + tid) * (long long)KK;
#pragma unroll
            for (int i = 0; i < KK; i++) o[i] = -1;
        }
        return;
    }

    const float lam_h = ld_par(lam, pflags & 1, h);
    const float sg = ld_par(sigv, pflags & 2, h);
    const float den = f_add(sg, sg);
    const float erf_margin = f_mul(__uint_as_float(XS_SAFE), den);

    {
        const unsigned short* qb = Q + b * qsB + h * qsH;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int v = tid + 256 * k, r = v >> 3, w = v & 7;
            const long long tok = n * Bq + r;
            const bool in = r < Bq && tok < N;
            cp16(qs + r * FROW + w * 8, in ? qb + tok * qsN + w * 8 : qb, in ? 16u : 0u);
        }
        cp_commit();
        cp_wait_all();
    }
    __syncthreads();

    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    const int row0 = 16 * warp + g;
    unsigned qa[4][4];
#pragma unroll
    for (int k4 = 0; k4 < 4; k4++) {
        ldsm_x4(qa[k4], qs + (16 * warp + (lane & 15)) * FROW + 16 * k4 + 8 * (lane >> 4));
    }
    __syncthreads();

    float bv[KK];
    int bj[KK];
#pragma unroll
    for (int i = 0; i < KK; i++) { bv[i] = __int_as_float(0xff800000); bj[i] = -1; }
    const float qpos = f_add((float)tid, bst);
    const uint4* kb_src = reinterpret_cast<const uint4*>(kbar + bh * M * 64);

    for (long long c0 = 0; c0 < mvis; c0 += CD) {
        const int nc = (int)(mvis - c0 < CD ? mvis - c0 : CD);
#pragma unroll
        for (int k = 0; k < 2; k++) {
            const int v = tid + 256 * k, j = v >> 3, w = v & 7;
            const bool in = j < nc;
            cp16(ks + j * FROW + w * 8, in ? kb_src + (c0 + j) * 8 + w : kb_src, in ? 16u : 0u);
        }
        cp_commit();
        if (tid == 0) nnear[0] = 0;
        __syncthreads();
        if (tid < CD) {
            if (tid < nc) {
                const long long c = c0 + tid;
                const float cen = f_mul(f_add((float)c, 0.5f), (float)s);
                const bool near = !(f_sub(bst, cen) >= erf_margin);
                ts[tid] = f_mul(lam_h, ceil[bh * M + c]);
                cs[tid] = cen;
                nr[tid] = (near && !content_only) ? 2 : 1;
                if (near && !content_only) nearl[atomicAdd(nnear, 1)] = tid;
            } else {
                ts[tid] = 0.0f;
                cs[tid] = 0.0f;
                nr[tid] = 0;
            }
        }
        cp_wait_all();
        __syncthreads();

        float acc[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            acc[nt][0] = 0.0f; acc[nt][1] = 0.0f; acc[nt][2] = 0.0f; acc[nt][3] = 0.0f;
        }
#pragma unroll
        for (int k4 = 0; k4 < 4; k4++) {
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                unsigned b0r, b1r;
                ldsm_x2(b0r, b1r, ks + (8 * nt + (lane & 7)) * FROW + 16 * k4 + 8 * ((lane >> 3) & 1));
                mma16816(acc[nt], qa[k4], b0r, b1r);
            }
        }
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = row0 + 8 * (e >> 1);
                const int j = 8 * nt + 2 * t4 + (e & 1);
                const unsigned cb = f2bf(acc[nt][e]);
                sx[row * SXH + j] = (unsigned short)f2bf(f_mul(bf2f(cb), scale));
            }
        }
        __syncthreads();

        if (tid < Bq) {
            const unsigned short* xr = sx + tid * SXH;
#pragma unroll 8
            for (int j = 0; j < CD; j++) {
                const float x = bf2f(xr[j]);
                const int kd = nr[j];
                float sc = content_only ? x : f_add(x, ts[j]);
                sc = (kd != 1) ? __int_as_float(0xff800000) : sc;
                const int cell = (int)c0 + j;
                bool gt[KK];
#pragma unroll
                for (int i = 0; i < KK; i++) gt[i] = sc > bv[i];
#pragma unroll
                for (int i = KK - 1; i > 0; i--) {
                    bv[i] = gt[i - 1] ? bv[i - 1] : (gt[i] ? sc : bv[i]);
                    bj[i] = gt[i - 1] ? bj[i - 1] : (gt[i] ? cell : bj[i]);
                }
                bv[0] = gt[0] ? sc : bv[0];
                bj[0] = gt[0] ? cell : bj[0];
            }
            const int nn = nnear[0];
            for (int u = 0; u < nn; u++) {
                const int j = nearl[u];
                const int cell = (int)c0 + j;
                const float d = __uint_as_float(__float_as_uint(f_sub(qpos, cs[j])) & 0x7fffffffu);
                const float sc = f_fma(ts[j], erf_tl(f_div_full(d, den)), bf2f(xr[j]));
                bool gt[KK];
#pragma unroll
                for (int i = 0; i < KK; i++) gt[i] = sc > bv[i] || (sc == bv[i] && cell < bj[i]);
#pragma unroll
                for (int i = KK - 1; i > 0; i--) {
                    bv[i] = gt[i - 1] ? bv[i - 1] : (gt[i] ? sc : bv[i]);
                    bj[i] = gt[i - 1] ? bj[i - 1] : (gt[i] ? cell : bj[i]);
                }
                bv[0] = gt[0] ? sc : bv[0];
                bj[0] = gt[0] ? cell : bj[0];
            }
        }
        __syncthreads();
    }
    if (tid < Bq) {
        int* o = sel + (bh * Npad + n * Bq + tid) * (long long)KK;
#pragma unroll
        for (int i = 0; i < KK; i++) o[i] = bj[i];
    }
}
#endif


typedef long long i64;
typedef unsigned long long u64;
#define FULL 0xffffffffu

// presence bit of cell c (test first: most slots repeat a cell another query of the row already set)
__device__ __forceinline__ void mark(unsigned* bm, const i64 c) {
    const unsigned bit = 1u << (unsigned)(c & 31);
    unsigned* w = bm + (c >> 5);
    if (!(*(volatile unsigned*)w & bit)) atomicOr(w, bit);
}

extern "C" __global__ void __launch_bounds__(NT)
fsb_rows_union(const i64* __restrict__ SELA, const i64* __restrict__ SELB, const int Bq, const int Mc, const int ud,
               int* __restrict__ CELLS, int* __restrict__ UCNT, int* __restrict__ POS, i64* __restrict__ DUP,
               unsigned char* __restrict__ OVF) {
    __shared__ unsigned bm[MCW];
    __shared__ int wsum[NT / 32];
    const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int nw = (Mc + 31) >> 5;
    for (int w = tid; w < nw; w += NT) bm[w] = 0u;
    __syncthreads();
    for (int q = tid; q < Bq; q += NT) {
        const i64 row = (i64)r * Bq + q;
        i64 a[CA];
        const longlong2* pa = reinterpret_cast<const longlong2*>(SELA + row * CA);
#pragma unroll
        for (int i = 0; i < CA / 2; i++) {
            const longlong2 v = pa[i];
            a[2 * i] = v.x;
            a[2 * i + 1] = v.y;
        }
#pragma unroll
        for (int i = 0; i < CA; i++)
            if (a[i] >= 0) mark(bm, a[i]);
#if CB > 0
        const longlong2* pb = reinterpret_cast<const longlong2*>(SELB + row * CB);
        u64 dup = 0ull;
#pragma unroll 2
        for (int j2 = 0; j2 < CB / 2; j2++) {
            const longlong2 v = pb[j2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const i64 c = h ? v.y : v.x;
                if (c >= 0) {
                    bool d = false;
#pragma unroll
                    for (int i = 0; i < CA; i++) d |= (a[i] == c);
                    if (d) dup |= 1ull << (2 * j2 + h);
                    else mark(bm, c);
                }
            }
        }
        DUP[row] = (i64)dup;
#endif
    }
    __syncthreads();
    // rank of every present cell: exclusive scan of the words' popcounts, each thread a contiguous run of words
    const int per = (nw + NT - 1) / NT;
    const int w0 = min(tid * per, nw), w1 = min(w0 + per, nw);
    int cnt = 0;
    for (int w = w0; w < w1; w++) cnt += __popc(bm[w]);
    int x = cnt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(FULL, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) wsum[wid] = x;
    __syncthreads();
    if (wid == 0) {
        int v = lane < NT / 32 ? wsum[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(FULL, v, o);
            if (lane >= o) v += y;
        }
        if (lane < NT / 32) wsum[lane] = v;
    }
    __syncthreads();
    int base = x - cnt + (wid > 0 ? wsum[wid - 1] : 0);
    if (tid == 0) {
        const int U = wsum[NT / 32 - 1];
        UCNT[r] = U;
        if (U > ud) *OVF = 1;
    }
    for (int w = w0; w < w1; w++) {
        unsigned bits = bm[w];
        while (bits) {
            const int b = __ffs(bits) - 1;
            bits &= bits - 1u;
            const int cell = (w << 5) + b;
            if (base < ud) CELLS[(i64)r * ud + base] = cell;
            POS[(i64)r * Mc + cell] = base;
            base++;
        }
    }
}


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

#ifdef KK
#define FROW 72

#if OUT_I32
typedef int out_t;
#else
typedef i64 out_t;
#endif

__device__ __forceinline__ float f_add(float a, float b) { float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_sub(float a, float b) { float r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_mul(float a, float b) { float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_fma_ftz(float a, float b, float c) { float r; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float f_max(float a, float b) { float r; asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_div_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_abs_ftz(float a) { float r; asm("abs.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_ex2_ftz(float a) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_lg2_ftz(float a) { float r; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_bits(unsigned u) { return __int_as_float((int)u); }

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

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    unsigned a; asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p)); return a;
}
__device__ __forceinline__ void ldsm_x2(unsigned& r0, unsigned& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], const unsigned b0, const unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

extern "C" __global__ void __launch_bounds__(128, MIN_BLOCKS)
fsb_rows_fused2(const i64* __restrict__ SELA, const i64* __restrict__ SELB,
                const unsigned short* __restrict__ QBF, const unsigned short* __restrict__ KEY,
                const float* __restrict__ CELLA, const i64 ca0, const i64 ca1,
                const i64* __restrict__ TIDX, const float* __restrict__ QPOS, const i64 sq0, const i64 sq1,
                const float* __restrict__ LAM, const i64 ls0, const float* __restrict__ SIG, const i64 ss0,
                out_t* __restrict__ OUT, signed char* __restrict__ PROV, unsigned char* __restrict__ OVF,
                const int Mc, const int ucap, const int Hh, const i64 sb, const i64 sh, const i64 sl,
                const float scale, const i64 QST0, const i64 QST1, const int NQR) {
    __shared__ unsigned short s_cells[512];        // 1 KB
    __shared__ short s_pu[CA + CB][128];           // <= 8 KB (stores rank pu in 0..511, or -1)
    __shared__ int wsum[4];
    __shared__ union {
        struct {
            unsigned bm[MCW];
            unsigned short wbase[MCW];
        } u;
        struct {
            unsigned short ks[2 * 32 * FROW];      // 9 KB
            float lse[UTILE][132];
        } s;
    } sm;

    const int r = blockIdx.x, tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    const int nw = (Mc + 31) >> 5;
    for (int w = tid; w < nw; w += 128) sm.u.bm[w] = 0u;
    __syncthreads();

    const i64 row = (i64)r * 128 + tid;
    {
        int a[CA];
        const longlong2* pa = reinterpret_cast<const longlong2*>(SELA + row * CA);
#pragma unroll
        for (int i = 0; i < CA / 2; i++) {
            const longlong2 v = pa[i];
            a[2 * i]     = (int)v.x;
            a[2 * i + 1] = (int)v.y;
        }
#pragma unroll
        for (int i = 0; i < CA; i++) {
            if (a[i] >= 0) mark(sm.u.bm, a[i]);
            s_pu[i][tid] = (short)a[i];
        }
#if CB > 0
        const longlong2* pb = reinterpret_cast<const longlong2*>(SELB + row * CB);
#pragma unroll
        for (int j2 = 0; j2 < CB / 2; j2++) {
            const longlong2 v = pb[j2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int c = (int)(h ? v.y : v.x);
                int val = -1;
                if (c >= 0) {
                    bool d = false;
#pragma unroll
                    for (int i = 0; i < CA; i++) d |= (a[i] == c);
                    if (!d) { mark(sm.u.bm, c); val = c; }
                }
                s_pu[CA + 2 * j2 + h][tid] = (short)val;
            }
        }
#endif
    }
    __syncthreads();

    const int per = (nw + 127) / 128;
    const int w0 = min(tid * per, nw), w1 = min(w0 + per, nw);
    int cnt = 0;
    for (int w = w0; w < w1; w++) cnt += __popc(sm.u.bm[w]);
    int x = cnt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(FULL, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) wsum[wid] = x;
    __syncthreads();
    if (wid == 0) {
        int v = lane < 4 ? wsum[lane] : 0;
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) {
            const int y = __shfl_up_sync(FULL, v, o);
            if (lane >= o) v += y;
        }
        if (lane < 4) wsum[lane] = v;
    }
    __syncthreads();
    int base = x - cnt + (wid > 0 ? wsum[wid - 1] : 0);
    const int U_raw = wsum[3];
    if (tid == 0 && U_raw > ucap) *OVF = 1;
    for (int w = w0; w < w1; w++) {
        sm.u.wbase[w] = (unsigned short)base;
        unsigned bits = sm.u.bm[w];
        while (bits) {
            const int b = __ffs(bits) - 1;
            bits &= bits - 1u;
            if (base < ucap) s_cells[base] = (unsigned short)((w << 5) + b);
            base++;
        }
    }
    __syncthreads();

#pragma unroll
    for (int c = 0; c < CA + CB; c++) {
        const int cid = (int)s_pu[c][tid];
        if (cid >= 0) {
            const int w = cid >> 5, b = cid & 31;
            const int pu = (int)sm.u.wbase[w] + __popc(sm.u.bm[w] & ((1u << b) - 1u));
            s_pu[c][tid] = (short)min(pu, ucap - 1);
        }
    }
    __syncthreads();

    // a strided stage view: rows >= NQR are pads and read zeros (the packed stage copy's fill values); QBF is
    // row-t contiguous when QST1 == 64, the packed layout
    const unsigned short* q_blk = QBF + (i64)r * QST0;
    unsigned qa[2][4][4];
    float qpos_r[2][2];
#pragma unroll
    for (int qm = 0; qm < 2; qm++) {
        const int qbase = (wid * 2 + qm) * 16;
        qpos_r[qm][0] = QPOS[r * sq0 + (qbase + g) * sq1];
        qpos_r[qm][1] = QPOS[r * sq0 + (qbase + g + 8) * sq1];
        const bool pok0 = qbase + g < NQR, pok1 = qbase + g + 8 < NQR;
        const unsigned* qr0 = reinterpret_cast<const unsigned*>(q_blk + (i64)(pok0 ? qbase + g : 0) * QST1);
        const unsigned* qr1 = reinterpret_cast<const unsigned*>(q_blk + (i64)(pok1 ? qbase + g + 8 : 0) * QST1);
#pragma unroll
        for (int dk = 0; dk < 4; dk++) {
            qa[qm][dk][0] = pok0 ? qr0[8 * dk + t4] : 0u;
            qa[qm][dk][1] = pok1 ? qr1[8 * dk + t4] : 0u;
            qa[qm][dk][2] = pok0 ? qr0[8 * dk + 4 + t4] : 0u;
            qa[qm][dk][3] = pok1 ? qr1[8 * dk + 4 + t4] : 0u;
        }
    }

    const i64 seq = TIDX[r];
    const i64 bi = seq / Hh, hi = seq - bi * Hh;
    const unsigned short* kb = KEY + bi * sb + hi * sh;
    const float qp_min = QPOS[r * sq0];
    const float lam_v = LAM[r * ls0];
    const float sig2_v = f_mul(2.0f, SIG[r * ss0]);
    const float erf_m = f_mul(f_bits(0x407ad8e6u), sig2_v);
    const float scale_lg2 = f_mul(scale, 1.4426950408889634f);

    float bv[KK];
    int bs[KK];
#pragma unroll
    for (int k = 0; k < KK; k++) {
        bv[k] = __int_as_float(0xff800000);
        bs[k] = CA + CB;
    }

    const int U = min(U_raw, ucap);
    unsigned short* ks0 = sm.s.ks;
    unsigned short* ks1 = sm.s.ks + 32 * FROW;

    for (int u0 = 0; u0 < U; u0 += UTILE) {
        const int u1 = min(u0 + UTILE, U);
        {
            const i64 cell = (i64)s_cells[u0];
            const unsigned short* ksrc = kb + cell * 32 * sl;
#pragma unroll
            for (int t = 0; t < 2; t++) {
                const int v = tid + 128 * t, kr = v >> 3, kc = (v & 7) * 8;
                cp16(ks0 + kr * FROW + kc, ksrc + kr * sl + kc);
            }
            cp_commit();
        }
        int stage = 0;
        for (int u = u0; u < u1; u++) {
            const int u_next = u + 1;
            unsigned short* ks_cur = stage ? ks1 : ks0;
            unsigned short* ks_nxt = stage ? ks0 : ks1;
            // ks_nxt held the previous cell, which slower warps may still be reading: every warp must be past it
            // before the prefetch overwrites it
            if (u > u0) __syncthreads();
            if (u_next < u1) {
                const i64 cell_n = (i64)s_cells[u_next];
                const unsigned short* ksrc_n = kb + cell_n * 32 * sl;
#pragma unroll
                for (int t = 0; t < 2; t++) {
                    const int v = tid + 128 * t, kr = v >> 3, kc = (v & 7) * 8;
                    cp16(ks_nxt + kr * FROW + kc, ksrc_n + kr * sl + kc);
                }
                cp_commit();
                cp_wait<1>();
            } else {
                cp_wait<0>();
            }
            __syncthreads();
            const int c32 = (int)s_cells[u];
#if POS_MODE
            const float a = CELLA[seq * ca0 + c32 * ca1];
            const float ts = f_mul(lam_v, a);
            const float cen = f_mul(f_add((float)c32, 0.5f), 32.0f);
            const bool far = f_sub(qp_min, cen) >= erf_m;
#endif
            float acc[2][4][4];
#pragma unroll
            for (int qm = 0; qm < 2; qm++)
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    acc[qm][nt][0] = 0.0f; acc[qm][nt][1] = 0.0f; acc[qm][nt][2] = 0.0f; acc[qm][nt][3] = 0.0f;
                }
#pragma unroll
            for (int dk = 0; dk < 4; dk++) {
                unsigned b0r[4], b1r[4];
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    ldsm_x2(b0r[nt], b1r[nt], ks_cur + (8 * nt + (lane & 7)) * FROW + dk * 16 + 8 * ((lane >> 3) & 1));
                }
#pragma unroll
                for (int qm = 0; qm < 2; qm++)
#pragma unroll
                    for (int nt = 0; nt < 4; nt++)
                        mma16816(acc[qm][nt], qa[qm][dk], b0r[nt], b1r[nt]);
            }
#pragma unroll
            for (int qm = 0; qm < 2; qm++) {
                const int qbase = (wid * 2 + qm) * 16;
#pragma unroll
                for (int e2 = 0; e2 < 2; e2++) {
                    float x[8];
#pragma unroll
                    for (int nt = 0; nt < 4; nt++) {
                        x[2 * nt]     = f_mul(acc[qm][nt][2 * e2], scale_lg2);
                        x[2 * nt + 1] = f_mul(acc[qm][nt][2 * e2 + 1], scale_lg2);
                    }
                    float m = f_max(x[0], x[1]);
#pragma unroll
                    for (int i = 2; i < 8; i++) m = f_max(m, x[i]);
                    m = f_max(m, __shfl_xor_sync(FULL, m, 2));
                    m = f_max(m, __shfl_xor_sync(FULL, m, 1));
#if LSE_MODE
                    float sum = 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; i++) sum = f_add(sum, f_ex2_ftz(f_sub(x[i], m)));
                    sum = f_add(sum, __shfl_xor_sync(FULL, sum, 2));
                    sum = f_add(sum, __shfl_xor_sync(FULL, sum, 1));
                    float lse_v = f_mul(f_add(m, f_lg2_ftz(sum)), 0.6931471805599453f);
#else
                    float lse_v = f_mul(m, 0.6931471805599453f);
#endif
                    if (t4 == 0) {
#if POS_MODE
                        if (far) {
                            lse_v = f_add(lse_v, ts);
                        } else {
                            const float dx = f_abs_ftz(f_sub(qpos_r[qm][e2], cen));
                            lse_v = f_add(lse_v, f_mul(ts, erf_tl(f_div_full(dx, sig2_v))));
                        }
#endif
                        sm.s.lse[u - u0][qbase + g + 8 * e2] = lse_v;
                    }
                }
            }
            stage ^= 1;
        }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < CA + CB; c++) {
            const int pu = (int)s_pu[c][tid];
            if (pu >= u0 && pu < u1) {
                const float sc = sm.s.lse[pu - u0][tid];
                bool gt[KK];
#pragma unroll
                for (int k = 0; k < KK; k++)
                    gt[k] = (sc > bv[k]) || (sc == bv[k] && c < bs[k]);
#pragma unroll
                for (int k = KK - 1; k > 0; k--) {
                    bv[k] = gt[k - 1] ? bv[k - 1] : (gt[k] ? sc : bv[k]);
                    bs[k] = gt[k - 1] ? bs[k - 1] : (gt[k] ? c  : bs[k]);
                }
                bv[0] = gt[0] ? sc : bv[0];
                bs[0] = gt[0] ? c  : bs[0];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int k = 0; k < KK; k++) {
        const bool ok = bv[k] > __int_as_float(0xff800000);
        const out_t val = ok ? (out_t)s_cells[s_pu[bs[k]][tid]] : (out_t)-1;
        OUT[row * KK + k] = val;
#if PROV_ON
        PROV[row * KK + k] = (signed char)(!ok ? -1 : (bs[k] >= CA ? PROV_BIT : 0));
#endif
    }
}
#endif


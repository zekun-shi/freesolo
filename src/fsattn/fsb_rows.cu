
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

#ifndef V3
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

#ifdef V3
// fsb_rows_v3: fused2's row union and (query, cell) scores with the same arithmetic in the same order (identical
// bits), restructured for throughput: the union from coalesced slot loads with plain presence bytes and a rank map,
// the key ring filled one cell ahead by all threads, per (cell, 16-query tile) need flags so a warp skips the tiles
// whose queries do not have the cell as a candidate, and each lane finishing one (query, cell) score (log2, prior)
// instead of a quarter of the lanes finishing four.
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// one 16-query row half (e2) of a cell's 32 token scores: fused2's scale, its max tree (the quad's max ends in all four
// lanes) and the lane's partial ex2 sum over its 8 tokens in fused2's order
__device__ __forceinline__ void v3_part(const float (&acc)[4][4], const int e2, const float scale_lg2, float& m_out,
                                        float& s_out) {
    float xv[8];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        xv[2 * nt]     = f_mul(acc[nt][2 * e2], scale_lg2);
        xv[2 * nt + 1] = f_mul(acc[nt][2 * e2 + 1], scale_lg2);
    }
    float m = f_max(xv[0], xv[1]);
#pragma unroll
    for (int i = 2; i < 8; i++) m = f_max(m, xv[i]);
    m = f_max(m, __shfl_xor_sync(FULL, m, 2));
    m = f_max(m, __shfl_xor_sync(FULL, m, 1));
    m_out = m;
#if LSE_MODE
    float sum = f_ex2_ftz(f_sub(xv[0], m));          // fused2 starts its sum at 0.0f: 0 + e0 == e0 (ex2 is never -0)
#pragma unroll
    for (int i = 1; i < 8; i++) sum = f_add(sum, f_ex2_ftz(f_sub(xv[i], m)));
    s_out = sum;
#else
    s_out = 0.0f;
#endif
}

// the quad's partial sums S[h] of four row halves -> lane t4 gets row half t4's total (S_0 + S_2) + (S_1 + S_3) over
// the quad's lanes: the additions of fused2's xor-2 then xor-1 butterfly, so the same bits, from 3 shuffles instead of 8
__device__ __forceinline__ float v3_sum_scatter(const float (&S)[4], const int t4) {
    const bool hi = (t4 & 2) != 0;
    const float k0 = hi ? S[2] : S[0], k1 = hi ? S[3] : S[1];
    const float x0 = hi ? S[0] : S[2], x1 = hi ? S[1] : S[3];
    const float p0 = f_add(k0, __shfl_xor_sync(FULL, x0, 2));
    const float p1 = f_add(k1, __shfl_xor_sync(FULL, x1, 2));
    const bool odd = (t4 & 1) != 0;
    const float kk = odd ? p1 : p0, xx = odd ? p0 : p1;
    return f_add(kk, __shfl_xor_sync(FULL, xx, 1));
}

// one ring cell against the warp's needed tiles (MASK bit t: its 16-query tile t) -> the (max, ex2 sum) of the row half
// the lane finishes: row half t4 is (tile t4 >> 1, rows g + 8 (t4 & 1)); one tile's two row halves sit at t4 = 2t, 2t + 1
template <int MASK>
__device__ __forceinline__ void v3_cell(const unsigned (&qa)[2][4][4], const unsigned kaddr, const unsigned o0,
                                        const unsigned o1, const float scale_lg2, const int t4, float& m_out,
                                        float& s_out) {
    unsigned bf[4][4][2];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        ldsm_x4(bf[nt][0][0], bf[nt][0][1], bf[nt][1][0], bf[nt][1][1], kaddr + o0 + nt * 1024);
        ldsm_x4(bf[nt][2][0], bf[nt][2][1], bf[nt][3][0], bf[nt][3][1], kaddr + o1 + nt * 1024);
    }
    float mg[4], sg[4];
    if (MASK == 3) {
        float acc[2][4][4];
#pragma unroll
        for (int qm = 0; qm < 2; qm++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                acc[qm][nt][0] = 0.0f; acc[qm][nt][1] = 0.0f; acc[qm][nt][2] = 0.0f; acc[qm][nt][3] = 0.0f;
            }
#pragma unroll
        for (int dk = 0; dk < 4; dk++)
#pragma unroll
            for (int qm = 0; qm < 2; qm++)
#pragma unroll
                for (int nt = 0; nt < 4; nt++)
                    mma16816(acc[qm][nt], qa[qm][dk], bf[nt][dk][0], bf[nt][dk][1]);
#pragma unroll
        for (int h = 0; h < 4; h++) v3_part(acc[h >> 1], h & 1, scale_lg2, mg[h], sg[h]);
    } else {
        const int qm = MASK == 2 ? 1 : 0;
        float acc[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            acc[nt][0] = 0.0f; acc[nt][1] = 0.0f; acc[nt][2] = 0.0f; acc[nt][3] = 0.0f;
        }
#pragma unroll
        for (int dk = 0; dk < 4; dk++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
                mma16816(acc[nt], qa[qm][dk], bf[nt][dk][0], bf[nt][dk][1]);
        v3_part(acc, 0, scale_lg2, mg[0], sg[0]);
        v3_part(acc, 1, scale_lg2, mg[1], sg[1]);
        mg[2] = mg[0]; mg[3] = mg[1]; sg[2] = sg[0]; sg[3] = sg[1];
    }
    m_out = (t4 & 2) ? ((t4 & 1) ? mg[3] : mg[2]) : ((t4 & 1) ? mg[1] : mg[0]);
#if LSE_MODE
    s_out = v3_sum_scatter(sg, t4);
#else
    s_out = 0.0f;
#endif
}

// the lane's (query, cell) score from its row half's (max, sum): fused2's LSE and lse_pos prior
__device__ __forceinline__ float v3_score(const float m, const float sm_, const float qp_l, const float qp_min,
                                          const float lam_v, const float a, const int c32, const float sig2_v,
                                          const float erf_m) {
#if LSE_MODE
    float lse_v = f_mul(f_add(m, f_lg2_ftz(sm_)), 0.6931471805599453f);
#else
    float lse_v = f_mul(m, 0.6931471805599453f);
#endif
#if POS_MODE
    const float ts = f_mul(lam_v, a);
    const float cen = f_mul(f_add((float)c32, 0.5f), 32.0f);
    if (f_sub(qp_min, cen) >= erf_m) {
        lse_v = f_add(lse_v, ts);
    } else {
        const float dx = f_abs_ftz(f_sub(qp_l, cen));
        lse_v = f_add(lse_v, f_mul(ts, erf_tl(f_div_full(dx, sig2_v))));
    }
#endif
    return lse_v;
}

// top-KK insertion of slot c's score, fused2's order (a higher score, then the earlier slot); FIRST: the row's first
// window, whose earlier insertions all have smaller slots, so a tie never displaces a kept entry
template <bool FIRST>
__device__ __forceinline__ void v3_insert(float (&bv)[KK], int (&bs)[KK], const float sc, const int c, const bool in) {
    bool gt[KK];
#pragma unroll
    for (int k = 0; k < KK; k++)
        gt[k] = in && (FIRST ? (sc > bv[k]) : ((sc > bv[k]) || (sc == bv[k] && c < bs[k])));
#pragma unroll
    for (int k = KK - 1; k > 0; k--) {
        bv[k] = gt[k - 1] ? bv[k - 1] : (gt[k] ? sc : bv[k]);
        bs[k] = gt[k - 1] ? bs[k - 1] : (gt[k] ? c  : bs[k]);
    }
    bv[0] = gt[0] ? sc : bv[0];
    bs[0] = gt[0] ? c  : bs[0];
}

// shared memory (dynamic; fsb_rows_cuda._v3_smem): the cell list, the per-rank tile need bits, the slot ranks, the
// scan totals, then region R: during the union the presence bytes [Mcp], the cell -> rank map [Mcp] and the
// (rank, tile) need bytes [UCAP][8]; afterwards the key ring (NST stages of NC cells, 4 KB each, and their cellA
// values), then the score window lse[UTILE][128] (each warp reads and writes only its own 32 query columns)
#define C_ALL (CA + CB)
#define V3_OFF_NEED 1024
#define V3_OFF_PU (V3_OFF_NEED + 512)
#define V3_OFF_MISC (V3_OFF_PU + C_ALL * 256)
#define V3_OFF_R (V3_OFF_MISC + 64)
#define V3_RING (NST * NC * (4096 + 16))

extern "C" __global__ void __launch_bounds__(128, MINB)
fsb_rows_v3(const i64* __restrict__ SELA, const i64* __restrict__ SELB,
            const unsigned short* __restrict__ QBF, const unsigned short* __restrict__ KEY,
            const float* __restrict__ CELLA, const i64 ca0, const i64 ca1,
            const i64* __restrict__ TIDX, const float* __restrict__ QPOS, const i64 sq0, const i64 sq1,
            const float* __restrict__ LAM, const i64 ls0, const float* __restrict__ SIG, const i64 ss0,
            out_t* __restrict__ OUT, signed char* __restrict__ PROV, unsigned char* __restrict__ OVF,
            const int Mc, const int ucap, const int Hh, const i64 sb, const i64 sh, const i64 sl,
            const float scale, const i64 QST0, const i64 QST1, const int NQR) {
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* s_cells = reinterpret_cast<unsigned short*>(smem);
    unsigned char* s_need = smem + V3_OFF_NEED;
    short (*s_pu)[128] = reinterpret_cast<short (*)[128]>(smem + V3_OFF_PU);
    int* s_misc = reinterpret_cast<int*>(smem + V3_OFF_MISC);
    unsigned char* R = smem + V3_OFF_R;
    const int Mcp = (Mc + 2047) & ~2047;
    unsigned char* s_pres = R;
    unsigned short* s_rank = reinterpret_cast<unsigned short*>(R + Mcp);
    unsigned char* s_nb = R + 3 * Mcp;
    unsigned short* ring = reinterpret_cast<unsigned short*>(R);
    float* s_rca = reinterpret_cast<float*>(R + NST * NC * 4096);
    float* lse = reinterpret_cast<float*>(R + V3_RING);

    const int r = blockIdx.x, tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;

    // the warp's two 16-query tiles as mma A fragments (fused2's loads, in flight during the union; rows >= NQR, the
    // pads of a strided stage view, read as zero queries)
    unsigned qa[2][4][4];
    {
        const unsigned short* q_blk = QBF + (i64)r * QST0;
#pragma unroll
        for (int qm = 0; qm < 2; qm++) {
            const int qbase = (wid * 2 + qm) * 16;
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
    }

    // row union: every valid slot sets its cell's presence byte (plain stores: concurrent writers write the same 1)
    {
        uint4* z = reinterpret_cast<uint4*>(s_pres);
        for (int i = tid; i < Mcp / 16; i += 128) z[i] = make_uint4(0u, 0u, 0u, 0u);
        uint4* zn = reinterpret_cast<uint4*>(s_nb);
        for (int i = tid; i < 4096 / 16; i += 128) zn[i] = make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();
    {
        // coalesced: chunk f (16 bytes, two slots) of the row's slots is query f / (CA / 2)'s slots 2 (f % (CA / 2)), + 1
        const longlong2* pa = reinterpret_cast<const longlong2*>(SELA + (i64)r * 128 * CA);
#pragma unroll
        for (int i = 0; i < CA / 2; i++) {
            const int f = tid + 128 * i, q = f / (CA / 2), k = 2 * (f % (CA / 2));
            const longlong2 v = pa[f];
            const int c0 = (int)v.x, c1 = (int)v.y;
            if (c0 >= 0) s_pres[c0] = 1;
            if (c1 >= 0) s_pres[c1] = 1;
            s_pu[k][q] = (short)c0;
            s_pu[k + 1][q] = (short)c1;
        }
#if CB > 0
        const longlong2* pb = reinterpret_cast<const longlong2*>(SELB + (i64)r * 128 * CB);
#pragma unroll
        for (int i = 0; i < CB / 2; i++) {
            const int f = tid + 128 * i, q = f / (CB / 2), k = CA + 2 * (f % (CB / 2));
            const longlong2 v = pb[f];
            const int c0 = (int)v.x, c1 = (int)v.y;
            if (c0 >= 0) s_pres[c0] = 1;
            if (c1 >= 0) s_pres[c1] = 1;
            s_pu[k][q] = (short)c0;
            s_pu[k + 1][q] = (short)c1;
        }
#endif
    }
    __syncthreads();
    // ranks: an exclusive scan of the presence counts of contiguous byte runs (per = Mcp / 128, a multiple of 16)
    const int per = Mcp / 128;
    int cnt = 0;
    {
        const uint4* pp = reinterpret_cast<const uint4*>(s_pres + tid * per);
        for (int i = 0; i < per / 16; i++) {
            const uint4 v = pp[i];
            cnt += __popc(__vcmpne4(v.x, 0u)) + __popc(__vcmpne4(v.y, 0u)) + __popc(__vcmpne4(v.z, 0u))
                 + __popc(__vcmpne4(v.w, 0u));
        }
        cnt >>= 3;
    }
    int x = cnt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(FULL, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_misc[wid] = x;
    __syncthreads();
    const int w0s = s_misc[0], w1s = s_misc[1], w2s = s_misc[2], w3s = s_misc[3];
    const int U_raw = w0s + w1s + w2s + w3s;
    if (tid == 0 && U_raw > ucap) *OVF = 1;
    {
        const int pre = (wid > 0 ? w0s : 0) + (wid > 1 ? w1s : 0) + (wid > 2 ? w2s : 0);
        int base = x - cnt + pre;
        const uint4* pp = reinterpret_cast<const uint4*>(s_pres + tid * per);
        for (int i = 0; i < per / 16 && base < x + pre; i++) {
            const uint4 v = pp[i];
            const unsigned wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int h = 0; h < 4; h++) {
                unsigned nz = __vcmpne4(wv[h], 0u) & 0x01010101u;     // bit 8 b: byte b is present
                while (nz) {
                    const int b = (__ffs(nz) - 1) >> 3;
                    nz &= nz - 1u;
                    const int cell = tid * per + 16 * i + 4 * h + b;
                    // the list keeps the first ucap cells and the ranks cap at ucap - 1 (fused2's overflowing rows)
                    s_rank[cell] = (unsigned short)min(base, ucap - 1);
                    if (base < ucap) s_cells[base] = (unsigned short)cell;
                    base++;
                }
            }
        }
    }
    __syncthreads();
    const int U = min(U_raw, ucap);
    {
        // thread tid = query tid: its slots' ranks, beam b's repeats of its beam-a cells -> -1 (union_candidates' rule;
        // a row of at most 128 cells tests them on a 128-bit rank set, larger rows compare the cells as fused2), and
        // the (rank, tile of 16 queries) need bytes
        int ac[CA], ra[CA];
        unsigned am0 = 0u, am1 = 0u, am2 = 0u, am3 = 0u;
#pragma unroll
        for (int c = 0; c < CA; c++) {
            ac[c] = (int)s_pu[c][tid];
            ra[c] = ac[c] >= 0 ? (int)s_rank[ac[c]] : -1;
            if (ra[c] >= 0) {
                const unsigned bit = 1u << (ra[c] & 31);
                const int w = (ra[c] >> 5) & 3;
                am0 |= w == 0 ? bit : 0u; am1 |= w == 1 ? bit : 0u; am2 |= w == 2 ? bit : 0u; am3 |= w == 3 ? bit : 0u;
            }
        }
        const int tb = tid >> 4;
#pragma unroll
        for (int c = 0; c < CA; c++) {
            s_pu[c][tid] = (short)ra[c];
            if (ra[c] >= 0) s_nb[ra[c] * 8 + tb] = 1;
        }
#if CB > 0
        const bool small = U_raw <= 128;
#pragma unroll
        for (int j = 0; j < CB; j++) {
            const int cid = (int)s_pu[CA + j][tid];
            int rb = cid >= 0 ? (int)s_rank[cid] : -1;
            if (rb >= 0) {
                bool dup;
                if (small) {
                    const int w = rb >> 5;
                    const unsigned wd = w == 0 ? am0 : (w == 1 ? am1 : (w == 2 ? am2 : am3));
                    dup = (wd >> (rb & 31)) & 1u;
                } else {
                    dup = false;
#pragma unroll
                    for (int i = 0; i < CA; i++) dup |= (ac[i] == cid);
                }
                if (dup) rb = -1;
            }
            s_pu[CA + j][tid] = (short)rb;
            if (rb >= 0) s_nb[rb * 8 + tb] = 1;
        }
#endif
    }
    __syncthreads();
    // need bytes -> one byte of tile bits per rank
    for (int u = tid; u < U; u += 128) {
        const uint2 v = *reinterpret_cast<const uint2*>(s_nb + u * 8);
        const unsigned lo = __vcmpne4(v.x, 0u), hi = __vcmpne4(v.y, 0u);
        unsigned b = 0u;
#pragma unroll
        for (int t = 0; t < 4; t++) {
            b |= ((lo >> (8 * t)) & 1u) << t;
            b |= ((hi >> (8 * t)) & 1u) << (t + 4);
        }
        s_need[u] = (unsigned char)b;
    }

    const i64 seq = TIDX[r];
    const i64 bi = seq / Hh, hi = seq - bi * Hh;
    const unsigned short* kb = KEY + bi * sb + hi * sh;
    const float* ca_row = CELLA + seq * ca0;
    const float qp_min = QPOS[r * sq0];
    const float lam_v = LAM[r * ls0];
    const float sig2_v = f_mul(2.0f, SIG[r * ss0]);
    const float erf_m = f_mul(f_bits(0x407ad8e6u), sig2_v);
    const float scale_lg2 = f_mul(scale, 1.4426950408889634f);
    // the lane finishes row half t4 of the warp's tiles: query q_l
    const int q_l = (wid * 2 + (t4 >> 1)) * 16 + g + 8 * (t4 & 1);
    const float qp_l = QPOS[r * sq0 + q_l * sq1];
    float* lse_l = lse + q_l;
    // per-lane ldmatrix offsets into a ring cell: key row kr's 16-byte chunk ch sits at chunk ch ^ (kr & 7) (no bank
    // conflicts for the 8 rows of a matrix)
    const unsigned csw = (unsigned)((lane >> 3) ^ (lane & 7));
    const unsigned o0 = (lane & 7) * 128 + csw * 16, o1 = (lane & 7) * 128 + (csw ^ 4u) * 16;
    const unsigned ring_a = smem_addr(ring);

    float bv[KK];
    int bs[KK];
#pragma unroll
    for (int k = 0; k < KK; k++) {
        bv[k] = __int_as_float(0xff800000);
        bs[k] = C_ALL;
    }

    const int NS = (U + NC - 1) / NC;
    // stage sg's NC cells into ring slot sg % NST: each thread copies 2 of a cell's 256 16-byte chunks, thread 0 the
    // cell's cellA value
#define V3_ISSUE(sg_)                                                                                                 \
    do {                                                                                                              \
        const int sgi = (sg_);                                                                                        \
        if (sgi < NS) {                                                                                               \
            unsigned short* dst0 = ring + (sgi % NST) * (NC * 2048);                                                  \
            _Pragma("unroll")                                                                                         \
            for (int j = 0; j < NC; j++) {                                                                            \
                const int uu = sgi * NC + j;                                                                          \
                if (uu < U) {                                                                                         \
                    const int cell_ = (int)s_cells[uu];                                                               \
                    const unsigned short* ksrc = kb + (i64)cell_ * 32 * sl;                                           \
                    _Pragma("unroll")                                                                                 \
                    for (int t = 0; t < 2; t++) {                                                                     \
                        const int v = tid + 128 * t, kr = v >> 3, ch = v & 7;                                         \
                        cp16(dst0 + j * 2048 + kr * 64 + ((ch ^ (kr & 7)) << 3), ksrc + kr * sl + ch * 8);            \
                    }                                                                                                 \
                    if (POS_MODE && tid == 0) cp4(s_rca + (sgi % NST) * NC + j, ca_row + (i64)cell_ * ca1);           \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
        cp_commit();                                                                                                  \
    } while (0)

    __syncthreads();             // the union scratch, which the ring aliases, is dead
#pragma unroll
    for (int s0 = 0; s0 < NST - 1; s0++) V3_ISSUE(s0);

    int u0 = 0;
    const int wsh = 2 * wid;
    for (int sg = 0; sg < NS; sg++) {
        cp_wait<NST - 2>();
        __syncthreads();
        V3_ISSUE(sg + NST - 1);
        const unsigned kst = ring_a + (unsigned)((sg % NST) * (NC * 4096));
#pragma unroll 1
        for (int j = 0; j < NC; j++) {
            const int u = sg * NC + j;
            if (u >= U) break;
            const unsigned nm = ((unsigned)s_need[u] >> wsh) & 3u;
            if (nm) {
                float m, s;
                const unsigned ka = kst + (unsigned)(j * 4096);
                if (nm == 3u) v3_cell<3>(qa, ka, o0, o1, scale_lg2, t4, m, s);
                else if (nm == 1u) v3_cell<1>(qa, ka, o0, o1, scale_lg2, t4, m, s);
                else v3_cell<2>(qa, ka, o0, o1, scale_lg2, t4, m, s);
                if ((nm >> (t4 >> 1)) & 1u)
                    lse_l[(u - u0) * 128] = v3_score(m, s, qp_l, qp_min, lam_v, s_rca[(sg % NST) * NC + j],
                                                     (int)s_cells[u], sig2_v, erf_m);
            }
            if (u + 1 - u0 == UTILE || u + 1 == U) {
                // the warp's queries keep their top KK over the window's candidates (scores written by the warp)
                __syncwarp();
                const int u1 = u + 1;
                const float* lt = lse + tid;
                if (u0 == 0) {
#pragma unroll
                    for (int c = 0; c < C_ALL; c++) {
                        const int pu = (int)s_pu[c][tid];
                        const bool in = (unsigned)(pu - u0) < (unsigned)(u1 - u0);
                        v3_insert<true>(bv, bs, lt[(in ? pu - u0 : 0) * 128], c, in);
                    }
                } else {
#pragma unroll
                    for (int c = 0; c < C_ALL; c++) {
                        const int pu = (int)s_pu[c][tid];
                        const bool in = (unsigned)(pu - u0) < (unsigned)(u1 - u0);
                        v3_insert<false>(bv, bs, lt[(in ? pu - u0 : 0) * 128], c, in);
                    }
                }
                __syncwarp();
                u0 = u1;
            }
        }
    }
#undef V3_ISSUE

    const i64 row = (i64)r * 128 + tid;
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
#endif

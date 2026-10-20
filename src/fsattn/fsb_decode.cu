// >>> SECTION: _DECODE_KERNEL <<<
// The FS-B decode read of one layer in one launch (fsb_decode_cuda, FSB_DECODE_V2=1): one block of NWARP warps per
// (batch, head) row runs the main beam descent (warps 0..3) and the union beam descent (warps 4..7), the union re-rank
// of their cells (warps 8.. re-rank the union beam's cells while the main descent goes on, then every warp the main
// beam's), the stable top-KK selection and the readout of the selected cells with lr_norm and the window combine.
// - descents: fsb_beam_pair_cuda's frontier, child scores, ranks and cells, bitwise (team_descend: four lanes per
//   child, the frontier in shared memory carrying every aux field the next level and the cells epilogue read; each
//   level stages the next level's candidate nodes in shared memory with cp.async while it is scored, so no global
//   round trip sits on a level's critical path, and prefetches into L2 the key rows of cells that may be read);
// - re-rank: _fsb_upar_int8_kernel + _fsb_selpv_kernel's arithmetic (the int8 rows dequantized as int8 * scale with one
//   rounding, libdevice expf / logf as Triton links them, CUDA's erff as torch calls it), so the read cells and their
//   provenance are bitwise the chain's;
// - readout: _fsb_decread_int8_kernel's math (the token scores are the re-rank's own q.k dot products, stored in bf16
//   and scaled), with its own reduction order for the softmax denominator and the value sum (F8-class, as that kernel
//   against the torch reference).
// DEC_V3 (FSB_DECODE_V3=1) shortens each descent's chain with the same cells and outputs bit for bit: the query's load
// leads the preamble, the spine rows (and the cached shape plan) arrive by cp.async in one round, and the root's aux
// row is read only where the level loop starts at the root; the keep-all top levels run as one batch (team_prefix:
// their nodes from the tree's shape, cached per row, staged in one round, scored in one pass, the frontier order
// replayed); and the ranks compare through register masks rather than predicate chains.
// Requires _HELPERS (fsb_beam.cu). Compile-time: DH (64), NA (9), NC, WM (<= 16), R, KK (cells read: 1, 2, 4 or 8),
// SCELL (cell size, 32), BOUND, RC_T, LSE_ON, POS_ON, HAS_SR, W_BF16, LRS_BF16, PROV_BIT, NWARP (16), DEC_V3.

#define C_ALL (2 * WM * R)
#define CA (WM * R)
#define TPW (KK * SCELL / NWARP)                    // readout tokens per warp
#define NEG_INF __int_as_float(0xff800000)
#define POS_INF __int_as_float(0x7f800000)
#ifndef DEC_TIMING
#define DEC_TIMING 0
#endif
// diagnostic build (DEC_TIMING=1): lane 0 of the recording warp stores the SM clock at phase boundaries
#if DEC_TIMING
#define TSTAMP(k) do { if (TIMES != nullptr && lane == 0) TIMES[r * 256 + (k)] = clock64(); } while (0)
#define LSTAMP(k) do { if (lt_out != nullptr && tt == 0 && lev < 16) lt_out[lev * 8 + (k)] = clock64(); } while (0)
#else
#define TSTAMP(k) do { } while (0)
#define LSTAMP(k) do { } while (0)
#endif

// one beam's tables (forest_beam_pair_cuda's BeamP without its outputs, plus the spines' heights)
struct DecBeam {
    const i64* ROOT; const i64* WBEAM; const i64* BUDGET; const i64* TIDX;
    const unsigned short* CHUNK; const float* CHUNK_AX; const float* SPINE; const i64* LEAF; const i64* ADM;
    const i64* HGT; int* PLAN;
    i64 Ntot; i64 Sp; i64 Ltot; i64 Mm; int KMAXP1; int pad_;
};

#define SPMAX 20                                    // spine rows a team stages (deeper spines read global memory)
#define NST (4 * WM)                                // staged nodes: the two children of each of a level's children

// a descent team's shared state: the frontier (double-buffered across levels: slot ref, the aux fields the next level
// and the cells epilogue read, and stg, the stage entries 2 stg + side of the node's children), the level's child
// scores, the next level's candidate chunk nodes (bf16 means and aux rows, staged while the level is scored) and the
// row's spine (fp32 means and aux, staged once)
struct __align__(16) FrontS {
    int ref[16], stg[16], chl[16], chr[16];
    float cnt[16], src[16], off[16], len[16];
};
struct __align__(16) TeamS {
    uint4 mu[NST][DH / 8];
    float4 ax[NST][3];                              // aux rows padded to 12 words: three 16-byte reads
    float spn[SPMAX][NC];
    FrontS f[2];
    float sc[32];
};

__device__ __forceinline__ float bf_lo(const unsigned w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(const unsigned w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ float bf2f(const unsigned short h) { return __uint_as_float(((unsigned)h) << 16); }
__device__ __forceinline__ unsigned short f2bf(const float x) {
    unsigned short r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x)); return r; }
__device__ __forceinline__ float bf_round(const float x) { return bf2f(f2bf(x)); }
__device__ __forceinline__ float f_fma_rn(float a, float b, float c) {
    float r; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float f_fma_rm_ftz(float a, float b, float c) {
    float r; asm("fma.rm.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float f_sat_ftz(float a) { float r; asm("cvt.ftz.sat.f32.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_div_rn(float a, float b) { float r; asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_max(float a, float b) { float r; asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_abs(float a) { float r; asm("abs.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_rsqrt_ftz(float a) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_i2f(int a) { float r; asm("cvt.rn.f32.s32 %0, %1;" : "=f"(r) : "r"(a)); return r; }
// comparison masks in a register (all ones / zero) rather than a predicate: a rank over many candidates then needs no
// chain of predicated moves through the few predicate registers
__device__ __forceinline__ unsigned m_gt(const float a, const float b) {
    unsigned r; asm("set.gt.u32.f32 %0, %1, %2;" : "=r"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ unsigned m_eq(const float a, const float b) {
    unsigned r; asm("set.eq.u32.f32 %0, %1, %2;" : "=r"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ unsigned m_lt(const int a, const int b) {
    unsigned r; asm("set.lt.u32.s32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
// 1 when (s2, c2) ranks ahead of (s, c): higher score, or the same score and a lower index (the ranks' tie order)
__device__ __forceinline__ int ahead(const float s2, const int c2, const float s, const int c) {
    return (int)((m_gt(s2, s) | (m_eq(s2, s) & m_lt(c2, c))) & 1u); }
__device__ __forceinline__ unsigned word_of(const uint4& u, const int k) {
    return k == 0 ? u.x : (k == 1 ? u.y : (k == 2 ? u.z : u.w));
}
__device__ __forceinline__ unsigned smem_u32(const void* p) {
    unsigned a; asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p)); return a;
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

// int8 byte b of word w times a cell scale, rounded once (= fp32(q) * scale): 12582912 + q is exact in fp32 with the
// byte sign-extended into its low bits, and c2 = -12582912 * scale (10 significant bits) is exact, so the FMA's single
// rounding is that of the product
__device__ __forceinline__ float deq8(const unsigned w, const int b, const float scale, const float c2) {
    int sx;                                          // prmt with sign replication (__byte_perm drops the mode bit)
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(sx) : "r"(w), "r"(b | ((b | 8) << 4) | ((b | 8) << 8) | ((b | 8) << 12)));
    return f_fma_rn(__int_as_float(sx + 0x4B400000), scale, c2);
}

// libdevice __nv_expf / __nv_logf (ftz) as Triton links them, and CUDA's erff (torch.erf), instruction for instruction
__device__ __forceinline__ float exp_tl(const float x) {
    const float t = f_sat_ftz(f_fma_ftz(x, f_bits(0x3BBB989Du), 0.5f));
    const float j = f_fma_rm_ftz(t, f_bits(0x437C0000u), f_bits(0x4B400001u));
    const float k = f_add(j, f_bits(0xCB40007Fu));
    float y = f_fma_ftz(x, f_bits(0x3FB8AA3Bu), -k);
    y = f_fma_ftz(x, f_bits(0x32A57060u), y);
    return f_mul(f_ex2_ftz(y), __int_as_float(__float_as_int(j) << 23));
}
__device__ __forceinline__ float log_tl(const float x) {
    const bool sub = x < f_bits(0x00800000u);
    const float xs = sub ? f_mul(x, f_bits(0x4B000000u)) : x;
    const float e0 = sub ? f_bits(0xC1B80000u) : 0.0f;
    const int ix = __float_as_int(xs);
    const int ie = (ix - 1059760811) & (int)0xFF800000;
    const float m = __int_as_float(ix - ie);
    const float e = f_fma_ftz(f_i2f(ie), f_bits(0x34000000u), e0);
    const float f = f_add(m, f_bits(0xBF800000u));
    float p = f_fma_ftz(f_bits(0xBE055027u), f, f_bits(0x3E1039F6u));
    p = f_fma_ftz(p, f, f_bits(0xBDF8CDCCu));
    p = f_fma_ftz(p, f, f_bits(0x3E0F2955u));
    p = f_fma_ftz(p, f, f_bits(0xBE2AD8B9u));
    p = f_fma_ftz(p, f, f_bits(0x3E4CED0Bu));
    p = f_fma_ftz(p, f, f_bits(0xBE7FFF22u));
    p = f_fma_ftz(p, f, f_bits(0x3EAAAA78u));
    p = f_fma_ftz(p, f, f_bits(0xBF000000u));
    p = f_fma_ftz(f_mul(f, p), f, f);
    float r = f_fma_ftz(e, f_bits(0x3F317218u), p);
    if (!((unsigned)ix < 2139095040u)) r = f_fma_ftz(xs, POS_INF, POS_INF);
    return xs == 0.0f ? NEG_INF : r;
}
__device__ __forceinline__ float erf_torch(const float x) {
    const float a = f_abs(x);
    const bool big = a >= f_bits(0x3F8060FEu);
    const float u = big ? a : f_mul(x, x);
    float r = f_fma_rn(big ? f_bits(0x38EB4C3Au) : f_bits(0x38B1E96Au), u, big ? f_bits(0xBAAE005Bu) : f_bits(0xBA574D20u));
    r = f_fma_rn(r, u, big ? f_bits(0x3C09919Fu) : f_bits(0x3BAAD5EAu));
    r = f_fma_rn(r, u, big ? f_bits(0xBD24D99Au) : f_bits(0xBCDC1BE7u));
    r = f_fma_rn(r, u, big ? f_bits(0x3E235519u) : f_bits(0x3DE718AFu));
    r = f_fma_rn(r, u, big ? f_bits(0x3F69B4F9u) : f_bits(0xBEC093ACu));
    r = f_fma_rn(r, u, big ? f_bits(0x3F210A14u) : f_bits(0x3E0375D3u));
    const float s = big ? -a : x;
    r = f_fma_rn(r, s, s);
    float rb = f_sub(1.0f, f_ex2_ftz(r));
    rb = __int_as_float((__float_as_int(x) & (int)0x80000000) | __float_as_int(rb));
    return big ? rb : r;
}

__device__ __forceinline__ const float* dec_aux(const float* cax, const float* sbase, const int r) {
    return r >= 0 ? cax + (i64)r * NA : sbase + (i64)(-r - 2) * NC + DH;
}

// the halving sum (_halve_rows' order) of 64 values held by the 4 lanes of a group, lane g holding x[i] = s1[8 g + i]
// (s1: the first halving step, columns j and j + 32 already added): reduce-scatter rounds for the pairs (j, j + 16),
// (j, j + 8), (j, j + 4), then butterflies for (j, j + 2) and (j, j + 1); every lane of the group ends with the sum
__device__ __forceinline__ float group_halve(const float (&x)[8], const int g) {
    float y[4], z[2];
    const bool u2 = (g & 2) != 0, u1 = (g & 1) != 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float send = u2 ? x[i] : x[i + 4], keep = u2 ? x[i + 4] : x[i];
        y[i] = f_add(keep, __shfl_xor_sync(FULL, send, 2));
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const float send = u1 ? y[i] : y[i + 2], keep = u1 ? y[i + 2] : y[i];
        z[i] = f_add(keep, __shfl_xor_sync(FULL, send, 1));
    }
    const float send = u2 ? z[0] : z[1], keep = u2 ? z[1] : z[0];
    float w = f_add(keep, __shfl_xor_sync(FULL, send, 2));
    w = f_add(w, __shfl_xor_sync(FULL, w, 1));
    return f_add(w, __shfl_xor_sync(FULL, w, 2));
}

__device__ __forceinline__ void team_bar(const int id) {
    asm volatile("bar.sync %0, 128;" :: "r"(id) : "memory");
}

// the 4 lanes of group c copy chunk node n (its bf16 mean and aux row) into stage entry e: lane g the mean's 16-byte
// pieces g and g + 4 and aux words g, g + 4 (and 8)
__device__ __forceinline__ void stage_node(TeamS& S, const int e, const int n, const int g,
                                           const unsigned short* __restrict__ cmub, const float* __restrict__ cax) {
    const uint4* row = reinterpret_cast<const uint4*>(cmub + (i64)n * DH);
    cp16(&S.mu[e][g], row + g);
    cp16(&S.mu[e][g + DH / 16], row + g + DH / 16);
    const float* a = cax + (i64)n * NA;
    float* d = reinterpret_cast<float*>(S.ax[e]);
    cp4(d + g, a + g);
    cp4(d + g + 4, a + g + 4);
    if (g == 0) cp4(d + 8, a + 8);
}

#ifndef DEC_V3
#define DEC_V3 0
#endif
#if DEC_V3
#define PMAX (2 * WM + 1)                           // prefix members: the root and up to 2 WM children
#define PDMAX 16                                    // prefix levels at most
// a descent team's keep-all prefix (team_prefix): its members (the nodes of the prefix levels: index, stage entry, chunk
// height << 16 | heap index or -1 for a spine node, count, parent member, slot in the replayed frontier, stage index of
// the children of a member that expands at the first per-level loop level), each level's member range, the frontier
// hand-over of the replay (member and count per slot) and the expanding members of the last prefix level
// The walk's part (key .. npm) is a pure function of the forest's shape at the row (root, its count, the budgets, the
// level bound) and is cached per row in global memory (PLAN): loaded with the spine rows, reused while its key holds.
struct __align__(16) PlanS {
    int key[8];
    int ref[PMAX], ent[PMAX], kh[PMAX];
    float cnt[PMAX];
    signed char mlv[2 * WM];                        // the level of member m at index m - 1 (word-aligned)
    signed char par[PMAX], eix[PMAX], fe[16];
    signed char lvs[PDMAX + 1], lvn[PDMAX + 1];
    int L0, nfe, npm;
};
struct __align__(16) PrefS : PlanS {
    float rsc[16];                                  // 16-byte aligned (read as float4): first after the plan
    int rcc[16];
    float ocnt[16];
    int bad;
    signed char slot[PMAX], fl[16], ord[16];
};
#define PLAN_BYTES ((int)sizeof(PlanS))
#define PLAN_CHUNKS ((PLAN_BYTES + 15) / 16)
#define PLAN_STRIDE 1024                            // bytes per row of the cache (fsb_decode_cuda._PLAN_INTS words)
#define PLAN_MARK 0x5EED0001
static_assert(PLAN_CHUNKS * 16 <= PLAN_STRIDE, "the prefix plan must fit its cache row");

// keep-all prefix of a descent (FSB_DECODE_V3): while a level keeps every child it visits (its carried slots plus
// its children fit in min(wbeam, WM)), its node set follows from the tree's shape alone: the spine rows (children and
// counts) and the chunks' heap layout (heap node h of a chunk of height kk has its children h + 1 and h + 2 entries
// after it and count 2^(kk - depth)), a lone chunk's height being the spine's (hgt). So the nodes of all such levels and
// the next level's candidates are staged in one round, the prefix nodes are scored in one pass with the per-level loop's
// arithmetic, and only the frontier order is replayed level by level with the loop's rules (the exit when every slot is
// carried, carried slots first, children by (score desc, c = 2 slot + side asc), the keep conditions). Every staged node
// is checked against the shape (count, heap children); on a mismatch, or with nothing to batch, it returns 0 and the loop
// starts at the root; else S.f[0] and the stage are what the loop would hold at the returned level. The shape walk and
// the replay run in the team's first warp with the frontier in registers (lane = slot). Team-uniform exits.
__device__ int team_prefix(const DecBeam& B, TeamS& S, PrefS& P, const int tt, const int bar_id, const i64 r,
                           const int root, const i64 hgt, const float bud, const int wbeam, const int nsp,
                           const unsigned short* __restrict__ cmub, const float* __restrict__ cax,
                           const i64* __restrict__ leafb, const i64* __restrict__ admb, const i64 Ltot, const i64 Mm,
                           const float (&qa)[8], const float (&qb)[8], const float qn, const float qpos,
                           const float lam, const float sig2, const float scale, const float cell_size,
                           const signed char* __restrict__ qk_row, const i64 sqk_n,
                           const unsigned short* __restrict__ sk_row, long long* pt_out) {
    const int c = tt >> 2, g = tt & 3, lane = tt & 31, gl = lane & ~3;
    const unsigned below = (1u << lane) - 1u;
    const int cap = min(wbeam, WM);
#if DEC_TIMING
#define PSTAMP(k) do { if (pt_out != nullptr && tt == 0) pt_out[k] = clock64(); } while (0)
#else
#define PSTAMP(k) do { } while (0)
#endif
    PSTAMP(0);
    // the key of the shape this row's walk depends on: the cached plan (loaded with the spine rows) holds if it matches.
    // Only the first warp reads the cached key (it alone walks on a miss, and its lane 0 then rewrites the key)
    int key[8];
    key[0] = PLAN_MARK; key[1] = root;
    key[2] = __float_as_int(root <= -2 && -root - 2 < nsp ? S.spn[-root - 2][DH + 3] : 0.0f);
    key[3] = __float_as_int(bud); key[4] = wbeam; key[5] = B.KMAXP1; key[6] = nsp; key[7] = (int)hgt;
    bool hit = false;
    if (tt < 32) {
        hit = B.PLAN != nullptr;
#pragma unroll
        for (int i = 0; i < 8; i++) hit = hit && P.key[i] == key[i];
    }
    // ---- the prefix levels' node sets from the shape (the team's first warp; lane j: member j of the walk's frontier,
    // carried members first, then the new children in member order) ----
    if (tt < 32 && !hit) {
        __syncwarp();                                   // every lane's key read before lane 0 writes the plan
        int p = -1, ref = 0, kh = -1, bad = 0;
        float cnt = 0.0f;
        if (lane == 0 && root != -1) {
            p = 0; ref = root;
            if (root <= -2) {
                if (-root - 2 < nsp) cnt = S.spn[-root - 2][DH + 3]; else bad = 1;
            } else if (hgt < 0 || hgt > 30) {
                bad = 1;
            } else {
                kh = (int)hgt << 16; cnt = (float)(1 << (int)hgt);
            }
            P.ref[0] = root; P.ent[0] = -1; P.kh[0] = kh; P.cnt[0] = cnt; P.par[0] = -1;
        }
        bad = __shfl_sync(FULL, bad, 0);
        int npm = 1, L0 = 0;
        for (int d = 1; d <= PDMAX && !bad; d++) {
            const bool expd = p >= 0 && cnt > bud;
            const bool car = p >= 0 && !expd;
            const unsigned be = __ballot_sync(FULL, expd), bc = __ballot_sync(FULL, car);
            const int nexp = __popc(be), ncar = __popc(bc);
            if (nexp == 0 || ncar + 2 * nexp > cap || npm + 2 * nexp > PMAX || d - 1 >= B.KMAXP1) break;
            int cref[2] = {0, 0}, ckh[2] = {-1, -1};
            float ccnt[2] = {0.0f, 0.0f};
            bool cbad = false;
            if (expd) {
                if (ref <= -2) {                                // a spine node: its chunk of height k and the rest
                    const int k = -ref - 2;
                    const float* row = S.spn[k];
                    const float ck = (float)(1 << k);
                    const int rr = (int)row[DH + 5];
                    cref[0] = (int)row[DH + 4]; ckh[0] = k << 16; ccnt[0] = ck;
                    cref[1] = rr;
                    if (rr <= -2) {
                        if (-rr - 2 < nsp) ccnt[1] = S.spn[-rr - 2][DH + 3]; else cbad = true;
                    } else {
                        const float rest = f_sub(row[DH + 3], ck);  // a lone chunk below: a power of two
                        const int ri = (int)rest;
                        if (ri < 1 || (ri & (ri - 1)) != 0 || (float)ri != rest) cbad = true;
                        else { ckh[1] = (31 - __clz(ri)) << 16; ccnt[1] = rest; }
                    }
                    if (cref[0] < 0 || rr == -1) cbad = true;
                } else {                                        // heap node h of a chunk of height kk
                    const int kk = kh >> 16, h = kh & 0xffff;
                    const int l = 31 - __clz(h + 1);
                    if (kh < 0 || l >= kk || 2 * h + 2 > 0xffff) cbad = true;
                    else {
                        cref[0] = ref + h + 1; cref[1] = ref + h + 2;
                        ckh[0] = (kk << 16) | (2 * h + 1); ckh[1] = (kk << 16) | (2 * h + 2);
                        ccnt[0] = ccnt[1] = (float)(1 << (kk - l - 1));
                    }
                }
            }
            if (__any_sync(FULL, cbad)) { bad = 1; break; }
            if (expd) {
                const int c0 = npm + 2 * __popc(be & below);
#pragma unroll
                for (int sd = 0; sd < 2; sd++) {
                    const int m = c0 + sd;
                    P.ref[m] = cref[sd]; P.kh[m] = ckh[sd]; P.cnt[m] = ccnt[sd]; P.par[m] = (signed char)p;
                    P.ent[m] = cref[sd] >= 0 ? 2 * WM + m - 1 : -1;
                    P.mlv[m - 1] = (signed char)d;
                }
            }
            if (car) P.fl[__popc(bc & below)] = (signed char)p;
            if (lane == 0) { P.lvs[d] = (signed char)npm; P.lvn[d] = (signed char)(2 * nexp); }
            __syncwarp();
            const int nm = lane < ncar ? (int)P.fl[lane] : (lane < ncar + 2 * nexp ? npm + lane - ncar : -1);
            p = nm;
            if (nm >= 0) { ref = P.ref[nm]; kh = P.kh[nm]; cnt = P.cnt[nm]; }
            __syncwarp();
            npm += 2 * nexp;
            L0 = d;
        }
        // the members that expand at the loop's first level: their children go to stage entries 2 e, 2 e + 1
        const bool expd = !bad && L0 > 0 && p >= 0 && cnt > bud;
        const unsigned be = __ballot_sync(FULL, expd);
        if (!bad && L0 > 0 && p >= 0) {
            const int e = expd ? __popc(be & below) : 0;
            P.eix[p] = (signed char)e;
            if (expd) P.fe[e] = (signed char)p;
        }
        if (lane == 0) {
            P.L0 = bad ? 0 : L0; P.nfe = __popc(be); P.npm = npm;
#pragma unroll
            for (int i = 0; i < 8; i++) P.key[i] = key[i];
        }
        __syncwarp();
        if (B.PLAN != nullptr) {                        // the plan into the row's cache for the next tokens
            const uint4* src = reinterpret_cast<const uint4*>(&P);
            uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<char*>(B.PLAN) + r * PLAN_STRIDE);
            for (int i = lane; i < PLAN_CHUNKS; i += 32) dst[i] = src[i];
        }
    }
    if (tt == 0) P.bad = 0;
    team_bar(bar_id);
    PSTAMP(1);
    if (P.L0 == 0) return 0;
    const int npm = P.npm, nfe = P.nfe, L0 = P.L0;
    // ---- one round of loads: the prefix members (group c: member c + 1) and the next level's candidates ----
    if (c + 1 < npm && P.ref[c + 1] >= 0) stage_node(S, P.ent[c + 1], P.ref[c + 1], g, cmub, cax);
    if ((c >> 1) < nfe) {
        const int m = P.fe[c >> 1], sd = c & 1, ref = P.ref[m];
        const int ch = ref <= -2 ? (int)S.spn[-ref - 2][DH + 4 + sd] : ref + (P.kh[m] & 0xffff) + 1 + sd;
        if (ch >= 0) stage_node(S, c, ch, g, cmub, cax);
    }
    cp_commit();
    cp_wait_all();
    team_bar(bar_id);
    PSTAMP(2);
    // ---- the prefix members' scores in one pass (the per-level loop's operations), each staged node checked; a member
    // that cannot expand loads its first cell for the L2 prefetch issued after the replay ----
    int lcell = -1;
    {
        const int m = c + 1;
        const bool has = m < npm;
        const int ref = has ? P.ref[m] : -1;
        float ax[NA], x[8];
#pragma unroll
        for (int f = 0; f < NA; f++) ax[f] = 0.0f;
        if (has && ref >= 0) {
            const int e = P.ent[m];
            const uint4 ma = S.mu[e][g], mb = S.mu[e][g + DH / 16];
            const float4 a0 = S.ax[e][0], a1 = S.ax[e][1], a2 = S.ax[e][2];
            ax[0] = a0.x; ax[1] = a0.y; ax[2] = a0.z; ax[3] = a0.w;
            ax[4] = a1.x; ax[5] = a1.y; ax[6] = a1.z; ax[7] = a1.w; ax[8] = a2.x;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const unsigned wa = word_of(ma, i >> 1), wb = word_of(mb, i >> 1);
                x[i] = f_add(f_mul(qa[i], (i & 1) ? bf_hi(wa) : bf_lo(wa)), f_mul(qb[i], (i & 1) ? bf_hi(wb) : bf_lo(wb)));
            }
            const int kk = P.kh[m] >> 16, h = P.kh[m] & 0xffff, l = 31 - __clz(h + 1);
            const bool ok = ax[3] == P.cnt[m] && (l < kk ? (ax[4] == (float)(ref + h + 1) && ax[5] == (float)(ref + h + 2))
                                                         : (ax[4] == -1.0f && ax[5] == -1.0f));
            if (!ok && g == 0) atomicOr(&P.bad, 1);         // one lane per member; members may flag at once
        } else if (has) {
            const float* sp32 = S.spn[-ref - 2];
#pragma unroll
            for (int f = 0; f < NA; f++) ax[f] = sp32[DH + f];
#pragma unroll
            for (int i = 0; i < 8; i++) x[i] = f_add(f_mul(qa[i], sp32[8 * g + i]), f_mul(qb[i], sp32[DH / 2 + 8 * g + i]));
        } else {
#pragma unroll
            for (int i = 0; i < 8; i++) x[i] = 0.0f;
        }
        const float v = group_halve(x, g);
        float content = f_mul(v, scale);
#if BOUND
        content = f_add(content, f_mul(f_mul(qn, ax[0]), scale));
#endif
        const float pmin_pos = f_mul(f_add(ax[2], 0.5f), cell_size);
        const float dx = __int_as_float(__float_as_int(f_sub(qpos, pmin_pos)) & 0x7fffffff);
        const float score = f_add(content, f_mul(f_mul(lam, ax[1]), erf_tl(f_div_full(dx, sig2))));
        if (has && g == 0) S.sc[c] = score;
        if (has && g == 0 && ax[3] <= bud && ax[6] > 0.0f && ax[8] > 0.0f) {
            const i64 ps = (i64)ax[7];
            lcell = (int)(ax[6] == 1.0f ? leafb[min(max(ps, (i64)0), Ltot - 1)] : admb[min(max(ps, (i64)0), Mm - 1)]);
        }
    }
    team_bar(bar_id);
    PSTAMP(3);
    if (P.bad) {
        if (tt == 0 && B.PLAN != nullptr)
            reinterpret_cast<PlanS*>(reinterpret_cast<char*>(B.PLAN) + r * PLAN_STRIDE)->L0 = 0;
        return 0;
    }
    // ---- the frontier order of every prefix level with the loop's rules, and the frontier it hands over (the team's
    // first warp; lane = slot, holding the member there and its count) ----
    if (tt < 32) {
        // with no two children of one level scoring alike and none dropped (score <= -5e8), every level orders its
        // children by score alone, so the frontier of level L0 is: the carried members (cnt <= bud, made before L0) by
        // (level made, score desc), then the members made at L0 by score desc; lane j: member j + 1
        int mo = -1;
        float mcnt = 0.0f;
        bool fast;
        {
            const int m = lane + 1;
            const bool has = m < npm;
            const int lv = has ? (int)P.mlv[lane] : 0x7f;
            const float sc = has ? S.sc[lane] : NEG_INF;
            // members are numbered level by level, so the lower levels are the members below the level's first; the
            // order within the level compares only its (at most WM) members, in four independent chains
            const int ls = has ? (int)P.lvs[lv] : 1, ln = has ? (int)P.lvn[lv] : 0;
            const unsigned lower = (1u << (ls - 1)) - 1u;
            unsigned ac[4] = {0u, 0u, 0u, 0u}, tc[4] = {0u, 0u, 0u, 0u};
#pragma unroll
            for (int i = 0; i < WM; i++) {
                const int j = ls - 1 + i;                   // member j + 1 of the level
                const float s2 = i < ln ? S.sc[j] : NEG_INF;
                const unsigned bit = i < ln ? (1u << j) : 0u;
                ac[i & 3] |= m_gt(s2, sc) & bit;
                tc[i & 3] |= m_eq(s2, sc) & bit;
            }
            const unsigned ahd = (ac[0] | ac[1]) | (ac[2] | ac[3]);
            unsigned tie = (tc[0] | tc[1]) | (tc[2] | tc[3]);
            tie &= ~(1u << lane);
            const bool carried = has && lv < L0 && P.cnt[m] <= bud;
            const unsigned bcar = __ballot_sync(FULL, carried);
            fast = !__any_sync(FULL, has && (tie != 0u || !(sc > -5e8f)));
            int pos = -1;
            if (carried) pos = __popc(bcar & (lower | ahd));
            else if (has && lv == L0) pos = __popc(bcar) + __popc(ahd);
            if (fast) {
                if (lane < 16) P.ord[lane] = -1;
                __syncwarp();
                if (pos >= 0 && pos < WM) { P.ord[pos] = (signed char)m; P.ocnt[pos] = P.cnt[m]; }
                __syncwarp();
                mo = lane < 16 ? (int)P.ord[lane] : -1;
                mcnt = mo >= 0 ? P.ocnt[lane] : 0.0f;
            }
        }
        int lr = L0;
        if (!fast) {
        mo = lane == 0 ? 0 : -1;
        mcnt = lane == 0 ? P.cnt[0] : 0.0f;
        if (lane < PMAX) P.slot[lane] = -1;
        if (lane + 32 < PMAX) P.slot[lane + 32] = -1;
        __syncwarp();
        for (int d = 1; d <= L0; d++) {
            const bool active = mo >= 0;
            const bool carried = active && mcnt <= bud;
            const unsigned bc = __ballot_sync(FULL, carried);
            const int ncar = __popc(bc);
            if (__ballot_sync(FULL, active && !carried) == 0u && ncar <= wbeam) { lr = d - 1; break; }
            if (active) P.slot[mo] = (signed char)lane;
            __syncwarp();
            const int cs = P.lvs[d], cn = P.lvn[d];
            const int cm = cs + lane;
            int cc = 0x7fffffff;
            float sc = NEG_INF, ccnt = 0.0f;
            bool valid = false;
            if (lane < cn) {
                const int ps = P.slot[P.par[cm]];
                if (ps >= 0) { valid = true; cc = 2 * ps + (lane & 1); sc = S.sc[cm - 1]; ccnt = P.cnt[cm]; }
            }
            if (lane < 16) { P.rsc[lane] = sc; P.rcc[lane] = cc; }
            __syncwarp();
            int rr;
            {
                const float4* s4p = reinterpret_cast<const float4*>(P.rsc);
                const int4* c4p = reinterpret_cast<const int4*>(P.rcc);
                int rk[4] = {0, 0, 0, 0};                   // four short chains instead of one of sixteen
#pragma unroll
                for (int h = 0; h < 4; h++) {
                    const float4 s4 = s4p[h];
                    const int4 c4 = c4p[h];
                    rk[0] += ahead(s4.x, c4.x, sc, cc);
                    rk[1] += ahead(s4.y, c4.y, sc, cc);
                    rk[2] += ahead(s4.z, c4.z, sc, cc);
                    rk[3] += ahead(s4.w, c4.w, sc, cc);
                }
                rr = (rk[0] + rk[1]) + (rk[2] + rk[3]);
            }
            const int crank = ncar + rr, ncb = __popc(bc & below);
            const bool keep = valid && sc > -5e8f && crank < wbeam && crank < WM;
            const bool ckeep = carried && ncb < wbeam && ncb < WM;
            if (lane < 16) P.ord[lane] = -1;
            __syncwarp();
            if (ckeep) { P.ord[ncb] = (signed char)mo; P.ocnt[ncb] = mcnt; }
            if (keep) { P.ord[crank] = (signed char)cm; P.ocnt[crank] = ccnt; }
            __syncwarp();
            mo = lane < 16 ? (int)P.ord[lane] : -1;
            mcnt = mo >= 0 ? P.ocnt[lane] : 0.0f;
        }
        }
        // the frontier the loop would hold at level lr
        if (lane < WM) {
            FrontS& F0 = S.f[0];
            if (mo < 0) {
                F0.ref[lane] = -1;
            } else {
                const int ref = P.ref[mo];
                float a[6];
                if (ref >= 0) {
                    const int e = P.ent[mo];
                    const float4 a0 = S.ax[e][0], a1 = S.ax[e][1], a2 = S.ax[e][2];
                    a[0] = a0.w; a[1] = a1.x; a[2] = a1.y; a[3] = a1.z; a[4] = a1.w; a[5] = a2.x;
                } else {
                    const float* sp = S.spn[-ref - 2] + DH + 3;
#pragma unroll
                    for (int f = 0; f < 6; f++) a[f] = sp[f];
                }
                F0.ref[lane] = ref; F0.stg[lane] = P.eix[mo]; F0.cnt[lane] = a[0]; F0.chl[lane] = (int)(i64)a[1];
                F0.chr[lane] = (int)(i64)a[2]; F0.src[lane] = a[3]; F0.off[lane] = a[4]; F0.len[lane] = a[5];
            }
        }
        if (lane == 0) P.L0 = lr;
    }
    team_bar(bar_id);
    PSTAMP(4);
    // the first cells of the members that cannot expand: their key rows into L2 for the re-rank
    {
        const int lc = __shfl_sync(FULL, lcell, gl);
        if (lc >= 0) {
            const signed char* kp = qk_row + (i64)lc * SCELL * sqk_n;
#pragma unroll
            for (int i = 0; i < SCELL * DH / 512; i++) prefetch_l2(kp + (i64)(g * (SCELL * DH / 512) + i) * 128);
            if (g == 0) prefetch_l2(sk_row + lc);
        }
    }
    PSTAMP(5);
    return P.L0;
#undef PSTAMP
}
#endif

// one beam descent by a team of 4 warps (fsb_beam_pair_cuda's warp_query and warp_descend, bitwise): the frontier
// slots live in shared memory, child c of a level (frontier slot c / 2, side c & 1: warp_descend's compacted child
// order is this order) belongs to the 4 lanes 4 c .. 4 c + 3, lane g of them holding columns 8 g .. 8 g + 7 and
// DH/2 + 8 g .. of the query and of the child's mean; the q.mu halving sum is group_halve, each lane ranks its child
// against children 8 g .. 8 g + 7 and the group sums the counts. A level reads its children from shared memory (the
// chunk nodes staged by the previous level, the spine staged once) and stages the children of every child that can
// still expand with cp.async, which lands while the level is scored and ranked: no global round trip on the level's
// critical path. A child that cannot expand (it may end in the frontier) loads its first cell and, once the level is
// ranked, prefetches that cell's key rows (qk_row, sqk_n, sk_row) into L2 for the re-rank. Three team barriers per
// level (children read; scores; next frontier). q_out (the query, fp32) and cells_out (the WM * R cells) are the
// team's outputs
__device__ __forceinline__ void team_descend(const DecBeam B, TeamS& S, int* __restrict__ cells_out,
                                             float* __restrict__ q_out, const int tt, const int bar_id, const i64 r,
                                             const void* __restrict__ Q, const int q_bf16, const float rcq,
                                             const float qpos, const float lam, const float sig2, const float scale,
                                             const float cell_size, const signed char* __restrict__ qk_row,
                                             const i64 sqk_n, const unsigned short* __restrict__ sk_row,
                                             long long* lt_out, int* __restrict__ lev0_out
#if DEC_V3
                                             , PrefS& P
#endif
                                             ) {
    const int c = tt >> 2, g = tt & 3;
    const int lane = tt & 31, gl = lane & ~3;
    const i64 tidx = B.TIDX[r];
    const int root = (int)B.ROOT[r];
    const float bud = (float)B.BUDGET[r];
    const i64 wbeam = B.WBEAM[r];
#if DEC_V3
    const i64 hgt = B.HGT[r];
#endif
    const unsigned short* cmub = B.CHUNK + tidx * B.Ntot * DH;
    const float* cax = B.CHUNK_AX + tidx * B.Ntot * NA;
    const float* sbase = B.SPINE + r * B.Sp * NC;
    const i64 Ltot = B.Ltot, Mm = B.Mm;
    const i64* leafb = B.LEAF + tidx * Ltot;
    const i64* admb = B.ADM + tidx * Mm;
    const int nsp = (int)min(B.Sp, (i64)SPMAX);
    // the spine rows into shared memory and the query into registers, in one round of loads
#if DEC_V3
    // the query's load first (the norm below waits for it), then the spine rows and the cached plan, all in flight
    uint4 qua = make_uint4(0u, 0u, 0u, 0u), qub = make_uint4(0u, 0u, 0u, 0u);
    if (q_bf16) {
        const uint4* q16 = reinterpret_cast<const uint4*>(reinterpret_cast<const unsigned short*>(Q) + r * DH);
        qua = q16[g]; qub = q16[g + DH / 16];
    }
    for (int i = tt; i < nsp * NC; i += 128) cp4(&S.spn[0][0] + i, sbase + i);
    if (B.PLAN != nullptr && tt < PLAN_CHUNKS)
        cp16(reinterpret_cast<char*>(&P) + 16 * tt, reinterpret_cast<const char*>(B.PLAN) + r * PLAN_STRIDE + 16 * tt);
    cp_commit();
    float qa[8], qb[8];
    if (q_bf16) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned wa = word_of(qua, i >> 1), wb = word_of(qub, i >> 1);
            qa[i] = (i & 1) ? bf_hi(wa) : bf_lo(wa);
            qb[i] = (i & 1) ? bf_hi(wb) : bf_lo(wb);
        }
    } else {
        const float* qf = reinterpret_cast<const float*>(Q) + r * DH;
#pragma unroll
        for (int i = 0; i < 8; i++) { qa[i] = qf[8 * g + i]; qb[i] = qf[DH / 2 + 8 * g + i]; }
    }
#else
    for (int i = tt; i < nsp * NC; i += 128) (&S.spn[0][0])[i] = sbase[i];
    float qa[8], qb[8];
    if (q_bf16) {
        const uint4* q16 = reinterpret_cast<const uint4*>(reinterpret_cast<const unsigned short*>(Q) + r * DH);
        const uint4 ua = q16[g], ub = q16[g + DH / 16];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned wa = word_of(ua, i >> 1), wb = word_of(ub, i >> 1);
            qa[i] = (i & 1) ? bf_hi(wa) : bf_lo(wa);
            qb[i] = (i & 1) ? bf_hi(wb) : bf_lo(wb);
        }
    } else {
        const float* qf = reinterpret_cast<const float*>(Q) + r * DH;
#pragma unroll
        for (int i = 0; i < 8; i++) { qa[i] = qf[8 * g + i]; qb[i] = qf[DH / 2 + 8 * g + i]; }
    }
#endif
    if (c == 0) {
#pragma unroll
        for (int i = 0; i < 8; i++) { q_out[8 * g + i] = qa[i]; q_out[DH / 2 + 8 * g + i] = qb[i]; }
    }
    float x[8];
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = f_add(f_mul(qa[i], qa[i]), f_mul(qb[i], qb[i]));
    const float qn = f_mul(f_sqrt_ftz(group_halve(x, g)), rcq);
    // the root's aux row (the spine stage or global memory), then its children staged by group 0
    float rx[NA];
#pragma unroll
    for (int f = 0; f < NA; f++) rx[f] = 0.0f;
#if !DEC_V3
    if (root != -1 && c == 0) {
        const float* p = (root <= -2 && -root - 2 < SPMAX) ? nullptr : dec_aux(cax, sbase, root);
#pragma unroll
        for (int f = 3; f < NA; f++) rx[f] = p != nullptr ? p[f] : 0.0f;
    }
#endif
    if (tt < WM) S.f[0].ref[tt] = -1;
#if DEC_V3
    cp_wait_all();
#endif
    team_bar(bar_id);                                     // the spine stage
    int lev0 = 0;
#if DEC_V3
    lev0 = team_prefix(B, S, P, tt, bar_id, r, root, hgt, bud, (int)wbeam, nsp, cmub, cax, leafb, admb, Ltot, Mm, qa, qb,
                       qn, qpos, lam, sig2, scale, cell_size, qk_row, sqk_n, sk_row,
                       lt_out != nullptr ? lt_out - 7 : nullptr);
#endif
    if (lev0_out != nullptr && tt == 0) *lev0_out = lev0;              // test hook: the levels the prefix covered
    if (lev0 == 0) {
#if DEC_V3
        if (root != -1 && c == 0 && !(root <= -2 && -root - 2 < SPMAX)) {
            const float* p = dec_aux(cax, sbase, root);
#pragma unroll
            for (int f = 3; f < NA; f++) rx[f] = p[f];
        }
#endif
        if (root != -1 && c == 0) {
            if (root <= -2 && -root - 2 < SPMAX) {
#pragma unroll
                for (int f = 3; f < NA; f++) rx[f] = S.spn[-root - 2][DH + f];
            }
            if (rx[3] > bud) {
#pragma unroll
                for (int side = 0; side < 2; side++) {
                    const int n = (int)(i64)(side ? rx[5] : rx[4]);
                    if (n >= 0) stage_node(S, side, n, g, cmub, cax);
                }
            }
            if (g == 0) {
                S.f[0].ref[0] = root; S.f[0].stg[0] = 0; S.f[0].cnt[0] = rx[3]; S.f[0].chl[0] = (int)(i64)rx[4];
                S.f[0].chr[0] = (int)(i64)rx[5]; S.f[0].src[0] = rx[6]; S.f[0].off[0] = rx[7]; S.f[0].len[0] = rx[8];
            }
        }
        cp_commit();
        cp_wait_all();
        team_bar(bar_id);
    }
    int cur = 0;
    for (int lev = lev0; lev < B.KMAXP1; lev++) {
        LSTAMP(0);
        FrontS& Fc = S.f[cur];
        FrontS& Fn = S.f[cur ^ 1];
        // the frontier's carried / active masks (every warp of the team derives the same ones; lane w: slot w)
        const int fr = lane < WM ? Fc.ref[lane] : -1;
        const bool active = fr != -1;
        const bool carried = active && Fc.cnt[lane] <= bud;
        const unsigned car = __ballot_sync(FULL, carried);
        const int ncar = __popc(car);
        // every active slot carried and kept: the frontier is a fixed point and no child is visited
        if (__ballot_sync(FULL, active && !carried) == 0u && ncar <= wbeam) break;
        const int slot = c >> 1;
        int child = -1, e = 0;
        if (slot < WM && Fc.ref[slot] != -1 && Fc.cnt[slot] > bud) {
            child = (c & 1) ? Fc.chr[slot] : Fc.chl[slot];
            e = 2 * Fc.stg[slot] + (c & 1);
        }
        if (tt < WM) Fn.ref[tt] = -1;                        // the next frontier starts empty (filled after the barrier)
        // the child from shared memory: a staged chunk node or a spine row (deeper spines: global memory)
        uint4 ma = make_uint4(0u, 0u, 0u, 0u), mb = make_uint4(0u, 0u, 0u, 0u);
        float ax[NA];
#pragma unroll
        for (int f = 0; f < NA; f++) ax[f] = (f == 4 || f == 5) ? -1.0f : 0.0f;
        const float* sp32 = nullptr;
        if (child >= 0) {
            ma = S.mu[e][g];
            mb = S.mu[e][g + DH / 16];
            const float4 a0 = S.ax[e][0], a1 = S.ax[e][1], a2 = S.ax[e][2];
            ax[0] = a0.x; ax[1] = a0.y; ax[2] = a0.z; ax[3] = a0.w;
            ax[4] = a1.x; ax[5] = a1.y; ax[6] = a1.z; ax[7] = a1.w; ax[8] = a2.x;
        } else if (child <= -2) {
            const int k = -child - 2;
            sp32 = k < SPMAX ? S.spn[k] : sbase + (i64)k * NC;
#pragma unroll
            for (int f = 0; f < NA; f++) ax[f] = sp32[DH + f];
        }
        const float rad = ax[0], amax = ax[1], pmin = ax[2], cnt = ax[3], chl = ax[4], chr = ax[5];
        const float src = ax[6], off = ax[7], len = ax[8];
        // a child that cannot expand may end in the frontier: its first cell now, its key rows into L2 once the level
        // is ranked (the re-rank reads them)
        int lcell = -1;
        if (g == 0 && child != -1 && cnt <= bud && src > 0.0f && len > 0.0f) {
            const i64 ps = (i64)off;
            lcell = (int)(src == 1.0f ? leafb[min(max(ps, (i64)0), Ltot - 1)] : admb[min(max(ps, (i64)0), Mm - 1)]);
        }
        float x[8];
        if (child >= 0) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const unsigned wa = word_of(ma, i >> 1), wb = word_of(mb, i >> 1);
                x[i] = f_add(f_mul(qa[i], (i & 1) ? bf_hi(wa) : bf_lo(wa)), f_mul(qb[i], (i & 1) ? bf_hi(wb) : bf_lo(wb)));
            }
        } else if (child <= -2) {
#pragma unroll
            for (int i = 0; i < 8; i++) x[i] = f_add(f_mul(qa[i], sp32[8 * g + i]), f_mul(qb[i], sp32[DH / 2 + 8 * g + i]));
        } else {
#pragma unroll
            for (int i = 0; i < 8; i++) x[i] = 0.0f;
        }
        LSTAMP(1);
        team_bar(bar_id);                                     // every child read: its stage entries can be reused
        // the next level's candidates (the children of a child that can still expand) into the stage
        if (child != -1 && cnt > bud) {
#pragma unroll
            for (int side = 0; side < 2; side++) {
                const int n = (int)(i64)(side ? chr : chl);
                if (n >= 0) stage_node(S, 2 * c + side, n, g, cmub, cax);
            }
        }
        cp_commit();
        // q.mu in _halve_rows' order: the first halving step in the lane, the rest across the group
        const float v = group_halve(x, g);
        LSTAMP(2);
        float content = f_mul(v, scale);
#if BOUND
        content = f_add(content, f_mul(f_mul(qn, rad), scale));
#endif
        const float pmin_pos = f_mul(f_add(pmin, 0.5f), cell_size);
        const float dx = __int_as_float(__float_as_int(f_sub(qpos, pmin_pos)) & 0x7fffffff);
        const float score = f_add(content, f_mul(f_mul(lam, amax), erf_tl(f_div_full(dx, sig2))));
        const float sc = child != -1 ? score : NEG_INF;
        if (g == 0) S.sc[c] = sc;
        team_bar(bar_id);
        LSTAMP(3);
        // carried slots keep their order ahead of every child; children rank by (score desc, slot asc) over the
        // valid children (idle slots hold -inf and never count; +0 == -0 ties by slot), as in warp_descend
        int rr = 0;
        {
            const float4* s4p = reinterpret_cast<const float4*>(S.sc) + 2 * g;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const float4 s4 = s4p[h];
                const int m0 = 8 * g + 4 * h;
#if DEC_V3
                rr += (ahead(s4.x, m0, sc, c) + ahead(s4.y, m0 + 1, sc, c)) + (ahead(s4.z, m0 + 2, sc, c)
                                                                            + ahead(s4.w, m0 + 3, sc, c));
#else
                rr += (int)(s4.x > sc) + ((int)(s4.x == sc) & (int)(m0 < c));
                rr += (int)(s4.y > sc) + ((int)(s4.y == sc) & (int)(m0 + 1 < c));
                rr += (int)(s4.z > sc) + ((int)(s4.z == sc) & (int)(m0 + 2 < c));
                rr += (int)(s4.w > sc) + ((int)(s4.w == sc) & (int)(m0 + 3 < c));
#endif
            }
            rr += __shfl_xor_sync(FULL, rr, 1);
            rr += __shfl_xor_sync(FULL, rr, 2);
        }
        const int crank = ncar + rr;
        LSTAMP(4);
        {
            const int lc = __shfl_sync(FULL, lcell, gl);
            if (lc >= 0) {
                const signed char* kp = qk_row + (i64)lc * SCELL * sqk_n;
#pragma unroll
                for (int i = 0; i < SCELL * DH / 512; i++) prefetch_l2(kp + (i64)(g * (SCELL * DH / 512) + i) * 128);
                if (g == 0) prefetch_l2(sk_row + lc);
            }
        }
        if (g == 0 && child != -1 && sc > -5e8f && crank < wbeam && crank < WM) {
            Fn.ref[crank] = child; Fn.stg[crank] = c; Fn.cnt[crank] = cnt; Fn.chl[crank] = (int)(i64)chl;
            Fn.chr[crank] = (int)(i64)chr;
            Fn.src[crank] = src; Fn.off[crank] = off; Fn.len[crank] = len;
        }
        if (tt < WM && carried) {
            const int ncb = __popc(car & ((1u << tt) - 1u));
            if (ncb < wbeam && ncb < WM) {
                Fn.ref[ncb] = fr; Fn.stg[ncb] = Fc.stg[tt]; Fn.cnt[ncb] = Fc.cnt[tt]; Fn.chl[ncb] = Fc.chl[tt];
                Fn.chr[ncb] = Fc.chr[tt]; Fn.src[ncb] = Fc.src[tt]; Fn.off[ncb] = Fc.off[tt]; Fn.len[ncb] = Fc.len[tt];
            }
        }
        cp_wait_all();
        team_bar(bar_id);
        LSTAMP(5);
        cur ^= 1;
    }
    // cells epilogue from the carried src / off / len: the leaf table (src == 1) or the admission table
    const FrontS& Ff = S.f[cur];
    for (int idx = tt; idx < WM * R; idx += 128) {
        const int w = idx / R, tr = idx - w * R;
        int cell = -1;
        if (Ff.ref[w] != -1) {
            const float src = Ff.src[w], ln = Ff.len[w];
            if (src > 0.0f && (float)tr < ln) {
                const i64 pos = (i64)Ff.off[w] + tr;
                cell = (int)(src == 1.0f ? leafb[min(max(pos, (i64)0), Ltot - 1)] : admb[min(max(pos, (i64)0), Mm - 1)]);
            }
        }
        cells_out[idx] = cell;
    }
}

// the halving sum (_halve_rows' order) of 64 columns in one lane, x[j] = column j + column j + DH/2
template <int H>
__device__ __forceinline__ void hstep(float (&x)[DH / 2]) {
#pragma unroll
    for (int j = 0; j < H; j++) x[j] = f_add(x[j], x[j + H]);
}
__device__ __forceinline__ float halve_x(float (&x)[DH / 2]) {
    if constexpr (DH / 4 >= 32) hstep<32>(x);
    if constexpr (DH / 4 >= 16) hstep<16>(x);
    if constexpr (DH / 4 >= 8) hstep<8>(x);
    hstep<4>(x);
    hstep<2>(x);
    hstep<1>(x);
    return x[0];
}

// u scores of candidates c_begin + wrank + nw i (one warp per candidate, lane j its token j): _fsb_upar_int8_kernel's
// token keys (int8 * cell scale with one rounding below n_q, the bf16 hot rows above), q.k by the halving order, lse and
// the lse_pos prior; the raw q.k of every token stays in s_raw for the readout. A warp takes G candidates at a time:
// their key rows load in one round (their value rows go into L2 for the readout), and each step of the computation
// runs for all G before the next, so their dependent shuffle / exp / log chains overlap
template <int G>
__device__ __forceinline__ void rerank_range(const int c_begin, const int c_end, const int wrank, const int nw,
                                             const int lane, const int* s_cell, float* s_u, float (*s_raw)[32],
                                             const float* s_q, const signed char* qk_row,
                                             const unsigned short* sk_row, const unsigned short* hk_row,
                                             const signed char* qv_row, const i64 sqk_n, const i64 shk_n,
                                             const i64 sqv_n, const i64 nq, const int Mc, const float* cella_row,
                                             const i64 ca1, const float qp, const float lam_r, const float sig2,
                                             const float scale) {
    const float4* q4 = reinterpret_cast<const float4*>(s_q);
    for (int c0 = c_begin + wrank; c0 < c_end; c0 += G * nw) {
        uint4 kr[G][DH / 16];
        float ksc[G], raw[G], m[G], res[G];
        int cells[G];
#pragma unroll
        for (int i = 0; i < G; i++) {
            const int c = c0 + nw * i;
            cells[i] = c < c_end ? s_cell[c] : -1;
            ksc[i] = 0.0f;
            if (cells[i] >= 0) {
                const i64 tok = (i64)cells[i] * SCELL + lane;
                if (tok < nq) {
                    const uint4* p = reinterpret_cast<const uint4*>(qk_row + tok * sqk_n);
#pragma unroll
                    for (int k = 0; k < DH / 16; k++) kr[i][k] = __ldg(p + k);
                    ksc[i] = bf2f(sk_row[min(cells[i], Mc - 1)]);
                }
                prefetch_l2(qv_row + tok * sqv_n);
            }
        }
#pragma unroll
        for (int i = 0; i < G; i++) {
            raw[i] = 0.0f;
            if (cells[i] < 0) continue;
            const i64 tok = (i64)cells[i] * SCELL + lane;
            float x[DH / 2];
            if (tok < nq) {
                const float c2 = f_mul(-12582912.0f, ksc[i]);
#pragma unroll
                for (int j4 = 0; j4 < DH / 8; j4++) {
                    const float4 qa = q4[j4], qb = q4[j4 + DH / 8];
                    const float qa_[4] = {qa.x, qa.y, qa.z, qa.w}, qb_[4] = {qb.x, qb.y, qb.z, qb.w};
#pragma unroll
                    for (int e4 = 0; e4 < 4; e4++) {
                        const int ja = 4 * j4 + e4, jb = ja + DH / 2;
                        const float ka = deq8(word_of(kr[i][ja >> 4], (ja >> 2) & 3), ja & 3, ksc[i], c2);
                        const float kb = deq8(word_of(kr[i][jb >> 4], (jb >> 2) & 3), jb & 3, ksc[i], c2);
                        x[ja] = f_add(f_mul(ka, qa_[e4]), f_mul(kb, qb_[e4]));
                    }
                }
            } else {
                const unsigned short* hp = hk_row + (tok - nq) * shk_n;
#pragma unroll
                for (int j = 0; j < DH / 2; j++)
                    x[j] = f_add(f_mul(bf2f(hp[j]), s_q[j]), f_mul(bf2f(hp[j + DH / 2]), s_q[j + DH / 2]));
            }
            raw[i] = halve_x(x);
            s_raw[c0 + nw * i][lane] = raw[i];
        }
#pragma unroll
        for (int i = 0; i < G; i++) m[i] = f_mul(raw[i], scale);
        float sc[G];
#pragma unroll
        for (int i = 0; i < G; i++) sc[i] = m[i];
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1)
#pragma unroll
            for (int i = 0; i < G; i++) m[i] = f_max(m[i], __shfl_xor_sync(FULL, m[i], o));
#if LSE_ON
        float e[G];
#pragma unroll
        for (int i = 0; i < G; i++) e[i] = exp_tl(f_sub(sc[i], m[i]));
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1)
#pragma unroll
            for (int i = 0; i < G; i++) e[i] = f_add(e[i], __shfl_xor_sync(FULL, e[i], o));
#pragma unroll
        for (int i = 0; i < G; i++) res[i] = f_add(m[i], log_tl(e[i]));
#else
#pragma unroll
        for (int i = 0; i < G; i++) res[i] = m[i];
#endif
#if POS_ON
#pragma unroll
        for (int i = 0; i < G; i++) {
            const int cell = max(cells[i], 0);
            const float xx = f_div_rn(f_abs(f_sub(qp, f_mul(f_add((float)cell, 0.5f), (float)SCELL))), sig2);
            const float a = cella_row[(i64)min(cell, Mc - 1) * ca1];
            res[i] = f_add(res[i], f_mul(f_mul(lam_r, a), erf_torch(xx)));
        }
#endif
        if (lane == 0) {
#pragma unroll
            for (int i = 0; i < G; i++) {
                const int c = c0 + nw * i;
                if (c < c_end) s_u[c] = cells[i] >= 0 ? res[i] : NEG_INF;
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(NWARP * 32, 1)
fsb_decode_fused(const DecBeam BA, const DecBeam BB, const void* __restrict__ Q, const int q_bf16,
                 const float* __restrict__ RC, const float rc_s, const float* __restrict__ QPOS, const i64 qpos_s,
                 const float* __restrict__ LAM, const float* __restrict__ SIG, const float scale, const float cell_size,
                 const signed char* __restrict__ QK, const unsigned short* __restrict__ SK,
                 const unsigned short* __restrict__ HOTK, const i64* __restrict__ NQ,
                 const float* __restrict__ CELLA, const i64 ca0, const i64 ca1, const int Mc,
                 const i64 sqk_b, const i64 sqk_h, const i64 sqk_n, const i64 ssk_b, const i64 ssk_h,
                 const i64 shk_b, const i64 shk_h, const i64 shk_n,
                 const signed char* __restrict__ QV, const unsigned short* __restrict__ SV,
                 const unsigned short* __restrict__ HOTV,
                 const i64 sqv_b, const i64 sqv_h, const i64 sqv_n, const i64 ssv_b, const i64 ssv_h,
                 const i64 shv_b, const i64 shv_h, const i64 shv_n,
                 const i64* __restrict__ POS, const void* __restrict__ LNW, const unsigned short* __restrict__ SR,
                 const i64 srb, const i64 srh, const void* __restrict__ LRS, unsigned short* __restrict__ OUT,
                 const i64 sob, const i64 soh, const float eps, const int Hh,
                 i64* __restrict__ SEL_OUT, signed char* __restrict__ PROV_OUT, i64* __restrict__ CELLS_OUT,
                 int* __restrict__ LEV0_OUT, long long* __restrict__ TIMES) {
    __shared__ __align__(16) float s_q[2][DH];
    __shared__ TeamS s_team[2];
#if DEC_V3
    __shared__ PrefS s_pref[2];
#endif
    __shared__ int s_cell[C_ALL];
    __shared__ float s_u[C_ALL];
    __shared__ float s_raw[C_ALL][32];
    __shared__ int s_slot[KK];
    __shared__ int s_sel[KK];
    __shared__ float s_w[KK][32];
    __shared__ float s_wm[KK], s_wden[KK];
    __shared__ int s_wok[KK];
    __shared__ float s_part[NWARP][DH];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const i64 r = blockIdx.x;
    if (wid == 0) TSTAMP(0);
    const float qp = QPOS[r * qpos_s];
    const float lam_r = LAM[r];
    const float sig2 = f_mul(2.0f, SIG[r]);
#if RC_T
    const float rcq = RC[r];
#else
    const float rcq = rc_s;
#endif
    // the re-rank's row scalars, in flight across the descents
    const i64 seq = BA.TIDX[r];
#if DEC_V3
    const i64 bi = (i64)((int)seq / Hh), hi = seq - bi * Hh;      // rows < 2^31: one 32-bit division
#else
    const i64 bi = seq / Hh, hi = seq - bi * Hh;
#endif
    const i64 nq = *NQ;
    const i64 posv = *POS;
    const signed char* qk_row = QK + bi * sqk_b + hi * sqk_h;
    const unsigned short* sk_row = SK + bi * ssk_b + hi * ssk_h;
    const unsigned short* hk_row = HOTK + bi * shk_b + hi * shk_h;
    const signed char* qv_row = QV + bi * sqv_b + hi * sqv_h;
    const float* cella_row = CELLA + seq * ca0;

    // ---- the main descent (team 0: warps 0..3) and the union descent (team 1: warps 4..7); the re-rank warps
    // (8 .. NWARP - 1) take the union beam's cells as soon as team 1 has them (no dependence on the main cells: the
    // duplicates drop out at the selection), overlapping team 0's longer descent; then every warp re-ranks the main
    // beam's cells ----
    const int team = wid >> 2, tt = tid & 127;
    constexpr int NRW = NWARP - 8;                            // re-rank warps
    if (wid == 0) TSTAMP(1);
    if (tid >= 256 && tid < 256 + KK) s_slot[tid - 256] = -1;
    if (team == 0) {
        team_descend(BA, s_team[0], s_cell, s_q[0], tt, 1, r, Q, q_bf16, rcq, qp, lam_r, sig2, scale, cell_size,
                     qk_row, sqk_n, sk_row, DEC_TIMING && TIMES != nullptr ? TIMES + r * 256 + 16 : nullptr,
                     LEV0_OUT != nullptr ? LEV0_OUT + 2 * r : nullptr
#if DEC_V3
                     , s_pref[0]
#endif
                     );
        if (tt == 0) TSTAMP(2);
    } else if (team == 1) {
        team_descend(BB, s_team[1], s_cell + CA, s_q[1], tt, 2, r, Q, q_bf16, rcq, qp, lam_r, sig2, scale, cell_size,
                     qk_row, sqk_n, sk_row, nullptr, LEV0_OUT != nullptr ? LEV0_OUT + 2 * r + 1 : nullptr
#if DEC_V3
                     , s_pref[1]
#endif
                     );
        if (tt == 0) TSTAMP(3);
        asm volatile("bar.arrive 3, %0;" :: "r"((NRW + 4) * 32) : "memory");
    } else {
        asm volatile("bar.sync 3, %0;" :: "r"((NRW + 4) * 32) : "memory");
        rerank_range<(CA + NRW - 1) / NRW < 4 ? (CA + NRW - 1) / NRW : 4>(
            CA, C_ALL, wid - 8, NRW, lane, s_cell, s_u, s_raw, s_q[1], qk_row, sk_row, hk_row, qv_row, sqk_n, shk_n,
            sqv_n, nq, Mc, cella_row, ca1, qp, lam_r, sig2, scale);
    }
    __syncthreads();
    if (wid == 0) TSTAMP(4);
    if (CELLS_OUT != nullptr && tid < C_ALL) CELLS_OUT[r * C_ALL + tid] = s_cell[tid];   // test hook: both beams' cells
    rerank_range<(CA + NWARP - 1) / NWARP < 4 ? (CA + NWARP - 1) / NWARP : 4>(
        0, CA, wid, NWARP, lane, s_cell, s_u, s_raw, s_q[1], qk_row, sk_row, hk_row, qv_row, sqk_n, shk_n, sqv_n, nq,
        Mc, cella_row, ca1, qp, lam_r, sig2, scale);
    // selpv's union deduplication: a second-beam cell already among the first beam's drops out (u = -inf)
    for (int c = CA + tid; c < C_ALL; c += NWARP * 32) {
        const int cell = s_cell[c];
        if (cell >= 0) {
            bool dup = false;
#pragma unroll 8
            for (int j = 0; j < CA; j++) dup |= (s_cell[j] == cell);
            if (dup) s_u[c] = NEG_INF;
        }
    }
    __syncthreads();
    if (wid == 0) TSTAMP(5);

    // ---- stable descending top-KK of u (ties by slot), the -inf mask and the second-beam provenance ----
    if (wid == 0) {
#pragma unroll
        for (int i = 0; i < (C_ALL + 31) / 32; i++) {
            const int c = lane + 32 * i;
            if (c < C_ALL) {
                const float uc = s_u[c];
                int rk[4] = {0, 0, 0, 0};
#pragma unroll
                for (int j = 0; j < C_ALL; j++) {
                    const float uj = s_u[j];
                    rk[j & 3] += (int)(uj > uc) + ((int)(uj == uc) & (int)(j < c));
                }
                const int rank = (rk[0] + rk[1]) + (rk[2] + rk[3]);
                if (rank < KK) s_slot[rank] = c;
            }
        }
        __syncwarp();
        if (lane < KK) {
            const int c = s_slot[lane];
            const int res = (c >= 0 && s_u[c] > NEG_INF) ? s_cell[c] : -1;
            s_sel[lane] = res;
            SEL_OUT[r * KK + lane] = (i64)res;
            PROV_OUT[r * KK + lane] = (signed char)(res < 0 ? -1 : (c >= CA ? PROV_BIT : 0));
        }
    }
    __syncthreads();
    if (wid == 0) TSTAMP(6);

    // ---- readout. Each warp's value rows first (warp w: tokens j0 .. j0 + TPW - 1 of cell kv, lane l: dims 2l,
    // 2l + 1, one 64-byte row per load), in flight across the softmax over the selected cells' tokens (warp k: cell
    // k, lane j: its token j) ----
    const int kv = (wid * TPW) / SCELL, j0 = wid * TPW - kv * SCELL;
    const int vcell = s_sel[kv];
    unsigned vw[TPW];
    float vsc = 0.0f;
    {
        const unsigned short* hv_row = HOTV + bi * shv_b + hi * shv_h;
#pragma unroll
        for (int t = 0; t < TPW; t++) {
            const i64 tok = (i64)max(vcell, 0) * SCELL + j0 + t;
            vw[t] = 0u;
            if (vcell >= 0 && tok < posv)
                vw[t] = tok < nq ? (unsigned)*reinterpret_cast<const unsigned short*>(qv_row + tok * sqv_n + 2 * lane)
                                 : *reinterpret_cast<const unsigned*>(hv_row + (tok - nq) * shv_n + 2 * lane);
        }
        if (vcell >= 0) vsc = bf2f(SV[bi * ssv_b + hi * ssv_h + vcell]);
    }
    float scr = -1e9f, e = 0.0f;
    int ok = 0;
    if (wid < KK) {
        const int slot = s_slot[wid], cell = s_sel[wid];
        const i64 tok = (i64)max(cell, 0) * SCELL + lane;
        ok = cell >= 0 && tok < posv;
        if (ok) scr = f_mul(bf_round(s_raw[slot][lane]), scale);     // the score einsum's store, then * scale
        float m = scr;
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) m = f_max(m, __shfl_xor_sync(FULL, m, o));
        const int nok = __popc(__ballot_sync(FULL, ok));
        if (lane == 0) { s_wm[wid] = m; s_wok[wid] = nok; }
    }
    __syncthreads();
    if (wid < KK) {
        float m_all = s_wm[0];
#pragma unroll
        for (int k = 1; k < KK; k++) m_all = f_max(m_all, s_wm[k]);
        e = exp_tl(f_sub(scr, m_all));
        float es = e;
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) es = f_add(es, __shfl_xor_sync(FULL, es, o));
        if (lane == 0) s_wden[wid] = es;
    }
    __syncthreads();
    if (wid < KK) {
        float den = s_wden[0];
#pragma unroll
        for (int k = 1; k < KK; k++) den = f_add(den, s_wden[k]);
        s_w[wid][lane] = ok ? bf_round(f_div_full(e, den)) : 0.0f;   // softmax stored, cast to the value dtype
    }
    __syncthreads();
    // ---- the value sum over the warp's tokens ----
    {
        float acc0 = 0.0f, acc1 = 0.0f;
        if (vcell >= 0) {
            const float c2 = f_mul(-12582912.0f, vsc);
#pragma unroll
            for (int t = 0; t < TPW; t++) {
                const i64 tok = (i64)vcell * SCELL + j0 + t;
                if (tok < posv) {
                    const float w = s_w[kv][j0 + t];
                    const float v0 = tok < nq ? deq8(vw[t], 0, vsc, c2) : bf_lo(vw[t]);
                    const float v1 = tok < nq ? deq8(vw[t], 1, vsc, c2) : bf_hi(vw[t]);
                    acc0 = f_add(acc0, f_mul(w, v0));
                    acc1 = f_add(acc1, f_mul(w, v1));
                }
            }
        }
        s_part[wid][2 * lane] = acc0;
        s_part[wid][2 * lane + 1] = acc1;
    }
    __syncthreads();
    if (wid == 0) TSTAMP(7);
    // ---- lr_norm in torch's order and dtypes, the window combine (warp 0, lane l: dims l, l + 32, ...) ----
    if (wid == 0) {
        int nok = 0;
#pragma unroll
        for (int k = 0; k < KK; k++) nok += s_wok[k];
        float xf[DH / 32];
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < DH / 32; i++) {
            const int d = lane + 32 * i;
            float acc = s_part[0][d];
#pragma unroll
            for (int w = 1; w < NWARP; w++) acc = f_add(acc, s_part[w][d]);
            xf[i] = nok > 0 ? bf_round(acc) : 0.0f;                     // the readout's store; empty row 0
            ss = f_add(ss, bf_round(f_mul(xf[i], xf[i])));
        }
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) ss = f_add(ss, __shfl_xor_sync(FULL, ss, o));
        const float ms = bf_round(f_mul(ss, 1.0f / DH));
        const float rms = bf_round(f_rsqrt_ftz(bf_round(f_add(ms, eps))));
#pragma unroll
        for (int i = 0; i < DH / 32; i++) {
            const int d = lane + 32 * i;
#if W_BF16
            const float wq = bf2f(reinterpret_cast<const unsigned short*>(LNW)[d]);
            float y = bf_round(f_mul(bf_round(f_mul(xf[i], rms)), wq));
#else
            const float wq = reinterpret_cast<const float*>(LNW)[d];
            float y = f_mul(bf_round(f_mul(xf[i], rms)), wq);
#endif
#if HAS_SR
            const float sr = bf2f(SR[bi * srb + hi * srh + d]);
#if LRS_BF16
            const float lrs = bf2f(reinterpret_cast<const unsigned short*>(LRS)[hi]);
#else
            const float lrs = reinterpret_cast<const float*>(LRS)[hi];
#endif
#if LRS_BF16 && W_BF16
            const float scaled = bf_round(f_mul(lrs, y));
#else
            const float scaled = f_mul(lrs, y);
#endif
            y = f_add(sr, scaled);
#endif
            OUT[bi * sob + hi * soh + d] = f2bf(y);
        }
        TSTAMP(8);
    }
}

#include "sha3.hpp"
#include <string.h>

static const uint64_t keccak_round_constants[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

static uint64_t rotl64(uint64_t x, int i) {
    return (x << i) | (x >> (64 - i));
}

static void keccak_f(uint8_t *state) {
    uint64_t *s = (uint64_t *)state;
    uint64_t A[5][5], B[5][5], C[5], D[5];
    int i, j, r;

    for (i = 0; i < 5; ++i) for (j = 0; j < 5; ++j) A[i][j] = s[5 * j + i];

    for (r = 0; r < 24; ++r) {
        for (i = 0; i < 5; ++i) C[i] = A[i][0] ^ A[i][1] ^ A[i][2] ^ A[i][3] ^ A[i][4];
        for (i = 0; i < 5; ++i) D[i] = C[(i + 4) % 5] ^ rotl64(C[(i + 1) % 5], 1);
        for (i = 0; i < 5; ++i) for (j = 0; j < 5; ++j) A[i][j] ^= D[i];

        B[0][0] = A[0][0];
        B[1][3] = rotl64(A[0][1], 36); B[2][1] = rotl64(A[0][2], 3); B[3][4] = rotl64(A[0][3], 105); B[4][2] = rotl64(A[0][4], 210);
        B[0][2] = rotl64(A[1][0], 1); B[1][0] = rotl64(A[1][1], 300); B[2][3] = rotl64(A[1][2], 10); B[3][1] = rotl64(A[1][3], 45); B[4][4] = rotl64(A[1][4], 66);
        B[0][4] = rotl64(A[2][0], 190); B[1][2] = rotl64(A[2][1], 6); B[2][0] = rotl64(A[2][2], 171); B[3][3] = rotl64(A[2][3], 15); B[4][1] = rotl64(A[2][4], 253);
        B[0][1] = rotl64(A[3][0], 28); B[1][4] = rotl64(A[3][1], 55); B[2][2] = rotl64(A[3][2], 25); B[3][0] = rotl64(A[3][3], 21); B[4][3] = rotl64(A[3][4], 120);
        B[0][3] = rotl64(A[4][0], 91); B[1][1] = rotl64(A[4][1], 276); B[2][4] = rotl64(A[4][2], 231); B[3][2] = rotl64(A[4][3], 136); B[4][0] = rotl64(A[4][4], 78);

        for (i = 0; i < 5; ++i) for (j = 0; j < 5; ++j) A[i][j] = B[i][j] ^ ((~B[(i + 1) % 5][j]) & B[(i + 2) % 5][j]);
        A[0][0] ^= keccak_round_constants[r];
    }
    for (i = 0; i < 5; ++i) for (j = 0; j < 5; ++j) s[5 * j + i] = A[i][j];
}

void shake128_init(sha3_context *ctx) {
    memset(ctx->state, 0, SHA3_STATE_SIZE);
    ctx->rate = 168; // 1344 bits
    ctx->pos = 0;
}

void shake128_update(sha3_context *ctx, const void *data, size_t len) {
    const uint8_t *d = (const uint8_t*)data;
    for (size_t i = 0; i < len; ++i) {
        ctx->state[ctx->pos++] ^= d[i];
        if (ctx->pos == ctx->rate) {
            keccak_f(ctx->state);
            ctx->pos = 0;
        }
    }
}

void shake128_xof(sha3_context *ctx) {
    ctx->state[ctx->pos] ^= 0x1F;
    ctx->state[ctx->rate - 1] ^= 0x80;
    keccak_f(ctx->state);
    ctx->pos = 0;
}

void shake128_out(sha3_context *ctx, void *out, size_t len) {
    uint8_t *o = (uint8_t*)out;
    for (size_t i = 0; i < len; ++i) {
        if (ctx->pos == ctx->rate) {
            keccak_f(ctx->state);
            ctx->pos = 0;
        }
        o[i] = ctx->state[ctx->pos++];
    }
}
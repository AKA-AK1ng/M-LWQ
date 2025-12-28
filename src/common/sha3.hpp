#pragma once
#include <cstdint>
#include <cstddef>

#define SHA3_STATE_SIZE 200

typedef struct sha3_context {
    uint8_t state[SHA3_STATE_SIZE];
    uint32_t pos;
    uint32_t rate;
} sha3_context;

void shake128_init(sha3_context *ctx);
void shake128_update(sha3_context *ctx, const void *data, size_t len);
void shake128_xof(sha3_context *ctx);
void shake128_out(sha3_context *ctx, void *out, size_t len);
#include "random.h"
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#if defined(__linux__) || defined(__APPLE__) || defined(__unix__)
#include <fcntl.h>
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sys/random.h>
#endif

void random_init() {
    // no-op: system RNG does not require initialization
}

#if defined(__linux__) || defined(__APPLE__) || defined(__unix__)
static void read_urandom(uint8_t *out, size_t len) {
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) {
        abort();
    }
    size_t offset = 0;
    while (offset < len) {
        ssize_t n = read(fd, out + offset, len - offset);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            close(fd);
            abort();
        }
        if (n == 0) {
            close(fd);
            abort();
        }
        offset += (size_t)n;
    }
    close(fd);
}
#endif

void random_bytes(uint8_t *out, size_t len) {
#if defined(__APPLE__)
    arc4random_buf(out, len);
#elif defined(__linux__)
    size_t offset = 0;
    while (offset < len) {
        ssize_t n = getrandom(out + offset, len - offset, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == ENOSYS) {
                read_urandom(out + offset, len - offset);
                return;
            }
            abort();
        }
        offset += (size_t)n;
    }
#else
#error "Unsupported platform for random_bytes"
#endif
}

void random_poly_uniform(poly *p) {
    for(int i=0; i<MLWQ_N; ++i) {
        uint16_t val = 0;
        random_bytes((uint8_t *)&val, sizeof(val));
        p->coeffs[i] = (int16_t)(val % MLWQ_Q);
    }
}


void random_poly_eta(poly *p) {
    for(int i=0; i<MLWQ_N; ++i) {
        // 模拟 CBD2: [-2, 2]
        // 概率: 0(37.5%), 1(25%), 2(6.25%)
        uint8_t byte = 0;
        random_bytes(&byte, sizeof(byte));
        int b0 = byte & 1;
        int b1 = (byte >> 1) & 1;
        int b2 = (byte >> 2) & 1;
        int b3 = (byte >> 3) & 1;
        p->coeffs[i] = (b0 + b1) - (b2 + b3);
    }
}

void random_poly_vec_eta(poly_vec *pv) {
    for(int i=0; i<MLWQ_K; ++i) {
        random_poly_eta(&pv->vec[i]);
    }
}

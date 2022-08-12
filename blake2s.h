#ifndef BLAKE2_H__
#define BLAKE2_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A BLAKE2s context, that keeps track of an ongoing computation.
 * Contents are not meant to be accessed directly.
 *
 * A context does not link to external resources, so it does not require
 * any specific release process. It also does not contain any pointer,
 * and thus can be copied and moved in RAM.
 */
typedef struct {
	uint8_t buf[64];
	uint32_t h[8];
	uint64_t ctr;
	size_t out_len;
} blake2s_context;

/*
 * Initialize a BLAKE2s context, using the specified output length (in
 * bytes). For the default BLAKE2s output (256 bits = 32 bytes), the
 * `out_len` parameter shall be set to 32.
 *
 * Note that the configured length impacts all output bytes; computing
 * BLAKE2s with a 256-bit output then truncating it to 128 bits does not
 * yield the same value as computing BLAKE2s directly with a 128-bit
 * output.
 */
void blake2s_init(blake2s_context *bc, size_t out_len);

/*
 * Initialize a BLAKE2s context, using the specified output length
 * (`out_len`, expressed in bytes) and a secret key (`key`, of length
 * `key_len` bytes). This uses BLAKE2s in MAC mode. If `key_len` is 0
 * then `key` is ignored, and a key-less initialization is performed
 * (as with blake2s_init()).
 */
void blake2s_init_key(blake2s_context *bc, size_t out_len,
	const void *key, size_t key_len);

/*
 * Inject some bytes (`data`, of length `len`) into an initialized
 * context. The to-hash data can be injected into a context through
 * an arbitrary number of calls to this function.
 */
void blake2s_update(blake2s_context *bc, const void *data, size_t len);

/*
 * Finalize a running computation and produce the output, which is
 * written into `dst`. The output length was configured when the
 * context was initialized.
 * After this call, the context is no longer usable, and must be
 * reinitialized before processing more bytes.
 */
void blake2s_final(blake2s_context *bc, void *dst);

/*
 * One-call BLAKE2s function: this function computes BLAKE2s with a
 * `dst_len` output length (in bytes). If `key_len` is non-zero, then
 * BLAKE2s is used in MAC mode with the provided `key`. The input
 * data is `src` (of length `src_len` bytes).
 */
void blake2s(void *dst, size_t dst_len, const void *key, size_t key_len,
	const void *src, size_t src_len);

#ifdef __cplusplus
}
#endif

#endif

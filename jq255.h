#ifndef jq255_h__
#define jq255_h__

#include <stddef.h>
#include <stdint.h>

/*
 * Types for a private key.
 * Type contents are opaque and MUST NOT be accessed directly.
 */
typedef union { uint32_t w32[8]; uint32_t w64[4]; } jq255e_private_key;
typedef union { uint32_t w32[8]; uint32_t w64[4]; } jq255s_private_key;

/*
 * Types for a public key.
 * Type contents are opaque and MUST NOT be accessed directly.
 */
typedef union { uint32_t w32[40]; uint32_t w64[20]; } jq255e_public_key;
typedef union { uint32_t w32[40]; uint32_t w64[20]; } jq255s_public_key;

/*
 * Types for a private/public key pair, which contains a private and
 * a public key. Key pairs are supposed to correspond to each other.
 */
typedef struct {
	jq255e_private_key private_key;
	jq255e_public_key public_key;
} jq255e_keypair;
typedef struct {
	jq255s_private_key private_key;
	jq255s_public_key public_key;
} jq255s_keypair;

/*
 * Generate a new private key, using the provided seed. The seed
 * must have at least 128 bits of entropy (seed length is not limited,
 * but a seed of less than 16 bytes cannot possibly hold 128 bits of
 * entropy).
 */
void jq255e_generate_private_key(jq255e_private_key *sk,
	const void *seed, size_t seed_len);
void jq255s_generate_private_key(jq255s_private_key *sk,
	const void *seed, size_t seed_len);

/*
 * (Re)compute a public key from its corresponding private key.
 */
void jq255e_make_public(jq255e_public_key *pk, const jq255e_private_key *sk);
void jq255s_make_public(jq255s_public_key *pk, const jq255s_private_key *sk);

/*
 * Generate a new key pair, using the provided seed. The seed
 * must have at least 128 bits of entropy (seed length is not limited,
 * but a seed of less than 16 bytes cannot possibly hold 128 bits of
 * entropy).
 */
void jq255e_generate_keypair(jq255e_keypair *jk,
	const void *seed, size_t seed_len);
void jq255s_generate_keypair(jq255s_keypair *jk,
	const void *seed, size_t seed_len);

/*
 * Decode a private key from bytes. Returned value is 1 on success, 0
 * on failure (invalid private key). On failure, the destination structure
 * (sk) is filled with a special "invalid key" value.
 * Encoded private keys have length 32 bytes exactly; if `len` is not 32,
 * then a failure is reported.
 */
int jq255e_decode_private_key(jq255e_private_key *sk,
	const void *src, size_t len);
int jq255s_decode_private_key(jq255s_private_key *sk,
	const void *src, size_t len);

/*
 * Decode a public key from bytes. Returned value is 1 on success, 0
 * on failure (invalid public key). On failure, the destination structure
 * (pk) is filled with a special "invalid key" value.
 * Encoded public keys have length 32 bytes exactly; if `len` is not 64,
 * then a failure is reported.
 */
int jq255e_decode_public_key(jq255e_public_key *pk,
	const void *src, size_t len);
int jq255s_decode_public_key(jq255s_public_key *pk,
	const void *src, size_t len);

/*
 * Decode a key pair, i.e. the concatenation of a private key
 * and a public key. Returned value is 1 on success, 0 on error.
 * On failure (invalid public or private key), both public and private
 * key parts of the destination structure (jk) are filled with
 * special "invalid key" values.
 * WARNING: this function does not verify whether the two keys match each
 * other.
 */
int jq255e_decode_keypair(jq255e_keypair *jk, const void *src, size_t len);
int jq255s_decode_keypair(jq255s_keypair *jk, const void *src, size_t len);

/*
 * Encode a private key. Output length is exactly 32 bytes. Output length
 * is returned. If the source key is invalid, then this function produces
 * 32 bytes of value 0x00.
 */
size_t jq255e_encode_private_key(void *dst, const jq255e_private_key *sk);
size_t jq255s_encode_private_key(void *dst, const jq255s_private_key *sk);

/*
 * Encode a public key. Output length is exactly 32 bytes. Output length
 * is returned. If the source key is invalid, then this function produces
 * 32 bytes of value 0x00.
 */
size_t jq255e_encode_public_key(void *dst, const jq255e_public_key *pk);
size_t jq255s_encode_public_key(void *dst, const jq255s_public_key *pk);

/*
 * Encode a key pair, i.e. the concatenation of a private key and a public
 * key. Output length is exactly 64 bytes. Output length is returned. If
 * either the source public or private key is invalid, then this function
 * produces an all-zero output for the invalid part.
 */
size_t jq255e_encode_keypair(void *dst, const jq255e_keypair *jk);
size_t jq255s_encode_keypair(void *dst, const jq255s_keypair *jk);

/*
 * Sign a message hash. The hash value is provided in `hv`, of size
 * `hv_len` bytes. The used hash function is identified by the provided
 * `hash_name`. If the raw unhashed message is used ("raw mode"), then
 * `hash_name` should be NULL or an empty string.
 *
 * The "sign_seeded()" functions accept an additional `seed`. The seed
 * is not necessary for security; providing a variable seed (e.g. the
 * output of a random generator, or a counter) randomizes the
 * signatures, in case strict deterministic signatures are not wished
 * for. If `seed_len` is 0, then it is assumed that no seed is provided,
 * and deterministic signatures are produced. The "sign()" functions
 * are equivalent to calling "sign_seeded()" with a zero-length seed.
 *
 * The signature is written into `sig`, and has length exactly 48 bytes.
 * The signature length is returned.
 */
size_t jq255e_sign(void *sig, const jq255e_keypair *jk,
	const char *hash_name, const void *hv, size_t hv_len);
size_t jq255e_sign_seeded(void *sig, const jq255e_keypair *jk,
	const char *hash_name, const void *hv, size_t hv_len,
	const void *seed, size_t seed_len);
size_t jq255s_sign(void *sig, const jq255s_keypair *jk,
	const char *hash_name, const void *hv, size_t hv_len);
size_t jq255s_sign_seeded(void *sig, const jq255s_keypair *jk,
	const char *hash_name, const void *hv, size_t hv_len,
	const void *seed, size_t seed_len);

/*
 * Standard hash function names.
 */
#define JQ255_HASHNAME_SHA224       "sha224"
#define JQ255_HASHNAME_SHA256       "sha256"
#define JQ255_HASHNAME_SHA384       "sha384"
#define JQ255_HASHNAME_SHA512       "sha512"
#define JQ255_HASHNAME_SHA512_224   "sha512224"
#define JQ255_HASHNAME_SHA512_256   "sha512256"
#define JQ255_HASHNAME_SHA3_224     "sha3224"
#define JQ255_HASHNAME_SHA3_256     "sha3256"
#define JQ255_HASHNAME_SHA3_384     "sha3384"
#define JQ255_HASHNAME_SHA3_512     "sha3512"
#define JQ255_HASHNAME_BLAKE2B      "blake2b"
#define JQ255_HASHNAME_BLAKE2S      "blake2s"
#define JQ255_HASHNAME_BLAKE3       "blake3"

/*
 * Verify a signature. The message hash (or raw message) is provided
 * with the same rules as in the signature generation function.
 * Returned value is 1 on success (signature is valid for the provided
 * message, relatively to the public key `pk`), 0 otherwise. If the
 * public key is in the special "invalid" state, then a failure is
 * reported.
 *
 * WARNING: verification is a variable-time process. It is assumed
 * that the signature, public key, and hashed message are all public
 * data.
 */
int jq255e_verify(const void *sig, size_t sig_len,
	const jq255e_public_key *pk,
	const char *hash_name, const void *hv, size_t hv_len);
int jq255s_verify(const void *sig, size_t sig_len,
	const jq255s_public_key *pk,
	const char *hash_name, const void *hv, size_t hv_len);

/*
 * Perform a key exchange between a local key pair, and a peer public
 * key. The resulting key has length 32 bytes and is written into the
 * destination array `shared_key`. The obtained key is the output of
 * a key derivation step, and has no discernable structure; if a shorter
 * key is required (e.g. 128 bits) then it can simply be truncated to
 * the right size.
 *
 * Returned value is 1 on success, 0 on failure. A failure is reported if
 * the peer public key is in the "invalid key" state. In that case, a
 * key is still generated; that key is unguessable by outsiders.
 */
int jq255e_ECDH(void *shared_key, const jq255e_keypair *jk_self,
	const jq255e_public_key *pk_peer);
int jq255s_ECDH(void *shared_key, const jq255s_keypair *jk_self,
	const jq255s_public_key *pk_peer);

#endif

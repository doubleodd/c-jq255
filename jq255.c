#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

/* ===================================================================== */
/*
 * CONFIGURATION
 *
 * JQ     If defined to JQ255E, implements jq255e.
 *        If defined to JQ255S, implements jq255s.
 *        If undefined, then it defaults to jq255e.
 *
 * W64    If defined to 1, uses 64-bit words.
 *        If defined to 0, uses 32-bit words.
 *        If undefined, then it autodetects the arch abilities.
 */

#ifndef JQ
#define JQ   JQ255E
#endif

#ifndef W64
#if defined _MSC_VER && defined _M_X64 \
	|| (((ULONG_MAX >> 31) >> 31) == 3 \
	&& (defined __GNUC__ || defined __clang__))
#define W64   1
#else
#define W64   0
#endif
#endif

#define JQ255E   1
#define JQ255S   2

/* ===================================================================== */
/*
 * We work in the finite field GF(2^255-MQ).
 * Rules:
 *    MQ < 32768
 *    MQ is odd
 *    MQ != 7 mod 8
 *    2^255 - MQ is a prime integer
 */
#if JQ == JQ255E
#define MQ   18651
#elif JQ == JQ255S
#define MQ   3957
#else
#error Unknown curve
#endif

static uint32_t
dec32le(const void *src)
{
	const uint8_t *buf = src;
	return (uint32_t)buf[0]
		| ((uint32_t)buf[1] << 8)
		| ((uint32_t)buf[2] << 16)
		| ((uint32_t)buf[3] << 24);
}

static void
enc32le(void *dst, uint32_t x)
{
	uint8_t *buf = dst;
	buf[0] = (uint8_t)x;
	buf[1] = (uint8_t)(x >> 8);
	buf[2] = (uint8_t)(x >> 16);
	buf[3] = (uint8_t)(x >> 24);
}

/* ===================================================================== */
/*
 * SECTION 1: FIELD ELEMENTS
 *
 * `gf` is defined to hold a field element in GF(2^255-MQ).
 */

#if W64

/* --------------------------------------------------------------------- */
/*
 * 64-bit implementation: the 256-bit value is split over four 64-bit
 * limbs.
 * Functions accept arbitrary input values (i.e. full range, up to
 * 2^256-1) and also return arbitrary values.
 */
typedef struct {
	uint64_t v0, v1, v2, v3;
} gf;

/*
 * Macro used to define constants. Eight 32-bit limbs are provided in
 * little-endian order.
 */
#define LGF(v0, v1, v2, v3, v4, v5, v6, v7) \
	{ (uint64_t)(v0) | ((uint64_t)(v1) << 32), \
	  (uint64_t)(v2) | ((uint64_t)(v3) << 32), \
	  (uint64_t)(v4) | ((uint64_t)(v5) << 32), \
	  (uint64_t)(v6) | ((uint64_t)(v7) << 32) }

static const gf gf_zero = LGF(0, 0, 0, 0, 0, 0, 0, 0);
static const gf gf_one = LGF(1, 0, 0, 0, 0, 0, 0, 0);
static const gf gf_minus_one = LGF(
	-(uint32_t)MQ - 1, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
	0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF);

/*
 * Add-with-carry and subtract-with-borrow primitives. On x86, there
 * are intrinsic functions for that; otherwise, we
 */
#if (defined _MSC_VER && defined _M_X64) \
	|| (defined __x86_64__ && (defined __GNUC__ || defined __clang__))
#include <immintrin.h>
#define adc(cc, a, b, d)   _addcarry_u64(cc, a, b, (unsigned long long *)(void *)d)
#define sbb(cc, a, b, d)   _subborrow_u64(cc, a, b, (unsigned long long *)(void *)d)
#else
static inline unsigned char
adc(unsigned char cc, uint64_t a, uint64_t b, uint64_t *d)
{
	unsigned __int128 t = (unsigned __int128)a + (unsigned __int128)b + cc;
	*d = (uint64_t)t;
	return (unsigned char)(t >> 64);
}
static inline unsigned char
sbb(unsigned char cc, uint64_t a, uint64_t b, uint64_t *d)
{
	unsigned __int128 t = (unsigned __int128)a - (unsigned __int128)b - cc;
	*d = (uint64_t)t;
	return (unsigned char)(-(uint64_t)(t >> 64));
}
#endif

/*
 * Multiplication and additions over 128 bits.
 */
#if defined _MSC_VER
#define UMUL(lo, hi, x, y)   do { \
		uint64_t umul_hi; \
		(lo) = _umul128((x), (y), &umul_hi); \
		(hi) = umul_hi; \
	} while (0)
#else
#define UMUL(lo, hi, x, y)   do { \
		unsigned __int128 umul_tmp; \
		umul_tmp = (unsigned __int128)(x) * (unsigned __int128)(y); \
		(lo) = (uint64_t)umul_tmp; \
		(hi) = (uint64_t)(umul_tmp >> 64); \
	} while (0)
#endif

/*
 * d <- a + b
 */
static inline void
gf_add(gf *d, const gf *a, const gf *b)
{
	uint64_t d0, d1, d2, d3;
	unsigned char cc;

	cc = adc(0, a->v0, b->v0, &d0);
	cc = adc(cc, a->v1, b->v1, &d1);
	cc = adc(cc, a->v2, b->v2, &d2);
	cc = adc(cc, a->v3, b->v3, &d3);

	/* On carry, add -2*q = -2^256 + 2*MQ */
	cc = adc(0, d0, -(uint64_t)cc & (2 * MQ), &d0);
	cc = adc(cc, d1, 0, &d->v1);
	cc = adc(cc, d2, 0, &d->v2);
	cc = adc(cc, d3, 0, &d->v3);

	/* Do it again if there is still a carry; but then the low word
	   must be close to 0, and it won't carry out again. */
	(void)adc(0, d0, -(uint64_t)cc & (2 * MQ), &d->v0);
}

/*
 * d <- a - b
 */
static inline void
gf_sub(gf *d, const gf *a, const gf *b)
{
	uint64_t d0, d1, d2, d3;
	unsigned char cc;

	cc = sbb(0, a->v0, b->v0, &d0);
	cc = sbb(cc, a->v1, b->v1, &d1);
	cc = sbb(cc, a->v2, b->v2, &d2);
	cc = sbb(cc, a->v3, b->v3, &d3);

	/* On borrow, subtract -2*q = -2^256 + 2*MQ */
	cc = sbb(0, d0, -(uint64_t)cc & (2 * MQ), &d0);
	cc = sbb(cc, d1, 0, &d->v1);
	cc = sbb(cc, d2, 0, &d->v2);
	cc = sbb(cc, d3, 0, &d->v3);

	/* Do it again if there is still a borrow; but then the low word
	   must be close to 2^64-1, and it won't carry out again. */
	(void)sbb(0, d0, -(uint64_t)cc & (2 * MQ), &d->v0);
}

/*
 * d <- -a
 */
static inline void
gf_neg(gf *d, const gf *a)
{
	uint64_t d0, d1, d2, d3, e;
	unsigned char cc;

	/* 2*q - a over 256 bits. */
	cc = sbb(0, -(uint64_t)(2 * MQ), a->v0, &d0);
	cc = sbb(cc, (uint64_t)-1, a->v1, &d1);
	cc = sbb(cc, (uint64_t)-1, a->v2, &d2);
	cc = sbb(cc, (uint64_t)-1, a->v3, &d3);

	/* On borrow, add q = 2^255 - MQ */
	e = -(uint64_t)cc;
	cc = sbb(0, d0, e & MQ, &d->v0);
	cc = sbb(cc, d1, e, &d->v1);
	cc = sbb(cc, d2, e, &d->v2);
	cc = sbb(cc, d3, e >> 1, &d->v3);
}

/*
 * If ctl == 0x00000000:  d <- a0
 * If ctl == 0xFFFFFFFF:  d <- a1
 * ctl MUST be either 0x00000000 or 0xFFFFFFFF.
 * Output is as reduced as the least reduced of the two inputs.
 */
static inline void
gf_select(gf *d, const gf *a0, const gf *a1, uint32_t ctl)
{
	uint64_t cm = (uint64_t)*(int32_t *)&ctl;
	d->v0 = a0->v0 ^ (cm & (a0->v0 ^ a1->v0));
	d->v1 = a0->v1 ^ (cm & (a0->v1 ^ a1->v1));
	d->v2 = a0->v2 ^ (cm & (a0->v2 ^ a1->v2));
	d->v3 = a0->v3 ^ (cm & (a0->v3 ^ a1->v3));
}

/*
 * If ctl == 0x00000000:  d <- a
 * If ctl == 0xFFFFFFFF:  d <- -a
 * ctl MUST be either 0x00000000 or 0xFFFFFFFF.
 */
static inline void
gf_condneg(gf *d, const gf *a, uint32_t ctl)
{
	gf c;
	gf_neg(&c, a);
	gf_select(d, a, &c, ctl);
}

/*
 * d <- 2*a
 */
static inline void
gf_mul2(gf *d, const gf *a)
{
	uint64_t d0, d1, d2, d3;
	unsigned char cc;

	/* Left-shift by 1 bit. */
	d0 = a->v0 << 1;
	d1 = (a->v1 << 1) | (a->v0 >> 63);
	d2 = (a->v2 << 1) | (a->v1 >> 63);
	d3 = (a->v3 << 1) | (a->v2 >> 63);

	/* Fold upper bit. */
	cc = adc(0, d0, -(uint64_t)(a->v3 >> 63) & (2 * MQ), &d0);
	cc = adc(cc, d1, 0, &d->v1);
	cc = adc(cc, d2, 0, &d->v2);
	cc = adc(cc, d3, 0, &d->v3);
	(void)adc(0, d0, -(uint64_t)cc & (2 * MQ), &d->v0);
}

/*
 * d <- a*(2^n)
 * 0 < n < 47
 */
static inline void
gf_lsh(gf *d, const gf *a, unsigned n)
{
	uint64_t d0, d1, d2, d3, d4;
	unsigned char cc;

	d0 = a->v0 << n;
	d1 = (a->v1 << n) | (a->v0 >> (64 - n));
	d2 = (a->v2 << n) | (a->v1 >> (64 - n));
	d3 = (a->v3 << n) | (a->v2 >> (64 - n));
	d4 = a->v3 >> (64 - n);

	cc = adc(0, d0, d4 * (2 * MQ), &d0);
	cc = adc(cc, d1, 0, &d->v1);
	cc = adc(cc, d2, 0, &d->v2);
	cc = adc(cc, d3, 0, &d->v3);
	(void)adc(cc, d0, -(uint64_t)cc & (2 * MQ), &d->v0);
}

/*
 * d <- a/2
 */
static inline void
gf_half(gf *d, const gf *a)
{
	uint64_t m = -(a->v0 & 1);
	unsigned char cc;

	/* Right-shift the integer by 1 bit; also add back (q+1)/2 if
	   the dropped bit was a 1. */
	cc = adc(0, (a->v0 >> 1) | (a->v1 << 63),
		m & -(uint64_t)(MQ >> 1), &d->v0);
	cc = adc(cc, (a->v1 >> 1) | (a->v2 << 63), m, &d->v1);
	cc = adc(cc, (a->v2 >> 1) | (a->v3 << 63), m, &d->v2);
	(void)adc(cc, a->v3 >> 1, m >> 2, &d->v3);
}

#if 0
/* unused */
/*
 * d <- a*m
 * m < 65536
 */
static inline void
gf_mul_small(gf *d, const gf *a, uint32_t m)
{
	uint64_t d0, d1, d2, d3, lo, hi;
	unsigned char cc;

	/* Multiplication over integers. */
	UMUL(d0, d1, a->v0, m);
	UMUL(d2, d3, a->v2, m);
	UMUL(lo, hi, a->v1, m);
	cc = adc(0, d1, lo, &d1);
	cc = adc(cc, d2, hi, &d2);
	UMUL(lo, hi, a->v3, m);
	cc = adc(cc, d3, lo, &d3);
	(void)adc(cc, 0, hi, &hi);

	/* Fold the upper bits. */
	hi = (hi << 1) | (d3 >> 63);
	d3 &= 0x7FFFFFFFFFFFFFFF;
	cc = adc(0, d0, hi * MQ, &d->v0);
	cc = adc(cc, d1, 0, &d->v1);
	cc = adc(cc, d2, 0, &d->v2);
	(void)adc(cc, d3, 0, &d->v3);
}
#endif

/*
 * d <- a*b
 */
static inline void
gf_mul(gf *d, const gf *a, const gf *b)
{
	uint64_t e0, e1, e2, e3, e4, e5, e6, e7;
	uint64_t h0, h1, h2, h3;
	uint64_t lo, hi, lo2, hi2;
	unsigned char cc;

	/* Multiplication over integers. */
	UMUL(e0, e1, a->v0, b->v0);
	UMUL(e2, e3, a->v1, b->v1);
	UMUL(e4, e5, a->v2, b->v2);
	UMUL(e6, e7, a->v3, b->v3);

	UMUL(lo, hi, a->v0, b->v1);
	cc = adc(0, e1, lo, &e1);
	cc = adc(cc, e2, hi, &e2);
	UMUL(lo, hi, a->v0, b->v3);
	cc = adc(cc, e3, lo, &e3);
	cc = adc(cc, e4, hi, &e4);
	UMUL(lo, hi, a->v2, b->v3);
	cc = adc(cc, e5, lo, &e5);
	cc = adc(cc, e6, hi, &e6);
	(void)adc(cc, e7, 0, &e7);

	UMUL(lo, hi, a->v1, b->v0);
	cc = adc(0, e1, lo, &e1);
	cc = adc(cc, e2, hi, &e2);
	UMUL(lo, hi, a->v3, b->v0);
	cc = adc(cc, e3, lo, &e3);
	cc = adc(cc, e4, hi, &e4);
	UMUL(lo, hi, a->v3, b->v2);
	cc = adc(cc, e5, lo, &e5);
	cc = adc(cc, e6, hi, &e6);
	(void)adc(cc, e7, 0, &e7);

	UMUL(lo, hi, a->v0, b->v2);
	cc = adc(0, e2, lo, &e2);
	cc = adc(cc, e3, hi, &e3);
	UMUL(lo, hi, a->v1, b->v3);
	cc = adc(cc, e4, lo, &e4);
	cc = adc(cc, e5, hi, &e5);
	cc = adc(cc, e6, 0, &e6);
	(void)adc(cc, e7, 0, &e7);

	UMUL(lo, hi, a->v2, b->v0);
	cc = adc(0, e2, lo, &e2);
	cc = adc(cc, e3, hi, &e3);
	UMUL(lo, hi, a->v3, b->v1);
	cc = adc(cc, e4, lo, &e4);
	cc = adc(cc, e5, hi, &e5);
	cc = adc(cc, e6, 0, &e6);
	(void)adc(cc, e7, 0, &e7);

	UMUL(lo, hi, a->v1, b->v2);
	UMUL(lo2, hi2, a->v2, b->v1);
	cc = adc(0, lo, lo2, &lo);
	cc = adc(cc, hi, hi2, &hi);
	(void)adc(cc, 0, 0, &hi2);
	cc = adc(0, e3, lo, &e3);
	cc = adc(cc, e4, hi, &e4);
	cc = adc(cc, e5, hi2, &e5);
	cc = adc(cc, e6, 0, &e6);
	(void)adc(cc, e7, 0, &e7);

	UMUL(lo, h0, e4, 2 * MQ);
	cc = adc(0, e0, lo, &e0);
	UMUL(lo, h1, e5, 2 * MQ);
	cc = adc(cc, e1, lo, &e1);
	UMUL(lo, h2, e6, 2 * MQ);
	cc = adc(cc, e2, lo, &e2);
	UMUL(lo, h3, e7, 2 * MQ);
	cc = adc(cc, e3, lo, &e3);
	(void)adc(cc, 0, h3, &h3);

	h3 = (h3 << 1) | (e3 >> 63);
	e3 &= 0x7FFFFFFFFFFFFFFF;
	cc = adc(0, e0, h3 * MQ, &e0);
	cc = adc(cc, e1, h0, &e1);
	cc = adc(cc, e2, h1, &e2);
	(void)adc(cc, e3, h2, &e3);

	d->v0 = e0;
	d->v1 = e1;
	d->v2 = e2;
	d->v3 = e3;
}

/*
 * d <- a*a
 */
static inline void
gf_square(gf *d, const gf *a)
{
	uint64_t e0, e1, e2, e3, e4, e5, e6, e7;
	uint64_t h0, h1, h2, h3;
	uint64_t lo, hi;
	unsigned char cc;

	UMUL(e1, e2, a->v0, a->v1);
	UMUL(e3, e4, a->v0, a->v3);
	UMUL(e5, e6, a->v2, a->v3);
	UMUL(lo, hi, a->v0, a->v2);
	cc = adc(0, e2, lo, &e2);
	cc = adc(cc, e3, hi, &e3);
	UMUL(lo, hi, a->v1, a->v3);
	cc = adc(cc, e4, lo, &e4);
	cc = adc(cc, e5, hi, &e5);
	(void)adc(cc, e6, 0, &e6);
	UMUL(lo, hi, a->v1, a->v2);
	cc = adc(0, e3, lo, &e3);
	cc = adc(cc, e4, hi, &e4);
	cc = adc(cc, e5, 0, &e5);
	(void)adc(cc, e6, 0, &e6);

	/* There cannot be extra carry here because the partial sum is
	   necessarily lower than 2^448 at this point. */

	e7 = e6 >> 63;
	e6 = (e6 << 1) | (e5 >> 63);
	e5 = (e5 << 1) | (e4 >> 63);
	e4 = (e4 << 1) | (e3 >> 63);
	e3 = (e3 << 1) | (e2 >> 63);
	e2 = (e2 << 1) | (e1 >> 63);
	e1 = e1 << 1;

	UMUL(e0, hi, a->v0, a->v0);
	cc = adc(0, e1, hi, &e1);
	UMUL(lo, hi, a->v1, a->v1);
	cc = adc(cc, e2, lo, &e2);
	cc = adc(cc, e3, hi, &e3);
	UMUL(lo, hi, a->v2, a->v2);
	cc = adc(cc, e4, lo, &e4);
	cc = adc(cc, e5, hi, &e5);
	UMUL(lo, hi, a->v3, a->v3);
	cc = adc(cc, e6, lo, &e6);
	(void)adc(cc, e7, hi, &e7);

	/* Reduction */

	UMUL(lo, h0, e4, 2 * MQ);
	cc = adc(0, e0, lo, &e0);
	UMUL(lo, h1, e5, 2 * MQ);
	cc = adc(cc, e1, lo, &e1);
	UMUL(lo, h2, e6, 2 * MQ);
	cc = adc(cc, e2, lo, &e2);
	UMUL(lo, h3, e7, 2 * MQ);
	cc = adc(cc, e3, lo, &e3);
	(void)adc(cc, 0, h3, &h3);

	h3 = (h3 << 1) | (e3 >> 63);
	e3 &= 0x7FFFFFFFFFFFFFFF;
	cc = adc(0, e0, h3 * MQ, &e0);
	cc = adc(cc, e1, h0, &e1);
	cc = adc(cc, e2, h1, &e2);
	(void)adc(cc, e3, h2, &e3);

	d->v0 = e0;
	d->v1 = e1;
	d->v2 = e2;
	d->v3 = e3;
}

/*
 * d <- a^(2^n) (n successive squarings)
 */
static void
gf_xsquare(gf *d, const gf *a, unsigned n)
{
	if (n == 0) {
		memmove(d, a, sizeof *a);
		return;
	}
	gf_square(d, a);
	while (n -- > 1) {
		gf_square(d, d);
	}
}

/*
 * Return 0xFFFFFFFF is a == 0 (as a field element), 0x00000000 otherwise.
 */
static uint32_t
gf_is_zero(const gf *a)
{
	/*
	 * Over 256 bits, there are three representations of zero:
	 * 0, q and 2*q. Value r0, r1 or r2 will be zero if and only if
	 * the input matches the corresponding representation.
	 */
	uint64_t r0, r1, r2;

	r0 = a->v0 | a->v1 | a->v2 | a->v3;
	r1 = (a->v0 ^ -(uint64_t)MQ) | ~(a->v1 & a->v2)
		| (a->v3 ^ 0x7FFFFFFFFFFFFFFF);
	r2 = (a->v0 ^ -(uint64_t)(2 * MQ)) | ~(a->v1 & a->v2 & a->v3);
	return (uint32_t)(((r0 | -r0) & (r1 | -r1) & (r2 | -r2)) >> 63) - 1;
}

/*
 * Return 0xFFFFFFFF is a == b (as field elements), 0x00000000 otherwise.
 * Input: full range
 */
static uint32_t
gf_equals(const gf *a, const gf *b)
{
	gf d;
	gf_sub(&d, a, b);
	return gf_is_zero(&d);
}

/*
 * Normalize a value into the 0 to q-1 range.
 */
static void
gf_normalize(gf *d, const gf *a)
{
	uint64_t d0, d1, d2, d3, e;
	unsigned char cc;

	/* Fold top bit. */
	e = -(a->v3 >> 63);
	cc = adc(0, a->v0, e & MQ, &d0);
	cc = adc(cc, a->v1, 0, &d1);
	cc = adc(cc, a->v2, 0, &d2);
	cc = adc(cc, a->v3, e << 63, &d3);

	/* Subtract q; add it back if that generates a borrow. */
	cc = sbb(0, d0, -(uint64_t)MQ, &d0);
	cc = sbb(cc, d1, (uint64_t)-1, &d1);
	cc = sbb(cc, d2, (uint64_t)-1, &d2);
	cc = sbb(cc, d3, (uint64_t)-1 >> 1, &d3);

	e = -(uint64_t)cc;
	cc = adc(0, d0, e & -MQ, &d->v0);
	cc = adc(cc, d1, e, &d->v1);
	cc = adc(cc, d2, e, &d->v2);
	(void)adc(cc, d3, e >> 1, &d->v3);
}

/*
 * Return 0xFFFFFFFF is x is "negative", or 0x00000000 if x is "non-negative".
 * A field element x is considered negative if the least significant bit of
 * its representation as an integer in the 0..q-1 range is 1.
 */
static uint32_t
gf_is_negative(const gf *x)
{
	gf y;
	gf_normalize(&y, x);
	return -((uint32_t)y.v0 & 1);
}

static inline uint64_t
dec64le(const void *src)
{
	const uint8_t *buf = src;
	return (uint64_t)buf[0]
		| ((uint64_t)buf[1] << 8)
		| ((uint64_t)buf[2] << 16)
		| ((uint64_t)buf[3] << 24)
		| ((uint64_t)buf[4] << 32)
		| ((uint64_t)buf[5] << 40)
		| ((uint64_t)buf[6] << 48)
		| ((uint64_t)buf[7] << 56);
}

static inline void
enc64le(void *dst, uint64_t x)
{
	uint8_t *buf = dst;
	buf[0] = (uint8_t)x;
	buf[1] = (uint8_t)(x >> 8);
	buf[2] = (uint8_t)(x >> 16);
	buf[3] = (uint8_t)(x >> 24);
	buf[4] = (uint8_t)(x >> 32);
	buf[5] = (uint8_t)(x >> 40);
	buf[6] = (uint8_t)(x >> 48);
	buf[7] = (uint8_t)(x >> 56);
}

/*
 * Decode a field element from exactly 32 bytes. Returned value is 0xFFFFFFFF
 * if the value was correct (i.e. in the 0..q-1 range), 0x00000000 otherwise.
 * On success, d is set to the field element; otherwise, d is set to zero.
 */
static uint32_t
gf_decode(gf *d, const void *src)
{
	const uint8_t *buf = src;
	uint64_t d0, d1, d2, d3, t;
	unsigned char cc;

	d0 = dec64le(buf);
	d1 = dec64le(buf + 8);
	d2 = dec64le(buf + 16);
	d3 = dec64le(buf + 24);
	cc = sbb(0, d0, -(uint64_t)MQ, &t);
	cc = sbb(cc, d1, (uint64_t)-1, &t);
	cc = sbb(cc, d2, (uint64_t)-1, &t);
	cc = sbb(cc, d3, (uint64_t)-1 >> 1, &t);

	t = -(uint64_t)cc;
	d->v0 = d0 & t;
	d->v1 = d1 & t;
	d->v2 = d2 & t;
	d->v3 = d3 & t;
	return (uint32_t)t;
}

#if 0
/* unused */
/*
 * Decode some bytes into a field element. Unsigned little-endian convention
 * is used for the source bytes. If the source decodes to an integer larger
 * than the modulus, then it is implicitly reduced modulo q. The process
 * never fails.
 */
static void
gf_decode_reduce(gf *d, const void *src, size_t len)
{
	const uint8_t *buf = src;

	/* Decode first block, to get a length multiple of 32. */
	if ((len & 31) != 0) {
		size_t clen = len & 31;
		uint8_t tmp[32];
		len -= clen;
		memcpy(tmp, buf + len, clen);
		memset(tmp + clen, 0, 32 - clen);
		d->v0 = dec64le(tmp);
		d->v1 = dec64le(tmp + 8);
		d->v2 = dec64le(tmp + 16);
		d->v3 = dec64le(tmp + 24);
	} else if (len == 0) {
		*d = gf_zero;
		return;
	} else {
		len -= 32;
		d->v0 = dec64le(buf + len);
		d->v1 = dec64le(buf + len + 8);
		d->v2 = dec64le(buf + len + 16);
		d->v3 = dec64le(buf + len + 24);
	}

	/* For all remaining blocks: multiply the current by 2^256 = 2*MQ,
	   then add the next block. */
	while (len > 0) {
		gf x;
		len -= 32;
		x.v0 = dec64le(buf + len);
		x.v1 = dec64le(buf + len + 8);
		x.v2 = dec64le(buf + len + 16);
		x.v3 = dec64le(buf + len + 24);
		gf_mul_small(d, d, 2 * MQ);
		gf_add(d, d, &x);
	}
}
#endif

/*
 * Encode a field element over exactly 32 bytes.
 */
static void
gf_encode(void *dst, const gf *a)
{
	uint8_t *buf = dst;
	gf x;
	gf_normalize(&x, a);
	enc64le(buf, x.v0);
	enc64le(buf + 8, x.v1);
	enc64le(buf + 16, x.v2);
	enc64le(buf + 24, x.v3);
}

#else /* W64 */

/* --------------------------------------------------------------------- */
/*
 * 32-bit implementation: the 256-bit value is split over eight 32-bit
 * limbs.
 * Full range: value may be up to 2^256-1
 * Partially reduced: value is in the 0 to 2^255 + 2^32 - 1 range.
 * Fully reduced: value is in the 0 to 2^255-MQ-1 range.
 *
 * GENERAL API RULES:
 *  - Functions always produce partially reduced outputs.
 *  - Functions accept partially reduced inputs.
 *  - Some functions can work with full-range inputs.
 *  - An output operand may be the same memory object as an input operand.
 */
typedef struct {
	uint32_t v[8];
} gf;

/*
 * Macro used to define constants. Eight 32-bit limbs are provided in
 * little-endian order.
 */
#define LGF(v0, v1, v2, v3, v4, v5, v6, v7) \
	{ { v0, v1, v2, v3, v4, v5, v6, v7 } }

static const gf gf_zero = LGF(0, 0, 0, 0, 0, 0, 0, 0);
static const gf gf_one = LGF(1, 0, 0, 0, 0, 0, 0, 0);
static const gf gf_minus_one = LGF(
	-(uint32_t)MQ - 1, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
	0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF);

/*
 * Partially reduce value d + extra*2^256.
 * Input: d is full range, extra < 65536
 * Output: d is partially reduced
 */
static void
gf_partial_reduce(gf *d, uint32_t extra)
{
	uint32_t cc = ((extra << 1) + (d->v[7] >> 31)) * MQ;
	d->v[7] &= 0x7FFFFFFF;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)d->v[i] + (uint64_t)cc;
		d->v[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}
}

/*
 * Finish reduction of d.
 * Input: d is partially reduced
 * Output: d is fully reduced
 */
static void
gf_finish_reduce(gf *d)
{
	static const uint32_t q[8] = {
		-(uint32_t)MQ, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
		0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF
	};

	/* Subtract q; if there is a borrow, add back q. */
	uint32_t cc = 0;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)d->v[i] - (uint64_t)q[i] - (uint64_t)cc;
		d->v[i] = (uint32_t)w;
		cc = -(uint32_t)(w >> 32);
	}
	uint32_t z = -cc;
	cc = 0;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)d->v[i] + (uint64_t)(z & q[i])
			+ (uint64_t)cc;
		d->v[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}
}

/*
 * d <- a + b
 * Input: full range
 * Output: d is partially reduced
 */
static void
gf_add(gf *d, const gf *a, const gf *b)
{
	uint32_t cc = 0;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)a->v[i]
			+ (uint64_t)b->v[i] + (uint64_t)cc;
		d->v[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}
	gf_partial_reduce(d, cc);
}

/*
 * d <- a - b
 * Input: full range
 * Output: d is partially reduced
 */
static void
gf_sub(gf *d, const gf *a, const gf *b)
{
	/*
	 * We compute 4*q + a - b = 2^257 + a - b - 4*MQ. This is
	 * necessarily positive so we get an output that can then be
	 * processed with gf_partial_reduce().
	 */
	uint32_t cc = 4 * MQ;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)a->v[i] - (uint64_t)b->v[i]
			- (uint64_t)cc;
		d->v[i] = (uint32_t)w;
		cc = -(uint32_t)(w >> 32);
	}
	gf_partial_reduce(d, 2 - cc);
}

/*
 * d <- -a
 * Input: full range
 * Output: partially reduced
 */
static void
gf_neg(gf *d, const gf *a)
{
	gf_sub(d, &gf_zero, a);
}

/*
 * If ctl == 0x00000000:  d <- a0
 * If ctl == 0xFFFFFFFF:  d <- a1
 * ctl MUST be either 0x00000000 or 0xFFFFFFFF.
 * Output is as reduced as the least reduced of the two inputs.
 */
static void
gf_select(gf *d, const gf *a0, const gf *a1, uint32_t ctl)
{
	for (int i = 0; i < 8; i ++) {
		uint32_t t0 = a0->v[i];
		uint32_t t1 = a1->v[i];
		d->v[i] = t0 ^ (ctl & (t0 ^ t1));
	}
}

/*
 * If ctl == 0x00000000:  d <- a
 * If ctl == 0xFFFFFFFF:  d <- -a
 * ctl MUST be either 0x00000000 or 0xFFFFFFFF.
 * Input: partially reduced
 * Output: partially reduced
 */
static void
gf_condneg(gf *d, const gf *a, uint32_t ctl)
{
	gf c;
	gf_neg(&c, a);
	gf_select(d, a, &c, ctl);
}

/*
 * d <- a*(2^n)
 * Input: full range; 0 < n <= 16
 * Output: partially reduced
 */
static void
gf_lsh(gf *d, const gf *a, unsigned n)
{
	uint32_t cc = 0;
	for (int i = 0; i < 8; i ++) {
		uint32_t t = a->v[i];
		d->v[i] = (t << n) | cc;
		cc = t >> (32 - n);
	}
	gf_partial_reduce(d, cc);
}

/*
 * d <- 2*a
 * Input: full range
 * Output: partially reduced
 */
#define gf_mul2(d, a)   gf_lsh(d, a, 1)

/*
 * d <- a/2
 * Input: partially reduced
 * Output: partially reduced
 */
static void
gf_half(gf *d, const gf *a)
{
	/* (q+1)/2 */
	static const uint32_t hq[8] = {
		-(uint32_t)(MQ >> 1), 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
		0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x3FFFFFFF
	};

	uint32_t z = -(a->v[0] & 1);
	uint32_t cc = 0;
	for (int i = 0; i < 7; i ++) {
		uint32_t t = (a->v[i] >> 1) | (a->v[i + 1] << 31);
		uint64_t w = (uint64_t)t + (uint64_t)(z & hq[i]) + (uint64_t)cc;
		d->v[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}
	d->v[7] = (a->v[7] >> 1) + (z & hq[7]) + cc;
}

#if 0
/* unused */
/*
 * d <- a*m
 * Input: full range, m < 65536
 * Output: partially reduced
 */
static void
gf_mul_small(gf *d, const gf *a, uint32_t m)
{
	uint32_t cc = 0;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)a->v[i] * (uint64_t)m + (uint64_t)cc;
		d->v[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}
	gf_partial_reduce(d, cc);
}
#endif

/*
 * d <- a*b
 * Input: full range
 * Output: d is partially reduced
 */
static void
gf_mul(gf *d, const gf *a, const gf *b)
{
	uint32_t t[16];
	uint32_t cc;

	/* t <- a*b (over integers) */
	memset(&t, 0, sizeof t);
	for (int i = 0; i < 8; i ++) {
		cc = 0;
		for (int j = 0; j < 8; j ++) {
			uint64_t w = (uint64_t)a->v[i] * (uint64_t)b->v[j]
				+ (uint64_t)t[i + j] + (uint64_t)cc;
			t[i + j] = (uint32_t)w;
			cc = (uint32_t)(w >> 32);
		}
		t[i + 8] = cc;
	}

	/* d <- t[0..7] + 2*MQ*t[8..15] */
	cc = 0;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)t[i] + (uint64_t)t[i + 8] * (2 * MQ)
			+ (uint64_t)cc;
		d->v[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}
	gf_partial_reduce(d, cc);
}

/*
 * d <- a*a
 * Input: full range
 * Output: d is partially reduced
 */
static void
gf_square(gf *d, const gf *a)
{
	uint32_t t[16];
	uint32_t cc;

	/* t <- a*a (over integers) */
	memset(&t, 0, sizeof t);

	/* 1. Compute a[i]*a[j] for j > i */
	for (int i = 0; i < 7; i ++) {
		cc = 0;
		for (int j = i + 1; j < 8; j ++) {
			uint64_t w = (uint64_t)a->v[i] * (uint64_t)a->v[j]
				+ (uint64_t)t[i + j] + (uint64_t)cc;
			t[i + j] = (uint32_t)w;
			cc = (uint32_t)(w >> 32);
		}
		t[i + 8] = cc;
	}

	/* 2. Double all computed values so far. */
	t[15] = t[14] >> 31;
	for (int i = 14; i >= 1; i --) {
		t[i] = (t[i] << 1) | (t[i - 1] >> 31);
	}

	/* 3. Add squares a[i]*a[i]. */
	cc = 0;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)a->v[i] * (uint64_t)a->v[i]
			+ (uint64_t)t[2 * i] + (uint64_t)cc;
		t[2 * i] = (uint32_t)w;
		w = (uint64_t)t[2 * i + 1] + (w >> 32);
		t[2 * i + 1] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}

	/* d <- t[0..7] + 2*MQ*t[8..15] */
	cc = 0;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)t[i] + (uint64_t)t[i + 8] * (2 * MQ)
			+ (uint64_t)cc;
		d->v[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}
	gf_partial_reduce(d, cc);
}

/*
 * d <- a^(2^n) (n successive squarings)
 * Input: full range
 * Output: d is partially reduced
 */
static void
gf_xsquare(gf *d, const gf *a, unsigned n)
{
	if (n == 0) {
		memmove(d, a, sizeof *a);
		return;
	}
	gf_square(d, a);
	while (n -- > 1) {
		gf_square(d, d);
	}
}

/*
 * Return 0xFFFFFFFF is a == 0 (as a field element), 0x00000000 otherwise.
 * Input: partially reduced
 */
static uint32_t
gf_is_zero(const gf *a)
{
	/* For partially reduces values, there are only two possible
	   representations of zero. */
	uint32_t r0, r1;

	r0 = a->v[0];
	r1 = a->v[0] ^ (-(uint32_t)MQ);
	for (int i = 1; i < 7; i ++) {
		r0 |= a->v[i];
		r1 |= ~a->v[i];
	}
	r0 |= a->v[7];
	r1 |= a->v[7] ^ 0x7FFFFFFF;

	/* If r0 == 0 or r1 == 0, then the value matches one of the
	   possible representations of zero; otherwise, the source
	   value is not a zero. */
	return (((r0 | -r0) & (r1 | -r1)) >> 31) - 1;
}

/*
 * Return 0xFFFFFFFF is a == b (as field elements), 0x00000000 otherwise.
 * Input: full range
 */
static uint32_t
gf_equals(const gf *a, const gf *b)
{
	gf d;
	gf_sub(&d, a, b);
	return gf_is_zero(&d);
}

/*
 * Return 0xFFFFFFFF is x is "negative", or 0x00000000 if x is "non-negative".
 * A field element x is considered negative if the least significant bit of
 * its representation as an integer in the 0..q-1 range is 1.
 *
 * Input: x is partially reduced
 */
static uint32_t
gf_is_negative(const gf *x)
{
	gf y = *x;
	gf_finish_reduce(&y);
	return -(y.v[0] & 1);
}

/*
 * Decode a field element from exactly 32 bytes. Returned value is 0xFFFFFFFF
 * if the value was correct (i.e. in the 0..q-1 range), 0x00000000 otherwise.
 * On success, d is set to the field element; otherwise, d is set to zero.
 *
 * Output: fully reduced
 */
static uint32_t
gf_decode(gf *d, const void *src)
{
	static const uint32_t q[8] = {
		-(uint32_t)MQ, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
		0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF
	};

	const uint8_t *buf = src;
	gf x;
	uint32_t cc = 0;

	for (int i = 0; i < 8; i ++) {
		uint32_t t = dec32le(buf + 4 * i);
		uint64_t w = (uint64_t)t - (uint64_t)q[i] - (uint64_t)cc;
		x.v[i] = t;
		cc = -(uint32_t)(w >> 32);
	}
	cc = -cc;
	gf_select(d, &gf_zero, &x, cc);
	return cc;
}

#if 0
/* unused */
/*
 * Decode some bytes into a field element. Unsigned little-endian convention
 * is used for the source bytes. If the source decodes to an integer larger
 * than the modulus, then it is implicitly reduced modulo q. The process
 * never fails.
 *
 * Output: partially reduced
 */
static void
gf_decode_reduce(gf *d, const void *src, size_t len)
{
	const uint8_t *buf = src;

	/* Decode first block, to get a length multiple of 32. */
	if ((len & 31) != 0) {
		size_t clen = len & 31;
		uint8_t tmp[32];
		len -= clen;
		memcpy(tmp, buf + len, clen);
		memset(tmp + clen, 0, 32 - clen);
		for (int i = 0; i < 8; i ++) {
			d->v[i] = dec32le(tmp + 4 * i);
		}
		/* No required reduction in that case (top byte was zero). */
	} else if (len == 0) {
		*d = gf_zero;
		return;
	} else {
		len -= 32;
		for (int i = 0; i < 8; i ++) {
			d->v[i] = dec32le(buf + len + 4 * i);
		}
		gf_partial_reduce(d, 0);
	}

	/* For all remaining blocks: multiply the current by 2^256 = 2*MQ,
	   then add the next block. */
	while (len > 0) {
		gf x;
		len -= 32;
		for (int i = 0; i < 8; i ++) {
			x.v[i] = dec32le(buf + len + 4 * i);
		}
		gf_mul_small(d, d, 2 * MQ);
		gf_add(d, d, &x);
	}
}
#endif

/*
 * Encode a field element over exactly 32 bytes.
 * Input: partially reduced
 */
static void
gf_encode(void *dst, const gf *a)
{
	uint8_t *buf = dst;
	gf x = *a;
	gf_finish_reduce(&x);
	for (int i = 0; i < 8; i ++) {
		enc32le(buf + 4 * i, x.v[i]);
	}
}

#endif /* W64 */

/*
 * Inversions and square roots use code common to the 32-bit and 64-bit
 * implementations of field elements.
 */

/*
 * Compute a^(2^240-1) (into a240); also write a, a^2 and a^3 into win[0..2].
 */
static void
gf_prep240(gf *a240, gf *win, const gf *a)
{
	gf x, y;

	/* Fill win[] */
	memmove(&win[0], a, sizeof *a);
	gf_square(&win[1], &win[0]);
	gf_mul(&win[2], &win[0], &win[1]);

	/* y <- a^(2^4-1) */
	gf_xsquare(&x, &win[2], 2);
	gf_mul(&y, &x, &win[2]);

	/* y <- a^(2^5-1) */
	gf_square(&x, &y);
	gf_mul(&y, &x, &win[0]);

	/* y <- a^(2^15-1) */
	gf_xsquare(&x, &y, 5);
	gf_mul(&x, &x, &y);
	gf_xsquare(&x, &x, 5);
	gf_mul(&y, &y, &x);

	/* y <- a^(2^30-1) */
	gf_xsquare(&x, &y, 15);
	gf_mul(&y, &y, &x);

	/* y <- a^(2^60-1) */
	gf_xsquare(&x, &y, 30);
	gf_mul(&y, &y, &x);

	/* y <- a^(2^120-1) */
	gf_xsquare(&x, &y, 60);
	gf_mul(&y, &y, &x);

	/* a240 <- a^(2^240-1) */
	gf_xsquare(&x, &y, 120);
	gf_mul(a240, &y, &x);
}

/*
 * d <- 1/a
 * If a == 0 then returned value is zero.
 * Input: full range
 * Output: d is partially reduced
 */
static void
gf_inv(gf *d, const gf *a)
{
	/*
	 * This is a perfunctory division with Fermat's little theorem.
	 * An optimized implementation may use a binary GCD or a
	 * similar plus/minus algorithm.
	 */
	gf x, win[3];
	uint32_t e;

	/*
	 * d <- a^(q-2)
	 * q-2 has length 255 bits; the top 240 bits are all equal to 1.
	 */
	gf_prep240(&x, win, a);
	e = -(uint32_t)MQ - 2;
	for (int i = 13; i >= 1; i -= 2) {
		unsigned k = (e >> i) & 3;
		gf_xsquare(&x, &x, 2);
		if (k != 0) {
			gf_mul(&x, &x, &win[k - 1]);
		}
	}
	gf_square(&x, &x);
	gf_mul(d, &x, &win[0]);
}

/*
 * d <- sqrt(a)
 * If a was a square, then the non-negative root is set in d, and
 * 0xFFFFFFFF is returned. Otherwise, d is set to zero, and 0x00000000
 * is returned.
 *
 * Input: full range
 * Output: d is partially reduced
 */
static uint32_t
gf_sqrt(gf *d, const gf *a)
{
	gf x, y, win[3];
	uint32_t e, r;

	/* If q = 5 mod 8, prepare (2*a)^(2^240-1); otherwise,
	   prepare a^(2^240-1). */
#if (MQ & 7) == 3
	gf_mul2(&x, a);
#else
	x = *a;
#endif
	gf_prep240(&x, win, &x);

#if (MQ & 3) == 1
	/* When q = 3 mod 4, candidate root is x = a^((q+1)/4) */
	e = (1 - (uint32_t)MQ) >> 2;
	for (int i = 11; i >= 1; i -= 2) {
		unsigned k = (e >> i) & 3;
		gf_xsquare(&x, &x, 2);
		if (k != 0) {
			gf_mul(&x, &x, &win[k - 1]);
		}
	}
	gf_square(&x, &x);
	if (e & 1) {
		gf_mul(&x, &x, &win[0]);
	}
#elif (MQ & 7) == 3
	/* When q = 5 mod 8, we use Atkin's algorithm:
	     b <- (2*a)^((q-5)/8)
	     c <- 2*a*b^2
	     x <- a*b*(c - 1) */
	e = (-(uint32_t)MQ - 5) >> 3;
	for (int i = 10; i >= 0; i -= 2) {
		unsigned k = (e >> i) & 3;
		gf_xsquare(&x, &x, 2);
		if (k != 0) {
			gf_mul(&x, &x, &win[k - 1]);
		}
	}
	gf_mul(&y, &x, a);    /* y = a*b */
	gf_mul(&x, &y, &x);   /* x = a*b^2 */
	gf_mul2(&x, &x);      /* y = 2*a*b^2 = c */
	gf_sub(&x, &x, &gf_one);
	gf_mul(&x, &y, &x);   /* x = a*b*(c - 1) */
#else
#error unimplemented sqrt for this modulus
#endif

	/* Replace x with -x if x is negative. */
	gf_condneg(&x, &x, gf_is_negative(&x));

	/* Check that x^2 matches a; replace it with zero otherwise. */
	gf_square(&y, &x);
	r = gf_equals(&y, a);
	gf_select(d, &gf_zero, &x, r);
	return r;
}

/* ===================================================================== */
/*
 * SECTION 2: SCALARS
 *
 * A scalar is an integer modulo the group order r. It is internally
 * represented over eight 32-bit limbs. Functions that operate on
 * scalars expect inputs to be fully reduced, and output fully reduced
 * values.
 */
typedef struct {
	uint32_t v[8];
} scalar;

/*
 * R = group order r
 * R0 = |R - 2^254|  (R0 < 2^127)
 * Note: r = 2^254 - R0 for jq255e, r = 2^254 + R0 for jq255s
 */
#if JQ == JQ255E

static const uint32_t R[8] = {
        0x74D84525, 0x1F52C8AE, 0x54078C53, 0x9D0C930F,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x3FFFFFFF
};
static const uint32_t R0[4] = {
        0x8B27BADB, 0xE0AD3751, 0xABF873AC, 0x62F36CF0
};

/* (r-1)/2 */
static const uint32_t HR[8] = {
	0x3A6C2292, 0x8FA96457, 0xAA03C629, 0xCE864987,
	0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x1FFFFFFF
};

#elif JQ == JQ255S

static const uint32_t R[8] = {
        0x396152C7, 0xDCF2AC65, 0x912B7F03, 0x2ACF567A,
	0x00000000, 0x00000000, 0x00000000, 0x40000000
};
static const uint32_t R0[4] = {
        0x396152C7, 0xDCF2AC65, 0x912B7F03, 0x2ACF567A
};

/* 4*r mod 2^256 */
static const uint32_t R_x4[8] = {
	0xE5854B1C, 0x73CAB194, 0x44ADFC0F, 0xAB3D59EA,
	0x00000000, 0x00000000, 0x00000000, 0x00000000
};

#else
#error Unknown curve
#endif

/*
 * If a < r, return 0xFFFFFFFF; otherwise, return 0x00000000.
 * a[] contains 8 limbs.
 */
static uint32_t
modr_is_reduced(const uint32_t *a)
{
	uint32_t cc = 0;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)a[i] - (uint64_t)R[i] - (uint64_t)cc;
		cc = -(uint32_t)(w >> 32);
	}
	return -cc;
}

/*
 * Input: a < 2*r
 * Output: a < r (fully reduced)
 * a[] contains 8 limbs.
 */
static void
modr_inner_reduce(uint32_t *a)
{
	uint32_t z = ~modr_is_reduced(a);
	uint32_t cc = 0;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)a[i] - (uint64_t)(z & R[i])
			- (uint64_t)cc;
		a[i] = (uint32_t)w;
		cc = -(uint32_t)(w >> 32);
	}
}

/*
 * d <- a*b
 * a and b are over 4 limbs each, d is over 8 limbs.
 * WARNING: d[] MUST NOT overlap with either a[] or b[].
 */
static void
mul128x128(uint32_t *d, const uint32_t *a, const uint32_t *b)
{
	memset(d, 0, 8 * sizeof(uint32_t));
	for (int i = 0; i < 4; i ++) {
		uint32_t cc = 0;
		for (int j = 0; j < 4; j ++) {
			uint64_t w = (uint64_t)a[i] * (uint64_t)b[j]
				+ (uint64_t)d[i + j] + (uint64_t)cc;
			d[i + j] = (uint32_t)w;
			cc = (uint32_t)(w >> 32);
		}
		d[i + 4] = cc;
	}
}

/*
 * Internal reduction modulo r functions. For each scalar modulus, there
 * is an internal "partially reduced" format, and a "fully reduced" formats,
 * both over 8 limbs.
 *
 *    modr_reduce256_partial(d, a, ah)
 *        a[]: 8 limbs (input)
 *        ah: uint32_t, less than 2^30 (input)
 *        d[]: 8 limbs (output)
 *        On output, d is partially reduced
 *
 *    modr_reduce384_partial(d, a)
 *        a[]: 12 limbs (input)
 *        d[]: 8 limbs (output)
 *        On output, d is partially reduced
 *
 *    modr_reduce256_finish(d)
 *        d[]: 8 limbs (input/output)
 *        d is partially reduced on input, fully reduced on output
 *
 *    modr_reduce256(d, a, ah):
 *        combines modr_reduce256_partial() and modr_reduce256_finish()
 *
 *    modr_reduce384(d, a):
 *        combines modr_reduce384_partial() and modr_reduce256_finish()
 *
 * Separate implementations are used depending on whether r = 2^254 - r0
 * (e.g. for jq255e) or r = 2^254 + r0 (e.g. for jq255s).
 */

#if JQ == JQ255E

/*
 * For jq255e, r = 2^254 - r0.
 * Partially reduced = value is less than 2^254 + 2^160
 */

static void
modr_reduce256_partial(uint32_t *d, const uint32_t *a, uint32_t ah)
{
	/* r = 2^254 - r0
	   Extract bits 254 to 285 and wrap them. */
	uint32_t cc;

	ah = (ah << 2) | (a[7] >> 30);
	cc = 0;
	for (int i = 0; i < 4; i ++) {
		uint64_t w = (uint64_t)ah * (uint64_t)R0[i]
			+ (uint64_t)a[i] + (uint64_t)cc;
		d[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}
	for (int i = 4; i < 7; i ++) {
		uint64_t w = (uint64_t)a[i] + (uint64_t)cc;
		d[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}
	d[7] = (a[7] & 0x3FFFFFFF) + cc;
}

#define modr_reduce256_finish   modr_inner_reduce

static void
modr_reduce384_partial(uint32_t *d, const uint32_t *a)
{
	uint32_t t[8], cc;

	/* t <- r0*floor(a/2^256) */
	mul128x128(t, a + 8, R0);

	/* t <- 4*t + (a mod 2^256) */
	cc = 0;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = ((uint64_t)t[i] << 2) + (uint64_t)a[i]
			+ (uint64_t)cc;
		t[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}

	/* Partial reduction. */
	modr_reduce256_partial(d, t, cc);
}

/*
 * d <- a*b mod 2^128
 * a, b and d are over 4 limbs each.
 * WARNING: d[] MUST NOT overlap with either a[] or b[].
 */
static void
mul128x128trunc(uint32_t *d, const uint32_t *a, const uint32_t *b)
{
	memset(d, 0, 4 * sizeof(uint32_t));
	for (int i = 0; i < 4; i ++) {
		uint32_t cc = 0;
		for (int j = 0; j < (4 - i); j ++) {
			uint64_t w = (uint64_t)a[i] * (uint64_t)b[j]
				+ (uint64_t)d[i + j] + (uint64_t)cc;
			d[i + j] = (uint32_t)w;
			cc = (uint32_t)(w >> 32);
		}
	}
}

#elif JQ == JQ255S

/*
 * For jq255s, r = 2^254 + r0
 * Functions *_partial() already output fully reduced values, so the
 * "finish" function is a no-op.
 */

static void
modr_reduce256_finish(uint32_t *d)
{
	(void)d;
}

static void
modr_reduce256_partial(uint32_t *d, const uint32_t *a, uint32_t ah)
{
	/* r = 2^254 + r0
	   Extract bits 254 to 285 and wrap them. */
	uint32_t cc1, cc2;
	uint64_t w;

	ah = (ah << 2) | (a[7] >> 30);
	cc1 = 0;
	cc2 = 0;
	for (int i = 0; i < 4; i ++) {
		w = (uint64_t)ah * (uint64_t)R0[i] + (uint64_t)cc1;
		cc1 = (uint32_t)(w >> 32);
		w = (uint64_t)a[i] - (uint64_t)(uint32_t)w - (uint64_t)cc2;
		d[i] = (uint32_t)w;
		cc2 = -(uint32_t)(w >> 32);
	}
	w = (uint64_t)a[4] - (uint64_t)cc1 - (uint64_t)cc2;
	d[4] = (uint32_t)w;
	cc2 = -(uint32_t)(w >> 32);
	for (int i = 5; i < 7; i ++) {
		w = (uint64_t)a[i] - (uint64_t)cc2;
		d[i] = (uint32_t)w;
		cc2 = -(uint32_t)(w >> 32);
	}
	w = (uint64_t)(a[7] & 0x3FFFFFFF) - (uint64_t)cc2;
	d[7] = (uint32_t)w;
	uint32_t z = (uint32_t)(w >> 32);

	/* If there is a borrow (z == 0xFFFFFFFF) then we have to add r (a
	   single addition is enough since we subtracted a value lower
	   than 2^192). */
	cc1 = 0;
	for (int i = 0; i < 8; i ++) {
		w = (uint64_t)d[i] + (uint64_t)(z & R[i]) + (uint64_t)cc1;
		d[i] = (uint32_t)w;
		cc1 = (uint32_t)(w >> 32);
	}
}

static void
modr_reduce384_partial(uint32_t *d, const uint32_t *a)
{
	uint32_t t[8], cc1, cc2;

	/* t <- 4*r0*floor(a/2^256)
	   Since r0 < 2^126, the result fits on 8 limbs. */
	mul128x128(t, a + 8, R_x4);

	/* t <- 4*r + (a mod 2^256) - t */
	cc1 = 0;
	cc2 = 0;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)R_x4[i] - (uint64_t)t[i] - (uint64_t)cc1;
		cc1 = -(uint32_t)(w >> 32);
		w = (uint64_t)(uint32_t)w + (uint64_t)a[i] + (uint64_t)cc2;
		t[i] = (uint32_t)w;
		cc2 = (uint32_t)(w >> 32);
	}

	/* cc1 = borrow from subtraction
	   cc2 = carry from addition
	   We also need to add 2^256 (since 4*r = 2^256 + R_x4). */
	modr_reduce256_partial(d, t, cc2 + 1 - cc1);
}

#else
#error Unknown curve
#endif

static const scalar scalar_zero = { { 0, 0, 0, 0, 0, 0, 0, 0 } };
static const scalar scalar_one = { { 1, 0, 0, 0, 0, 0, 0, 0 } };

/*
 * Return 0xFFFFFFFF is d is zero; return 0x00000000 otherwise.
 */
static uint32_t
scalar_is_zero(const scalar *d)
{
	/*
	 * Since scalars are kept in fully reduced format, we just have
	 * to check whether all limbs are zero or not.
	 */
	uint32_t r = d->v[0];
	for (int i = 1; i < 8; i ++) {
		r |= d->v[i];
	}
	return ((r | -r) >> 31) - 1;
}

/*
 * Encode a scalar into exactly 32 bytes. Encoding uses the unsigned
 * little-endian convention and is always canonical (the encoded integer
 * is always in the 0 to r-1 range).
 */
static void
scalar_encode(void *dst, const scalar *a)
{
	uint8_t *buf = dst;
	for (int i = 0; i < 8; i ++) {
		enc32le(buf + 4 * i, a->v[i]);
	}
}

/*
 * Decode a scalar from exactly 32 bytes. Returned value is 0xFFFFFFFF on
 * success (the bytes encoded an integer in the proper 0 to r-1 range),
 * or 0x00000000 otherwise.
 */
static uint32_t
scalar_decode(scalar *d, const void *src)
{
	const uint8_t *buf = src;
	uint32_t r;

	for (int i = 0; i < 8; i ++) {
		d->v[i] = dec32le(buf + 4 * i);
	}
	r = modr_is_reduced(d->v);
	for (int i = 0; i < 8; i ++) {
		d->v[i] &= r;
	}
	return r;
}

/*
 * Decode a scalar by interpreting the provided bytes in unsigned
 * little-endian convention, and reducing the integer modulo r. This
 * process never fails.
 */
static void
scalar_decode_reduce(scalar *d, const void *src, size_t len)
{
	const uint8_t *buf = src;
	size_t clen;

	/* Input lengths of less than 32 bytes are easy; they don't need
	   any reduction. */
	if (len < 32) {
		uint8_t tmp[32];
		memcpy(tmp, buf, len);
		memset(tmp + len, 0, 32 - len);
		for (int i = 0; i < 8; i ++) {
			d->v[i] = dec32le(tmp + 4 * i);
		}
		return;
	}

	/* Decode up to 17 to 32 bytes, to get a length multiple of 16. */
	clen = 17 + ((len - 1) & 15);
	len -= clen;
	if (clen < 32) {
		uint8_t tmp[32];
		memcpy(tmp, buf + len, clen);
		memset(tmp + clen, 0, 32 - clen);
		for (int i = 0; i < 8; i ++) {
			d->v[i] = dec32le(tmp + 4 * i);
		}
		/* Smaller than 32 bytes -> no reduction needed. */
	} else {
		for (int i = 0; i < 8; i ++) {
			d->v[i] = dec32le(buf + len + 4 * i);
		}
		modr_reduce256_partial(d->v, d->v, 0);
	}

	/* Process all other chunks of 16 bytes. */
	while (len > 0) {
		uint32_t t[12];
		len -= 16;
		for (int i = 0; i < 4; i ++) {
			t[i] = dec32le(buf + len + 4 * i);
		}
		memcpy(&t[4], &d->v, 8 * sizeof(uint32_t));
		modr_reduce384_partial(d->v, t);
	}

	/* Finish reduction. */
	modr_reduce256_finish(d->v);
}

/*
 * If ctl == 0x00000000, then d <- a0
 * If ctl == 0xFFFFFFFF, then d <- a1
 * ctl MUST be either 0x00000000 or 0xFFFFFFFF.
 */
static void
scalar_select(scalar *d, const scalar *a0, const scalar *a1, uint32_t ctl)
{
	for (int i = 0; i < 8; i ++) {
		d->v[i] = a0->v[i] ^ (ctl & (a0->v[i] ^ a1->v[i]));
	}
}

/* d <- a + b */
static void
scalar_add(scalar *d, const scalar *a, const scalar *b)
{
	uint32_t cc = 0;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)a->v[i] + (uint64_t)b->v[i]
			+ (uint64_t)cc;
		d->v[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}
	modr_inner_reduce(d->v);
}

/* d <- a*b */
static void
scalar_mul(scalar *d, const scalar *a, const scalar *b)
{
	uint32_t t[16];

	/* Compute integer product over 512 bits. */
	memset(t, 0, sizeof t);
	for (int i = 0; i < 8; i ++) {
		uint32_t cc = 0;
		for (int j = 0; j < 8; j ++) {
			uint64_t w = (uint64_t)a->v[i] * (uint64_t)b->v[j]
				+ (uint64_t)t[i + j] + (uint64_t)cc;
			t[i + j] = (uint32_t)w;
			cc = (uint32_t)(w >> 32);
		}
		t[i + 8] = cc;
	}

	/* Reduce. */
	modr_reduce384_partial(t + 4, t + 4);
	modr_reduce384_partial(d->v, t);
	modr_reduce256_finish(d->v);
}

/*
 * Recode an integer: from s, return digits s_i such that:
 *   -15 <= s_i <= +16
 *   s = \sum_{i=0}^{...} s_i * 2^(5*i)
 * Exactly sd_len digits are produced.
 * Assumptions:
 *    sd_len > 0
 *    s contains at least ceil(5*sd_len/32) limbs
 */
static void
uint_recode(int8_t *sd, size_t sd_len, const uint32_t *s)
{
	uint32_t acc = s[0];
	int acc_len = 32;
	int j = 1;
	uint32_t cc = 0;

	/*
	 * We explore bits by chunks of 5. If v is the value of the chunk,
	 * and c is the carry (0 or 1), then:
	 *   v' <- v + c
	 *   if v' <= 16, then produce v', and set c to 0
	 *   otherwise, product v' - 32, and set c to 1
	 */
	for (size_t i = 0; i < sd_len; i ++) {
		uint32_t v;
		if (acc_len < 5) {
			uint32_t nw = s[j ++];
			v = (acc | (nw << acc_len)) & 0x1F;
			acc = nw >> (5 - acc_len);
			acc_len += 27;
		} else {
			v = acc & 0x1F;
			acc_len -= 5;
			acc >>= 5;
		}
		v += cc;
		cc = (16 - v) >> 31;
		sd[i] = (int8_t)v - ((int8_t)cc << 5);
	}
}

/*
 * Recode a scalar: from s, return digits s_i for i = 0 to 50. Rules:
 *   -15 <= s_i <= +16
 *   s = \sum_{i=0}^{50} s_i * 2^(5*i)
 * Top digit s_50 is in the 0 to +16 range.
 */
static void
scalar_recode(int8_t *sd, const scalar *s)
{
	/*
	 * Since r is very close to 2^254, top chunk can have v = 16 only
	 * if the previous chunk had v = 0, in which case there won't be
	 * any input carry; otherwise, top chunk is v <= 15 and even if
	 * there is an input carry we'll just get at most 16, and there
	 * won't be an output carry. Thus, 51 digits are enough to cover
	 * the whole scalar range.
	 *
	 * 51*5 = 255 < 8*32, therefore we have enough limbs as well.
	 */
	uint_recode(sd, 51, s->v);
}

/*
 * Recode a big integer in wNAF. Exactly `sd_len` digits are produced,
 * out of the integer `u` consisting of `u_len` limbs. Each digit has
 * value either 0, or an odd integer between -15 and +15.
 * Warning: u_len MUST be greater than 0.
 *
 * wNAF representation is such that:
 *   u = \sum_{i=0}^{...} sd_i*2^i
 *   non-zero digits are separated by at least 5 zeros
 */
static void
uint_recode_wNAF(int8_t *sd, size_t sd_len, const uint32_t *u, size_t u_len)
{
	size_t lim = (u_len << 5) - 4;
	uint32_t x = u[0] & 0x0000FFFF;
	for (size_t j = 0; j < sd_len; j ++) {
		if ((j & 31) == 12) {
			if (j < lim) {
				x += ((*u ++) & 0xFFFF0000) >> 12;
			}
		} else if ((j & 31) == 28) {
			if (j < lim) {
				x += (*u & 0x0000FFFF) << 4;
			}
		}
		uint32_t m = -(x & 1);
		uint32_t v = x & m & 0x1F;
		uint32_t c = (v & 0x10) << 1;
		int32_t d = (int32_t)v - (int32_t)c;
		sd[j] = (int8_t)d;
		x = (x - (uint32_t)d) >> 1;
	}
}

/*
 * Recode a scalar in wNAF. 256 digits are produced.
 */
static void
scalar_recode_wNAF(int8_t *sd, const scalar *s)
{
	uint_recode_wNAF(sd, 256, s->v, 8);
}

#if JQ == JQ255E

/*
 * d <- a - b
 * a, b and d are over 4 limbs
 * d may be equal to a and/or b, but partial overlap is forbidden.
 */
static void
sub128(uint32_t *d, const uint32_t *a, const uint32_t *b)
{
	uint32_t cc = 0;
	for (int i = 0; i < 4; i ++) {
		uint64_t w = (uint64_t)a[i] - (uint64_t)b[i] - (uint64_t)cc;
		d[i] = (uint32_t)w;
		cc = -(uint32_t)(w >> 32);
	}
}

/*
 * Input:
 *   k < r         (8 limbs)
 *   e < 2^127 -2  (4 limbs)
 * Output:
 *   d = round(k*e / r)  (4 limbs)
 */
static void
mul_divr_rounded(uint32_t *d, const uint32_t *k, const uint32_t *e)
{
	uint32_t z[12], y[4], t[8];
	uint32_t cc;

	/* z <- k*e */
	memset(z, 0, sizeof z);
	for (int i = 0; i < 4; i ++) {
		cc = 0;
		for (int j = 0; j < 8; j ++) {
			uint64_t w = (uint64_t)e[i] * (uint64_t)k[j]
				+ (uint64_t)z[i + j] + (uint64_t)cc;
			z[i + j] = (uint32_t)w;
			cc = (uint32_t)(w >> 32);
		}
		z[i + 8] = cc;
	}

	/* z <- z + (r-1)/2 */
	cc = 0;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)z[i] + (uint64_t)HR[i] + (uint64_t)cc;
		z[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}
	for (int i = 8; i < 12; i ++) {
		uint64_t w = (uint64_t)z[i] + (uint64_t)cc;
		z[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}

	/* y <- floor(z / 2^254) + 1 */
	cc = 1;
	for (int i = 0; i < 4; i ++) {
		uint32_t yw = (z[i + 7] >> 30) | (z[i + 8] << 2);
		uint64_t w = (uint64_t)yw + (uint64_t)cc;
		y[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}

	/* t <- y*r0 */
	mul128x128(t, y, R0);

	/* Compute high limb of t + z0 */
	z[7] &= 0x3FFFFFFF;
	cc = 0;
	uint32_t hi;
	for (int i = 0; i < 8; i ++) {
		uint64_t w = (uint64_t)z[i] + (uint64_t)t[i] + (uint64_t)cc;
		hi = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}

	/* High limb is lower than 2^31. If it is lower than 2^30, then y
	   is too large and we must decrement it; otherwise, we keep it
	   unchanged. */
	cc = 1 - (hi >> 30);
	for (int i = 0; i < 4; i ++) {
		uint64_t w = (uint64_t)y[i] - (uint64_t)cc;
		d[i] = (uint32_t)w;
		cc = -(uint32_t)(w >> 32);
	}
}

/*
 * d <- abs(d)
 * d is over four limbs; on input, it is assumed to use two's complement
 * representation.
 * Returned value is the original sign of d (1 = negative, 0 = non-negative).
 */
static uint32_t
abs128(uint32_t *d)
{
	uint32_t s = d[3] >> 31;
	uint32_t m = -s;
	uint32_t cc = s;
	for (int i = 0; i < 4; i ++) {
		uint64_t w = (uint64_t)(d[i] ^ m) + (uint64_t)cc;
		d[i] = (uint32_t)w;
		cc = (uint32_t)(w >> 32);
	}
	return s;
}

/*
 * Split scalar k into k0 and k1, such that k = k0 + k1*mu mod r (with
 * mu being a specific square root of -1 modulo r).
 *
 * k0 and k1 are returned as their absolute values (less than 2^127
 * each, over 4 limbs) and their signs; the signs are combined into
 * bits 0 and 1 of the returned value (1 = negative, 0 = non-negative).
 */
static uint32_t
scalar_split(uint32_t *k0, uint32_t *k1, const scalar *k)
{
	static const uint32_t eU[] = {
		0xC93F6111, 0x2ACCF9DE, 0x53C2C6E6, 0x1A509F7A
	};
	static const uint32_t eV[] = {
		0x5466F77E, 0x0B7A3130, 0xFFBB3A93, 0x7D440C6A
	};

	uint32_t c[4], d[4], t[4];
	uint32_t r;

	/*
	 * c = round(k*v/r)
	 * d = round(k*u/r)
	 */
	mul_divr_rounded(c, k->v, eV);
	mul_divr_rounded(d, k->v, eU);

	/*
	 * k0 = k - d*u - c*v
	 * k1 = d*v - c*u
	 */
	mul128x128trunc(t, d, eU);
	sub128(k0, k->v, t);
	mul128x128trunc(t, c, eV);
	sub128(k0, k0, t);
	r = abs128(k0);

	mul128x128trunc(k1, d, eV);
	mul128x128trunc(t, c, eU);
	sub128(k1, k1, t);
	r |= abs128(k1) << 1;

	return r;
}

#endif

/* ===================================================================== */
/*
 * SECTION 3: CURVE
 *
 * A group element is represented by a point, which uses the extended
 * (E:Z:U:T) representation. Rules:
 *   E != 0
 *   Z != 0
 *   E^2*Z^2 = (a^2-4*b)*U^4 - 2*a*U^2*Z^2 + Z^4
 *   U^2 = T*Z
 * A point in (extended) affine coordinates has Z == 1.
 */

typedef struct {
	gf E, Z, U, T;
} point;

typedef struct {
	gf E, U, T;
} point_affine;

#if 0
/* unused */
/*
 * Conventional base point.
 */
static const point point_base = {
#if JQ == JQ255E
	/* Standard is (-3,-1), but (3,1) is a valid representant of
	   the same point. */
	LGF(3, 0, 0, 0, 0, 0, 0, 0),
	LGF(1, 0, 0, 0, 0, 0, 0, 0),
	LGF(1, 0, 0, 0, 0, 0, 0, 0),
	LGF(1, 0, 0, 0, 0, 0, 0, 0)
#elif JQ == JQ255S
	LGF(0xA2789410, 0x104220CD, 0x348CC437, 0x6D7386B2,
	    0x4612D10E, 0x55E452A6, 0xA747ADAC, 0x0F520B1B),
	LGF(1, 0, 0, 0, 0, 0, 0, 0),
	LGF(3, 0, 0, 0, 0, 0, 0, 0),
	LGF(9, 0, 0, 0, 0, 0, 0, 0)
#else
#error Unknown curve
#endif
};
#endif

/*
 * Neutral element, in extended and affine coordinates.
 */
static const point point_neutral = {
	LGF(1, 0, 0, 0, 0, 0, 0, 0),
	LGF(1, 0, 0, 0, 0, 0, 0, 0),
	LGF(0, 0, 0, 0, 0, 0, 0, 0),
	LGF(0, 0, 0, 0, 0, 0, 0, 0)
};
static const point_affine point_affine_neutral = {
	LGF(1, 0, 0, 0, 0, 0, 0, 0),
	LGF(0, 0, 0, 0, 0, 0, 0, 0),
	LGF(0, 0, 0, 0, 0, 0, 0, 0)
};

/*
 * Decode a point from exactly 32 bytes. If the source is a validly
 * encoded point, then d is set to that point, and 0xFFFFFFFF is
 * returned; otherwise, d is set to the neutral, and 0x00000000 is
 * returned.
 *
 * Note: decoding returns a point in affine coordinates (Z == 1),
 * and the e coordinate is non-negative.
 */
static uint32_t
point_decode(point *d, const void *src)
{
	gf e, u, ee, uu;
	uint32_t r;

	/* Decode the source as the field element u. */
	r = gf_decode(&u, src);

	/* Compute ee = (a^2-4*b)*u^4 - 2*a*u^2 + 1. */
	gf_square(&uu, &u);
	gf_square(&ee, &uu);
#if JQ == JQ255E
	/* jq255e: a = 0, b = -2 */
	gf_lsh(&ee, &ee, 3);
#elif JQ == JQ255S
	/* jq255e: a = -1, b = 1/2 */
	gf_sub(&ee, &uu, &ee);
	gf_add(&ee, &ee, &uu);
#else
#error Unknown curve
#endif
	gf_add(&ee, &ee, &gf_one);

	/* Extract e as a square root of ee. The gf_sqrt() function already
	   takes care to return the non-negative root. */
	r &= gf_sqrt(&e, &ee);

	/* Set d to (e:1:u:u^2) on success, to (-1:1:0:0) on error. */
	gf_select(&d->E, &gf_minus_one, &e, r);
	d->Z = gf_one;
	gf_select(&d->U, &gf_zero, &u, r);
	gf_select(&d->T, &gf_zero, &uu, r);
	return r;
}

/*
 * Encode a point p into exactly 32 bytes.
 */
static void
point_encode(void *dst, const point *p)
{
	gf iZ, e, u;

	/* Get the affine (e,u) coordinates. If e is negative, then
	   choose the other representant P+N = (-e,-u). */
	gf_inv(&iZ, &p->Z);
	gf_mul(&e, &p->E, &iZ);
	gf_mul(&u, &p->U, &iZ);
	gf_condneg(&u, &u, gf_is_negative(&e));
	gf_encode(dst, &u);
}

/*
 * Add two points together: P3 <- P1 + P2.
 */
static void
point_add(point *p3, const point *p1, const point *p2)
{
	const gf *E1 = &p1->E, *Z1 = &p1->Z, *U1 = &p1->U, *T1 = &p1->T;
	const gf *E2 = &p2->E, *Z2 = &p2->Z, *U2 = &p2->U, *T2 = &p2->T;
	gf *E3 = &p3->E, *Z3 = &p3->Z, *U3 = &p3->U, *T3 = &p3->T;
	gf e1e2, u1u2, z1z2, t1t2, eu, zt, hd, g1, g2, g3;

	gf_mul(&e1e2, E1, E2);  /* e1e2 <- E1*E2 */
	gf_mul(&u1u2, U1, U2);  /* u1u2 <- U1*U2 */
	gf_mul(&z1z2, Z1, Z2);  /* z1z2 <- Z1*Z2 */
	gf_mul(&t1t2, T1, T2);  /* t1t2 <- T1*T2 */

	/* eu <- E1*U2 + E2*U1 */
	gf_add(&g1, E1, U1);
	gf_add(&g2, E2, U2);
	gf_mul(&eu, &g1, &g2);
	gf_add(&g3, &e1e2, &u1u2);
	gf_sub(&eu, &eu, &g3);

	/* zt <- Z1*T2 + Z2*T1 */
	gf_add(&g1, Z1, T1);
	gf_add(&g2, Z2, T2);
	gf_mul(&zt, &g1, &g2);
	gf_add(&g3, &z1z2, &t1t2);
	gf_sub(&zt, &zt, &g3);

	/* hd <- z1z2 - b'*t1t2
	   E3 <- (z1z2 + b'*t1t2)*(e1e2 + a'*u1u2) + 2*b'*u1u2*zt
	   with a' = -2*a and b' = a^2-4*b */
#if JQ == JQ255E
	/* jq255e: a' = 0, b' = 8 */
	gf_lsh(&g1, &t1t2, 3);
	gf_sub(&hd, &z1z2, &g1);
	gf_add(&g1, &z1z2, &g1);
	gf_mul(&g1, &g1, &e1e2);
	gf_lsh(&g2, &u1u2, 4);
	gf_mul(&g2, &g2, &zt);
	gf_add(E3, &g1, &g2);
#elif JQ == JQ255S
	/* jq255e: a' = 2, b' = -1 */
	gf_add(&hd, &z1z2, &t1t2);
	gf_sub(&g1, &z1z2, &t1t2);
	gf_mul2(&g2, &u1u2);
	gf_add(&g3, &e1e2, &g2);
	gf_mul(&g1, &g3, &g1);
	gf_mul(&g2, &g2, &zt);
	gf_sub(E3, &g1, &g2);
#else
#error Unknown curve
#endif

	gf_square(Z3, &hd);  /* Z3 <- hd^2 */
	gf_square(T3, &eu);  /* T3 <- eu^2 */

	/* U3 <- hd*eu = ((hd + eu)^2 - hd^2 - eu^2)/2
	   A direct multiply may be faster than using a square here,
	   depending on the relative speed of mul, square, add, sub
	   and half. */
	gf_add(&g1, &hd, &eu);
	gf_square(&g1, &g1);
	gf_add(&g2, Z3, T3);
	gf_sub(&g1, &g1, &g2);
	gf_half(U3, &g1);
}

/*
 * Add two points together: P3 <- P1 + P2.
 * Point P2 is in extended affine coordinates (i.e. Z2 == 1).
 */
static void
point_add_affine(point *p3, const point *p1, const point_affine *p2)
{
	const gf *E1 = &p1->E, *Z1 = &p1->Z, *U1 = &p1->U, *T1 = &p1->T;
	const gf *E2 = &p2->E, /* Z2 == 1 */ *U2 = &p2->U, *T2 = &p2->T;
	gf *E3 = &p3->E, *Z3 = &p3->Z, *U3 = &p3->U, *T3 = &p3->T;
	gf e1e2, u1u2, /* z1z2 == Z1 */ t1t2, eu, zt, hd, g1, g2, g3;

	gf_mul(&e1e2, E1, E2);  /* e1e2 <- E1*E2 */
	/* skip z1z2, since Z2 == 1 by assumption */
	gf_mul(&u1u2, U1, U2);  /* u1u2 <- U1*U2 */
	gf_mul(&t1t2, T1, T2);  /* t1t2 <- T1*T2 */

	/* eu <- E1*U2 + E2*U1 */
	gf_add(&g1, E1, U1);
	gf_add(&g2, E2, U2);
	gf_mul(&eu, &g1, &g2);
	gf_add(&g3, &e1e2, &u1u2);
	gf_sub(&eu, &eu, &g3);

	/* zt <- Z1*T2 + Z2*T1
	   By the assumption Z2 == 1, we have zt = Z1*T2 + T1 */
	gf_mul(&g1, Z1, T2);
	gf_add(&zt, &g1, T1);

	/* hd <- z1z2 - b'*t1t2
	   E3 <- (z1z2 + b'*t1t2)*(e1e2 + a'*u1u2) + 2*b'*u1u2*zt
	   with a' = -2*a and b' = a^2-4*b */
#if JQ == JQ255E
	/* jq255e: a' = 0, b' = 8 */
	gf_lsh(&g1, &t1t2, 3);
	gf_sub(&hd, Z1, &g1);
	gf_add(&g1, Z1, &g1);
	gf_mul(&g1, &g1, &e1e2);
	gf_lsh(&g2, &u1u2, 4);
	gf_mul(&g2, &g2, &zt);
	gf_add(E3, &g1, &g2);
#elif JQ == JQ255S
	/* jq255e: a' = 2, b' = -1 */
	gf_add(&hd, Z1, &t1t2);
	gf_sub(&g1, Z1, &t1t2);
	gf_mul2(&g2, &u1u2);
	gf_add(&g3, &e1e2, &g2);
	gf_mul(&g1, &g3, &g1);
	gf_mul(&g2, &g2, &zt);
	gf_sub(E3, &g1, &g2);
#else
#error Unknown curve
#endif

	gf_square(Z3, &hd);  /* Z3 <- hd^2 */
	gf_square(T3, &eu);  /* T3 <- eu^2 */

	/* U3 <- hd*eu = ((hd + eu)^2 - hd^2 - eu^2)/2
	   A direct multiply may be faster than using a square here,
	   depending on the relative speed of mul, square, add, sub
	   and half. */
	gf_add(&g1, &hd, &eu);
	gf_square(&g1, &g1);
	gf_add(&g2, Z3, T3);
	gf_sub(&g1, &g1, &g2);
	gf_half(U3, &g1);
}

/*
 * Negate a point: P2 <- -P1
 */
static void
point_neg(point *p2, const point *p1)
{
	p2->E = p1->E;
	p2->Z = p1->Z;
	gf_neg(&p2->U, &p1->U);
	p2->T = p1->T;
}

/*
 * Subtract points: P3 <- P1 - P2
 */
static void
point_sub(point *p3, const point *p1, const point *p2)
{
	point np2;
	point_neg(&np2, p2);
	point_add(p3, p1, &np2);
}

/*
 * Subtract points: P3 <- P1 - P2
 * Point P2 is in extended affine coordinates (i.e. Z2 == 1).
 */
static void
point_sub_affine(point *p3, const point *p1, const point_affine *p2)
{
	point_affine np2;
	np2.E = p2->E;
	gf_neg(&np2.U, &p2->U);
	np2.T = p2->T;
	point_add_affine(p3, p1, &np2);
}

/*
 * Double a point repeatedly: d <- 2^n*P
 */
static void
point_xdouble(point *d, const point *p, unsigned n)
{
	if (n == 0) {
		if (d != p) {
			*d = *p;
		}
		return;
	}

#if JQ == JQ255E
	gf X, W, J, g1, g2;

	/*
	 * First doubling: P ezut -> 2*P xwj
	 *   ee = E^2
	 *   X' = ee^2
	 *   W' = 2*Z^2 - ee
	 *   J' = 2*E*U
	 */

	gf_square(&g1, &p->E);
	gf_mul(&J, &p->E, &p->U);
	gf_square(&X, &g1);
	gf_square(&W, &p->Z);
	gf_mul2(&J, &J);
	gf_mul2(&W, &W);
	gf_sub(&W, &W, &g1);

	/* extra doublings in xwj coordinates */
	while (n -- > 1) {
		/*
		 * ww = W^2
		 * t1 = ww - 2*X
		 * t2 = t1^2
		 * J' = ((W + t1)^2 - ww - t2)*J  # Or: J' = 2*W*t1*J
		 * W' = t2 - 2*ww^2
		 * X' = t2^2
		 */
		gf ww;

		gf_square(&ww, &W);
		gf_mul2(&g1, &X);
		gf_sub(&g1, &ww, &g1);
		gf_square(&g2, &g1);

		/* J' <- 2*W*t1*J = ((W + t1)^2 - ww - t2)*J
		   Using a W*t1 multiplication may be faster,
		   depending on the relative speeds of mul, square,
		   add and sub. */
		gf_mul(&g1, &g1, &W);
		gf_mul2(&g1, &g1);
		/* 
		gf_add(&g1, &g1, &W);
		gf_square(&g1, &g1);
		gf_sub(&g1, &g1, &ww);
		gf_sub(&g1, &g1, &g2);
		*/

		gf_mul(&J, &J, &g1);

		/* W' <- t2 - 2*ww^2 */
		gf_square(&ww, &ww);
		gf_mul2(&ww, &ww);
		gf_sub(&W, &g2, &ww);

		/* X' <- t2^2 */
		gf_square(&X, &g2);
	}

	/*
	 * Conversion xwj -> ezut
	 *   Z = W^2
	 *   T = J^2
	 *   U = W*J = ((W + J)^2 - Z - T)/2
	 *   E = 2*X - Z
	 */

	gf_square(&d->Z, &W);
	gf_square(&d->T, &J);

	/* U <- W*J = ((W + J)^2 - ww - jj)^2
	   A direct multiplication may be faster, depending on the
	   relative speeds of mul, square, add, sub and half. */
	gf_mul(&d->U, &W, &J);
	/*
	gf_add(&g2, &d->Z, &d->T);
	gf_add(&g1, &W, &J);
	gf_square(&g1, &g1);
	gf_sub(&g1, &g1, &g2);
	gf_half(&d->U, &g1);
	*/

	gf_mul2(&g1, &X);
	gf_sub(&d->E, &g1, &d->Z);

#elif JQ == JQ255S
	gf X, W, J, g1, g2, g3;

	/*
	 * First doubling: P ezut -> 2*P+N xwj
	 * Note: adding +N is fine in the group, since that only moves
	 * to another representant of the same group element.
	 *   uu = U^2
	 *   X' = 8*(uu^2)
	 *   W' = 2*uu - (T + Z)^2
	 *   J' = 2*E*U
	 */

	/* uu <- U^2 */
	gf_square(&g1, &p->U);

	/* J <- 2*E*U */
	gf_mul(&J, &p->E, &p->U);
	gf_mul2(&J, &J);

	/* X <- 8*(uu^2) */
	gf_square(&X, &g1);
	gf_lsh(&X, &X, 3);

	/* W <- 2*uu - (T + Z)^2 */
	gf_add(&g2, &p->T, &p->Z);
	gf_mul2(&W, &g1);
	gf_square(&g2, &g2);
	gf_sub(&W, &W, &g2);

	/* extra doublings in xwj coordinates */
	while (n -- > 1) {
		/*
		 * t1 = W*J
		 * t2 = t1^2
		 * X' = 2*t2^2
		 * t3 = (W + J)^2 - 2*t1
		 * W' = t2 - (t3^2)/2
		 * J' = t1*(2*X - t2)
		 */

		gf_mul(&g1, &W, &J);
		gf_add(&g3, &W, &J);
		gf_mul2(&g2, &g1);
		gf_square(&g3, &g3);
		gf_mul2(&J, &X);
		gf_sub(&g3, &g3, &g2);
		gf_sub(&J, &J, &g3);
		gf_square(&g2, &g1);
		gf_mul(&J, &J, &g1);
		gf_square(&g3, &g3);
		gf_square(&X, &g2);
		gf_half(&g3, &g3);
		gf_mul2(&X, &X);
		gf_sub(&W, &g2, &g3);
	}

	/*
	 * Conversion xwj -> ezut
	 *   Z = W^2
	 *   T = J^2
	 *   U = W*J = ((W + J)^2 - Z - T)/2
	 *   E = 2*X - Z - T
	 */
	gf_square(&d->Z, &W);
	gf_square(&d->T, &J);

	/* U <- W*J = ((W + J)^2 - ww - jj)/2
	   A direct multiplication may be faster, depending on the
	   relative speeds of mul, square, add, sub and half. */
	gf_mul(&d->U, &W, &J);
	/*
	gf_add(&g2, &d->Z, &d->T);
	gf_add(&g1, &W, &J);
	gf_square(&g1, &g1);
	gf_sub(&g1, &g1, &g2);
	gf_half(&d->U, &g1);
	*/

	gf_mul2(&g1, &X);
	gf_sub(&g1, &g1, &d->Z);
	gf_sub(&d->E, &g1, &d->T);

#else
#error Unknown curve
#endif
}

/*
 * Double a point: d <- 2*P
 */
static void
point_double(point *d, const point *p)
{
	point_xdouble(d, p, 1);
}

/*
 * Test whether a point is the group neutral.
 * Output: 0xFFFFFFFF is the point is the neutral, 0x00000000 otherwise.
 */
static uint32_t
point_is_neutral(const point *p)
{
	return gf_is_zero(&p->U);
}

#if 0
/* unused */
/*
 * Test whether two points represent the same group element.
 * Output: 0xFFFFFFFF is the two group elements are equal to each other,
 * 0x00000000 otherwise.
 */
static uint32_t
point_equals(const point *p1, const point *p2)
{
	gf g1, g2;

	gf_mul(&g1, &p1->U, &p2->E);
	gf_mul(&g2, &p1->E, &p2->U);
	return gf_equals(&g1, &g2);
}
#endif

/*
 * If ctl == 0x00000000, then d <- p0.
 * If ctl == 0xFFFFFFFF, then d <- p1.
 * ctl MUST be either 0x00000000 or 0xFFFFFFFF.
 */
static void
point_select(point *d, const point *p0, const point *p1, uint32_t ctl)
{
	gf_select(&d->E, &p0->E, &p1->E, ctl);
	gf_select(&d->Z, &p0->Z, &p1->Z, ctl);
	gf_select(&d->U, &p0->U, &p1->U, ctl);
	gf_select(&d->T, &p0->T, &p1->T, ctl);
}

/*
 * Lookup a point from a window, with sign management.
 * Input:
 *   win[i] = (i+1)*P1
 *   -16 <= k <= 16
 * Output:
 *   P2 <- k*P1
 */
static void
point_lookup(point *p2, const point *win, int8_t k)
{
	/*
	 * Get abs(k) but retain the sign.
	 */
	uint32_t m = (uint32_t)(uint8_t)k;
	uint32_t sk = -(m >> 7);
	m = ((m ^ sk) - sk) & 0xFF;

	/*
	 * Constant-time lookup through the window.
	 */
	*p2 = point_neutral;
	for (uint32_t j = 0; j < 16; j ++) {
		uint32_t c = m - j - 1;
		c = ((c | -c) >> 31) - 1;
		gf_select(&p2->E, &p2->E, &win[j].E, c);
		gf_select(&p2->Z, &p2->Z, &win[j].Z, c);
		gf_select(&p2->U, &p2->U, &win[j].U, c);
		gf_select(&p2->T, &p2->T, &win[j].T, c);
	}

	/*
	 * Adjust the sign.
	 */
	gf_condneg(&p2->U, &p2->U, sk);
}

/*
 * Lookup a point from a window (in affine coordinates), with sign management.
 * Input:
 *   win[i] = (i+1)*P1
 *   -16 <= k <= 16
 * Output:
 *   P2 <- k*P1
 */
static void
point_affine_lookup(point_affine *p2, const point_affine *win, int8_t k)
{
	/*
	 * Get abs(k) but retain the sign.
	 */
	uint32_t m = (uint32_t)(uint8_t)k;
	uint32_t sk = -(m >> 7);
	m = ((m ^ sk) - sk) & 0xFF;

	/*
	 * Constant-time lookup through the window.
	 */
	*p2 = point_affine_neutral;
	for (uint32_t j = 0; j < 16; j ++) {
		uint32_t c = m - j - 1;
		c = ((c | -c) >> 31) - 1;
		gf_select(&p2->E, &p2->E, &win[j].E, c);
		gf_select(&p2->U, &p2->U, &win[j].U, c);
		gf_select(&p2->T, &p2->T, &win[j].T, c);
	}

	/*
	 * Adjust the sign.
	 */
	gf_condneg(&p2->U, &p2->U, sk);
}

/*
 * Point multiplication by a scalar (general case).
 * P2 <- s*P1
 */
static void
point_mul(point *p2, const point *p1, const scalar *s)
{
#if JQ == JQ255E
	/*
	 * On jq255e, we use the endomorphism to split the scalar s
	 * into two half-width (signed) scalars k0 and k1. The
	 * endomorphism is:
	 *   zeta(e,u) = (e, eta*u)
	 * for eta = sqrt(-1). The split was computed for a specific value
	 * of eta; we also change its sign conditionally to account for
	 * the signs of the scalars k0 and k1.
	 */

	/* Square root of -1 in the field. */
	static const gf ETA = LGF(
		0xAA938AEE, 0xD99E0F1B, 0xB30E6336, 0xA60D864F,
		0xE53688E3, 0xE414983F, 0x3C69B85F, 0x10ED2DB3);

	uint32_t k0[5], k1[5];
	uint32_t sk;
	point win[16], p;
	gf eta;
	int8_t sd0[26], sd1[26];

	/* Split the scalar.
	   We need an extra limb for both k0 and k1 for easier recoding. */
	sk = scalar_split(k0, k1, s);
	k0[4] = 0;
	k1[4] = 0;

	/*
	 * Build the window over P1 (if k0 >= 0) or -P1 (if k0 < 0).
	 */
	win[0] = *p1;
	gf_condneg(&win[0].U, &win[0].U, -(sk & 1));
	for (int i = 1; i < 15; i += 2) {
		point_double(&win[i], &win[i >> 1]);
		point_add(&win[i + 1], &win[i], &win[0]);
	}
	point_double(&win[15], &win[7]);

	/*
	 * Get the proper square root of -1 (eta or -eta) to account
	 * for the signs of k0 and k1.
	 */
	gf_condneg(&eta, &ETA, -((sk ^ (sk >> 1)) & 1));

	/*
	 * Recode the two half-width scalars.
	 */
	uint_recode(sd0, 26, k0);
	uint_recode(sd1, 26, k1);

	/*
	 * Perform a double-and-add algorithm with the 5-bit window.
	 */
	point_lookup(p2, win, sd0[25]);
	point_lookup(&p, win, sd1[25]);
	gf_mul(&p.U, &p.U, &eta);
	gf_neg(&p.T, &p.T);
	point_add(p2, p2, &p);
	for (int i = 24; i >= 0; i --) {
		point_xdouble(p2, p2, 5);
		point_lookup(&p, win, sd0[i]);
		point_add(p2, p2, &p);
		point_lookup(&p, win, sd1[i]);
		gf_mul(&p.U, &p.U, &eta);
		gf_neg(&p.T, &p.T);
		point_add(p2, p2, &p);
	}

#else
	/*
	 * Generic implementation: a double-and-add algorithm with
	 * a 5-bit window and Booth recoding (i.e. digits are in the
	 * -15..+16 range instead of 0..+31).
	 */

	point win[16];
	int8_t sd[51];

	/*
	 * Fill window: win[i] = (i+1)*P
	 */
	win[0] = *p1;
	for (int i = 1; i < 15; i += 2) {
		point_double(&win[i], &win[i >> 1]);
		point_add(&win[i + 1], &win[i], &win[0]);
	}
	point_double(&win[15], &win[7]);

	/*
	 * Recode the scalar into digits.
	 */
	scalar_recode(sd, s);

	/*
	 * Perform a double-and-add algorithm with the 5-bit window.
	 */
	point_lookup(p2, win, sd[50]);
	for (int i = 49; i >= 0; i --) {
		point p;

		point_xdouble(p2, p2, 5);
		point_lookup(&p, win, sd[i]);
		point_add(p2, p2, &p);
	}
#endif
}

/* Forward declaration of the precomputed point tables. */
static const point_affine point_win_base[];
static const point_affine point_win_base65[];
static const point_affine point_win_base130[];
static const point_affine point_win_base195[];

/*
 * Multiplication of the fixed base point by a scalar.
 * P <- s*G
 */
static void
point_mulgen(point *p, const scalar *s)
{
	int8_t sd[51];
	point_affine qa;

	/*
	 * Recode the scalar into digits.
	 */
	scalar_recode(sd, s);

	/*
	 * Perform a double-and-add algorithm with the four precomputed
	 * 5-bit windows (affine).
	 */
	point_affine_lookup(&qa, point_win_base, sd[12]);
	p->E = qa.E;
	p->Z = gf_one;
	p->U = qa.U;
	p->T = qa.T;
	point_affine_lookup(&qa, point_win_base65, sd[25]);
	point_add_affine(p, p, &qa);
	point_affine_lookup(&qa, point_win_base130, sd[38]);
	point_add_affine(p, p, &qa);
	for (int i = 11; i >= 0; i --) {
		point_xdouble(p, p, 5);
		point_affine_lookup(&qa, point_win_base, sd[i]);
		point_add_affine(p, p, &qa);
		point_affine_lookup(&qa, point_win_base65, sd[i + 13]);
		point_add_affine(p, p, &qa);
		point_affine_lookup(&qa, point_win_base130, sd[i + 26]);
		point_add_affine(p, p, &qa);
		point_affine_lookup(&qa, point_win_base195, sd[i + 39]);
		point_add_affine(p, p, &qa);
	}
}

/*
 * Signature verification helper: given point P1, 128-bit integer `u`
 * (expressed over 4 limbs), and scalar `v`, compute:
 * P2 <- u*P1 + v*G
 * THIS FUNCTION IS NOT CONSTANT-TIME. It is assumed that it will be used
 * in signature verification, and that signature verification uses only
 * public data.
 */
static void
point_mul128_add_mulgen_vartime(point *p2,
	const point *p1, uint32_t *u, const scalar *v)
{
	point win[8];
	int8_t sdu[130], sdv[256];

	/*
	 * Make a wNAF window: win[i] = (2*i+1)*P1
	 */
	point_double(&win[0], p1);
	point_add(&win[1], &win[0], p1);
	for (int i = 2; i < 8; i ++) {
		point_add(&win[i], &win[i - 1], &win[0]);
	}
	win[0] = *p1;

	/*
	 * Recode u and v into wNAF.
	 */
	uint_recode_wNAF(sdu, 130, u, 4);
	scalar_recode_wNAF(sdv, v);

	/*
	 * zz = 1 when the accumulator is still the neutral, 0 afterwards.
	 * ndbl is the number of pending doublings.
	 */
	int zz = 1;
	unsigned ndbl = 0;
	for (int i = 129; i >= 0; i --) {
		/* Schedule one more doubling. */
		ndbl ++;

		/* Get next digits; if all three are zeros, then skip
		   to the next iteration. */
		int eu = sdu[i];
		int ev0 = sdv[i];
		int ev1 = i < 126 ? sdv[i + 130] : 0;
		if ((eu | ev0 | ev1) == 0) {
			continue;
		}

		/* Apply pending doublings. */
		if (zz) {
			zz = 0;
			*p2 = point_neutral;
		} else {
			point_xdouble(p2, p2, ndbl);
		}
		ndbl = 0;

		/* Process digits. */
		if (eu != 0) {
			if (eu > 0) {
				point_add(p2, p2, &win[eu >> 1]);
			} else {
				point_sub(p2, p2, &win[(-eu) >> 1]);
			}
		}
		if (ev0 != 0) {
			if (ev0 > 0) {
				point_add_affine(p2, p2,
					&point_win_base[ev0 - 1]);
			} else {
				point_sub_affine(p2, p2,
					&point_win_base[-ev0 - 1]);
			}
		}
		if (ev1 != 0) {
			if (ev1 > 0) {
				point_add_affine(p2, p2,
					&point_win_base130[ev1 - 1]);
			} else {
				point_sub_affine(p2, p2,
					&point_win_base130[-ev1 - 1]);
			}
		}
	}

	if (zz) {
		*p2 = point_neutral;
	} else {
		point_xdouble(p2, p2, ndbl);
	}
}

/* ===================================================================== */
/*
 * SECTION 4: PRECOMPUTED POINT WINDOWS
 *
 * We store here some precomputed tables for multiples of the base point.
 */

#if JQ == JQ255E

/* Points i*G for i = 1 to 16, affine extended format */
static const point_affine point_win_base[] = {
	/* G * 1 */
	{ LGF(0x00000003, 0x00000000, 0x00000000, 0x00000000,
	      0x00000000, 0x00000000, 0x00000000, 0x00000000),
	  LGF(0x00000001, 0x00000000, 0x00000000, 0x00000000,
	      0x00000000, 0x00000000, 0x00000000, 0x00000000),
	  LGF(0x00000001, 0x00000000, 0x00000000, 0x00000000,
	      0x00000000, 0x00000000, 0x00000000, 0x00000000) },
	/* G * 2 */
	{ LGF(0xD6342FD1, 0xD0FAC687, 0xEB1A1F58, 0x687D6343,
	      0xF58D0FAC, 0x343EB1A1, 0xFAC687D6, 0x1A1F58D0),
	  LGF(0xDB6D97A3, 0xB6DB6DB6, 0x6DB6DB6D, 0xDB6DB6DB,
	      0xB6DB6DB6, 0x6DB6DB6D, 0xDB6DB6DB, 0x36DB6DB6),
	  LGF(0x97827791, 0x0A72F053, 0xCBC14E5E, 0x05397829,
	      0xE5E0A72F, 0x829CBC14, 0x72F05397, 0x414E5E0A) },
	/* G * 3 */
	{ LGF(0xD61A9E2F, 0x26AFA803, 0x7F0E7D48, 0xAD827302,
	      0xBA270925, 0x4D5065C0, 0xDB3919FD, 0x6E6BA44D),
	  LGF(0xC4043E79, 0xC2F21347, 0x066D4156, 0x6B1CEBA6,
	      0xA3E20224, 0xAB617909, 0xD30336A0, 0x12358E75),
	  LGF(0x2BDDB3C7, 0xC4FAF544, 0xF0485A50, 0xC58EF652,
	      0x71E284EF, 0x0509961D, 0xDC59141C, 0x7287BBB2) },
	/* G * 4 */
	{ LGF(0xF8390C16, 0xBCE7BEB9, 0xE825BA04, 0xCBFF478E,
	      0x95BDA924, 0x96E4BB9C, 0x9561D944, 0x0F137176),
	  LGF(0x130DB4AD, 0x65A29F71, 0xFA47C8BB, 0x9F71130D,
	      0xC8BB65A2, 0x130DFA47, 0x65A29F71, 0x7A47C8BB),
	  LGF(0x4402088D, 0x4D0A213B, 0xF44E59F2, 0x853223D7,
	      0x2101F311, 0x03ADCBE2, 0x9918E929, 0x2375E811) },
	/* G * 5 */
	{ LGF(0xE875C86A, 0xD23D2C8B, 0x73C41197, 0x1BD81557,
	      0xBCDB09C0, 0x74304444, 0x980D6493, 0x3A3E1251),
	  LGF(0xDA5B43EE, 0x1F2B6B08, 0xC44A0C63, 0xE40F8B8B,
	      0xB35FB70C, 0x5866F1F8, 0x50F768D7, 0x185034D2),
	  LGF(0x3D361051, 0xC9192749, 0xC1C66FF4, 0xE00C1E20,
	      0x724B43CC, 0x8982206A, 0xBB5DF4DA, 0x3E3560E7) },
	/* G * 6 */
	{ LGF(0x229AD5F2, 0x643AD390, 0x07471E77, 0xC7120748,
	      0x6A6D2E2A, 0x1673C5B9, 0x5844C804, 0x124FA9DF),
	  LGF(0xF91D6B18, 0x0BD0C5F1, 0x263610A7, 0xBB4A410D,
	      0x98F35F00, 0xA1AB0B9D, 0xAFDDC92B, 0x4FA6D8B6),
	  LGF(0xAEB11ACD, 0x355D1614, 0xAEE9D26F, 0x76ED99CC,
	      0xE94A460E, 0xD7991971, 0xDA88753E, 0x34F3562F) },
	/* G * 7 */
	{ LGF(0x678229FB, 0xE8D6CBC7, 0x232C88D0, 0xD86308C5,
	      0x4BFC77E7, 0xC36F6969, 0x82AAF7AC, 0x4C5C3FA3),
	  LGF(0x159EF4EA, 0x7EB52414, 0xEB4CC9E1, 0xB885C9D1,
	      0xEE64BF7F, 0x350914B3, 0x520AED5A, 0x6DD8CDFA),
	  LGF(0x34C75544, 0x59DAD0E6, 0x0C2A0899, 0x818C7393,
	      0x60AC1520, 0x0957AB7A, 0x0A217C1C, 0x56861F4D) },
	/* G * 8 */
	{ LGF(0x53187B28, 0x4E11C113, 0x515E9113, 0xAD881FFB,
	      0x2FD5BA49, 0x86AD97E4, 0xB2A6146C, 0x2C922B2C),
	  LGF(0xEB513A8B, 0x99DA8C93, 0x5DEDFC87, 0x0706B8B9,
	      0x1F778CE9, 0xC54D8F47, 0xFA2E63E5, 0x4766315B),
	  LGF(0x39729F9A, 0x4D9B8F56, 0x8B0077A8, 0x89B68A9C,
	      0xFCA311FD, 0xF3C520B8, 0xB811270A, 0x532698BC) },
	/* G * 9 */
	{ LGF(0xB52FBCC4, 0x6FB66DF6, 0x38AA1784, 0x675E5BCC,
	      0x852C1B0B, 0x55B6D3E8, 0xFA293050, 0x2289F3AB),
	  LGF(0xD0A08E61, 0xA84A27A9, 0x132CCAC1, 0x27E9084D,
	      0x01F68C40, 0x498C7D8B, 0x940E4159, 0x6957FDFF),
	  LGF(0x815F2EFF, 0x8D2F2DE6, 0x88C812F9, 0x76CA668F,
	      0x32B42796, 0x56244B8A, 0x72CB2D3C, 0x431DA1A6) },
	/* G * 10 */
	{ LGF(0xACC048BE, 0x8EB44683, 0xAAD5CD46, 0x803FC1C6,
	      0x0762C505, 0xAF553973, 0x61B06E3A, 0x30290CF9),
	  LGF(0xB889903E, 0x3AA366BB, 0xCC140A37, 0x55838146,
	      0xA9B6AD5E, 0x4AA37581, 0x916F803C, 0x7B37113C),
	  LGF(0xE1C6283E, 0xD912EBC4, 0x8AE5C163, 0xC70EAC51,
	      0xE828C438, 0x9EDDA370, 0x189ECFD9, 0x252DC97C) },
	/* G * 11 */
	{ LGF(0x99D93A3B, 0x38A7E995, 0xA3919A65, 0x09DF0EB0,
	      0xF643AF23, 0x6F385F29, 0x2424A548, 0x467C84CA),
	  LGF(0x864E7F82, 0xA9A8911D, 0xAB741725, 0x65CF6B9C,
	      0xE772B327, 0x8C133221, 0x8CD1F209, 0x15852107),
	  LGF(0x8F92D685, 0x41583C9A, 0x53E938EB, 0xFAE4DC55,
	      0x6C5406EA, 0xC3FC1F02, 0xBC1F036B, 0x5D4A07E9) },
	/* G * 12 */
	{ LGF(0x83BF7B93, 0xA49A13F8, 0x8DC7807E, 0x06E6D0CF,
	      0x238DA4B1, 0x4B3B87C8, 0x5A9F7719, 0x66FF84AF),
	  LGF(0x087640AA, 0xE4C1C725, 0x3EF5D5E0, 0xFB902D6A,
	      0x2E1297EB, 0x53EF3593, 0xE1787343, 0x67E65CF7),
	  LGF(0xFF9309B4, 0x4203ACE2, 0x8E506208, 0xAE5BB531,
	      0x3DEB52CB, 0x4742F3CB, 0x3959DA85, 0x2213A3D9) },
	/* G * 13 */
	{ LGF(0x2ED04723, 0xA9962278, 0xB5D2A716, 0x22587604,
	      0xFFB6AD2D, 0x3E7BB13D, 0x9D4C885B, 0x6E7036BE),
	  LGF(0x81061965, 0x90F88398, 0xF2BFCB98, 0x67D0394F,
	      0xCD1396D8, 0x913200FC, 0x306A3580, 0x17F96D76),
	  LGF(0x4099DC93, 0xF1F0EC98, 0xE43361F5, 0xE02396E9,
	      0xAB0AE384, 0x028EBB02, 0x2DB22F61, 0x0E236467) },
	/* G * 14 */
	{ LGF(0x0F2521F0, 0xC6C60BB3, 0x5CB6D116, 0xA59973DD,
	      0x706DD30D, 0x069708CC, 0xF8A08989, 0x74C367AB),
	  LGF(0xD2AC4172, 0x05B49673, 0x0D77E4E6, 0xA016A689,
	      0x0635E1C0, 0x7C6DAA97, 0x47A6A04A, 0x42C80345),
	  LGF(0x75DEF4DC, 0x0266FAA8, 0x05C5A659, 0x41B211E5,
	      0x39E5E234, 0xE13C4A76, 0xE6AF9B7D, 0x0C4AC28D) },
	/* G * 15 */
	{ LGF(0xE86224A7, 0x543AA085, 0x9EA21055, 0x626226C0,
	      0xEE7E01D9, 0x257B5FE5, 0x82C497CD, 0x1110D927),
	  LGF(0x19120727, 0x7FFA4AF7, 0x1BF74984, 0x705D1257,
	      0x9FAE1F07, 0x4AD1FA64, 0x265D7456, 0x2F4CA2B6),
	  LGF(0xF7C6525E, 0xB111F1B5, 0xC1B29AC7, 0x54BD0CFF,
	      0x7009957D, 0xCC7CCE32, 0x0D563132, 0x0CCF7FF0) },
	/* G * 16 */
	{ LGF(0x24F31B2E, 0x97DA44A0, 0xDB5120DD, 0xF8FAE043,
	      0xD7F5F415, 0x03D9F770, 0xA296F053, 0x676824C9),
	  LGF(0xE1227049, 0x2D808316, 0x32683177, 0x15064C91,
	      0x41E90ED8, 0x706D8A1F, 0x1A6DB76E, 0x251A1931),
	  LGF(0x9E05F91A, 0x95191DCA, 0x0C6EE0A8, 0x0DB49CC1,
	      0x7BF95128, 0x7C16D8FF, 0xB15D04AE, 0x2C8D5EC4) }
};

/* Points i*(2^65)*G for i = 1 to 16, affine extended format */
static const point_affine point_win_base65[] = {
	/* (2^65)*G * 1 */
	{ LGF(0x8E08211F, 0x886FF627, 0xF94734AB, 0x4629856E,
	      0xB41DD08C, 0xAB3BAC1A, 0x634A53B5, 0x6CE0B77B),
	  LGF(0xC596FA71, 0xAD5F8FBF, 0x9DE223FF, 0x41589354,
	      0xE50A4384, 0x395D2181, 0xA8A7626E, 0x1B313D36),
	  LGF(0xEB9D832B, 0x17E71DD1, 0xD599D9CB, 0x222B7C0C,
	      0xFF13EC0E, 0x48E8393C, 0x21A5AAD2, 0x6E78594A) },
	/* (2^65)*G * 2 */
	{ LGF(0x2716A209, 0x111922C7, 0x3417B3E6, 0xE4584C1E,
	      0xCE6C2582, 0x8AA629F4, 0x8DD93292, 0x7521310F),
	  LGF(0x889E3F33, 0x2B7C2F71, 0xA5E65CF4, 0x4CA4C049,
	      0x9976BFE7, 0x5CD27D90, 0x9985D602, 0x0BE56F35),
	  LGF(0x1BE2027B, 0xC699D8CD, 0x19ECC70E, 0xC9233D82,
	      0xABAECD17, 0x6C805A5D, 0x6985620E, 0x40534B30) },
	/* (2^65)*G * 3 */
	{ LGF(0xD6A45A0E, 0x33F6746E, 0xCDA92791, 0x5A40972F,
	      0x6C9CC38D, 0x7B11CD4B, 0x8D0885BF, 0x390940AD),
	  LGF(0xED61CB7F, 0x4A504A5D, 0x42D7801D, 0xAF415083,
	      0xAB4295EB, 0x0519A68A, 0x0B09C2B4, 0x098D3AB9),
	  LGF(0x59A456C6, 0xA06F9DAD, 0x3FFE78C9, 0x92D53426,
	      0x45C181DE, 0x0696A395, 0x7B5F6FD7, 0x41A59092) },
	/* (2^65)*G * 4 */
	{ LGF(0x32A76153, 0x6C4E581C, 0xD6E7A570, 0xEADD9BBE,
	      0xC0DF4A3B, 0xBD9A0206, 0xFA934AC1, 0x10D60536),
	  LGF(0x81B4D89F, 0xD98BB32E, 0xE1067F08, 0x5A0FCD3A,
	      0xAFAD3191, 0x3B845EF3, 0xAA6E2C23, 0x3540EB32),
	  LGF(0xA76336E1, 0x1D5909B1, 0x47CE6A39, 0x4563C7AD,
	      0xA3CB3C98, 0x893FFC0C, 0x11381CFF, 0x6470D110) },
	/* (2^65)*G * 5 */
	{ LGF(0x8591C275, 0x648443E4, 0x18227B9E, 0x74427A88,
	      0x6E87C51E, 0x9AE9B073, 0xB9985165, 0x742236A2),
	  LGF(0xD80EB7AB, 0xDB2EB15D, 0x633AC0BB, 0x695C5863,
	      0x5FB45810, 0x9DD36D92, 0x374E74CB, 0x4B36B3BE),
	  LGF(0x97F19A1B, 0x3B020304, 0xCBC99BC3, 0xBCF6E190,
	      0x3280BAD3, 0x52217A6F, 0x685E3293, 0x7563AAD0) },
	/* (2^65)*G * 6 */
	{ LGF(0x0C14DC83, 0xD8AD585D, 0xD6669E2A, 0x748A8C77,
	      0x6BC1A5D5, 0x9318569D, 0x42C4587D, 0x637BA56A),
	  LGF(0x1425C001, 0x2473D6DB, 0x6E58108F, 0x76306718,
	      0xF6ED32DE, 0x0C0776C9, 0x90ACB085, 0x143CF8A0),
	  LGF(0x05579C38, 0x73CDF9E5, 0x9040AF87, 0xDE3B04BA,
	      0x8418DD26, 0x3A3345F9, 0xCC1F09C1, 0x2C301B6B) },
	/* (2^65)*G * 7 */
	{ LGF(0x9D051B56, 0xE14526AA, 0x8273B21D, 0x0E16666D,
	      0x6F56FA43, 0x1094A46B, 0x96A1BB91, 0x5C780EAA),
	  LGF(0x27D26209, 0xE95E2181, 0xE8FB612B, 0x33088969,
	      0xF2BF788C, 0xBAC06821, 0xD61C83DD, 0x7EE7B8C1),
	  LGF(0x1F3502A9, 0x0474E697, 0x5FA59240, 0x624FFDB3,
	      0xA1A25303, 0x617E1C89, 0x16C33383, 0x72AB912B) },
	/* (2^65)*G * 8 */
	{ LGF(0xD67CC3ED, 0xE800A094, 0xF26076D9, 0xD6C684ED,
	      0x17DD5343, 0xE95C514D, 0x473732FF, 0x6B3865BE),
	  LGF(0x01944CB8, 0x4C033B23, 0xA5D337CB, 0x3260DA08,
	      0xA6C39FFC, 0x34DFEFB2, 0x9B5B142E, 0x23587229),
	  LGF(0xD04090FB, 0x439324FC, 0x6FC23C15, 0x08AA0426,
	      0xBBFAA0EE, 0x7B1EF0DE, 0xE34C35A1, 0x2961CB37) },
	/* (2^65)*G * 9 */
	{ LGF(0xA19EF549, 0xE7847250, 0x94B8F6C1, 0xDD56EA48,
	      0x9C0105B4, 0xF8815AEC, 0x916B15C7, 0x32138C48),
	  LGF(0x6086DF92, 0x6742B67C, 0x8D75331F, 0x53193A71,
	      0xCA14E352, 0xADD356D1, 0xD7C9E88D, 0x07DF0CC5),
	  LGF(0xB8594D9B, 0x5C9C41C3, 0xE6FADF0D, 0x643500CD,
	      0x03003B68, 0x0D31937E, 0x3766078D, 0x713B783B) },
	/* (2^65)*G * 10 */
	{ LGF(0xB299A5D0, 0x25F90BCE, 0x24E25FBE, 0x4FD5981D,
	      0xBEFEC84E, 0x83A3CE92, 0x6A90B198, 0x5C932978),
	  LGF(0x51C4DF08, 0x7982C540, 0xEBBFE2BB, 0x476A1D01,
	      0x7E287D25, 0x3F6064CC, 0x0C17C457, 0x79E82E42),
	  LGF(0xE8B54114, 0x8CE01EA2, 0xE2EF99F8, 0x73165189,
	      0x7B6F0AD1, 0x27984BF8, 0x70DAFA38, 0x1813C6ED) },
	/* (2^65)*G * 11 */
	{ LGF(0x0884167D, 0x7CAD3302, 0xCDDA791E, 0x383BFB84,
	      0x16879C64, 0x8D0D667D, 0xAEAFD937, 0x4515BBA1),
	  LGF(0x13403E58, 0x00FF1286, 0x2D02F041, 0x854CBABF,
	      0x65593890, 0xB97069DD, 0x68B5A376, 0x3CC8BC31),
	  LGF(0xF65D98BA, 0xC303EBB8, 0x5BEF4381, 0xF8F2F98E,
	      0x9FE2783E, 0x206DA25D, 0xD462C39D, 0x058DA5FE) },
	/* (2^65)*G * 12 */
	{ LGF(0x7FDDE52D, 0x7118E357, 0x6D616E5E, 0x4F20F40E,
	      0x4D7E5167, 0x68592F91, 0xD95C2805, 0x3E658213),
	  LGF(0x6588D284, 0x3688782C, 0x163B51DB, 0xA50B5985,
	      0xF7311FA5, 0x23447D42, 0x5B751D79, 0x51CBB2B6),
	  LGF(0xEB5BEC17, 0x877FB541, 0xD77A0305, 0xDC1EF290,
	      0x43025B6A, 0x15EB451E, 0x0CB85A8A, 0x60570F44) },
	/* (2^65)*G * 13 */
	{ LGF(0x45295FBE, 0xBF9758CD, 0xBE8D1017, 0x0DAC5E2E,
	      0x89E728B3, 0x3CA545A9, 0x1F644B88, 0x40255DC8),
	  LGF(0x7A9FAA68, 0x6AF277E6, 0x9009703F, 0xFDEE555C,
	      0x8D5B502A, 0xAA36C89F, 0xAFC87131, 0x052051ED),
	  LGF(0x86AAB04C, 0xDED77C44, 0x9AA3A8E1, 0x9B85D45C,
	      0xC4F2C4BA, 0x0F6D233D, 0x319D2237, 0x50C64067) },
	/* (2^65)*G * 14 */
	{ LGF(0x4910DE30, 0x13D6BB10, 0x76848E4E, 0x52126D43,
	      0x0130A4FF, 0x8E7CF5FB, 0xF7F8E0F4, 0x631EFE8E),
	  LGF(0x08EB56DD, 0x86A494FC, 0xD62219A0, 0xF694DF6E,
	      0x22098B86, 0x214282B0, 0xA95620EB, 0x7182E92F),
	  LGF(0xF4D51B6A, 0x8A96C7C2, 0xFBF79614, 0x6E06A2EF,
	      0x36FFDEC3, 0x2F8383D0, 0x275A04FE, 0x136A3753) },
	/* (2^65)*G * 15 */
	{ LGF(0x2F78D8DC, 0x71477471, 0x2A4254F0, 0x533027C6,
	      0x8C3DB802, 0xBF13A51C, 0xA671868A, 0x09C0518D),
	  LGF(0xE71417FD, 0xD25E4076, 0xD617E037, 0xAE53B9A6,
	      0xBB08D620, 0xEDF6131A, 0x7206C6C4, 0x406E8857),
	  LGF(0xE5261DBE, 0xA7B19AB7, 0x93780470, 0x5FD69501,
	      0xD3DDA8C2, 0x69CB583F, 0x106BD4A6, 0x7A35DA9A) },
	/* (2^65)*G * 16 */
	{ LGF(0x3E152D96, 0xB7E60EBD, 0x3CBC9CE8, 0x16B77AD0,
	      0xA8F300F7, 0xA8D3D5EB, 0x6203938C, 0x0DA857B8),
	  LGF(0xD0781C73, 0x7D37BBB6, 0x82DAF6AB, 0x1ADF4CFE,
	      0xDFBA601C, 0x6CB90D1D, 0x3C090D35, 0x7B916A82),
	  LGF(0x1875E47C, 0xFE1CAB3D, 0xE298321B, 0x75D85076,
	      0x623721E5, 0xAA62937A, 0xE382EC23, 0x78BE6830) }
};

/* Points i*(2^130)*G for i = 1 to 16, affine extended format */
static const point_affine point_win_base130[] = {
	/* (2^130)*G * 1 */
	{ LGF(0xDE63606E, 0x869B548F, 0x9BE27159, 0x04DA93AE,
	      0xBC2E0657, 0x9FBB8D6B, 0x0E47A525, 0x06CB9DE7),
	  LGF(0xCAF58BCB, 0x6A6F8767, 0xCD520CD9, 0xEF9A2E5A,
	      0xEE40437C, 0x2B998E19, 0xF3E02AB1, 0x1E3A7692),
	  LGF(0xC50BD952, 0x12DCF83E, 0x82CB44C6, 0xC1FEC4C7,
	      0x2158E8D4, 0x2FCDD9E7, 0x46FAC0AE, 0x6383719F) },
	/* (2^130)*G * 2 */
	{ LGF(0x3D54F276, 0xB08E5A16, 0xADF1CAD3, 0x137941C5,
	      0x1B6AA715, 0xE12975DE, 0xE1492E65, 0x69E07759),
	  LGF(0xC28259E8, 0x97643EA7, 0xA0416456, 0x64C33BBE,
	      0xAFFFBCEB, 0xAC5EBA85, 0x8936CEEA, 0x1DE0359F),
	  LGF(0x1879EA59, 0xF32ACE06, 0xAB9427B0, 0xC8A5F632,
	      0xE081587F, 0xF83CBCAC, 0x87D69006, 0x3A18C765) },
	/* (2^130)*G * 3 */
	{ LGF(0x8C9DC498, 0xA6954493, 0x73152188, 0x9225D9C9,
	      0x85C3FC53, 0x7A630559, 0xA431B60A, 0x7F9D9B87),
	  LGF(0x185A63C2, 0x4A8F6858, 0x105E6338, 0xC29D0227,
	      0x57313D72, 0x4FA122A3, 0x3C842009, 0x62BAF3EF),
	  LGF(0x7B30E0BC, 0x18390845, 0x76EF453F, 0x799295C3,
	      0xAF33DE83, 0xC5FE42DE, 0x628B654A, 0x54C34BDF) },
	/* (2^130)*G * 4 */
	{ LGF(0xEC32F903, 0x607C64DA, 0xA9BA461C, 0x8777F23B,
	      0x92C3BE8D, 0x6E7804AC, 0x5F951D11, 0x180B366A),
	  LGF(0xFFC35397, 0x17625F88, 0xDA783099, 0xEF697901,
	      0xEF458EDC, 0xDCCCD3E1, 0xCFBBEDE2, 0x37EDC360),
	  LGF(0xA0B94FB0, 0x5F7FC0E7, 0x38D69625, 0x3E9CC116,
	      0x720B072F, 0x62C64632, 0x5D836913, 0x51589183) },
	/* (2^130)*G * 5 */
	{ LGF(0xCEA3BDDC, 0xB5DA3AAF, 0xAE7AD735, 0x75DE8E82,
	      0x5446291A, 0xA458CD16, 0x5A7FF671, 0x552022F7),
	  LGF(0xE9EC95DF, 0x055D6CCD, 0xDFCADBF5, 0xEB86F24A,
	      0x9F7AE17C, 0x6DDAC4AB, 0x09ADD692, 0x282E2094),
	  LGF(0x6B02F394, 0x60BA3A7F, 0x387EA15D, 0x5AF736BE,
	      0x3EC38691, 0x541CE0C5, 0x32FB79E4, 0x775F5966) },
	/* (2^130)*G * 6 */
	{ LGF(0xD222B565, 0x7C2346E4, 0x258AEB93, 0xA45999A2,
	      0x64D273CD, 0x66ECF438, 0xB5A5952D, 0x5EF32CC1),
	  LGF(0xDCAD7528, 0xBA9DA77E, 0x892C93FF, 0xB6FA24A8,
	      0xFA1ACD6B, 0xF406AD1F, 0x131E195C, 0x7E4F46D5),
	  LGF(0xEA698452, 0x85A61A2D, 0xB9C3F150, 0xEB5AD1F6,
	      0x89B1BA23, 0xA68B3F01, 0xCA2FEDC4, 0x3306A93B) },
	/* (2^130)*G * 7 */
	{ LGF(0x988BD912, 0x70FD1203, 0x5FC76584, 0xE9BDF6AD,
	      0xDB704F5D, 0x5E05B4C4, 0x189FDCFE, 0x4830813F),
	  LGF(0xEBEF1413, 0x9E343B6F, 0xB7F95D67, 0xC1293137,
	      0x05CA0B5B, 0x6FB5672D, 0x2AEC172D, 0x37223948),
	  LGF(0x28A8DD36, 0xEDC0FD96, 0x89BAA4F4, 0xF796A60E,
	      0xCF12747E, 0x21A50BBD, 0xCED8D90B, 0x35266C9A) },
	/* (2^130)*G * 8 */
	{ LGF(0xD1F7E9CE, 0x68EE4317, 0xC8731082, 0x27214239,
	      0x24ED9795, 0x5D86EBC5, 0x4D486C20, 0x303B08CF),
	  LGF(0xEA446499, 0x21232B19, 0xEDDC580B, 0x0F110F3B,
	      0x7F928B59, 0x7BFEE516, 0xC48289F4, 0x4E1CF6CC),
	  LGF(0x04969A8C, 0x1768F10E, 0x60B519A9, 0x0834BAF8,
	      0xCEC7737D, 0x94AAE19E, 0xB5360B34, 0x1DFD019E) },
	/* (2^130)*G * 9 */
	{ LGF(0x454A9789, 0xFA3E5E1F, 0xCE8DDF8F, 0x9259F250,
	      0x2FBB9A1B, 0x2830FF4E, 0xF57EA2FC, 0x4262554C),
	  LGF(0xA28FF940, 0xFFFA0532, 0xE791A9D0, 0x06134380,
	      0x934E696A, 0x24120A75, 0x2C83BC57, 0x6F671B02),
	  LGF(0x9180368F, 0x0BE5ED63, 0xF6E447C2, 0xBB03E3FD,
	      0x83E29BB6, 0x4CA31E3B, 0x35166005, 0x03BBDFC6) },
	/* (2^130)*G * 10 */
	{ LGF(0xE93AA158, 0xE1B24ECC, 0x3202BCE6, 0xBC5D29A7,
	      0xA4D753B1, 0x2970E4C0, 0x18E51DF5, 0x776F6C1C),
	  LGF(0x3B071E37, 0x0EE1B500, 0x8D80BCA7, 0x3EC32044,
	      0x5EC7534A, 0x108D5A1C, 0xEBA1A511, 0x64ECAAC0),
	  LGF(0x1535C514, 0x10E50013, 0x6D74A32A, 0xD00F9D21,
	      0xA13E6050, 0x8C125FB0, 0xA3AC991A, 0x4BC50BF7) },
	/* (2^130)*G * 11 */
	{ LGF(0x3EB40DBD, 0x10DC6E26, 0x1F4F95D8, 0x31DC4E88,
	      0x5D45B50A, 0x2C33E16E, 0x2A3899E8, 0x3622E7BF),
	  LGF(0xD4AB70A1, 0x6CC3477A, 0x712BA4D4, 0x0BC92225,
	      0x4564D8F4, 0xD51C733D, 0x599B372A, 0x029CAFA5),
	  LGF(0x6518ADF3, 0x3D63011A, 0x816D4B31, 0x84FD6E87,
	      0x67CD8FA6, 0x396102BA, 0xE547B2AD, 0x30F0D235) },
	/* (2^130)*G * 12 */
	{ LGF(0x82E719F6, 0x54246632, 0x779DEFF8, 0xB3E38AE4,
	      0x3EE0C99A, 0x3F2A874D, 0x8561373F, 0x70F7A39B),
	  LGF(0xB7854025, 0xCB9E1D5C, 0xC064493E, 0x960067A5,
	      0xA91ED717, 0x41F420B5, 0x7E661D8D, 0x6E81F9FA),
	  LGF(0xD1B3FF89, 0x96737B3B, 0x8B260B44, 0xFA5B9975,
	      0xD5EB0507, 0x4E40C5DC, 0xE37DDC45, 0x797141D1) },
	/* (2^130)*G * 13 */
	{ LGF(0xDE84F742, 0x3A2AA626, 0xE24C4D0F, 0xC52014D6,
	      0x665947AC, 0x87620DD6, 0x38239FA8, 0x4489AC65),
	  LGF(0xE489DDB8, 0x1A722773, 0x3D4AABD6, 0xF9498389,
	      0xBB3DFDCC, 0x59F3D4C5, 0xD2801E6A, 0x653EE371),
	  LGF(0x108032A6, 0xA14B344A, 0x99975786, 0x336E96DD,
	      0x6ED6198C, 0x3AF72BF1, 0x23DFA5BC, 0x6E8DE137) },
	/* (2^130)*G * 14 */
	{ LGF(0x21EA7956, 0x5BDDDC01, 0xEE996605, 0x30C254D3,
	      0xBE2A729B, 0xE23DA12D, 0xD7B177EE, 0x0698AEC8),
	  LGF(0x09693F50, 0x460E164D, 0x44D22EC2, 0x96FABF77,
	      0x595E868E, 0x216A1928, 0xAC402680, 0x50E1BEE9),
	  LGF(0x5FE17CBC, 0x2EA3B442, 0x8227BB81, 0x3076D3BE,
	      0x99779B03, 0x73999AF9, 0x287FBDD0, 0x52BC8B51) },
	/* (2^130)*G * 15 */
	{ LGF(0x33632097, 0x4DE169D2, 0x2711C087, 0x7269B654,
	      0x544AC635, 0xC7562833, 0x09C7A4DA, 0x67C8ED8C),
	  LGF(0x64C6EE61, 0x442C914C, 0xAA3D41CD, 0x5486463A,
	      0x744FB271, 0x2323BA05, 0xB63D2983, 0x5CE94782),
	  LGF(0x41CFEA05, 0x5E85E0F8, 0xC8449D15, 0xFE575987,
	      0x40C3632A, 0x4B8F046B, 0xC85A090C, 0x79B75334) },
	/* (2^130)*G * 16 */
	{ LGF(0x45B50B50, 0xA1638BEC, 0x669E52E3, 0xB956B5A6,
	      0x6F53165A, 0xAFF58E0E, 0xEDB8A088, 0x5F00BEB6),
	  LGF(0x560BD063, 0x20DDE2D9, 0x9386B815, 0x68337F97,
	      0xB5F9B94C, 0x9CAE33A6, 0x8B17674E, 0x0F2ED841),
	  LGF(0x8690FF50, 0x42082E61, 0x5901899E, 0x3721E53E,
	      0x342DE052, 0xBB88653D, 0xCF10FA1C, 0x2EED8F30) }
};

/* Points i*(2^195)*G for i = 1 to 16, affine extended format */
static const point_affine point_win_base195[] = {
	/* (2^195)*G * 1 */
	{ LGF(0x2FD3D48B, 0x259381D5, 0xCA5B1805, 0x05A13379,
	      0x83A60AC4, 0x4F9CCA64, 0x2FF7C515, 0x26FE2B2D),
	  LGF(0x864B8022, 0xC6C4EA52, 0x27F2FE05, 0x3FACF030,
	      0xAFE0F2B2, 0x5A78F8FD, 0x2117A352, 0x7A205868),
	  LGF(0xF869915D, 0x827479FB, 0x2A0FAC70, 0xD2369B35,
	      0xAA299C4A, 0x759EDBD5, 0x3C1C85C0, 0x6CF49CB7) },
	/* (2^195)*G * 2 */
	{ LGF(0xF6BA6D06, 0xBEA01D24, 0x8F9811D4, 0x2F4C099F,
	      0x3EE54433, 0xFEFB9425, 0x3F986A71, 0x26E11FA3),
	  LGF(0xFDCCE69F, 0xE7A9C455, 0x23C52866, 0xB043C24E,
	      0x3179B032, 0xCBC1DD8A, 0x4E366C38, 0x597FE7EC),
	  LGF(0x28BB5A33, 0xF41D8A29, 0xD79CEDB6, 0xB52F9C48,
	      0x62DF9B38, 0x31EC395A, 0x2AECDB76, 0x5D82A362) },
	/* (2^195)*G * 3 */
	{ LGF(0x90F2D551, 0x9902C1E0, 0xA8811C74, 0xE3C20E9F,
	      0xB83A88A9, 0x5D9FD304, 0x152AC3D0, 0x7A97A73E),
	  LGF(0x782AD7E7, 0xEBF1C192, 0x0228990B, 0xDAC867CE,
	      0x39C2A9BB, 0xB0C0AEB8, 0x2E3222F2, 0x5D529C2B),
	  LGF(0x8F1AE0C2, 0xDF8E6392, 0x0BB45ACF, 0x8692B805,
	      0xE7017825, 0x4CC66CF8, 0x70747870, 0x25F396E8) },
	/* (2^195)*G * 4 */
	{ LGF(0xD9017500, 0x825E9E80, 0xC34493C0, 0x532E7F73,
	      0x55EB285B, 0x23A615DE, 0x6278C4C5, 0x68BEB6AD),
	  LGF(0x050CE7CF, 0xE5AD1FDA, 0x398FE9E2, 0xD5179DCB,
	      0xA2B23DE9, 0x880F0F9C, 0x7C583AB6, 0x73E9DA1D),
	  LGF(0x161AD03A, 0x1BED8C4A, 0x631A3736, 0x56A385AC,
	      0xE2FDED3F, 0x55CA5A73, 0x5751514B, 0x2F2A1084) },
	/* (2^195)*G * 5 */
	{ LGF(0xA10674B8, 0xEF721415, 0x520B23D9, 0x39A98FB9,
	      0x3583A50F, 0xDD94B182, 0x359D5D64, 0x1A980F7E),
	  LGF(0xCED00FE0, 0x81533FD0, 0x457375B0, 0x2B41B323,
	      0x0B0B6412, 0x3428954D, 0x656FCDE7, 0x3FB05C6B),
	  LGF(0xCD1FF35F, 0xD6A2CFBE, 0x3E59FA2B, 0x3EB933A6,
	      0xD6CF146F, 0x0156D1B1, 0x53FBE8B0, 0x1FA8E207) },
	/* (2^195)*G * 6 */
	{ LGF(0x0A7FA85F, 0xC3201380, 0xA647E0CA, 0xCE6BBA92,
	      0xC1969616, 0xD1328954, 0xD146C39F, 0x53505DC8),
	  LGF(0xAA4DEFF3, 0xABECE8DE, 0x70FF8BED, 0xA6B25F53,
	      0x8B95875D, 0xC70C1F01, 0x80019FB8, 0x1EEE7F43),
	  LGF(0xCC1E0562, 0xCC3FC741, 0x42B13EF8, 0x93B664F2,
	      0x16798EB4, 0x5D617DE8, 0x3CAC8CA8, 0x35C68ADE) },
	/* (2^195)*G * 7 */
	{ LGF(0x7745E14A, 0x293F3FD5, 0xEE207D6F, 0x023E52D8,
	      0xA7918F82, 0xC95A3B2F, 0x3594BCCC, 0x203E41FE),
	  LGF(0xCE5A6CD2, 0xC2513D53, 0x4C9ADB58, 0x8AF5B5BD,
	      0x56292D78, 0xDF748C18, 0xC147EB47, 0x1C54D437),
	  LGF(0xA5ABCB56, 0x02C3EA61, 0xBA7BA956, 0xB56CC897,
	      0x80AA5525, 0xB8E346F8, 0x7F925E67, 0x791ECB5E) },
	/* (2^195)*G * 8 */
	{ LGF(0x67A86CB5, 0x9E436CB7, 0xF1F6D3FE, 0x9EAB45BE,
	      0xE4A81FA1, 0xA2654498, 0xB7FBF10D, 0x5A50303B),
	  LGF(0xA4853698, 0x961441BF, 0x425429D3, 0x76396E28,
	      0x49399AB4, 0x23187D9C, 0x1C754A72, 0x47EC8934),
	  LGF(0x6AF04917, 0xB5B36F77, 0x8CDEC91B, 0x4E7E5985,
	      0x56A70427, 0x47DBD9D7, 0xB2E0E163, 0x00D090A5) },
	/* (2^195)*G * 9 */
	{ LGF(0x3B6C3E84, 0x1390F65E, 0xE8EDE015, 0xD1492FB1,
	      0xF4D52DD6, 0x3ADC11E0, 0x1EF8D7C0, 0x4E1EFB69),
	  LGF(0x56DA5EBB, 0x790DDCB6, 0x8A6C1157, 0x899CA30F,
	      0xE160FF52, 0xB055E943, 0xE97A3F02, 0x0C4E4B67),
	  LGF(0x421C783F, 0xE665E42E, 0x6D8BF1FD, 0x90F6162E,
	      0x5667BD29, 0x0E7DEA66, 0xBCA04267, 0x5C4551BB) },
	/* (2^195)*G * 10 */
	{ LGF(0x76E757EA, 0x28CA2A1B, 0xB218D2B4, 0x82E93F7D,
	      0x5317BE46, 0x433AC8B1, 0xBEFBB5CF, 0x0BBAE0E5),
	  LGF(0x04D1F6B9, 0x7CBAD6A2, 0x999A4399, 0xEB725D50,
	      0x104B0670, 0x7C80807D, 0x5DF07889, 0x44B8942C),
	  LGF(0x6D315D12, 0x617C6A39, 0x86404CA2, 0x4555C3E7,
	      0x9819C28B, 0x4584662C, 0x144BDDCC, 0x3B052B97) },
	/* (2^195)*G * 11 */
	{ LGF(0x7FFDE613, 0x0D101C91, 0x631EC7C4, 0x200EE9E2,
	      0xA8A29B73, 0xD28F560E, 0xE40BF4D0, 0x2CCB2CF7),
	  LGF(0x6E7DF796, 0x30FE9671, 0x9A98317F, 0xB6A21496,
	      0x7543C3F6, 0xEB5423DB, 0xD475BB65, 0x7DD81F0A),
	  LGF(0x5FC133E0, 0x6E9D90B8, 0x4652F82F, 0x4ED76031,
	      0x89A2673B, 0x76EEE284, 0x851B3032, 0x69561B43) },
	/* (2^195)*G * 12 */
	{ LGF(0xB1499911, 0x50C89E4A, 0x513DFDB2, 0xA66737D8,
	      0x3EE958E7, 0xB3A61F72, 0x27D42687, 0x2DFA0BA1),
	  LGF(0x4BB15593, 0x041881DC, 0x90F070A8, 0x06206286,
	      0xA6239BFD, 0xF6647FFF, 0xE0A8E484, 0x60406418),
	  LGF(0xCF384958, 0xB186E710, 0x53A27045, 0xCC5BCAEE,
	      0x0823ABA8, 0x5F7FF4B1, 0x24508710, 0x3C01E53D) },
	/* (2^195)*G * 13 */
	{ LGF(0x4921B43D, 0xA5498175, 0x169BCBD8, 0x039A3283,
	      0xF7E6E8DA, 0x16210166, 0x2C1C58B5, 0x0B005155),
	  LGF(0xE250DF40, 0x02A5C58C, 0x60313C1E, 0x80A9A629,
	      0xA9A286D9, 0xEDC81102, 0x0E5DE932, 0x03C8B061),
	  LGF(0x329A1F5C, 0x9B656127, 0x90853286, 0x7F61E2E1,
	      0x901D370E, 0x220189A7, 0xE72A0992, 0x5C2FDF1B) },
	/* (2^195)*G * 14 */
	{ LGF(0x45249491, 0x1725A4D4, 0xD83A2644, 0x117438E7,
	      0x06740CDF, 0xB7E063EA, 0x9D2BB8D8, 0x6021348E),
	  LGF(0x188EE74D, 0x2359C265, 0xDB7611FC, 0x3498D65B,
	      0x286A6BD1, 0x97D9FFD6, 0x4E36165F, 0x63F77522),
	  LGF(0x5B583047, 0xD5B3AA1D, 0x7FE06CED, 0xA3107630,
	      0xA98D71BD, 0x6492A9AF, 0xB782F441, 0x534AFBC1) },
	/* (2^195)*G * 15 */
	{ LGF(0x1937FD6E, 0x66C6224B, 0xA12C63DB, 0x07778715,
	      0xB3E6947A, 0xD35D4667, 0x27D08486, 0x6C435D4A),
	  LGF(0xD34FC573, 0x14BD04EF, 0x7B42BEA3, 0x58F89E26,
	      0x083CC9BF, 0xDD4D47FA, 0x8DA29629, 0x5C69FCC3),
	  LGF(0x623C4605, 0x98A6F2B6, 0xDD9551F4, 0x4F944F4C,
	      0xD1C9E81B, 0x5A90BF07, 0x6A326820, 0x0B9C0955) },
	/* (2^195)*G * 16 */
	{ LGF(0x173AD989, 0xF4448E39, 0xAC6C7FE8, 0xB4BD88C5,
	      0xC5C899F0, 0x6217606A, 0x2CBA888E, 0x4DA95AF1),
	  LGF(0x81B3D10D, 0x805E0284, 0x39FCEFCB, 0x3E6A069F,
	      0x7FED771B, 0xDA636B90, 0xB675A4E1, 0x162581D9),
	  LGF(0x02F702CE, 0x52668A89, 0x39061B8D, 0x44FDFE7C,
	      0x252FD554, 0xEEAA462F, 0xB083E563, 0x02FA9D8F) }
};

#elif JQ == JQ255S

/* Points i*G for i = 1 to 16, affine extended format */
static const point_affine point_win_base[] = {
	/* G * 1 */
	{ LGF(0xA2789410, 0x104220CD, 0x348CC437, 0x6D7386B2,
	      0x4612D10E, 0x55E452A6, 0xA747ADAC, 0x0F520B1B),
	  LGF(0x00000003, 0x00000000, 0x00000000, 0x00000000,
	      0x00000000, 0x00000000, 0x00000000, 0x00000000),
	  LGF(0x00000009, 0x00000000, 0x00000000, 0x00000000,
	      0x00000000, 0x00000000, 0x00000000, 0x00000000) },
	/* G * 2 */
	{ LGF(0x0E9EA08D, 0x155215A0, 0xF46D0A34, 0xC474598B,
	      0xDE7F0296, 0xA479391B, 0xEFDB7348, 0x7DCAB2C9),
	  LGF(0x0D1657FC, 0xB3E22F8D, 0x219E490E, 0xE5427944,
	      0x5887FD30, 0x256C2BE7, 0x9C5869ED, 0x6F44EC74),
	  LGF(0x69B07916, 0x84CC11AA, 0x16FEEF18, 0xEC33C759,
	      0x78762D61, 0x2501ACD9, 0xD5F7C6BD, 0x42B401D3) },
	/* G * 3 */
	{ LGF(0x369F681F, 0x1C2AE84D, 0xA657BB05, 0xE7931687,
	      0x8780329D, 0xFAC101D1, 0x346F39AA, 0x407A13E6),
	  LGF(0x36F06441, 0x7204233F, 0x233F36F0, 0x36F07204,
	      0x7204233F, 0x233F36F0, 0x36F07204, 0x7204233F),
	  LGF(0x64ED4BDD, 0xB582FD50, 0xAE3D0228, 0x9F6417AA,
	      0x79D25F29, 0x10163C9F, 0xD9DADBFA, 0x3E21C102) },
	/* G * 4 */
	{ LGF(0x0A0C8EFE, 0xFCB9E72D, 0xCBC752B5, 0x879B715F,
	      0x585A38A4, 0xADBF0638, 0x83525815, 0x1153E6DF),
	  LGF(0x69223E39, 0x9204A59E, 0x4B12D8E7, 0x4E645F87,
	      0x9F5D3475, 0xF2A1145C, 0x662F1657, 0x54E64904),
	  LGF(0xF87650FD, 0x330B2867, 0x0D1CB6FA, 0x013E9EBA,
	      0x7FC59C00, 0xAB4BE93A, 0x8B86CA68, 0x11F4E281) },
	/* G * 5 */
	{ LGF(0xB7768768, 0xDE7D4909, 0xC57ADC57, 0x89E1BB43,
	      0x36A0B118, 0x850ECBC9, 0xD764C85E, 0x67D128B9),
	  LGF(0x0667B64D, 0xDF0337C0, 0x2FBA673A, 0x58856B29,
	      0x33A6D7CE, 0xC17C3E93, 0x9F0CC65D, 0x52932B9A),
	  LGF(0x423C38C3, 0x1C824AAF, 0x19778FD9, 0x635DC688,
	      0xDCBD289A, 0x3B8EBB2D, 0x3D12426F, 0x591738A9) },
	/* G * 6 */
	{ LGF(0xD0DC5ED9, 0xDB908132, 0x563AB60C, 0xBAF1C9AE,
	      0x47C0974E, 0x3F0A5A36, 0xE4758D2C, 0x1AE40CE8),
	  LGF(0x659F8304, 0x2378FCE7, 0xE92DA598, 0xF71199A7,
	      0xEE1D7E76, 0x05D59CEC, 0x18E39726, 0x7B55CE1D),
	  LGF(0xDBE905C9, 0x5C964B33, 0x22A48DBC, 0xFB8008F3,
	      0xD3C7EDD4, 0x002552F2, 0x7F5A5422, 0x42B37300) },
	/* G * 7 */
	{ LGF(0x5A305CD1, 0xBCF61643, 0x7A24E5FE, 0x8FF2EAAA,
	      0x0FFADF5D, 0x1F836A92, 0x8A25A05C, 0x77869103),
	  LGF(0x9712F248, 0xBB70A309, 0xB5C7CED6, 0x62AE8CAB,
	      0xC3D060D0, 0x150E35D2, 0x47D95DA4, 0x6EB76B26),
	  LGF(0xE42EC752, 0x5429F9F2, 0x72EDC8DB, 0x64F44E4B,
	      0xBC1B7A38, 0x6A2EACE6, 0x44734AB4, 0x59A07700) },
	/* G * 8 */
	{ LGF(0x56AF9D20, 0x1EA227FB, 0xA435E2AC, 0x342B4BA3,
	      0xE4A5FB16, 0x188E0332, 0xE64F8650, 0x4CB7E380),
	  LGF(0x93F6EEA0, 0x2DE04D6F, 0x4BD9AB93, 0x13A2A2DE,
	      0xACF9CC03, 0x4BA8485F, 0xEC331ED1, 0x26DCD74F),
	  LGF(0xFF3A091D, 0x008F881C, 0xAC529912, 0xD79E9196,
	      0xB91D9EE2, 0x27641FC9, 0x791D0D36, 0x08F2F46F) },
	/* G * 9 */
	{ LGF(0x3867E249, 0x7064AA19, 0x1BBD9331, 0x9E0262DD,
	      0x47465363, 0x28DAB54C, 0xF58996EC, 0x186E71D5),
	  LGF(0x28B2AEF7, 0xF293B014, 0x9F64AADF, 0x6544BF67,
	      0x3DC0038B, 0xC25DB7DB, 0xE8B16348, 0x641BC3EA),
	  LGF(0x553D878E, 0xC9769297, 0xF210B104, 0x5AFEBCBF,
	      0xEBC5E399, 0xC30E7E91, 0xF78889DE, 0x5344F6CB) },
	/* G * 10 */
	{ LGF(0x69A38BAF, 0xEC41FB98, 0xB257A28D, 0x55781708,
	      0x46435A2C, 0xF1BEAC33, 0xF2E5C1DF, 0x24D9FE2B),
	  LGF(0x35E0E33E, 0xAC2450CD, 0x4BE098FF, 0xA5FBAFCA,
	      0xF6CC3634, 0x15C5CFFB, 0x4DD0CC8A, 0x11EB002E),
	  LGF(0x0B3E9878, 0xB57551F4, 0x087CD2FE, 0xA1D9E1AC,
	      0xE072548A, 0xFE443CA5, 0x431C24BB, 0x3361CE26) },
	/* G * 11 */
	{ LGF(0xFDB68B79, 0x640788E8, 0xD4645856, 0xA02A4D6F,
	      0x6A54EB86, 0xB18CAB0F, 0x814FCFF1, 0x01511890),
	  LGF(0x9FDD13DE, 0x4845EE2F, 0x84D73CA8, 0x343EAC12,
	      0x33C3A9ED, 0xD7761D75, 0xD6AF1B44, 0x44204FCF),
	  LGF(0xEA04605B, 0x88C209C7, 0xB025867C, 0xDEC8E3E4,
	      0xAB8AE0C4, 0xF25BD01A, 0xBFBE67E2, 0x10E31B1D) },
	/* G * 12 */
	{ LGF(0xAC5599AB, 0xC31D48EE, 0x69F1BBA3, 0x3E87D668,
	      0x1165365F, 0x8AB43867, 0x2E25AB56, 0x42B17278),
	  LGF(0xAD8160D1, 0xEDFD8B4B, 0x5457C4A7, 0x9BBB9D36,
	      0x3BA7F9DB, 0xC2A3F962, 0x296F8D92, 0x1EA722FF),
	  LGF(0x4727F1F8, 0xDBD377FD, 0xB05F1736, 0x060E2D9D,
	      0x9422C162, 0xA86F76D5, 0x92E58063, 0x466ED475) },
	/* G * 13 */
	{ LGF(0xC42DF019, 0xFFAD311F, 0x6B1677A5, 0xCC5F0E31,
	      0x724C430C, 0xDB27E2AA, 0x7910438F, 0x72491B47),
	  LGF(0x76049D71, 0x4DEC2338, 0xA8EA7661, 0x12E5410D,
	      0x1E78A79B, 0xA8C80C9E, 0x071FC1D5, 0x56187141),
	  LGF(0x61AB363D, 0x5F87D7B1, 0x6751C8BA, 0xFADED89F,
	      0xBA3C4BCD, 0x37A78C21, 0x9C950F96, 0x3E2BE3EF) },
	/* G * 14 */
	{ LGF(0x4FE98F30, 0xB1620E56, 0x92E72DD0, 0xB119CC91,
	      0xE6CCAC7F, 0xA0040674, 0xDD41A2D1, 0x5E430A47),
	  LGF(0x87390957, 0x06930693, 0x567E3C9A, 0xC3E0CD34,
	      0x8E0CEEAE, 0x2EDB8253, 0x6D3123EB, 0x60EF0D21),
	  LGF(0xEC603124, 0xCEB292E0, 0x09C8339D, 0x18B961A3,
	      0x5D3B524A, 0x917BC541, 0x257F38AE, 0x681E3041) },
	/* G * 15 */
	{ LGF(0xCBB41565, 0x871FFDDB, 0x1DE23B6B, 0x98B52686,
	      0xC016B5CD, 0x51EBFEAA, 0xB08C48A2, 0x3BE6B06E),
	  LGF(0x57EA5D5A, 0x30100939, 0xAE95E20B, 0x99E3DE25,
	      0x449B9EFD, 0x5594A4E0, 0xCFB9A5A5, 0x100615BC),
	  LGF(0xD534F12E, 0x2CB15E4A, 0x49523FF8, 0x07A2C5FB,
	      0x8E59F7E5, 0x79BCF046, 0x8A8B99AA, 0x172AAB62) },
	/* G * 16 */
	{ LGF(0x94943C22, 0x18BF1FBA, 0xC9AF901F, 0xEE888DF3,
	      0x083B2FE3, 0xE6355FA5, 0x5B5F3676, 0x2DE21A5F),
	  LGF(0xBEE08226, 0xFE9991C8, 0xB6132ED0, 0x70631241,
	      0x930CC3DF, 0xF17032C1, 0x5E61EDCA, 0x1880F6E6),
	  LGF(0x5053E953, 0x7D5A5028, 0xE2C9B21E, 0xB0627DFE,
	      0xE54A4C3C, 0xB6548C27, 0xA2BDDB5E, 0x5C75B961) }
};

/* Points i*(2^65)*G for i = 1 to 16, affine extended format */
static const point_affine point_win_base65[] = {
	/* (2^65)*G * 1 */
	{ LGF(0xFC850F1F, 0xAD671792, 0x10F4FB5A, 0xB7D10E97,
	      0xF1498B64, 0xE59F700C, 0xA555FEFD, 0x145C3F79),
	  LGF(0x301870EC, 0xC504DF70, 0x59BEB30B, 0xC3F57575,
	      0x0E041627, 0x59EF9CAA, 0x1FDF29EA, 0x3AA6C124),
	  LGF(0xA73A5D82, 0x6D1B6960, 0xB7B83754, 0x7E905634,
	      0x66E7AD73, 0x5728E3A8, 0x4DD20707, 0x6F9938F0) },
	/* (2^65)*G * 2 */
	{ LGF(0xF769D548, 0xAEA44289, 0xD9C85032, 0xF3B3A271,
	      0x19A10A0D, 0x450F6EC6, 0xD8A60452, 0x2A078A32),
	  LGF(0xA49AB090, 0xE395C342, 0xD994BBBA, 0xBEEAEA12,
	      0x316756DE, 0xE555C4B2, 0x2A2835EA, 0x0C4B1FDF),
	  LGF(0xE71A2679, 0xACC1B6C0, 0x6A20F966, 0x3B828FBF,
	      0xF305035E, 0x2B40BE24, 0x5F9EE71D, 0x5138EFFC) },
	/* (2^65)*G * 3 */
	{ LGF(0x61C3B389, 0xFCB87370, 0x5FB663E6, 0x2EB5074C,
	      0x727B63A1, 0xC58DE35B, 0x00B2FBF9, 0x6F9AECFE),
	  LGF(0x64D47F35, 0xDD2736E0, 0x4CCCB7AD, 0x8235DE8C,
	      0x937FCD8B, 0x84993635, 0xFEACCC60, 0x05DADA9D),
	  LGF(0xCCDEEA2C, 0x041CF837, 0x8F430D4E, 0x128FA18D,
	      0xF6EC5EF7, 0x146AA2BA, 0x7A3314D7, 0x367779B5) },
	/* (2^65)*G * 4 */
	{ LGF(0x6AB767DD, 0x02E35314, 0xF6DB6D43, 0x07B12ACE,
	      0x8291AF02, 0xB844A3A3, 0xBFF84166, 0x168FE09E),
	  LGF(0x2E0DD4AF, 0x20CCC894, 0x40E6146F, 0xB05A1446,
	      0x747B5584, 0x39FB3501, 0xAA0DCC5D, 0x637DD68B),
	  LGF(0xC8D19D78, 0xF858BE49, 0xC049D501, 0x1CD2F537,
	      0x72C504DC, 0x5B82CA8E, 0xD8F74B16, 0x081DAA0C) },
	/* (2^65)*G * 5 */
	{ LGF(0xE492D81E, 0xEAAC278E, 0xDF4B5715, 0x9B679600,
	      0xF7D251E0, 0x0930AFE6, 0x4B0A57E8, 0x12B18E68),
	  LGF(0xE0B61BA4, 0x8E3257DD, 0x87333A80, 0x034912BE,
	      0x6FE3860D, 0x35538974, 0xB72FDAE3, 0x0430DD4D),
	  LGF(0x10703FE2, 0x8F69AB4C, 0xB9CA5DEA, 0x9DB06D02,
	      0xE9A31DF3, 0x80E1FCAC, 0x833EE541, 0x6CE5827F) },
	/* (2^65)*G * 6 */
	{ LGF(0xD0BE8BD1, 0x296B5AC7, 0x17A925CB, 0xF9ABC6FE,
	      0xFE2755E0, 0x9A382C9B, 0xA45DEDDC, 0x18BD1D68),
	  LGF(0x17995DE4, 0x41C60373, 0xFD0E5177, 0xEB9F7DAE,
	      0xEBEFEA23, 0x3C89F25A, 0xE2A6FEA1, 0x6BC51220),
	  LGF(0xDC0850D8, 0xA8BE5154, 0x9117FF61, 0x46B4997F,
	      0xFD05C7A5, 0xF3632B4C, 0x3A06C5F9, 0x5198473D) },
	/* (2^65)*G * 7 */
	{ LGF(0xE54EC194, 0x0C025505, 0x5E54A341, 0x16DFAAD1,
	      0x9BAC268D, 0x4E0E0938, 0x3AE0F758, 0x2C841A6C),
	  LGF(0x0B45CB36, 0x6E0C6EC0, 0xA09D7B7D, 0xDE22614D,
	      0x86C5311B, 0x68532DB3, 0x90BF3721, 0x6C76A366),
	  LGF(0x61021E16, 0xEF4DC9C5, 0xEE045148, 0xBF20C5FE,
	      0xF7E93EA0, 0xCC0911EB, 0xB042AE8F, 0x2C44B584) },
	/* (2^65)*G * 8 */
	{ LGF(0xC7DFAAFF, 0x0D409883, 0x44A2B7AC, 0xA9BB3C13,
	      0x23AA330B, 0xADEE45CF, 0x38DF8DF9, 0x32F82151),
	  LGF(0x98027EEE, 0xCF5B38B8, 0x6C358DFB, 0x2E0E5F28,
	      0xB54C83A0, 0x77F9F481, 0xD4AD38D8, 0x320289A4),
	  LGF(0x2D050368, 0x850595A4, 0x1A87EED4, 0x69338F65,
	      0xC63C7E31, 0xB56B41AA, 0xF28D875D, 0x0A7DECD4) },
	/* (2^65)*G * 9 */
	{ LGF(0x0CD6C445, 0x3EA9CA53, 0xB11E2F78, 0xF4121ED9,
	      0xDF4CBE2B, 0x32D2ACB4, 0x5C93F764, 0x6BFAD453),
	  LGF(0x30D70239, 0xE9F73ACE, 0x73FFD2AB, 0x2D6EB283,
	      0x14264D84, 0xE362CB1B, 0x47F58EDA, 0x6A58C4C4),
	  LGF(0xEB81C4F0, 0xD421A3CD, 0x973B8F0F, 0x8C5750AC,
	      0xB3184DFA, 0x43D7F20C, 0xA00332B5, 0x05EA57EA) },
	/* (2^65)*G * 10 */
	{ LGF(0x8C55EFC5, 0x6E80B93F, 0xAC82CA7F, 0xCFA05077,
	      0x997B623A, 0xFC3A41A8, 0xF388397E, 0x594955BF),
	  LGF(0x1D23F58F, 0xBD4ABD3A, 0x13D824B7, 0xFDF2BE7E,
	      0x6DA9D7C5, 0xFDF4EBAD, 0x0A760CE6, 0x6F6BF0FD),
	  LGF(0x55EADF3C, 0xB81C2566, 0xD18BC1A6, 0xBDD6F1B6,
	      0x471E8803, 0x3160D36F, 0x809C36C6, 0x5B172E38) },
	/* (2^65)*G * 11 */
	{ LGF(0xF53D8310, 0x80DDD82D, 0xFB0E0E43, 0x9FE3E489,
	      0x92FEAA7B, 0xE1EC96F4, 0x8CA0A54C, 0x0D9EFEC2),
	  LGF(0x5BEF5D82, 0xAC82BC83, 0x26768EEA, 0x1ED56E4F,
	      0xAE0F6520, 0x7CB1DF78, 0x1A793D69, 0x3F27081B),
	  LGF(0xF93EB90E, 0x057471C1, 0xFB299F2D, 0xDA034A47,
	      0xB0691BD2, 0xC51142A5, 0xFCF6A3EE, 0x5FD614C1) },
	/* (2^65)*G * 12 */
	{ LGF(0x52CB2A5C, 0xDBEFD63B, 0xC7FCE11E, 0xCA182B05,
	      0xB7E0B2B0, 0x8173D517, 0x6EBCC5BD, 0x1A664CA5),
	  LGF(0x00488C1B, 0x3C2CD45E, 0xD5AA1E98, 0x744209C1,
	      0x3F628FFF, 0xB0A6FCE8, 0x1138F9B1, 0x770D858C),
	  LGF(0x88D37A03, 0x6A96C565, 0x14153EC3, 0x733C7CA7,
	      0x78D61F3C, 0xA04AF9D1, 0xFFE4F21E, 0x10420882) },
	/* (2^65)*G * 13 */
	{ LGF(0x1CCCC730, 0x85581131, 0x61E9960D, 0x7B6D5AAC,
	      0xE26F9B9C, 0xC3729090, 0x61BABE1E, 0x035DE04A),
	  LGF(0x159B7C7E, 0x2B0FECC1, 0x3AD75522, 0x5176B224,
	      0x6C3BD8ED, 0xF578275E, 0x4A4335B5, 0x3EA45CDA),
	  LGF(0xFB8296AD, 0x399CEF5F, 0x0203F23D, 0x7BB775A1,
	      0x009BFFEE, 0xEA10679B, 0xFA69E967, 0x70977639) },
	/* (2^65)*G * 14 */
	{ LGF(0x2E20E4A3, 0x969B92FA, 0xDBFB267E, 0xC10608CE,
	      0x1B964F50, 0x03B07217, 0x4D4EAAF8, 0x513E12D6),
	  LGF(0x2B792081, 0xDCABCD1C, 0x97317B12, 0xC0B470A2,
	      0x5D26A03D, 0xB2C96A08, 0x7D7202E9, 0x52A2BF2E),
	  LGF(0x4111A128, 0x92EA95A7, 0x195F29D9, 0x8164FFF4,
	      0x39E5051E, 0x8A3F1593, 0x7E870D74, 0x43FC2262) },
	/* (2^65)*G * 15 */
	{ LGF(0xC3DF479B, 0x5B4B80D1, 0x1817958B, 0x26E5E74F,
	      0x3AED803B, 0x899B72B3, 0xFAB7DFA1, 0x07D749F9),
	  LGF(0x82ED1F56, 0xCB4E8E86, 0xE6F438C6, 0x95E7B3DA,
	      0x11E4EAE5, 0x436574C3, 0x6B6557D1, 0x5B29A9D4),
	  LGF(0x8E2A8D1F, 0xA0FF9F12, 0x527B4E88, 0xD3C936A9,
	      0xF2B0454A, 0xB279CEBE, 0x6130D747, 0x04B77CA1) },
	/* (2^65)*G * 16 */
	{ LGF(0xDB9A5B83, 0xD82E7D79, 0xDB1386D1, 0xE55F1507,
	      0x0B19751E, 0x4E2ED8BC, 0x6AD8C891, 0x191D614E),
	  LGF(0x07947CAE, 0xA0A662EF, 0x407ADF90, 0xCFD4B9B4,
	      0x4CDA26D4, 0x9698A42E, 0xFA1C5F43, 0x00D98818),
	  LGF(0x20D474E3, 0xF048C3E4, 0xCDA1B6AB, 0xA09CE0B0,
	      0x9727A4CA, 0x31ED70E8, 0xB2D89DF0, 0x214C5B3C) }
};

/* Points i*(2^130)*G for i = 1 to 16, affine extended format */
static const point_affine point_win_base130[] = {
	/* (2^130)*G * 1 */
	{ LGF(0x287B7A8C, 0xC547E3D2, 0x2E9A0A54, 0x907B6425,
	      0x55F5D398, 0x7A7BA223, 0x492D6D08, 0x5FEB95EE),
	  LGF(0x247C18A0, 0xC17AB82D, 0x6973F13C, 0x95542A3E,
	      0x9E957BD2, 0xB14CDFC7, 0xBADE4F32, 0x661229C7),
	  LGF(0x28E441B7, 0x5541C599, 0x442F368E, 0x5B912D60,
	      0x5BC7D073, 0x2EC47581, 0x03E1A1E7, 0x02802A6A) },
	/* (2^130)*G * 2 */
	{ LGF(0x0F28E7B4, 0x15A644D9, 0x4550787C, 0x2AEBD607,
	      0x83354F29, 0x319418ED, 0x2688A0B0, 0x2B3CECA0),
	  LGF(0x87129578, 0x72C27DF1, 0x94A3CAEB, 0xF443CA1C,
	      0x35C3A22D, 0xEA24368F, 0x7283DADF, 0x17A619CA),
	  LGF(0x3459B171, 0x0EC06DB4, 0x8FD7DD2D, 0x1A6558F9,
	      0xA82A6DEE, 0x4B3F9D68, 0x340F9542, 0x649637A5) },
	/* (2^130)*G * 3 */
	{ LGF(0x8F86DA38, 0xA065A103, 0x66338796, 0xBF1723E4,
	      0x18633561, 0xD922BF8C, 0x97CE7EDA, 0x168AAAD7),
	  LGF(0x2CB400ED, 0x31D729A1, 0xCEEAA6A9, 0x7A195520,
	      0x192BE4B6, 0x86F51D53, 0x1CDCF306, 0x22A23BF6),
	  LGF(0xE212FCC2, 0xCACA700A, 0x3381880A, 0x8E87630B,
	      0xBC73A980, 0xC73BF67B, 0x1AA1CCE1, 0x66C538B7) },
	/* (2^130)*G * 4 */
	{ LGF(0x7B396AC1, 0xA7C98580, 0x050B1CC0, 0xF5B27B3B,
	      0xF3BB8550, 0x199AF9DF, 0xCCECE2DE, 0x0DD0B15B),
	  LGF(0x08A28FA3, 0x767FEA82, 0x37DCA7EE, 0xA2DC84A0,
	      0x33E1DE6F, 0xEBD2AC82, 0x21AD21CA, 0x188F7085),
	  LGF(0xBA423FFD, 0x3AC52264, 0x3B333AA4, 0xEF5048FF,
	      0xEC2F8A1A, 0x2FBD912D, 0x670533CF, 0x2BEF1130) },
	/* (2^130)*G * 5 */
	{ LGF(0x9FC7024F, 0xCD8A44DA, 0xBC29AA93, 0xF737A550,
	      0x1E9106B9, 0x0D895281, 0x2DA132F4, 0x2392F5F4),
	  LGF(0x64875239, 0xB8201817, 0x342A7459, 0x5B4DD1BB,
	      0x0BE03B56, 0xC66B48D8, 0x636EA208, 0x7F08F18F),
	  LGF(0x72019CF2, 0x2CC60206, 0x05B43C11, 0x14E86A76,
	      0x5DE4C795, 0xB9CE10F5, 0xBA073D7D, 0x67B29276) },
	/* (2^130)*G * 6 */
	{ LGF(0xF864F06F, 0x43D22D5E, 0x3BE07C7C, 0x69548FFF,
	      0x3EDD77B8, 0xEBB4F5A3, 0x2CA0CD5B, 0x5C5A9324),
	  LGF(0x2E2ECC30, 0x9FFA28B9, 0x78DA5C00, 0xFB480B56,
	      0xB14804D7, 0xE1C94F9A, 0x48C9FA1D, 0x7D453CC9),
	  LGF(0x84BD090A, 0x99B10D26, 0x702948B9, 0xAA8361C9,
	      0x15F7685C, 0x8FE833DB, 0x46BA5349, 0x5A40858A) },
	/* (2^130)*G * 7 */
	{ LGF(0xC806064F, 0xFE714DB9, 0x96052BB3, 0xEC532CBC,
	      0x62B03E4E, 0xFEB56676, 0x52734277, 0x7D201AFC),
	  LGF(0x55F5DEF5, 0x87848119, 0x33271AC2, 0x48D26621,
	      0xA6C81237, 0xAF3F2A3F, 0xE01B2E7D, 0x1A086A6F),
	  LGF(0xAA0289C2, 0xE60717C5, 0x465F8A77, 0x06226B3A,
	      0xC053632B, 0xAAE6A9C9, 0x16332425, 0x74441004) },
	/* (2^130)*G * 8 */
	{ LGF(0xC3D8E685, 0x9C901178, 0x5E9E8C93, 0x3FD38D2D,
	      0x3AECC482, 0xA0478BE3, 0x263956AC, 0x507DF5A8),
	  LGF(0x59DBD92B, 0x1DC9A2A8, 0x6245BBB3, 0xBF15C84C,
	      0xB2CAB143, 0x2A8D65D0, 0x17A5525E, 0x3DE43A80),
	  LGF(0x13E8DD16, 0x20400890, 0x9E66D5ED, 0x98B67294,
	      0x76BC1D61, 0x2468A4B8, 0x4A76A7C6, 0x1D2C24FA) },
	/* (2^130)*G * 9 */
	{ LGF(0xA15BFEC5, 0x4E0B3042, 0xB78EFCC3, 0x25107B05,
	      0x0AB84B6A, 0x895F5129, 0x6CD0570A, 0x2875F452),
	  LGF(0x891C8CF1, 0xDB52105D, 0xF01D372F, 0xA1E811D5,
	      0x0ADED951, 0x5CF867DE, 0xE0B4DA8B, 0x17052800),
	  LGF(0xD990BDFE, 0x465AF151, 0xED9253B0, 0x61C1C858,
	      0xA3FBDE66, 0x565E0443, 0x53E17835, 0x12DB650B) },
	/* (2^130)*G * 10 */
	{ LGF(0x29D2E833, 0x01496672, 0xED0CA017, 0x01916903,
	      0x293857A0, 0xB4E997E9, 0x85A3DB38, 0x52B3D9F4),
	  LGF(0x46496B31, 0x126C8477, 0x626487D7, 0x23D37316,
	      0x982B39EB, 0x46505E02, 0x2143582E, 0x23434575),
	  LGF(0xA0420BAA, 0x6BBD04A3, 0x425C1B2E, 0x9684CAEB,
	      0x89C6CAF4, 0x2D86F805, 0xA6568CBE, 0x625467C6) },
	/* (2^130)*G * 11 */
	{ LGF(0x228C2235, 0x8444EE09, 0xBBF1E2D0, 0xC67ADEED,
	      0x16B2CFD7, 0x78F69BE7, 0xB95FF650, 0x7B0AF5F2),
	  LGF(0xBECE2651, 0xD51EDD52, 0xD6FA4957, 0x792F3E2E,
	      0x8E6F750E, 0x8349ED26, 0xEF6F6D88, 0x6BB94CDF),
	  LGF(0x5088C5A8, 0xE4774F54, 0xFFA8E04D, 0x43CAA7EC,
	      0x0644B58D, 0x2CBAD6B9, 0xA1180C6F, 0x66EC80D6) },
	/* (2^130)*G * 12 */
	{ LGF(0x8591C1AD, 0xAF4B65BA, 0x17CEAF23, 0x34946626,
	      0xD86050C8, 0x67D0F7B0, 0x2F7E3A00, 0x7DD5EFEF),
	  LGF(0xDE2129A6, 0x89340045, 0xE46000DB, 0x10CF7923,
	      0x2C69ABE2, 0xA17D1BEB, 0x203C2DA4, 0x1F3592DE),
	  LGF(0xC9A50DDD, 0x946A6220, 0x90529CD9, 0xB19B06C5,
	      0xC2D4D2BA, 0x932CF781, 0x2633F97D, 0x643EC048) },
	/* (2^130)*G * 13 */
	{ LGF(0x02E72852, 0x681F8DB2, 0x6FBEA675, 0x1321374B,
	      0x23DB7E8C, 0xD874B177, 0x0693A8A2, 0x19430AF8),
	  LGF(0x57355C90, 0x0F4767AD, 0x19124EA8, 0xCAD900D8,
	      0xE702318A, 0xB4B045E5, 0x3A7E1058, 0x7AB6CB35),
	  LGF(0x0505358F, 0xF5572A21, 0x20133EC8, 0x0C528F0B,
	      0xB3606AA3, 0xC425B464, 0x6CAB14B8, 0x33E0CFF1) },
	/* (2^130)*G * 14 */
	{ LGF(0x862E3AF0, 0x1739D8DE, 0xB3D5ED7B, 0xEAEA8FAE,
	      0x4B06CE7A, 0x394668C3, 0x5FC95D4F, 0x26B43F06),
	  LGF(0x01A95CF8, 0xE62B26D3, 0x48398A95, 0x84BC9EEC,
	      0x267875A4, 0x08F76F6E, 0x9A0A50A1, 0x32D3B1A4),
	  LGF(0xEAB69B8C, 0x1E1D0050, 0x31875B73, 0x5FAE0C5B,
	      0x29295DB8, 0x5FD8D77B, 0x8F5136E0, 0x720DA33C) },
	/* (2^130)*G * 15 */
	{ LGF(0xC609C455, 0x53614A3B, 0xD229E43B, 0x1FDF44E6,
	      0xEC35FF3A, 0xC69D5CC6, 0xD4386D2C, 0x40D8B2ED),
	  LGF(0x58819185, 0x9A0B63FF, 0x1086D125, 0xF789BF22,
	      0xB1E1776A, 0xAD341ED8, 0x46EBF733, 0x082617E1),
	  LGF(0xB0AF72AF, 0x94BAFDC6, 0x67032F80, 0xCE7FD958,
	      0x42E3E324, 0x60A041C7, 0xE0C48A92, 0x52BAFECD) },
	/* (2^130)*G * 16 */
	{ LGF(0x6686F776, 0xB5972247, 0x42C88D93, 0xFCAABBE4,
	      0xFA237DD5, 0x0A7A9E0D, 0x43FCED53, 0x2DBB938F),
	  LGF(0xC1409964, 0xF50D8973, 0xECBBDB1D, 0x7E7D988F,
	      0x0DE07B83, 0x64BAB168, 0xE277A32E, 0x588C179E),
	  LGF(0xF1FFD3B3, 0x62A6B63B, 0xA4AEC308, 0xAA7B23F4,
	      0xBC75755C, 0x52196FBD, 0xD4635289, 0x60C42361) }
};

/* Points i*(2^195)*G for i = 1 to 16, affine extended format */
static const point_affine point_win_base195[] = {
	/* (2^195)*G * 1 */
	{ LGF(0x52EF088D, 0xEB4B028B, 0x511D0BF8, 0x7BFFA172,
	      0x4AA70C44, 0x20A72891, 0x23A0A46D, 0x1982687F),
	  LGF(0x4F6950E1, 0xD5386C02, 0xF2C711D7, 0xB46F6A19,
	      0x7F90F84F, 0xD13612F7, 0xC35B1108, 0x02B1FE40),
	  LGF(0x3C8937C9, 0x24BC6E86, 0xD4AE4DC9, 0xA15165FB,
	      0x4BAA772F, 0xF54207D1, 0x734C65E6, 0x4ECAC7AF) },
	/* (2^195)*G * 2 */
	{ LGF(0xF5DF3C6B, 0x6736CB36, 0x9E622FFF, 0xAAC155C4,
	      0x6C9AF425, 0x08BF6AEC, 0x3E754687, 0x32453BFF),
	  LGF(0x01D9BE69, 0x4D205228, 0x30005684, 0x491F6D95,
	      0x61348B5B, 0x03948795, 0xC7DA4C9E, 0x207807A3),
	  LGF(0x5081D09A, 0x39C46014, 0xEEDD6F53, 0xE3BB41D1,
	      0x2AECE484, 0x8BAB5AC4, 0x073E261D, 0x7BE8AD2F) },
	/* (2^195)*G * 3 */
	{ LGF(0x0240D5AB, 0xE2163ED4, 0xD2EDE213, 0x0D36DDAF,
	      0x84DA7A94, 0xB5DA9A3A, 0x81AD5800, 0x616BB179),
	  LGF(0x42399CCB, 0x62329BF6, 0x54BDD8E5, 0xD15830A7,
	      0x45F76715, 0x052C83B7, 0x77A075E5, 0x2C2FEC42),
	  LGF(0x050E4AFB, 0xCA181F67, 0x3CABC581, 0x8FA110DC,
	      0x46B0A4C3, 0xEC7B337F, 0x8F72400F, 0x61465925) },
	/* (2^195)*G * 4 */
	{ LGF(0xC01F036B, 0x11A5ECC6, 0x3DCFAB11, 0x00905EC7,
	      0xBE833EB9, 0xBD182106, 0x520D6147, 0x21345004),
	  LGF(0x823F9793, 0x33F1C936, 0x8646E9E2, 0x45EB4CD7,
	      0x49110514, 0xB28C3C8D, 0x39F3AA65, 0x20B293F7),
	  LGF(0xF465E491, 0xECE0E7C9, 0x3E295315, 0x6E3D99DD,
	      0x0D23AB0D, 0x11CAE9FC, 0x3D65F08E, 0x395BB5CD) },
	/* (2^195)*G * 5 */
	{ LGF(0xB04EFBCB, 0xC3465879, 0x7DF1D9F8, 0xDDD7400D,
	      0xBB68E927, 0x3FC45AA3, 0x7CC560D9, 0x0D654BC9),
	  LGF(0x4D4F0DAA, 0x3F09E8FE, 0xB553CA96, 0xE47EE2E2,
	      0xAC578533, 0x9FFA3C3A, 0xA08113C1, 0x356B0675),
	  LGF(0x6CF3E78E, 0x2F2E5ED1, 0xDD5F3D89, 0x3739979B,
	      0x9FB489BE, 0x6DA80DAB, 0xBDA14A65, 0x431D2978) },
	/* (2^195)*G * 6 */
	{ LGF(0xE71501A7, 0x93683AAA, 0x01C67235, 0x7A150F18,
	      0x300E35B7, 0x5F345C36, 0x605D8B43, 0x7A900684),
	  LGF(0xC8E7B387, 0xA143C257, 0x6CC108D4, 0x62D88913,
	      0xBB72837C, 0xE32F52E0, 0x12F67001, 0x076CCDEE),
	  LGF(0x262D9E46, 0x02F43BF7, 0x5B679BDB, 0xC600EF04,
	      0x5A115558, 0x75B858E1, 0x25934593, 0x580DEEF0) },
	/* (2^195)*G * 7 */
	{ LGF(0x09001CAE, 0xCAF3D56D, 0x7F61B1E3, 0x56CCAFC5,
	      0x3BA1A346, 0x6BE0EB31, 0xB592478D, 0x4AB3F0BE),
	  LGF(0x8AEDB101, 0x971F761E, 0x3B802862, 0xD10EB0D7,
	      0xDE5AD3F6, 0x100B543B, 0x27445B8E, 0x2D2D355F),
	  LGF(0x90669B3D, 0xBB6B5140, 0x9368BB18, 0x6AF73229,
	      0xC4662114, 0xD33B393F, 0x2E5F9548, 0x44B393C4) },
	/* (2^195)*G * 8 */
	{ LGF(0xC96EBE06, 0x073F794D, 0x6642E5AB, 0xF9AAA789,
	      0x18DC4F59, 0x47F09728, 0x70B031C5, 0x30E51AB0),
	  LGF(0x79CB6EA2, 0x6AC99539, 0x43904E3A, 0xB754075F,
	      0x6C51FC54, 0x0F5C5D70, 0x6A945237, 0x72F0811D),
	  LGF(0x7656E362, 0xBB17DDD3, 0x405FA3BF, 0x9F321433,
	      0x72A57EE6, 0x326CCE63, 0x0F2A9875, 0x2CEB4E80) },
	/* (2^195)*G * 9 */
	{ LGF(0xE1675B2C, 0x39611A8F, 0x7A122549, 0xFF7E1AF2,
	      0x7B6325F6, 0xBF6F1A75, 0x55883EDC, 0x074A5FED),
	  LGF(0xCAFEB583, 0xF2B5C06F, 0xA1A2C97B, 0x2D0EFBE6,
	      0x1442F46D, 0x466067CE, 0xC999DD67, 0x13349ABF),
	  LGF(0x2D09A54C, 0xA4D40FF4, 0x16DF1800, 0xC003C77D,
	      0xE6393676, 0xA0761711, 0x64969117, 0x2C1C0AAC) },
	/* (2^195)*G * 10 */
	{ LGF(0xA4EEFBF4, 0xA3062A05, 0x726FAB1E, 0xE5D5C2E0,
	      0x531F05A1, 0x60DC0E92, 0x89936E48, 0x4643A40C),
	  LGF(0x453A8A8A, 0x1F87C746, 0x5DA27586, 0xDDCF1D00,
	      0xC7664BF3, 0xC77308C7, 0x6436D94B, 0x5774724E),
	  LGF(0x6CB73C25, 0xAA14EFB7, 0x3DC2B252, 0xAFD7FD33,
	      0xCCBF5685, 0x58B8811A, 0x0471A372, 0x1C9D11E4) },
	/* (2^195)*G * 11 */
	{ LGF(0x3B886BC0, 0xD13062D6, 0xB8D4C07B, 0x59FB0704,
	      0x6A5FE90E, 0x012B93CE, 0xA4404810, 0x5884A598),
	  LGF(0x9987EBED, 0x08A8CB0C, 0xC4D7168A, 0x05428BBD,
	      0x0207E6AA, 0x5FDD560F, 0x2497FC6D, 0x4CD603F7),
	  LGF(0xBAB59BD5, 0x3F08842C, 0x0024CECC, 0x1502C744,
	      0x6BE0B6EA, 0x7241E80F, 0xFB557160, 0x6942504C) },
	/* (2^195)*G * 12 */
	{ LGF(0x7B7A2079, 0x94CCC8EE, 0x9368F9D1, 0x98E6462E,
	      0x3D44A6C4, 0xF97BFF02, 0x4D545564, 0x5396F045),
	  LGF(0x7F72D537, 0xB4754DE2, 0x2AC5B1C3, 0xC0F01E80,
	      0x4B64FC09, 0x6492E0F0, 0xE1CDCAD1, 0x14C0F255),
	  LGF(0x516E0A97, 0x392EB3FD, 0x05B71FB1, 0x4EC3DC56,
	      0x7BDCED64, 0x7F75ADBF, 0xC96482D6, 0x6FC86668) },
	/* (2^195)*G * 13 */
	{ LGF(0xBD6BFE31, 0xEAD383AB, 0x9F6E73B8, 0xBCB0813E,
	      0xBA25DFF8, 0xC58E3723, 0x6C6C9E5B, 0x4F9DA6DF),
	  LGF(0xF580DF1C, 0xCF158917, 0xD3E90C45, 0x0E14A880,
	      0xF882B1C7, 0x60D780AF, 0xED165A4B, 0x420C0614),
	  LGF(0xB121A745, 0xCBDDA904, 0xC9A6880F, 0x31611DAE,
	      0x31FEC743, 0xB43A2ACF, 0xE4E1F3CE, 0x5F142474) },
	/* (2^195)*G * 14 */
	{ LGF(0xD884B154, 0x37DADA6D, 0x051FC1FE, 0x9BD20465,
	      0xD36E5167, 0xF618BF0E, 0x85F6BE8F, 0x6713DDBB),
	  LGF(0xA182F658, 0xE3DF626A, 0xD410C765, 0x0AEB0A55,
	      0x9DD03BA1, 0x645809F3, 0x5AE057DB, 0x40C86021),
	  LGF(0xCF581D58, 0xB2C0705F, 0xFC64B6A1, 0xB136DC9F,
	      0x919623A2, 0xB0115208, 0x83722184, 0x15D34F32) },
	/* (2^195)*G * 15 */
	{ LGF(0x2ECAC269, 0xA304489B, 0x3145A960, 0x8482C85F,
	      0x2CDBD9D0, 0x4A240F13, 0x29DB3D90, 0x2AE80644),
	  LGF(0x8436B829, 0xBA4D3B43, 0xB0906610, 0xFD932027,
	      0x56D1C683, 0x43EFF537, 0xFF8301C5, 0x0C078789),
	  LGF(0x0F98A082, 0x206A563B, 0xC630CCDC, 0xBFEFB28B,
	      0x5BE2309A, 0xE3FA9751, 0x904FD7A5, 0x71698D07) },
	/* (2^195)*G * 16 */
	{ LGF(0x494F6306, 0x320F03CA, 0x8E79B929, 0x5AA58711,
	      0x3C3A18CF, 0x4BAD3FD4, 0x0DF3849B, 0x255DDF7E),
	  LGF(0x18A991C5, 0x2C949105, 0x18F00C14, 0xAAC1F71E,
	      0x9768A122, 0x2E1D217E, 0xBB0B3E0B, 0x50A69209),
	  LGF(0x3C888F1E, 0xF93D8C08, 0x59145CCC, 0x57112855,
	      0x44969A4C, 0x1F0F91EC, 0x4C260615, 0x7AB91A18) }
};

#else
#error Unknown curve
#endif

/* ===================================================================== */
/*
 * SECTION 5: CRYPTOGRAPHIC SCHEMES
 *
 * This part implements the high-level schemes (signature, key exchange...)
 * as documented in the jq255.h API.
 */

#include "blake2s.h"
#include "jq255.h"

#if JQ == JQ255E
#define jq_private_key            jq255e_private_key
#define jq_public_key             jq255e_public_key
#define jq_keypair                jq255e_keypair
#define jq_generate_private_key   jq255e_generate_private_key
#define jq_make_public            jq255e_make_public
#define jq_generate_keypair       jq255e_generate_keypair
#define jq_decode_private_key     jq255e_decode_private_key
#define jq_decode_public_key      jq255e_decode_public_key
#define jq_decode_keypair         jq255e_decode_keypair
#define jq_encode_private_key     jq255e_encode_private_key
#define jq_encode_public_key      jq255e_encode_public_key
#define jq_encode_keypair         jq255e_encode_keypair
#define jq_sign                   jq255e_sign
#define jq_sign_seeded            jq255e_sign_seeded
#define jq_verify                 jq255e_verify
#define jq_ECDH                   jq255e_ECDH
#elif JQ == JQ255S
#define jq_private_key            jq255s_private_key
#define jq_public_key             jq255s_public_key
#define jq_keypair                jq255s_keypair
#define jq_generate_private_key   jq255s_generate_private_key
#define jq_make_public            jq255s_make_public
#define jq_generate_keypair       jq255s_generate_keypair
#define jq_decode_private_key     jq255s_decode_private_key
#define jq_decode_public_key      jq255s_decode_public_key
#define jq_decode_keypair         jq255s_decode_keypair
#define jq_encode_private_key     jq255s_encode_private_key
#define jq_encode_public_key      jq255s_encode_public_key
#define jq_encode_keypair         jq255s_encode_keypair
#define jq_sign                   jq255s_sign
#define jq_sign_seeded            jq255s_sign_seeded
#define jq_verify                 jq255s_verify
#define jq_ECDH                   jq255s_ECDH
#else
#error Unknown curve
#endif

/*
 * A private key is a scalar; a public key is a point, and its encoded
 * version (32 bytes). The API structure contains blobs into which we
 * copy our internal structures when approprivate.
 * In some cases, we get "invalid key" values. They use the following
 * representations:
 *  - Invalid private key: the scalar value is zero.
 *  - Invalid public key: the point is
 *    original invalid encoding is kept in the slot for the Z coordinate.
 */

/* see jq255.h */
void
jq_generate_private_key(jq_private_key *sk, const void *seed, size_t seed_len)
{
	blake2s_context bc;
	uint8_t tmp[32];
	scalar s;

	/* Hash the seed to produce 32 bytes, which are then decoded as a
	   scalar. */
	blake2s_init(&bc, 32);
	blake2s_update(&bc, seed, seed_len);
	blake2s_final(&bc, tmp);
	scalar_decode_reduce(&s, tmp, 32);

	/* It is very improbable that a zero is obtained, but just for
	   completeness, in that case, we add 1. */
	scalar_select(&s, &s, &scalar_one, scalar_is_zero(&s));
	memcpy(sk, &s, sizeof s);
}

/* see jq255.h */
void
jq_make_public(jq_public_key *pk, const jq_private_key *sk)
{
	scalar s;
	point p;

	memcpy(&s, sk, sizeof s);
	point_mulgen(&p, &s);
	memcpy(pk, &p, sizeof p);
	point_encode((unsigned char *)pk + sizeof p, &p);
}

/* see jq255.h */
void
jq_generate_keypair(jq_keypair *jk, const void *seed, size_t seed_len)
{
	jq_generate_private_key(&jk->private_key, seed, seed_len);
	jq_make_public(&jk->public_key, &jk->private_key);
}

/* see jq255.h */
int
jq_decode_private_key(jq_private_key *sk, const void *src, size_t len)
{
	scalar s;
	uint32_t r;

	if (len != 32) {
		s = scalar_zero;
		r = 0;
	} else {
		r = scalar_decode(&s, src);
		r &= ~scalar_is_zero(&s);
	}
	memcpy(sk, &s, sizeof s);
	return (int)(r & 1);
}

/* see jq255.h */
int
jq_decode_public_key(jq_public_key *pk, const void *src, size_t len)
{
	point p;
	uint32_t r;

	if (len != 32) {
		p = point_neutral;
		memset((unsigned char *)pk + sizeof p, 0, 32);
		r = 0;
	} else {
		r = point_decode(&p, src);
		r &= ~point_is_neutral(&p);
		memcpy((unsigned char *)pk + sizeof p, src, 32);
	}
	memcpy(pk, &p, sizeof p);
	return (int)(r & 1);
}

/* see jq255.h */
int
jq_decode_keypair(jq_keypair *jk, const void *src, size_t len)
{
	uint32_t r;
	scalar s;
	point p;

	if (len != 64) {
		memcpy(&jk->private_key, &scalar_zero, sizeof scalar_zero);
		memcpy(&jk->public_key, &point_neutral, sizeof point_neutral);
		memset((unsigned char *)&jk->public_key + sizeof(point), 0, 32);
		return 0;
	}
	r = scalar_decode(&s, src);
	r &= ~scalar_is_zero(&s);
	r &= point_decode(&p, (const uint8_t *)src + 32);
	r &= ~point_is_neutral(&p);
	scalar_select(&s, &scalar_zero, &s, r);
	point_select(&p, &point_neutral, &p, r);
	memcpy(&jk->private_key, &s, sizeof s);
	memcpy(&jk->public_key, &p, sizeof p);
	memcpy((unsigned char *)&jk->public_key + sizeof(point),
		(const uint8_t *)src + 32, 32);
	return (int)(r & 1);
}

/* see jq255.h */
size_t
jq_encode_private_key(void *dst, const jq_private_key *sk)
{
	scalar s;

	memcpy(&s, sk, sizeof s);
	scalar_encode(dst, &s);
	return 32;
}

/* see jq255.h */
size_t
jq_encode_public_key(void *dst, const jq_public_key *pk)
{
	uint8_t tmp[32];
	point p;
	uint32_t r;

	/* If the key is valid, then we already have a copy of its encoding,
	   so we can just copy that value. If the key is invalid, then
	   the encoding is invalid and we must produce from zeros. */
	memcpy(&p, pk, sizeof p);
	memcpy(tmp, (const uint8_t *)pk + sizeof(point), 32);
	r = ~point_is_neutral(&p);
	for (int i = 0; i < 32; i ++) {
		tmp[i] &= r;
	}
	memcpy(dst, tmp, 32);
	return 32;
}

/* see jq255.h */
size_t
jq_encode_keypair(void *dst, const jq_keypair *jk)
{
	jq_encode_private_key(dst, &jk->private_key);
	jq_encode_public_key((uint8_t *)dst + 32, &jk->public_key);
	return 64;
}

/*
 * Compute the per-signature secret scalar k. The private key is provided
 * as the scalar `sec`; the public key is provided as `epub` (encoded).
 */
static void
make_sign_k(scalar *k, const scalar *sec, const void *epub,
	const char *hash_name, const void *hv, size_t hv_len,
	const void *seed, size_t seed_len)
{
	blake2s_context bc;
	unsigned char tmp[32];

	blake2s_init(&bc, 32);
	scalar_encode(tmp, sec);
	blake2s_update(&bc, tmp, 32);
	blake2s_update(&bc, epub, 32);
	for (int i = 0; i < 8; i ++) {
		tmp[i] = (uint8_t)((uint64_t)seed_len >> (8 * i));
	}
	blake2s_update(&bc, tmp, 8);
	blake2s_update(&bc, seed, seed_len);
	if (hash_name == NULL || hash_name[0] == 0) {
		tmp[0] = 0x52;
		blake2s_update(&bc, tmp, 1);
	} else {
		tmp[0] = 0x48;
		blake2s_update(&bc, tmp, 1);
		blake2s_update(&bc, hash_name, strlen(hash_name) + 1);
	}
	blake2s_update(&bc, hv, hv_len);
	blake2s_final(&bc, tmp);
	scalar_decode_reduce(k, tmp, 32);
}

/*
 * Compute the "challenge" part of the signature. The challenge has
 * length exactly 16 bytes.
 */
static void
make_challenge(void *dst, const point *r, const void *epub,
	const char *hash_name, const void *hv, size_t hv_len)
{
	blake2s_context bc;
	unsigned char tmp[32];

	blake2s_init(&bc, 32);
	point_encode(tmp, r);
	blake2s_update(&bc, tmp, 32);
	blake2s_update(&bc, epub, 32);
	if (hash_name == NULL || hash_name[0] == 0) {
		tmp[0] = 0x52;
		blake2s_update(&bc, tmp, 1);
	} else {
		tmp[0] = 0x48;
		blake2s_update(&bc, tmp, 1);
		blake2s_update(&bc, hash_name, strlen(hash_name) + 1);
	}
	blake2s_update(&bc, hv, hv_len);
	blake2s_final(&bc, tmp);
	memcpy(dst, tmp, 16);
}

/* see jq255.h */
size_t
jq_sign(void *sig, const jq_keypair *jk,
	const char *hash_name, const void *hv, size_t hv_len)
{
	return jq_sign_seeded(sig, jk, hash_name, hv, hv_len, NULL, 0);
}

/* see jq255.h */
size_t
jq_sign_seeded(void *sig, const jq_keypair *jk,
	const char *hash_name, const void *hv, size_t hv_len,
	const void *seed, size_t seed_len)
{
	scalar sec, k, s;
	point r;
	const void *epub;
	unsigned char tmp[32];

	memcpy(&sec, &jk->private_key, sizeof sec);
	epub = (const uint8_t *)&jk->public_key + sizeof(point);

	/* Per-signature secret scalar k. */
	make_sign_k(&k, &sec, epub, hash_name, hv, hv_len, seed, seed_len);

	/* R = k*G */
	point_mulgen(&r, &k);

	/* c = H(R, Q, m) */
	make_challenge(tmp, &r, epub, hash_name, hv, hv_len);

	/* s = k + sec*c */
	scalar_decode_reduce(&s, tmp, 16);
	scalar_mul(&s, &s, &sec);
	scalar_add(&s, &s, &k);

	memcpy(sig, tmp, 16);
	scalar_encode((uint8_t *)sig + 16, &s);
	return 48;
}

/* see jq255.h */
int
jq_verify(const void *sig, size_t sig_len, const jq_public_key *pk,
	const char *hash_name, const void *hv, size_t hv_len)
{
	point p;
	const void *epub;
	scalar s;
	unsigned char tmp[16];
	uint32_t c[4];

	/* Valid signatures have length 48 bytes exactly. */
	if (sig_len != 48) {
		return 0;
	}

	/* If the public key is invalid, report a failure. */
	memcpy(&p, pk, sizeof p);
	epub = (const uint8_t *)pk + sizeof(point);
	if (point_is_neutral(&p)) {
		return 0;
	}

	/* Decode scalar s and challenge c. */
	if (!scalar_decode(&s, (const uint8_t *)sig + 16)) {
		return 0;
	}
	for (int i = 0; i < 4; i ++) {
		c[i] = dec32le((const uint8_t *)sig + 4 * i);
	}

	/* Recompute R = s*G - c*Q */
	point_neg(&p, &p);
	point_mul128_add_mulgen_vartime(&p, &p, c, &s);

	/* Recompute the challenge c. Signature is valid if that value
	   matches what was received as part of the signature. */
	make_challenge(tmp, &p, epub, hash_name, hv, hv_len);
	return memcmp(tmp, sig, 16) == 0;
}

/* see jq255.h */
int
jq_ECDH(void *shared_key,
	const jq_keypair *jk_self, const jq_public_key *pk_peer)
{
	point p;
	scalar s;
	const uint8_t *epub_self;
	const uint8_t *epub_peer;
	uint32_t bad;
	uint8_t shared[32], tmp[64];
	blake2s_context bc;

	/*
	 * Get peer public key; set the 'bad' flag to true if the peer
	 * public key was invalid.
	 */
	memcpy(&p, pk_peer, sizeof p);
	epub_peer = (const uint8_t *)pk_peer + sizeof(point);
	bad = point_is_neutral(&p);

	/*
	 * Get our private key, and multiply the peer public key with it.
	 * The encoded output point is the candidate shared secret.
	 */
	memcpy(&s, &jk_self->private_key, sizeof s);
	point_mul(&p, &p, &s);
	point_encode(shared, &p);

	/*
	 * If the peer key was not valid, replace the shared secret with
	 * our own private key. This will lead to an output key unguessable
	 * by outsiders, but will otherwise not leak whether the process
	 * worked or not.
	 */
	scalar_encode(tmp, &s);
	for (int i = 0; i < 32; i ++) {
		shared[i] ^= bad & (shared[i] ^ tmp[i]);
	}

	/*
	 * Derive the key with BLAKE2s.
	 * We need to order the two public keys lexicographically.
	 */
	epub_self = (const uint8_t *)&jk_self->public_key + sizeof(point);
	uint32_t cc = 0;
	for (int i = 31; i >= 0; i --) {
		cc = ((uint32_t)epub_self[i]
			- (uint32_t)epub_peer[i] - cc) >> 31;
	}
	uint32_t z1 = -cc;
	uint32_t z2 = ~z1;
	for (int i = 0; i < 32; i ++) {
		tmp[i]      = (epub_self[i] & z1) | (epub_peer[i] & z2);
		tmp[i + 32] = (epub_self[i] & z2) | (epub_peer[i] & z1);
	}
	blake2s_init(&bc, 32);
	blake2s_update(&bc, tmp, 64);
	tmp[0] = 0x53 - (bad & (0x53 - 0x46));
	blake2s_update(&bc, tmp, 1);
	blake2s_update(&bc, shared, 32);
	blake2s_final(&bc, shared_key);
	return (int)(bad + 1);
}

# jq255e and j255s

The jq255e and jq255s are prime-order groups defined over some
[double-odd elliptic curves](https://doubleodd.group/) (i.e. elliptic
curves over finite fields, whose order is twice an odd integer). This
code implements cryptographic signatures and key exchange (ECDH).

The [jq255.h](jq255.h) file implements the API for these operations. The
code is meant to be portable, yet efficient; internally, it
automatically chooses between a 32-bit and a 64-bit backends, depending
on the local architecture. All operations (exception signature
verification) are constant-time. The code performs no heap memory
allocation; everything is contained on the stack. The [jq255.c](jq255.c)
file can be compiled for one or the other of the two groups, depending
on the value og hte `JQ` macro at the time of compilation; compile it
twice to use both curves (jq255e is the default and preferred group).

These groups offer **128-bit security level**. All encodings are
canonical and verified. A public key (group element) encodes over 32
bytes. A private key (scalar) encodes over 32 bytes. Signatures have
length **48 bytes** (i.e. they are shorter than the usual ECDSA or
Ed25519 on other curves with comparable security, which use 64-byte
signatures). Operations are fast (on an Intel "Coffee Lake" CPU in
64-bit mode, compiled with Clang 14.0.0, jq255e can sign in 56300
cycles, verify in 96500 cycles, and perform a key exchange in 95000
cycles; these figures include the cost of decoding the peer's public key
from bytes).

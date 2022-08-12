CC = clang
CFLAGS = -Wall -Wextra -Wundef -Wshadow -O2 -march=native
LD = clang
LDFLAGS =
LIBS =

OBJ_JQ255E = blake2s.o jq255e.o
OBJ_JQ255S = blake2s.o jq255s.o
OBJ_TEST_JQ255E = test_jq255e.o
OBJ_TEST_JQ255S = test_jq255s.o

all: test_jq255e test_jq255s

clean:
	-rm -f test_jq255e test_jq255s $(OBJ_JQ255E) $(OBJ_JQ255S) $(OBJ_TEST_JQ255E) $(OBJ_TEST_JQ255S)

test_jq255e: $(OBJ_JQ255E) $(OBJ_TEST_JQ255E)
	$(LD) $(LDFLAGS) -o test_jq255e $(OBJ_JQ255E) $(OBJ_TEST_JQ255E)

test_jq255s: $(OBJ_JQ255S) $(OBJ_TEST_JQ255S)
	$(LD) $(LDFLAGS) -o test_jq255s $(OBJ_JQ255S) $(OBJ_TEST_JQ255S)

blake2s.o: blake2s.c blake2s.h
	$(CC) $(CFLAGS) -c -o blake2s.o blake2s.c

jq255e.o: jq255.c jq255.h blake2s.h
	$(CC) $(CFLAGS) -DJQ=JQ255E -c -o jq255e.o jq255.c

jq255s.o: jq255.c jq255.h blake2s.h
	$(CC) $(CFLAGS) -DJQ=JQ255S -c -o jq255s.o jq255.c

test_jq255e.o: test_jq255.c jq255.h blake2s.h
	$(CC) $(CFLAGS) -DJQ=JQ255E -c -o test_jq255e.o test_jq255.c

test_jq255s.o: test_jq255.c jq255.h blake2s.h
	$(CC) $(CFLAGS) -DJQ=JQ255S -c -o test_jq255s.o test_jq255.c

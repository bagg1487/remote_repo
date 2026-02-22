CC=gcc
CFLAGS=-O2
TEST=test_cblas_level2.c
REAL_EXE=level2_real
FAKE_EXE=level2_fake

all: real fake

real:
	$(CC) $(CFLAGS) $(TEST) -lopenblas -lm -o $(REAL_EXE)

fake: libfakeblas.a
	$(CC) $(CFLAGS) $(TEST) libfakeblas.a -lm -o $(FAKE_EXE)

libfakeblas.a: fake_blas.o
	ar rcs libfakeblas.a fake_blas.o

fake_blas.o: fake_blas.c
	$(CC) -c fake_blas.c -o fake_blas.o

run_real:
	OPENBLAS_NUM_THREADS=4 ./$(REAL_EXE)

run_fake:
	./$(FAKE_EXE)

clean:
	rm -f *.o *.a $(REAL_EXE) $(FAKE_EXE)
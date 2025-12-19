CC = gcc
CFLAGS = -I../include -O3 -Wall -Wno-gnu-folding-constant -fsanitize=address -fno-omit-frame-pointer -g
LIBS = -lm

ASAN_ENV = ASAN_OPTIONS=verbosity=1

all: test_density_matrix test_wavefunction test_pauli test_fermionic

test_density_matrix: test_density_matrix.c density_matrix.c pauli.c wavefunction.c
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)
	$(ASAN_ENV) ./$@
	rm -f test_density_matrix
	rm -rf $@.dSYM

test_wavefunction: test_wavefunction.c wavefunction.c pauli.c
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)
	$(ASAN_ENV) ./$@
	rm -f test_wavefunction
	rm -rf *.dSYM

test_pauli: test_pauli.c pauli.c
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)
	$(ASAN_ENV) ./$@
	rm -f test_pauli
	rm -rf $@.dSYM

test_fermionic: test_fermionic.c fermionic.c wavefunction.c pauli.c
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)
	$(ASAN_ENV) ./$@
	rm -f test_fermionic
	rm -rf $@.dSYM

run: all
	@for t in test_density_matrix test_wavefunction test_pauli test_fermionic; do \
		$(ASAN_ENV) ./$$t; \
	done

clean:
	rm -f test_density_matrix test_wavefunction test_pauli test_fermionic
	rm -rf *.dSYM
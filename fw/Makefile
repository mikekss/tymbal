# fw/Makefile — HOST build and tests of the DSP building blocks against golden (§2.4).
# Target build (N6) — from the CubeIDE project ported from the CubeN6 example;
# the same .c files are included there unchanged (except main_target.c: -DN6_TARGET).
CC = cc
CFLAGS = -O2 -Wall -Wextra -std=c11 -lm

SRC = src/pqmf_synth.c src/skeleton_b.c src/wowflutter.c src/midi.c \
      src/voice.c src/pipeline.c src/npu_stub.c src/n6_fir.c

test: build/host_test
	cd $(CURDIR) && ./build/host_test

# Checkpoint 4 (H1): host reference for MVE validation -> build/ck4_ref.bin (src/ck4.h)
ck4: build/host_ck4
	cd $(CURDIR) && ./build/host_ck4

# n6_fir_coeffs.h must be listed in the dependencies: it sits in the fw/ root
# and is not covered by the src/*.h mask — otherwise replacing it (stub ->
# real) does NOT rebuild the tests and the CK4 reference (caught 4 Aug: make
# ck4 returned the old CRC from the old binary, and that looked like "the
# reference did not move").
build/host_ck4: test/host_ck4.c $(SRC) src/ck4.c src/*.h pqmf_coeffs.h n6_bandresp.h n6_fir_coeffs.h
	mkdir -p build
	$(CC) -o $@ test/host_ck4.c $(SRC) src/ck4.c $(CFLAGS)

build/host_test: test/host_test.c $(SRC) src/*.h pqmf_coeffs.h n6_bandresp.h n6_fir_coeffs.h
	mkdir -p build
	$(CC) -o $@ test/host_test.c $(SRC) $(CFLAGS)

# --- Cortex-M55 model in QEMU (mps3-an547) ------------------------------
# Why: the host tests run the SCALAR branches, while the hot path on the
# board is vector, and until now it could only be checked by flashing.
# QEMU gives the same MVE decoder:
#   make qemu      — instructions per hop by section (wt/pro/bod/dec/ph)
#   make qemu-ck4  — the CK4 score through the vector branches, cross-checked
#                    against the host
# Absolute cycles must not be taken from there (the cache and dual issue are
# not modelled) — the numbers are comparable between VARIANTS, not with the
# board. Syntax check of the TARGET-ONLY files (npu_neuralart/npu_boot/
# n6_weights): they are built neither by the host nor by QEMU — only by
# CubeIDE. Requires the ST headers alongside (see the header of
# test/check_target.sh); without them it is silently skipped.
check-target:
	@./test/check_target.sh

qemu: check-target
	./test/qemu/build.sh prof

qemu-ck4: ck4
	./test/qemu/build.sh ck4
	@python3 -c "import numpy as np; \
a=np.fromfile('build/ck4_qemu.bin',dtype=np.float32).astype(np.float64); \
b=np.fromfile('build/ck4_ref.bin',dtype=np.float32).astype(np.float64); \
r=np.sqrt(((a-b)**2).mean())/np.sqrt((b**2).mean()); \
print('[ck4 qemu] MVE vs scalar: rel RMS = %.3e (threshold 1e-4) %s'%(r,'OK' if r<1e-4 else 'FAIL')); \
raise SystemExit(0 if r<1e-4 else 1)"

# Render the chain to WAV without the board (D-19 A/B and listening tests before the DAC):
#   make play && ./build/host_play out.wav [uni_cents bloom_k drift_cents]
play: build/host_play
build/host_play: test/host_play.c $(SRC) src/*.h pqmf_coeffs.h n6_bandresp.h n6_fir_coeffs.h
	mkdir -p build
	$(CC) -o $@ test/host_play.c $(SRC) $(CFLAGS)

clean:
	rm -rf build
.PHONY: test ck4 qemu qemu-ck4 check-target play clean

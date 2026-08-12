#!/bin/sh
# build.sh — build and run the firmware DSP on the Cortex-M55 model (QEMU).
#   ./test/qemu/build.sh prof   — instruction counter by hot-path section
#   ./test/qemu/build.sh ck4    — the CK4 score through the VECTOR branches -> build/ck4_qemu.bin
# Run from fw/. Requires arm-none-eabi-gcc and qemu-system-arm >= 8.
set -e
Q=test/qemu
CF="-O2 -std=c11 -mcpu=cortex-m55 -mfpu=auto -mfloat-abi=hard -mthumb -DN6_TARGET"
CF="$CF -Wall -Wextra -ffunction-sections -fdata-sections -I src -I ."
O=/tmp/n6q; mkdir -p $O build
case "${1:-prof}" in
  prof) MAIN=$Q/qemu_prof.c; SRC="pqmf_synth skeleton_b wowflutter midi voice pipeline npu_stub n6_fir" ;;
  ck4)  MAIN=$Q/qemu_ck4.c;  SRC="pqmf_synth skeleton_b wowflutter midi voice pipeline npu_stub ck4 n6_fir" ;;
  *) echo "prof | ck4"; exit 2 ;;
esac
OBJ=""
arm-none-eabi-gcc $CF -c $Q/startup.c -o $O/startup.o; OBJ="$OBJ $O/startup.o"
arm-none-eabi-gcc $CF -c $Q/qemu_io.c -o $O/io.o;      OBJ="$OBJ $O/io.o"
arm-none-eabi-gcc $CF -c $MAIN        -o $O/main.o;    OBJ="$OBJ $O/main.o"
for f in $SRC; do arm-none-eabi-gcc $CF -c src/$f.c -o $O/$f.o; OBJ="$OBJ $O/$f.o"; done
arm-none-eabi-gcc $CF -T $Q/an547.ld -nostartfiles -Wl,--gc-sections -specs=nano.specs \
                  -o $O/n6q.elf $OBJ -lm
timeout 600 qemu-system-arm -M mps3-an547 -cpu cortex-m55 -nographic -monitor none \
     -serial none -semihosting-config enable=on,target=native -icount shift=0 -kernel $O/n6q.elf || true

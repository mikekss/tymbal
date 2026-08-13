#!/bin/sh
# check_target.sh — a local syntax check of the TARGET-ONLY sources.
#
# Why: n6_weights.c, npu_boot.c, npu_neuralart.c are built neither by the host
# nor by the QEMU rig — they pull in HAL, CMSIS, BSP and ll_aton. Until 4 Aug
# nothing compiled them outside the CubeIDE project, and a definition-order
# error went out to the board and burned a build round. The flags here are a
# copy of the production ones from CubeIDE (-mcmse matters: without it the
# secure aliases RISAFx_S are not visible).
#
# The ST headers are not kept in the repository (29 MB of someone else's
# code). To rebuild them:
#   on the device:  tar czf hdrs.tar.gz  -C <STEdgeAI>/Middlewares/ST/AI/Npu .
#                   tar czf hdrs2.tar.gz -C <Cube>/Drivers CMSIS/Include \
#                       CMSIS/Device/ST/STM32N6xx/Include STM32N6xx_HAL_Driver/Inc \
#                       -C <project>/FSBL/Core Inc -C <repo>/models/t1 \
#                       gen_app_safe/network.h gen_app_safe/stai_network.h
#                   tar czf hdrs3.tar.gz -C <Cube>/Drivers BSP
#   locally:        unpack into $SDK/{ll_aton_root,drv}
SDK=${SDK:-../sdk}
if [ ! -d "$SDK/ll_aton_root/ll_aton" ]; then
  echo "[check-target] no ST headers in $SDK — skipping (see the script header)"
  exit 0
fi
CF="-fsyntax-only -O2 -std=gnu11 -Wall -Wextra -mcpu=cortex-m55 -mfpu=auto"
CF="$CF -mfloat-abi=hard -mthumb -mcmse"
CF="$CF -DUSE_HAL_DRIVER -DN6_TARGET -DSTM32N657xx"
CF="$CF -DLL_ATON_PLATFORM=LL_ATON_PLAT_STM32N6 -DLL_ATON_OSAL=LL_ATON_OSAL_BARE_METAL"
INC="-I $SDK/drv/STM32N6xx_HAL_Driver/Inc -I $SDK/drv/CMSIS/Device/ST/STM32N6xx/Include"
INC="$INC -I $SDK/drv/CMSIS/Include -I $SDK/drv/Inc -I $SDK/drv/BSP/STM32N6xx_Nucleo"
INC="$INC -I $SDK/drv/BSP/Components/mx25um51245g -I $SDK/ll_aton_root/ll_aton"
INC="$INC -I $SDK/ll_aton_root/Devices/STM32N6xx -I $SDK/drv/gen_app_safe -I src -I ."
rc=0
for f in npu_neuralart npu_boot n6_weights; do
  if arm-none-eabi-gcc $CF $INC src/$f.c; then echo "[check-target] $f.c OK"
  else echo "[check-target] $f.c FAIL"; rc=1; fi
done
exit $rc

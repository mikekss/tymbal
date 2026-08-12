@echo off
rem M0 scaling probe: e1 (d30-clone, C=32) with the FINAL r14 settings
rem (all-onchip map + int8-io + epoch controller). Reference: DevCloud e1 = 2.74 ms.
setlocal enabledelayedexpansion
set STEDGEAI=C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe
cd /d "%~dp0\.."
set RUN=models\t0\out\m0_e1_%RANDOM%
mkdir %RUN%
echo Output: %RUN%
call :one e1_ctrl_c32_l12_qdq n6-nucleo-full-onchip "--input-data-type int8 --output-data-type int8 --enable-epoch-controller" e1_allonchip_ec
echo Done. The output folder is named above.
pause
exit /b

:one
echo === %~4 ===
"%STEDGEAI%" analyze --model models\t0\diag\%~1.onnx --target stm32n6 --st-neural-art %~2@fw\neural_art.json %~3 --workspace %RUN%\ws\%~4 --output %RUN%\%~4 > %RUN%\log_%~4.txt 2>&1
findstr /C:"TOOL ERROR" /C:"INTERNAL ERROR" /C:"CliRuntimeError" %RUN%\log_%~4.txt >nul
if not errorlevel 1 (echo    FAIL) else (echo    ok)
exit /b

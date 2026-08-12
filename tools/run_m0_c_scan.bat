@echo off
rem M0 C-scan: d32 (C=128) + d33 (C=96), gather-form, r14 settings.
setlocal enabledelayedexpansion
set STEDGEAI=C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe
cd /d "%~dp0\.."
set RUN=models\t0\out\m0_cscan_%RANDOM%
mkdir %RUN%
echo Output: %RUN%
call :one d32_bh_c128_v3_l12_qdq n6-nucleo-full-onchip "--input-data-type int8 --output-data-type int8 --enable-epoch-controller" d32_allonchip_ec
call :one d33_bh_c96_v3_l12_qdq n6-nucleo-full-onchip "--input-data-type int8 --output-data-type int8 --enable-epoch-controller" d33_allonchip_ec
echo Done. The output folder is named above.
pause
exit /b

:one
echo === %~4 ===
"%STEDGEAI%" analyze --model models\t0\diag\%~1.onnx --target stm32n6 --st-neural-art %~2@fw\neural_art.json %~3 --workspace %RUN%\ws\%~4 --output %RUN%\%~4 > %RUN%\log_%~4.txt 2>&1
findstr /C:"TOOL ERROR" /C:"INTERNAL ERROR" /C:"CliRuntimeError" %RUN%\log_%~4.txt >nul
if not errorlevel 1 (echo    FAIL) else (echo    ok)
exit /b

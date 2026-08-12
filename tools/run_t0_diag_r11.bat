@echo off
rem T0 round 11: smaller architectures (L=12 / L=18, k=3) on canonical ORT-QDQ path.
setlocal enabledelayedexpansion
set STEDGEAI=C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe
cd /d "%~dp0\.."
set RUN=models\t0\out\r11_%RANDOM%
mkdir %RUN%
echo Output: %RUN%
call :one d27_bh_c128_v12_l12_qdq n6-nucleo        "--input-data-type int8 --output-data-type int8" d27_nucleo_i8io
call :one d27_bh_c128_v12_l12_qdq n6-nucleo-onchip "--input-data-type int8 --output-data-type int8" d27_onchip_i8io
call :one d28_bh_c128_v12_l18_qdq n6-nucleo        "--input-data-type int8 --output-data-type int8" d28_nucleo_i8io
call :one d28_bh_c128_v12_l18_qdq n6-nucleo-onchip "--input-data-type int8 --output-data-type int8" d28_onchip_i8io
call :one d27_bh_c128_v12_l12_qdq n6-nucleo "--enable-epoch-controller --input-data-type int8 --output-data-type int8" d27_nucleo_ec
echo Done. The output folder is named above.
pause
exit /b

:one
echo === %~4 ===
"%STEDGEAI%" analyze --model models\t0\diag\%~1.onnx --target stm32n6 --st-neural-art %~2@fw\neural_art.json %~3 --workspace %RUN%\ws\%~4 --output %RUN%\%~4 > %RUN%\log_%~4.txt 2>&1
findstr /C:"TOOL ERROR" /C:"INTERNAL ERROR" /C:"CliRuntimeError" %RUN%\log_%~4.txt >nul
if not errorlevel 1 (echo    FAIL) else (echo    ok)
exit /b

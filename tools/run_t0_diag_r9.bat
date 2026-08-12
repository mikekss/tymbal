@echo off
rem T0 round 9: final-form candidates (ORT conv conventions + int8 movement + quantized y).
setlocal enabledelayedexpansion
set STEDGEAI=C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe
cd /d "%~dp0\.."
set RUN=models\t0\out\r9_%RANDOM%
mkdir %RUN%
echo Output: %RUN%
call :one d23_bh_c128_v12_i8f n6-nucleo        "" d23_nucleo
call :one d23_bh_c128_v12_i8f n6-nucleo        "--input-data-type int8 --output-data-type int8" d23_nucleo_i8io
call :one d23_bh_c128_v12_i8f n6-nucleo-onchip "--input-data-type int8 --output-data-type int8" d23_onchip_i8io
call :one d23_bh_c128_v12_i8f n6-audio         "" d23_audio
call :one d24_bh_c192_v3_i8f  n6-nucleo        "--input-data-type int8 --output-data-type int8" d24_nucleo_i8io
echo Done. The output folder is named above.
pause
exit /b

:one
echo === %~4 ===
"%STEDGEAI%" analyze --model models\t0\diag\%~1.onnx --target stm32n6 --st-neural-art %~2@fw\neural_art.json %~3 --workspace %RUN%\ws\%~4 --output %RUN%\%~4 > %RUN%\log_%~4.txt 2>&1
findstr /C:"TOOL ERROR" /C:"INTERNAL ERROR" /C:"CliRuntimeError" %RUN%\log_%~4.txt >nul
if not errorlevel 1 (echo    FAIL) else (echo    ok)
exit /b

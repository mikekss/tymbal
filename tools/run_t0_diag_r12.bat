@echo off
rem T0 round 12 (final static): candidates to close T0.
rem d16 = canonical wiring-B graph from r5 (ORT-QDQ straight, matcher-approved).
setlocal enabledelayedexpansion
set STEDGEAI=C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe
cd /d "%~dp0\.."
set RUN=models\t0\out\r12_%RANDOM%
mkdir %RUN%
echo Output: %RUN%
call :one d16_bh_c192_v3_qdq2  n6-nucleo "--input-data-type int8 --output-data-type int8" d16_nucleo_i8io
call :one d16_bh_c192_v3_qdq2  n6-nucleo "--input-data-type int8 --output-data-type int8 --enable-epoch-controller" d16_nucleo_ec
call :one d29_bh_c96_v8_l24_qdq n6-nucleo "--input-data-type int8 --output-data-type int8" d29_nucleo_i8io
call :one d30_bh_c192_v3_l12_qdq n6-nucleo "--input-data-type int8 --output-data-type int8" d30_nucleo_i8io
call :one d30_bh_c192_v3_l12_qdq n6-nucleo-onchip "--input-data-type int8 --output-data-type int8" d30_onchip_i8io
echo Done. The output folder is named above.
pause
exit /b

:one
echo === %~4 ===
"%STEDGEAI%" analyze --model models\t0\diag\%~1.onnx --target stm32n6 --st-neural-art %~2@fw\neural_art.json %~3 --workspace %RUN%\ws\%~4 --output %RUN%\%~4 > %RUN%\log_%~4.txt 2>&1
findstr /C:"TOOL ERROR" /C:"INTERNAL ERROR" /C:"CliRuntimeError" %RUN%\log_%~4.txt >nul
if not errorlevel 1 (echo    FAIL) else (echo    ok)
exit /b

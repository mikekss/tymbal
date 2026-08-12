@echo off
rem M0 candidate matrix: S (strict T=48) + R (relaxed T=96), gather2, r14 settings.
setlocal enabledelayedexpansion
set STEDGEAI=C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe
cd /d "%~dp0\.."
set RUN=models\t0\out\m0_matrix_%RANDOM%
mkdir %RUN%
echo Output: %RUN%
call :one d37_bh_c80_v3_l12_g2_qdq       d37_s_c80v3
call :one d38_bh_c88_v3_l12_g2_qdq       d38_s_c88v3
call :one d39_bh_c96_v2_l12_g2_qdq       d39_s_c96v2
call :one d40_bh_c128_v2_l12_g2_qdq      d40_s_c128v2
call :one d41_bh_c128_v2_l12_g2_t96_qdq  d41_r_c128v2t96
call :one d42_bh_c88_v3_l12_g2_t96_qdq   d42_r_c88v3t96
call :one d43_bh_c64_v3_l24_g2_t96_qdq   d43_r_c64v3l24t96
echo Done. The output folder is named above.
pause
exit /b

:one
echo === %~2 ===
"%STEDGEAI%" analyze --model models\t0\diag\%~1.onnx --target stm32n6 --st-neural-art n6-nucleo-full-onchip@fw\neural_art.json --input-data-type int8 --output-data-type int8 --enable-epoch-controller --workspace %RUN%\ws\%~2 --output %RUN%\%~2 > %RUN%\log_%~2.txt 2>&1
findstr /C:"TOOL ERROR" /C:"INTERNAL ERROR" /C:"CliRuntimeError" %RUN%\log_%~2.txt >nul
if not errorlevel 1 (echo    FAIL) else (echo    ok)
exit /b

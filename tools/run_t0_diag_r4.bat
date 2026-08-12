@echo off
rem T0 round 4: epoch-controller flag + k5/L12 shape + nucleo mpool (no hyperRAM).
setlocal enabledelayedexpansion
set STEDGEAI=C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe
cd /d "%~dp0\.."
set RUN=models\t0\out\r4_%RANDOM%
mkdir %RUN%
echo Output: %RUN%
call :one d12_bh_c128_v12_qdq  n6-audio  "--enable-epoch-controller" d12_ec
call :one d14_bh_c128_v12_k5l12_qdq n6-audio "" d14_base
call :one d14_bh_c128_v12_k5l12_qdq n6-audio "--enable-epoch-controller" d14_ec
call :one d12_bh_c128_v12_qdq  n6-nucleo "" d12_nucleo
call :one d14_bh_c128_v12_k5l12_qdq n6-nucleo "" d14_nucleo
echo Done. The output folder is named above.
pause
exit /b

:one
echo === %~4 ===
"%STEDGEAI%" analyze --model models\t0\diag\%~1.onnx --target stm32n6 --st-neural-art %~2@fw\neural_art.json %~3 --workspace %RUN%\ws\%~4 --output %RUN%\%~4 > %RUN%\log_%~4.txt 2>&1
findstr /C:"TOOL ERROR" /C:"INTERNAL ERROR" %RUN%\log_%~4.txt >nul
if not errorlevel 1 (echo    FAIL) else (echo    ok)
exit /b

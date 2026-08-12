@echo off
rem T0 round 6: unified-scale QDQ graphs (d17/d18).
setlocal enabledelayedexpansion
set STEDGEAI=C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe
cd /d "%~dp0\.."
set RUN=models\t0\out\r6_%RANDOM%
mkdir %RUN%
echo Output: %RUN%
call :one d17_bh_c128_v12_qdqu n6-audio         "" d17_audio
call :one d17_bh_c128_v12_qdqu n6-nucleo        "" d17_nucleo
call :one d17_bh_c128_v12_qdqu n6-nucleo-onchip "" d17_onchip
call :one d17_bh_c128_v12_qdqu n6-nucleo "--enable-epoch-controller" d17_nucleo_ec
call :one d18_bh_c192_v3_qdqu  n6-nucleo        "" d18_nucleo
echo Done. The output folder is named above.
pause
exit /b

:one
echo === %~4 ===
"%STEDGEAI%" analyze --model models\t0\diag\%~1.onnx --target stm32n6 --st-neural-art %~2@fw\neural_art.json %~3 --workspace %RUN%\ws\%~4 --output %RUN%\%~4 > %RUN%\log_%~4.txt 2>&1
findstr /C:"TOOL ERROR" /C:"INTERNAL ERROR" /C:"CliRuntimeError" %RUN%\log_%~4.txt >nul
if not errorlevel 1 (echo    FAIL) else (echo    ok)
exit /b

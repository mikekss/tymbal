@echo off
rem T0 diag round 3 (fix): analyze QDQ batch-as-height graphs d12 and d13.
setlocal enabledelayedexpansion
set STEDGEAI=C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe
cd /d "%~dp0\.."
set RUN=models\t0\out\qdq_%RANDOM%
mkdir %RUN%
echo Output: %RUN%
for %%m in (models\t0\diag\d12_bh_c128_v12_qdq.onnx models\t0\diag\d13_bh_c192_v3_qdq.onnx) do (
  echo === %%~nm ===
  "%STEDGEAI%" analyze --model "%%m" --target stm32n6 --st-neural-art n6-audio@fw\neural_art.json --workspace %RUN%\ws\%%~nm --output %RUN%\%%~nm > %RUN%\log_%%~nm.txt 2>&1
  findstr /C:"TOOL ERROR" /C:"INTERNAL ERROR" %RUN%\log_%%~nm.txt >nul
  if not errorlevel 1 (echo    FAIL) else (echo    ok)
)
echo Done. The output folder is named above.
pause

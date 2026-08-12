@echo off
rem T0 diag: stedgeai analyze for models\t0\diag\*.onnx. Each graph runs twice:
rem with the neural-art profile and plain (isolates profile vs model issues).
setlocal enabledelayedexpansion
set STEDGEAI=C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe
cd /d "%~dp0\.."
set RUN=models\t0\out\diag_%RANDOM%
mkdir %RUN%
echo Output: %RUN%
for %%m in (models\t0\diag\*.onnx) do (
  echo === %%~nm ===
  "%STEDGEAI%" analyze --model "%%m" --target stm32n6 --st-neural-art n6-audio@fw\neural_art.json --workspace %RUN%\ws\%%~nm --output %RUN%\%%~nm > %RUN%\log_%%~nm.txt 2>&1
  findstr /C:"TOOL ERROR" /C:"INTERNAL ERROR" %RUN%\log_%%~nm.txt >nul
  if not errorlevel 1 (echo    FAIL-art) else (echo    ok-art)
  "%STEDGEAI%" analyze --model "%%m" --target stm32n6 --workspace %RUN%\ws\%%~nm_plain --output %RUN%\%%~nm_plain > %RUN%\log_%%~nm_plain.txt 2>&1
  findstr /C:"TOOL ERROR" /C:"INTERNAL ERROR" %RUN%\log_%%~nm_plain.txt >nul
  if not errorlevel 1 (echo    FAIL-plain) else (echo    ok-plain)
)
echo Done. The output folder is named above.
pause

@echo off
rem T0: stedgeai analyze for all ONNX in models\t0 (pipeline sec. 3.2). Run: double click.
rem Each run writes to a FRESH folder models\t0\out\run_<N> (FS-bridge cache workaround).
setlocal enabledelayedexpansion
set STEDGEAI=C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe
cd /d "%~dp0\.."
set RUN=models\t0\out\run_%RANDOM%
mkdir %RUN%
echo Output: %RUN%
"%STEDGEAI%" --version > %RUN%\stedgeai_version.txt 2>&1
set NOK=0
for %%m in (models\t0\*.onnx) do (
  echo === %%~nm ===
  "%STEDGEAI%" analyze --model "%%m" --target stm32n6 --st-neural-art n6-audio@fw\neural_art.json --workspace models\t0\ws\%%~nm --output %RUN%\%%~nm > %RUN%\log_%%~nm.txt 2>&1
  findstr /C:"TOOL ERROR" /C:"INTERNAL ERROR" /C:"NOT IMPLEMENTED" %RUN%\log_%%~nm.txt >nul
  if not errorlevel 1 (
    echo    FAIL - see %RUN%\log_%%~nm.txt
    set NOK=1
  ) else (
    if exist %RUN%\%%~nm\*report*.txt (echo    OK) else (echo    WARN - no report file & set NOK=1)
  )
)
echo.
if %NOK%==0 (echo ALL GREEN. Reports in %RUN%) else (echo SOME FAILED - see logs in %RUN%)
pause

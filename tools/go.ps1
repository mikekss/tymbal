# go.ps1 -- one button: build, dump NOR, capture UART. ASCII only on purpose.
#
# WHY. Build, NOR dump and UART capture were three separate commands typed by
# hand in the right order every time. This script wraps that whole cycle into
# one call and tees the output into C:\ST\Projects\N6\run.log, so one run can
# be compared against the previous one instead of scrolling the console back.
#
# USAGE
#   & .\tools\go.ps1                 # build only
#   & .\tools\go.ps1 -uart 20        # + capture UART (board must be running)
#   & .\tools\go.ps1 -dump           # + read NOR (no debug session!)
# If execution policy blocks it:  powershell -ep bypass -f tools\go.ps1
#
# -Uart needs the board already running (IDE Debug launched, or flash boot).

param(
    [switch]$Dump,
    [int]$Uart = 0,
    [string]$Port = "COM8",
    [int]$Baud = 115200
)

$ErrorActionPreference = "Continue"

$ROOT = "C:\ST\Projects\N6"
$LOG  = Join-Path $ROOT "run.log"
$DEBUGDIR = "C:\Users\<user>\STM32Cube\Repository\STM32Cube_FW_N6_V1.4.0\Projects\NUCLEO-N657X0-Q\Examples\SAI\N6_m1\STM32CubeIDE\FSBL\Debug"
$PLUG = "C:\ST\STM32CubeIDE_2.1.1\STM32CubeIDE\plugins"
$MAKE = "$PLUG\com.st.stm32cube.ide.mcu.externaltools.make.win32_2.2.200.202604021615\tools\bin"
$GCC  = "$PLUG\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740\tools\bin"
$PROG = "C:\ST\STM32CubeProgrammer\bin"
$LOADER = "$PROG\ExternalLoader\MX25UM51245G_STM32N6570-NUCLEO.stldr"

function Log($s) {
    Write-Host $s
    $s | Out-File -FilePath $LOG -Append -Encoding utf8
}

"" | Out-File -FilePath $LOG -Encoding utf8
Log ("=== go.ps1 " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " ===")

# ---- build ---------------------------------------------------------------
$env:PATH = "$MAKE;$GCC;$env:PATH"
Push-Location $DEBUGDIR
Log "--- build ---"
$out = & make -j8 all 2>&1
$rc = $LASTEXITCODE
# keep the log readable: full command lines are noise, errors are not
$out | Where-Object { $_ -notmatch '^arm-none-eabi-gcc "' } | ForEach-Object { Log $_ }
Pop-Location
Log ("--- build exit code: " + $rc + " ---")
if ($rc -ne 0) { Log "BUILD FAILED, stopping"; exit 1 }

# ---- NOR dump ------------------------------------------------------------
if ($Dump) {
    Log "--- NOR dump ---"
    foreach ($a in @("0x70000000", "0x70400000", "0x70400010")) {
        $d = & "$PROG\STM32_Programmer_CLI.exe" -c port=SWD mode=HOTPLUG -el "$LOADER" -r32 $a 32 2>&1
        $d | Where-Object { $_ -match '^0x7' } | ForEach-Object { Log $_ }
    }
}

# ---- UART capture --------------------------------------------------------
if ($Uart -gt 0) {
    if ($Port -eq "") {
        $names = [System.IO.Ports.SerialPort]::GetPortNames()
        Log ("--- COM ports found: " + ($names -join ", ") + " ---")
        if ($names.Count -gt 0) { $Port = $names[-1] }
    }
    if ($Port -eq "") { Log "no COM port, skipping UART" }
    else {
        Log ("--- UART " + $Port + " @" + $Baud + ", " + $Uart + " s ---")
        $sp = New-Object System.IO.Ports.SerialPort $Port, $Baud, "None", 8, "one"
        $sp.ReadTimeout = 500
        $sp.Encoding = [System.Text.Encoding]::UTF8
        try {
            $sp.Open()
            $deadline = (Get-Date).AddSeconds($Uart)
            while ((Get-Date) -lt $deadline) {
                try { $line = $sp.ReadLine(); Log $line.TrimEnd() } catch { }
            }
        } catch { Log ("UART error: " + $_.Exception.Message) }
        finally { if ($sp.IsOpen) { $sp.Close() } }
    }
}

Log "=== done ==="

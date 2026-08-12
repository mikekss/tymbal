# VERSIONS.md — exact versions of everything

Rule: any change to a tool version gets a line here, with the date.

## Windows machine (C:\ST) — read off the install directories; confirm by hand
| Tool | Version | Source / check |
|---|---|---|
| ST Edge AI Core (`stedgeai`) | 4.0 | `C:\ST\STEdgeAI\4.0` (stedgeai.exe under Utilities\windows); confirm with `stedgeai --version`; the stm32n6 target is present (Projects\STM32N6570-DK, N6_scripts, mpool profiles) |
| Neural-ART pool profiles | — | `C:\ST\STEdgeAI\4.0\scripts\N6_scripts\my_mpools\stm32n6*.mpool` — the template for the fw/neural_art profile in T0 |
| STM32CubeIDE | 2.1.1 | `C:\ST\STM32CubeIDE_2.1.1` (an older `STM32CubeIDE` sits next to it) |
| STM32CubeMX | 6.18.0 (confirmed 29 Jul, Help → About) | `C:\ST\STM32CubeMX` |
| STM32CubeProgrammer | 2.23.0 (confirmed 29 Jul, bin/version) | `C:\ST\STM32CubeProgrammer` |
| STM32Cube_FW_N6 package | **V1.4.0** (found 18 Jul: Users\<user>\STM32Cube\Repository) | port list for H0; LL_ATON ships with STEdgeAI, not here |
| ST Edge AI Developer Cloud | service available | https://stedgeai-dc.st.com — check for an N6 board in the farm after logging in with myST (first day of T0) |

## Training machine: laptop, GTX 1060 Max-Q **3 GB** (Pascal sm_61; corrected 2 Aug — not 6 GB)
| Tool | Version | Note |
|---|---|---|
| torch | **2.4.1+cu118** (installed 2 Aug) | GLOBAL environment, not a venv, by choice; cuda.is_available()=True, device reported as "GTX 1060 with Max-Q Design" |
| numpy | **2.5.0** (2 Aug, pulled in by the onnx install) | torch 2.4.1 imports and `cuda.is_available()=True` — verified. numpy 2.x works with torch 2.4, but the pairing is off the beaten path: if anything looks strange, fall back with `pip install "numpy<2"` |
| scipy / soundfile | installed (2 Aug) | versions to be pinned down when convenient |
| onnx | **1.22.0** (installed 2 Aug) | builds the gather2 graph, opset 17 (train/export_gather2.py) |
| onnxruntime | **1.28.0** (installed 2 Aug) | CPU build; needed both for graph cross-checks and for `onnxruntime.quantization.quantize_static` |
| sympy | being installed (2 Aug) | HIDDEN dependency of `onnxruntime.quantization`: without it the import fails in `symbolic_shape_infer`. `pip install sympy` |
| brevitas / auraloss | not installed | not needed under the current plan: quantization is PTQ through ORT, and the loss is our own |

## Verification sandbox (Linux, phase 0 runs — 2026-07-15)
| Tool | Version |
|---|---|
| Python | 3.10.12 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| git | 2.34.1 |

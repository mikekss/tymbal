#!/bin/sh
# Requires: linux/atonn from the ST Edge AI installation (Utilities/linux — a static
# x86-64 ELF that lives inside the WINDOWS installation), configs/*.mdesc|cdesc,
# fw/*.mpool and the preprocessed ONNX + Q.json from the generation directory.
# run_variant.sh <name> [extra atonn options...]
# Running the Neural-ART compiler locally. The base is the exact command line from
# network_generate_report.txt of the current graph; everything after the name is added.
#
# SCOPE BOUNDARY. atonn has 95 options, and it is ONLY the backend — the mapping
# of an already preprocessed and quantized graph onto the engines. Frontend
# switches are not here and never will be:
#   --no-inputs-allocation / --no-outputs-allocation   (user-allocated IO)
#   --inputs-ch-position chfirst|chlast
#   --outputs-ch-position chfirst|chlast
# These are `stedgeai generate` switches, i.e. the Python wrapper; variants with
# them are only run on the Windows side. The order: locally we throw out everything
# the epoch table (epochs.py) lets us throw out, and take the survivors to stedgeai.
#
# Sifted on 4 Aug and did NOT help (Concat did not budge): eliminate_concat_split,
# fuse_consecutive_concats_new, --ec-optimize, -S. --Ox is much worse:
# 172 blocks against 70, estimate 2 725 403 against 1 263 056.
set -e
V=$1; shift
mkdir -p ws_$V
./linux/atonn \
  -i models/t1/gen_app_safe/n6_gather2_qdq_OE_3_3_1.onnx \
  --json-quant-file models/t1/gen_app_safe/n6_gather2_qdq_OE_3_3_1_Q.json \
  -g network.c \
  --load-mdesc configs/stm32n6.mdesc \
  --load-mpool fw/stm32n6_nucleo_app_safe.mpool \
  --load-cdesc configs/cortex-m55.cdesc \
  --save-mpool-file used.mpool \
  --out-dir-prefix ws_$V/ \
  --network-name network \
  --optimization 3 --all-buffers-info --cache-maintenance --Oauto-sched \
  --native-float --enable-virtual-mem-pools --Omax-ca-pipe 4 --Ocache-opt --Os \
  --enable-epoch-controller \
  --output-info-file c_info.json --generate-stai "$@" 2>&1 | tail -3

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pack_weights.py — the NPU weights blob to be flashed into the external NOR.

The atonn compiler emits the weights as a separate file
(network_atonbuf.AXISRAM5.raw), addressing them absolutely at 0x342E0000. They
do not fit into the FSBL image — the ROM region is 255 kB, 207 already taken.
So the blob lives in NOR and is copied into AXISRAM5 at startup
(fw/src/n6_weights.c).

WHY THE HEADER. The weights are rigidly tied to the graph layout in the
firmware: regenerate the graph, forget to reflash the weights — and inference
will run off WITHOUT any diagnostics at all. ST's documentation warns about
this in plain words. Here a signature, a length and a CRC32 are put in front of
the data, and the firmware checks them before copying and prints the result to
the terminal.

Blob layout (little-endian, the way w_hdr_t reads it):
    +0   uint32  magic  = 'N6W1' (0x3157364E)
    +4   uint32  bytes  = length of the payload
    +8   uint32  crc32  = IEEE 802.3 over the payload
    +12  uint32  reserved = 0
    +16  data

RUN
  python pack_weights.py
  python pack_weights.py --raw ../models/t1/gen_app_safe/network_atonbuf.AXISRAM5.raw
  python pack_weights.py --selftest
"""
import argparse
import binascii
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAGIC = 0x3157364E                     # 'N6W1'
NOR_OFFSET = 0x00400000                # offset inside the NOR
NOR_ABS = 0x70000000 + NOR_OFFSET      # address for STM32_Programmer_CLI
AXISRAM5 = 448 * 1024                  # size of the destination bank


def pack(raw: bytes) -> bytes:
    crc = binascii.crc32(raw) & 0xFFFFFFFF
    hdr = struct.pack("<IIII", MAGIC, len(raw), crc, 0)
    return hdr + raw, crc


def selftest():
    import random
    rng = random.Random(20260803)
    for n in (1, 15, 16, 17, 1024, 273 * 1024):
        raw = bytes(rng.randrange(256) for _ in range(min(n, 4096))) * (
            max(1, n // 4096))
        raw = raw[:n] if len(raw) >= n else raw + b"\0" * (n - len(raw))
        blob, crc = pack(raw)
        m, b, c, r = struct.unpack("<IIII", blob[:16])
        assert m == MAGIC and b == n and r == 0, (m, b, r)
        assert c == crc == (binascii.crc32(raw) & 0xFFFFFFFF)
        assert blob[16:] == raw
        # the same CRC that the nibble table in n6_weights.c computes
        tab = [0] * 16
        for i in range(16):
            v = i
            for _ in range(4):
                v = (v >> 1) ^ (0xEDB88320 if v & 1 else 0)
            tab[i] = v
        cc = 0xFFFFFFFF
        for byte in raw:
            cc ^= byte
            cc = (cc >> 4) ^ tab[cc & 0xF]
            cc = (cc >> 4) ^ tab[cc & 0xF]
        assert (~cc) & 0xFFFFFFFF == crc, "the nibble CRC diverged from binascii"
        print("  %7d B -> CRC32 %08X, the C implementation agrees" % (n, crc))
    print("\nSELFTEST OK: the header format and both CRC32 implementations agree")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(
        HERE, "..", "models", "t1", "gen_app_safe",
        "network_atonbuf.AXISRAM5.raw"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    if not os.path.exists(a.raw):
        sys.exit("no weights file: %s\n"
                 "it appears next to network.c after stedgeai generate" % a.raw)
    raw = open(a.raw, "rb").read()
    if len(raw) > AXISRAM5:
        sys.exit("the weights %d B do not fit into the AXISRAM5 bank (%d B)" % (len(raw), AXISRAM5))

    blob, crc = pack(raw)
    out = a.out or os.path.join(os.path.dirname(a.raw), "n6_weights.bin")
    open(out, "wb").write(blob)

    print("weights   : %s" % os.path.basename(a.raw))
    print("size      : %d B (%.1f kB), header +16 B" % (len(raw), len(raw) / 1024))
    print("CRC32     : %08X" % crc)
    print("blob      : %s (%d B)" % (out, len(blob)))
    print()
    print("FLASH IT (the address must match W_NOR_OFFSET in fw/src/n6_weights.c):")
    print()
    print('  & "$env:PROGRAMMER" -c port=SWD mode=HOTPLUG '
          '-el "<...>\\ExternalLoader\\MX25UM51245G_STM32N6570-NUCLEO.stldr" '
          '-hardRst -w "%s" 0x%08X' % (out, NOR_ABS))
    print()
    print("After flashing, at startup the terminal must print")
    print("  weights: %d B, CRC %08X — OK" % (len(raw), crc))
    print("Any other result means the blob and the graph have diverged.")


if __name__ == "__main__":
    main()

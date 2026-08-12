#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ck4_compare.py — cross-check of the checkpoint 4 UART dump (board, MVE) against the
host reference (fw/build/ck4_ref.bin, scalar). See fw/src/ck4.h.

Input: a text terminal log (save the VCP session to a file; lines outside the
CK4 BEGIN..END block — heartbeat and so on — are ignored), and the reference .bin.
Criterion: rel RMS < 1e-4 (PASS). 1e-4..1e-3 — suspect libm sinf on
host/target, look at the breakdown by time/bands, do not rush to blame MVE.

Run:     python3 ck4_compare.py <dump.log> [--ref ../fw/build/ck4_ref.bin]
Smoke:   python3 ck4_compare.py --selftest   (dump made from the reference: PASS,
         a corrupted one: FAIL, the CRC catches the corruption)
"""
import argparse
import os
import struct
import sys
import zlib

import numpy as np

TOL = 1e-4
HERE = os.path.dirname(os.path.abspath(__file__))
REF_DEFAULT = os.path.join(HERE, "..", "fw", "build", "ck4_ref.bin")


def load_ref(path):
    a = np.fromfile(path, dtype="<f4")
    if len(a) == 0:
        sys.exit(f"reference is empty/not found: {path} (first cd fw && make ck4)")
    return a


def parse_dump(lines):
    """-> (np.float32 array, crc_from_dump | None). Silently skips garbage."""
    words, crc, inside, n_expect = [], None, False, None
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("CK4 BEGIN"):
            inside = True
            try:
                n_expect = int(ln.split()[2])
            except (IndexError, ValueError):
                n_expect = None
            continue
        if ln.startswith("CK4 CRC"):
            crc = int(ln.split()[2], 16)
            continue
        if ln.startswith("CK4 END"):
            inside = False
            continue
        if not inside:
            continue
        try:
            words.extend(int(t, 16) for t in ln.split())
        except ValueError:
            pass                                     # a heartbeat that slipped in
    if not words:
        sys.exit("the log has no CK4 BEGIN..END block with data")
    if n_expect is not None and len(words) != n_expect:
        sys.exit(f"got {len(words)} words, expected {n_expect} — the dump is incomplete "
                 "(did the terminal truncate lines? save the log again)")
    raw = struct.pack(f"<{len(words)}I", *words)
    if crc is not None:
        got = zlib.crc32(raw) & 0xFFFFFFFF
        if got != crc:
            sys.exit(f"CRC mismatch: dump {crc:08x}, recomputed {got:08x} — "
                     "corrupted bytes in the log")
    return np.frombuffer(raw, dtype="<f4"), crc


def compare(board, ref):
    if len(board) != len(ref):
        sys.exit(f"lengths: board {len(board)}, reference {len(ref)}")
    b64, r64 = board.astype(np.float64), ref.astype(np.float64)
    rel = np.sqrt(np.mean((b64 - r64) ** 2) / (np.mean(r64 ** 2) + 1e-30))
    print(f"rel RMS (whole second) = {rel:.3e}   [threshold {TOL:.0e}]")
    print(f"max |diff| = {np.max(np.abs(b64 - r64)):.3e}, "
          f"reference peak = {np.max(np.abs(r64)):.3f}")
    seg = len(ref) // 10
    print("per 100 ms:", " ".join(
        f"{np.sqrt(np.mean((b64[i*seg:(i+1)*seg]-r64[i*seg:(i+1)*seg])**2) / (np.mean(r64[i*seg:(i+1)*seg]**2)+1e-30)):.1e}"
        for i in range(10)))
    ok = rel < TOL
    print("== CK4 PASS: MVE ≡ scalar within tolerance ==" if ok else
          "== CK4 FAIL — breakdown above; 1e-4..1e-3 => suspect libm ==")
    return ok


def make_dump_text(a):
    w = a.view("<u4")
    lines = [f"CK4 BEGIN {len(w)}"]
    lines += [" ".join(f"{x:08x}" for x in w[i:i + 12])
              for i in range(0, len(w), 12)]
    lines += [f"CK4 CRC {zlib.crc32(a.tobytes()) & 0xFFFFFFFF:08x}", "CK4 END"]
    return lines


def selftest(ref_path):
    ref = load_ref(ref_path)
    noise = ["[hb] hops=250 miss=0", ""]                     # garbage in the log is legal
    a, _ = parse_dump(noise + make_dump_text(ref) + noise)
    assert compare(a, ref), "a clean dump must PASS"
    bad = ref.copy()
    bad[1000:2000] += 0.01
    lines = make_dump_text(ref)                              # CRC of the REFERENCE...
    w = bad.view("<u4")
    lines[1 + 1000 // 12] = " ".join(                        # ...but the words are corrupt
        f"{x:08x}" for x in w[(1000 // 12) * 12:(1000 // 12) * 12 + 12])
    try:
        parse_dump(lines)
        raise AssertionError("the CRC must catch the corruption")
    except SystemExit:
        print("[selftest] corruption caught by CRC: OK")
    assert not compare(bad, ref), "a corrupted buffer must FAIL"
    print("[selftest] ALL OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?", help="terminal log with the CK4 block")
    ap.add_argument("--ref", default=REF_DEFAULT)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.ref)
    elif args.dump:
        board, _ = parse_dump(open(args.dump, encoding="utf-8",
                                   errors="replace").read().splitlines())
        sys.exit(0 if compare(board, load_ref(args.ref)) else 1)
    else:
        ap.error("need a dump or --selftest")

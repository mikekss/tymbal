# fw/target/ — mirror of the Cube project files (review backlog item 16, closed 8 Aug)

> **The licence here is a different one.** `main.c`,
> `STM32N657X0HXQ_AXISRAM2_fsbl.ld`, `N6_m1.ioc` and `FSBL.project.xml`
> originate from an STM32Cube example and remain under SLA0048 from
> STMicroelectronics. The Apache-2.0 licence at the root of the repository
> does NOT apply to them — the terms are in `LICENSE.md` next to this file,
> and `NOTICE` at the root explains the whole picture.

## Why

The firmware is built by CubeIDE from the project in
`C:\Users\<user>\STM32Cube\Repository\STM32Cube_FW_N6_V1.4.0\Projects\`
`NUCLEO-N657X0-Q\Examples\SAI\N6_m1\`, which is **outside** this repository.
Until 8 Aug things that exist nowhere else lived there in a single copy:
the D-19 preset, the D-22 master level, the D-24 A/B button, the hop handler,
the 24-in-32 packing, the heartbeat. The handover audit of 7 Aug flagged this
as a separate item: four of the six findings of the review against industry
practice pointed at a file the reader of the repository never sees.

## What is here and WHO IS IN CHARGE

What is here is a MIRROR, not the source. **The authority is the files in the
Cube project**: the firmware is built from them, CubeIDE edits them, and they
are the ones sitting under the debugger. The direction of synchronisation is
always the same: **project → here**. Editing files in `fw/target/` is
pointless — the build will not see them.

(This is the THIRD copy of the sources in the project, after `fw/src` ↔
`FSBL/DSP/src`. Starting it without a synchronisation rule would have been
worse than not starting it at all, so the rule is below and the checksum
snapshot is in the table.)

## Snapshot as of 8 Aug

| file | md5 | bytes | endings |
|---|---|---|---|
| `main.c` | `83c0b601196813de677b22db8be77a2f` | 48389 | CRLF |
| `N6_m1.ioc` | `aae7540dfed5d3a8ef3553db4252319c` | 49087 | LF |
| `STM32N657X0HXQ_AXISRAM2_fsbl.ld` | `5c353a603ac98e22cbe9796c89ed818f` | 8521 | CRLF |
| `FSBL.project.xml` | `0c611f28e7cf88bdb985f40dddd9d5e6` | 11716 | LF |

To check for divergence (PowerShell, from the repository root):

```
$p = "C:\Users\<user>\STM32Cube\Repository\STM32Cube_FW_N6_V1.4.0\Projects\NUCLEO-N657X0-Q\Examples\SAI\N6_m1"
get-filehash -algorithm md5 "$p\FSBL\Core\Src\main.c", "fw\target\main.c" | % { $_.hash }
```

Two identical hashes mean the mirror is fresh. Different ones mean: first
update the mirror (`copy` from the project to here), then commit.

## The rule

**Any edit in the Cube project means a copy to here IN THE SAME MOVE and an
update of the table above.** Exactly as with the pair `fw/src` ↔
`FSBL/DSP/src`: there divergence is caught by comparing md5 by hand, here by
this table. Line endings are to be preserved as in the source (`main.c` and
`.ld` are CRLF, `.project` is LF).

## What is deliberately NOT here

The HAL drivers, `Middlewares`, the startup code and the rest of the
STM32Cube_FW_N6_V1.4.0 package: they are not ours, the version is pinned in
`VERSIONS.md`, and copying them into the repository would bloat it with vendor
code in exactly the same way as the STEdgeAI compiler output did (D-21).

# About this mirror

There is no history here. This repository is a build: a snapshot of the working
project's tracked tree, with everything that does not belong in public stripped
out, and the text translated from Russian.

Why no history. The working repository's history is over a gigabyte and carries
training corpora, weights and intermediate artefacts. I did not rewrite it with
filters, because the decision and findings journals reference specific commit
hashes, and after a rewrite those references would mean nothing. So commit links
inside the documents point at my working branch, not at this repository.

What is deliberately not here: training corpora, audio, weights and checkpoints,
ST Edge AI compiler output, and ST's own datasheets and user manuals — those are
ST's documents, cited by number where they matter rather than re-hosted here.
Also absent is the working layer: the project handover notes, the review
backlogs and the decision log, which are about process rather than engineering.

Everything absent is reproducible by running what is here — see `README.md`.

What can be checked without the board: `cd fw && make test` runs 11 host tests
against the Python references. After that, `make qemu-ck4` runs the same score
on a Cortex-M55 model.

The English here is a translation and the original is Russian. Every number,
identifier and file name is preserved exactly.

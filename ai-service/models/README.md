# Model Artifacts

Large neural-network checkpoints are intentionally not tracked in git. A fresh
clone contains this manifest only; production and trainer nodes sync the actual
`.pth` files from the model artifact store described in
`docs/REPRODUCIBILITY.md`.

Before running production inference, model-integrity checks, or training jobs
that require a seed checkpoint, sync these files into this directory:

- `canonical_hex8_2p.pth`
- `canonical_hex8_3p.pth`
- `canonical_hex8_4p.pth`
- `canonical_hexagonal_2p.pth`
- `canonical_hexagonal_3p.pth`
- `canonical_hexagonal_4p.pth`
- `canonical_square8_2p.pth`
- `canonical_square8_3p.pth`
- `canonical_square8_4p.pth`
- `canonical_square19_2p.pth`
- `canonical_square19_3p.pth`
- `canonical_square19_4p.pth`

The Python contract suite treats this manifest as the source of truth in clean
clones where the checkpoint files are absent. On machines with synced models,
the same contract asserts that local checkpoint files are non-empty.

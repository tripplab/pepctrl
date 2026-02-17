# pepctrl

`pepctrl.py` generates peptide control sequences for a **peptide of interest (POI)**.

It supports three modes:
- **`decoy`**: matched decoys with composition/physicochemical constraints.
- **`scramble`**: permutations of the POI (same residues, reordered).
- **`random`**: random AA20 peptides of the same length.

Author: trippm@tripplab.com [Feb 2026]
---

## Dependencies

`pepctrl.py` uses only Python standard library modules:

- `argparse`
- `csv`
- `json`
- `math`
- `random`
- `re`
- `sys`
- `time`
- `dataclasses`
- `typing`

So there are **no third-party Python package dependencies** for normal use.

### Required runtime

- Python 3.8+ recommended

---

## Installation

### Option 1: run directly (recommended)

Clone the repo and run the script with Python:

```bash
git clone <your-repo-url>
cd pepctrl
python3 pepctrl.py --help
```

### Option 2: make it executable

```bash
chmod +x pepctrl.py
./pepctrl.py --help
```

---

## Run in a CLI terminal

From the project directory:

```bash
# Show all options
python3 pepctrl.py --help

# Generate 10 decoys and print CSV to terminal
python3 pepctrl.py --poi ACDEFGHIKLMNPQRSTVWY --type decoy --n 10

# Write output file instead of terminal
python3 pepctrl.py --poi ACDEFGHIKLMNPQRSTVWY --type random --n 25 --out random.csv
```

---

## Testing

There is no separate test suite in this repository right now. A practical CLI smoke test is:

```bash
# 1) Verify CLI parsing/help
python3 pepctrl.py --help

# 2) Verify generation/output shape
python3 pepctrl.py --poi ACDEFGHIKLMNPQRSTVWY --type scramble --n 3 --seed 1 --format csv | head
```

If those commands succeed and output looks correct, your local setup is working.

---

## Quick start

```bash
# 1) Generate 25 matched decoys (CSV to stdout)
python3 pepctrl.py --poi ACDEFGHIKLMNPQRSTVWY --type decoy --n 25

# 2) Generate 20 scrambles and write TSV
python3 pepctrl.py --poi ACDEFGHIKLMNPQRSTVWY --type scramble --n 20 --format tsv --out scrambles.tsv

# 3) Generate 50 random controls with fixed seed (reproducible)
python3 pepctrl.py --poi ACDEFGHIKLMNPQRSTVWY --type random --n 50 --seed 42 --out random.csv
```

---

## Input rules

- Provide the POI via `--poi` **or** `--poi-file` (first non-empty line is used).
- Allowed residues are the standard AA20 letters: `ACDEFGHIKLMNPQRSTVWY`.
- `--type` and `--n` are required.

---

## Modes

### 1) `--type decoy`

Best when you want **fair controls** that are not simple shuffles.

Decoy generation can match:
- mean hydrophobicity (`--tol-hydro-mean`),
- aromatic content (`--match-arom`, `--tol-arom-frac`),
- special residues C/P/G (`--match-special`),
- charge-shape and polarization (`--charge-shape`, `--tol-charge-shape`, `--tol-charge-polarization`),
- identity/complexity constraints (`--max-identity`, `--max-run`, `--min-shannon`, motif exclusions, etc.).

Generation strategy is configurable with `--method` (`mutate`, `compose`, `hybrid`) plus tuning flags like `--batch-size`, `--max-attempts`, and `--refine-steps`.

### 2) `--type scramble`

Generates sequence permutations of the POI (same composition, same length).

By default, this mode does **not** apply full decoy filtering. Enable optional filters with:

```bash
--filter-sr
```

### 3) `--type random`

Generates random AA20 peptides with POI-matched length.

Like scramble mode, optional filtering can be enabled with `--filter-sr`.

---

## Output formats

`--format` supports:
- `csv` (default)
- `tsv`
- `fasta`
- `jsonl`

Use `--out -` (default) for stdout, or set a path.

### POI row behavior

- CSV/TSV include a **POI row by default** so POI metrics are visible alongside generated controls.
- Disable with `--no-poi-row`.

---

## Common options

- `--seed <int>`: reproducible outputs.
- `--max-identity-between-decoys <float>`: diversity constraint among returned sequences.
- `--exclude-motif <motif>`: reject sequences containing motif (repeatable).
  - Prefix with `re:` to use regex (example: `--exclude-motif 're:N[^P][ST]'`).
- `--exclude-motif-file <path>`: motif list file, one per line.
- `--allow-partial`: write fewer than `--n` sequences if target cannot be met.
- `-v`, `-vv`, `-vvv`: stderr manifest verbosity.
- `--timing`: include stage timings in manifest output.

---

## Exit codes

- `0`: success, requested output count met.
- `2`: partial result produced (or insufficient candidates without `--allow-partial`).
- `3`: configuration/input error.

---

## Example workflows

### Matched decoys with stricter identity and motifs excluded

```bash
python3 pepctrl.py \
  --poi-file poi.txt \
  --type decoy \
  --n 100 \
  --max-identity 0.25 \
  --exclude-motif RP \
  --exclude-motif 're:N[^P][ST]' \
  --relax \
  --allow-partial \
  --seed 2025 \
  --format csv \
  --out decoys.csv \
  -vv --timing
```

### FASTA output for random controls

```bash
python3 pepctrl.py --poi ACDEFGHIKLMNPQRSTVWY --type random --n 30 --format fasta --out random.fasta
```

---

## Notes

- Charge handling uses a simplified model with configurable histidine partial charge (`--his-charge`) and optional terminal charges (`--include-termini`).
- Hydrophobicity scale currently supports Kyte-Doolittle (`--hydro-scale kd`).

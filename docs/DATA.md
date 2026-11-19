# Data

The RNADGG experiments use three processed regulatory RNA datasets.

## Expected files

Place the following files under `data/processed/`:

```text
rbs_data.csv
rbs_data_f.csv
utr_data.csv
toehold_data.csv
toehold_data_f.csv
```

Place the raw 5-prime UTR library under `data/raw/` if preprocessing from the raw GEO table:

```text
GSM3130443_designed_library.csv
```

## Provenance

- Synthetic 50-nt 5-prime UTRs measured in human cells: Sample et al., Nature Biotechnology 2019, DOI: `10.1038/s41587-019-0164-5`, GEO accession `GSE114002`.
- RBS data: "Large-scale DNA-based phenotypic recording and deep learning enable highly accurate sequence-function mapping", DOI: `10.1038/s41467-020-17222-4`.
- Toehold-switch data: "A deep learning approach to programmable RNA switches", DOI: `10.1038/s41467-020-18677-1`.

## Input handling

CSV loading strips surrounding whitespace and converts sequences to uppercase. Rows with missing or empty sequences, nonnumeric labels, or infinite labels are excluded. A table without any usable rows raises an error.

The default one-hot alphabet is A/C/G/T. RNA U is mapped to the T channel during encoding. Selecting `RNA_ALPHABET` instead maps T to U. Ambiguous bases such as N raise an error. Batches supplied to the model must have a common sequence length.

## Git policy

Full data tables are excluded from git through `.gitignore`. This keeps the repository lightweight and reduces the risk of committing derived data files by accident. For archival publication, deposit full datasets and generated sequence libraries in a data repository such as Zenodo, Figshare, OSF or an institutional repository, then add the DOI to this file.

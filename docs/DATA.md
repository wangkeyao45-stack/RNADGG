# Data

The RNADGG experiments use three processed regulatory RNA datasets.

## Expected files

Place the following files under `data/processed/`:

```text
rbs_data.csv
rbs_data_f.csv
processed_data.csv
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

The UTR manuscript experiments require `processed_data.csv`, containing 280,000 unique 50-nt sequences and the label column `r1`. The older `utr_data.csv` has 100,017 rows and an `rl` column and is a different processed library. The benchmark driver rejects a different row count unless `--smoke` is used. It does not pad, truncate or silently substitute datasets. The original Chinese sequence-column header is supported.

The grouped toehold driver requires `full_sequence`, `ON` and `OFF`. The manuscript input contains 44,859 rows representing 44,660 unique full sequences. Rows sharing a full sequence remain in the same partition. A matching raw `toehold_data.csv` can also be supplied explicitly if it has the same rows and labels as the filtered benchmark input; verify the exported split manifest against the archive.

CSV loading strips surrounding whitespace and converts sequences to uppercase. Rows with missing or empty sequences, nonnumeric labels, or infinite labels are excluded. A table without any usable rows raises an error.

The default one-hot alphabet is A/C/G/T. RNA U is mapped to the T channel during encoding. Selecting `RNA_ALPHABET` instead maps T to U. Ambiguous bases such as N raise an error. Batches supplied to the model must have a common sequence length.

## Git policy

Full data tables are excluded from git through `.gitignore`. This keeps the repository lightweight and reduces the risk of committing derived data files by accident. For archival publication, deposit full datasets and generated sequence libraries in a data repository such as Zenodo, Figshare, OSF or an institutional repository, then add the DOI to this file.

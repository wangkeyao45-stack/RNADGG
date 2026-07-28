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

- 5-prime UTR MPRA data: Sample et al., Nature Biotechnology 2019, DOI: `10.1038/s41587-019-0164-5`, GEO accession `GSE114002`.
- RBS data: "Large-scale DNA-based phenotypic recording and deep learning enable highly accurate sequence-function mapping", DOI: `10.1038/s41467-020-17222-4`.
- Toehold-switch data: "A deep learning approach to programmable RNA switches", DOI: `10.1038/s41467-020-18677-1`.

## Git policy

Full data tables are intentionally excluded from git through `.gitignore`. This keeps the repository lightweight and avoids committing derived data files by accident. For archival publication, deposit full datasets and generated sequence libraries in a data repository such as Zenodo, Figshare, OSF or an institutional repository, then add the DOI to this file.

# Data sources

Every dataset used by Prism is public and intended for research or teaching.
Two files are bundled in this folder; the other three ship inside scikit-learn.
Nothing is downloaded while the app is running.

---

## Bundled in this folder

### `co2_daily_mlo.csv`

Daily atmospheric carbon dioxide concentration measured at Mauna Loa
Observatory, Hawaii.

| | |
|---|---|
| Rows | 18,304 |
| Columns | `date`, `value` (parts per million) |
| Span | 30 March 1958 to 9 August 2025 |
| Gaps | 6,301 calendar days have no reading; the longest unbroken gap is 131 days |

The record was begun by Charles David Keeling at the Scripps Institution of
Oceanography and is now maintained jointly with NOAA's Global Monitoring
Laboratory. This copy is redistributed through the Frictionless Data
`co2-ppm-daily` package.

The gaps are genuine instrument downtime rather than anything introduced for
this project, which is why the Signal Refinery page can demonstrate gap filling
on real missing data instead of holes punched in an otherwise complete file.

- Scripps CO₂ Program: https://scrippsco2.ucsd.edu/
- NOAA Global Monitoring Laboratory: https://gml.noaa.gov/ccgg/trends/

> Keeling, C. D., et al. *Exchanges of atmospheric CO₂ and ¹³CO₂ with the
> terrestrial biosphere and oceans from 1978 to 2000.* Scripps Institution of
> Oceanography, 2001.

### `winequality-red.csv`

Physicochemical measurements and blind tasting scores for Portuguese *Vinho
Verde* red wines.

| | |
|---|---|
| Rows | 1,599 as distributed, 1,359 after duplicate removal |
| Columns | 11 physicochemical measures plus a `quality` score from 0 to 10 |
| Separator | semicolon, as published |

Each score is the median of assessments made by at least three judges who could
not see the chemistry. The 240 duplicate rows are removed by the cleaning
pipeline and reported in the app.

- UCI record: https://archive.ics.uci.edu/dataset/186/wine+quality

> Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. *Modeling wine
> preferences by data mining from physicochemical properties.* Decision Support
> Systems, 47(4):547–553, 2009.

---

## Supplied by scikit-learn

These load from the installed package, so there is no file to keep in step with
an upstream source.

### Wine cultivars — `sklearn.datasets.load_wine`

178 wines grown in the same Italian region by three different cultivars, with
13 chemical measurements each. The cultivar labels exist but are withheld from
the clustering on the Cultivar Segmentation page and only used afterwards to
mark the result.

- UCI record: https://archive.ics.uci.edu/dataset/109/wine

### Breast cancer diagnostic — `sklearn.datasets.load_breast_cancer`

569 fine-needle aspirate biopsies described by 30 features computed from
digitised images of cell nuclei. Prism inverts scikit-learn's original encoding
so that malignant is the positive class, because recall on malignancy is the
number that carries clinical weight.

- UCI record: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

> Street, W. N., Wolberg, W. H., and Mangasarian, O. L. *Nuclear feature
> extraction for breast tumor diagnosis.* IS&T/SPIE International Symposium on
> Electronic Imaging: Science and Technology, 1993.

**This dataset is used here for teaching only.** The model trained on it has no
regulatory approval, has never been validated against current clinical
practice, and must not inform any decision about any real person.

### Handwritten digits — `sklearn.datasets.load_digits`

1,797 handwritten digits reduced to 8×8 grids with intensities from 0 to 16.
A copy of the UCI *Optical Recognition of Handwritten Digits* test set.

- UCI record: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits

---

## Licensing

The UCI datasets are distributed for research and educational use under the
terms recorded on their respective UCI pages. The Mauna Loa record is a public
scientific dataset; Scripps asks that use of the record be cited as above.
Prism's own source code is MIT licensed, which covers the code only and not the
datasets.

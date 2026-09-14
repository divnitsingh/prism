# Prism

**An interactive data science atlas.** Six analyses across five public
datasets, from the first pass over a raw archive through to a convolutional
neural network that trains in the browser while you watch the loss curve move.

Every chart is live. Nothing is a saved image — change an input and the pipeline
behind it recomputes.

---

## What is in it

| Analysis | Question it answers | Methods |
|---|---|---|
| **Signal Refinery** | Can 67 years of raw atmospheric readings be turned into something trustworthy? | validation, gap interpolation, outlier rules, feature engineering |
| **Atmospheric Trends** | How much of the carbon dioxide record is seasonal, and how much is the climb? | seasonal decomposition, climatology, anomaly scoring, growth rates |
| **Cultivar Segmentation** | Can wine varieties be recovered from chemistry alone, with no labels? | k-means, Ward linkage, PCA, silhouette analysis |
| **Diagnostic Intelligence** | Where should the line between benign and malignant be drawn, and what does moving it cost? | logistic regression, random forest, threshold tuning, ROC and PR curves |
| **Digit Recognition** | Can a convolutional network be built without a framework? | convolution, max pooling, dropout, Adam |
| **Quality Lab** | What chemistry makes a wine score well? | gradient boosting, cross-validation, permutation importance, segmentation |

### A few things worth opening

- **Signal Refinery → "Add faults to the source"** puts duplicate rows, blank
  cells and sentinel values back into a copy of the archive, so you can watch
  the pipeline catch each one and still pass all five integrity checks.
- **Atmospheric Trends → "Year on year"** draws all 67 years as rings on one
  polar axis. The rings move outward because concentration is rising, and
  spread apart because the rise is accelerating.
- **Diagnostic Intelligence → threshold slider** updates the confusion matrix,
  both curves and every metric as you drag it. At the conventional 0.50 the
  default model misses three cancers and raises one false alarm; dropping the
  cut-off to 0.07 misses nothing but raises ten. The page shows that trade
  rather than picking for you.
- **Digit Recognition → "Draw a digit"** lets you click out a digit on an 8×8
  grid and have the hand-written network read it back.
- **Quality Lab → "Pour a glass"** loads a real bottle, lets you change its
  chemistry, and shows which measurement is pushing the predicted score around.

### The neural network has no framework underneath it

`prism/convnet.py` implements convolution (via im2col), max pooling, dropout,
softmax cross-entropy and the Adam optimiser directly in NumPy. There is no
PyTorch or TensorFlow in `requirements.txt`.

That is a deliberate engineering decision rather than a stunt. A CPU build of
PyTorch is a large install that frequently times out on free hosting, and it
would have been the only reason this app could not deploy reliably. Writing the
network by hand keeps the whole dependency list to six packages, and the
network still reaches **about 98.5% test accuracy in roughly three seconds** —
fast enough to retrain live whenever you change a hyperparameter.

The backward pass is checked against finite-difference gradients in the test
suite, which is the standard way to prove a hand-written network computes the
gradients it claims to.

---

## Running it on Windows

### What you need

**Python 3.10 or newer.** Check by opening PowerShell and running:

```powershell
py --version
```

If that fails, install Python from
[python.org/downloads/windows](https://www.python.org/downloads/windows/) and
**tick "Add python.exe to PATH"** on the first screen of the installer.

### The quick way

Unzip the folder, then **double-click `run_app.bat`**.

It creates a virtual environment, installs the dependencies and starts the app.
The first run takes a minute or two; after that it starts in seconds. Your
browser should open at <http://localhost:8501>. Leave the black window open
while you use the app, and press `Ctrl+C` in it to stop.

### The manual way

Open PowerShell in the project folder (Shift + right-click inside the folder →
*Open PowerShell window here*) and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Using Command Prompt instead of PowerShell? Only the activate line differs:

```bat
.venv\Scripts\activate.bat
```

### If PowerShell blocks the activate script

Windows disables script execution by default. This allows it for your own
account only:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Answer `Y`, then run the activate line again.

### Other things that can go wrong

| Symptom | Fix |
|---|---|
| `'py' is not recognized` | Python is not on your PATH. Reinstall and tick *Add python.exe to PATH*, or use the full path to `python.exe`. |
| `'streamlit' is not recognized` | The virtual environment is not active. Look for `(.venv)` at the start of your prompt, and re-run the activate line if it is missing. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Windows Firewall prompt on first run | Allow access on private networks, or just dismiss it — the app only needs localhost. |
| Blank page in the browser | Hard refresh with `Ctrl+F5`. |

### Running the tests

```powershell
pip install -r requirements-dev.txt
pytest -q
```

49 tests, about seven seconds. To skip the one that trains the network end to
end:

```powershell
pytest -q -m "not slow"
```

---

## Putting it online

The app is built to deploy to
[Streamlit Community Cloud](https://share.streamlit.io), which is free.

1. **Push this folder to a public GitHub repository.** From the project folder:

   ```powershell
   git init
   git add .
   git commit -m "Prism: an interactive data science atlas"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/prism.git
   git push -u origin main
   ```

2. **Sign in at [share.streamlit.io](https://share.streamlit.io)** with the same
   GitHub account and choose *Create app → Deploy a public app from GitHub*.

3. **Point it at the repository:**

   - Repository: `YOUR-USERNAME/prism`
   - Branch: `main`
   - Main file path: `app.py`

4. **Click Deploy.** The first build takes two or three minutes. After that
   every push to `main` redeploys automatically.

Both data files live inside the repository, so there is nothing to configure
and no secrets to set. Total install size is well inside the free tier.

If the build fails, open the log from the app's menu — it is nearly always a
dependency line rather than the app itself.

---

## How the project is laid out

```
prism/
├── app.py                    Entry point: page registry and navigation
├── requirements.txt          Six runtime packages, no ML framework
├── requirements-dev.txt      Adds pytest
├── run_app.bat               Windows one-click launcher
├── pytest.ini
├── .streamlit/
│   └── config.toml           Dark theme matching the in-app styling
├── data/
│   ├── co2_daily_mlo.csv     Mauna Loa daily record, 1958–2025
│   ├── winequality-red.csv   UCI red wine quality
│   └── ATTRIBUTION.md        Sources, citations and licensing
├── prism/
│   ├── convnet.py            The NumPy convolutional network
│   ├── datasets.py           Loaders and cleaning pipelines
│   ├── models.py             Cached clustering, classification, regression
│   ├── theme.py              Design tokens, stylesheet, Plotly styling
│   ├── ui.py                 Shared interface components
│   └── modules/
│       ├── home.py           Overview and the spectral band
│       ├── refinery.py       Signal Refinery
│       ├── atmosphere.py     Atmospheric Trends
│       ├── cultivars.py      Cultivar Segmentation
│       ├── diagnostics.py    Diagnostic Intelligence
│       ├── digits.py         Digit Recognition
│       └── quality.py        Quality Lab
└── tests/
    └── test_prism.py         49 tests
```

### Notes on the design

The interface borrows its logic from a spectrograph: the instrument itself is
neutral graphite and colour is reserved for data. Each analysis owns one
spectral line, and that line is the only place its colour appears.

The band on the overview page is six sparklines, one per analysis, each
computed from that analysis's own dataset at page load — the Keeling curve, the
seasonal cycle, cluster density along the first principal component, the
classifier's sorted scores, mean ink per pixel, and the spread of alcohol across
the cellar. None of it is drawn by hand.

Type is Space Grotesk for headings, IBM Plex Sans for body text, and IBM Plex
Mono for anything numeric, so figures line up in columns the way an instrument
readout should.

### Notes on the analysis

- Test data is held out before any model is fitted, and scaling is learned on
  training folds only.
- Cross-validation is reported next to single-split results, so the spread is
  visible rather than hidden behind one flattering number.
- Cluster labels are withheld from clustering and only used afterwards to mark
  the result.
- Where a method does not work well, the app says so. The interquartile outlier
  rule has almost nothing to catch on a trending series; silhouette scores on
  the wine cellar never rise far above 0.2 because the wines form one continuous
  cloud; the wine quality model is timid at the extremes because squared error
  rewards guessing near the middle when the signal is weak.

---

## A word on the medical model

The Diagnostic Intelligence page trains on a public research dataset of 569
biopsies collected decades ago at a single institution. It is a teaching
exercise about decision thresholds. It has no regulatory approval, has never
been validated against current clinical practice, and must not inform any real
decision about any real person.

---

## Data and licensing

Source, citation and licensing details for all five datasets are in
[`data/ATTRIBUTION.md`](data/ATTRIBUTION.md).

The code is MIT licensed — see [`LICENSE`](LICENSE). That covers the code only,
not the datasets.

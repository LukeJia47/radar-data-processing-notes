# radar-algo-notes

Pure Python notes and demos for common millimeter-wave radar data processing
algorithms. The project focuses on readable algorithm implementations and small
visual demos that are easy to run and modify.

## Features

- **Clustering**: DBSCAN clustering on a two-moon dataset.
- **Filtering**: CV-KF, CTRV-EKF, and IMM tracking demo.
- **Classification**: Hand-written CART-based random forest classifier.

## Project Structure

```text
radar-algo-notes/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── clustering/
│   ├── dbscan.py
│   ├── demo_dbscan.py
│   └── assets/
│
├── filtering/
│   ├── ekf.py
│   ├── demo_ekf.py
│   └── assets/
│
└── classification/
    ├── classifier.py
    ├── demo.py
    └── assets/
```

## Installation

```bash
git clone https://github.com/LukeJia47/radar-algo-notes.git
cd radar-algo-notes
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Demos

```bash
python clustering/demo_dbscan.py
python filtering/demo_ekf.py
python classification/demo.py
```

Generated figures are saved under each module's `assets/` directory.

## Modules

### Clustering

`clustering/dbscan.py` implements pairwise Euclidean distance calculation and a
basic DBSCAN clustering algorithm.

### Filtering

`filtering/ekf.py` implements:

- Constant Velocity Kalman Filter (CV-KF)
- Constant Turn Rate and Velocity Extended Kalman Filter (CTRV-EKF)
- Interacting Multiple Model filter (IMM)

### Classification

`classification/classifier.py` implements a random forest classifier from basic
NumPy operations:

- Bootstrap sampling
- CART-style binary split
- Gini impurity
- Multi-tree voting


## Notes

This repository is intended for learning and experimentation. The algorithms are
kept compact and readable, so they may not include all optimizations used in
production radar systems.


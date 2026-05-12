## Project structure
- **`abaqus code/`** — Python scripts used inside ABAQUS to generate and run
  FE simulations across three geometries (Geom1, Geom2, Geom3).
- **`data/`** — Datasets generated from the FE simulations, plus the analysis
  scripts:
  - Mesh convergence study
  - Polynomial regression model
  - Gradient boosted decision tree model

## Requirements

- Python 3.14
- ABAQUS (for running the simulation scripts in `abaqus code/`)
- Python packages: `numpy`, `pandas`, `scikit-learn`


## Usage

The ABAQUS scripts in `abaqus code/` can be run inside ABAQUS to produce the
simulation datasets (CSV files in `data/report/`). The analysis scripts in
`data/` then load these CSVs to train and evaluate the regression models.


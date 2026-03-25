# ML Assignment Solutions (CA-I)

This repository contains solved problem statements from:
- `7 K-Means Clustering (1).pdf`
- `8 PCA_Problem Statement (1).pdf`

All solutions were executed on datasets available in `/home/pavan/Pictures/ML-Ass`.

## Deliverables Created

- `CA-I_All_Problem_Statements_Solved.doc`
: Consolidated assignment write-up for all K-means and PCA problem statements.

- `KMeans_Clustering_Solutions.ipynb`
: Notebook solving all 5 K-means problem statements.

- `PCA_Problem_Statement_Solution.ipynb`
: Notebook solving PCA + clustering problem statement on heart disease data.

- `assignment_results.json`
: Computed metrics for all tasks (k selection, silhouette, cluster sizes, profiles).

- `heart_pca_3_components.csv`
: New dataset with first 3 principal components.

- `solve_all_assignments.py`
: Python script used to compute all outputs.

## Datasets Used

1. `EastWestAirlines (1) (1).xlsx` (sheet: `data`)
2. `crime_data (1).xlsx`
3. `Insurance Dataset.xlsx`
4. `Telco_customer_churn (1) (1).xlsx`
5. `AutoInsurance (1).xlsx`
6. `heart disease.xlsx`

## Key Results Summary

### K-Means Clustering PDF

1. Airlines dataset:
- Best K: `6`
- KMeans silhouette: `0.3334`

2. Crime dataset:
- Best K: `2`
- KMeans silhouette: `0.4085`

3. Insurance dataset:
- Best K: `2`
- KMeans silhouette: `0.3970`

4. Telco mixed dataset:
- Best K: `2`
- KMeans silhouette: `0.2170`

5. AutoInsurance mixed dataset:
- Best K: `2`
- KMeans silhouette: `0.0831`

### PCA PDF (Heart Disease)

- Best K on original scaled data: `2`
- First 3 PCs explained variance ratio: `[0.2125, 0.1182, 0.0941]`
- Cumulative variance (3 PCs): `0.4248`
- KMeans ARI (original vs PCA): `0.9474` (very similar cluster structure)

## How To Run

```bash
cd /home/pavan/Pictures/ML-Ass
/home/pavan/Pictures/ML-Ass/.venv/bin/python solve_all_assignments.py
```

This regenerates:
- `assignment_results.json`
- `heart_pca_3_components.csv`

## Notes

- Airlines workbook has two sheets (`Description`, `data`); actual clustering uses sheet `data`.
- Mixed-data problems (Telco, AutoInsurance) use one-hot encoding for categorical features and scaling/imputation for robust clustering.
# ML-Ass

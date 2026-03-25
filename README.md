# K-Means Clustering CA1 Report

## Project Overview
This repository contains a complete clustering analysis for five datasets using K-Means:

1. Airlines Dataset
2. Crime Dataset
3. Insurance Policy Dataset
4. Telecom Customer Dataset (mixed features)
5. Auto Insurance Dataset (mixed features)

The notebook used for analysis is:
- K-Means_Clustering_CA1.ipynb

## Objective
The objective is to identify meaningful clusters in each dataset, interpret segment behavior, and derive practical insights for decision making.

## Datasets Used
- EastWestAirlines (1) (1).xlsx (sheet: data)
- crime_data (1).xlsx
- Insurance Dataset.xlsx
- Telco_customer_churn (1) (1).xlsx
- AutoInsurance (1).xlsx

## Tools and Libraries
- Python
- pandas, numpy
- scikit-learn (KMeans, preprocessing, PCA, silhouette score)
- matplotlib, seaborn

## Common Methodology
For each problem statement, the following workflow was applied:

1. Load dataset and inspect structure.
2. Perform preprocessing:
   - Handle missing values
   - Scale numeric features using StandardScaler
   - For mixed data: encode categorical columns using OneHotEncoder
3. Determine optimal number of clusters using:
   - Elbow Method (inertia trend)
   - Silhouette Score
4. Train K-Means with selected k.
5. Assign cluster labels.
6. Profile clusters using summary statistics.
7. Visualize clusters using PCA (2D scatter).
8. Write business/domain insights.

## Problem-Wise Report

## Q1. Airlines Dataset
### Goal
Identify customer segments based on flight and loyalty behavior.

### Preprocessing
- Loaded data sheet from airlines file.
- Removed ID column from modeling features.
- Converted features to numeric and imputed missing values.
- Standardized features.

### Optimal Clusters
- Selected k = 6 (best silhouette among tested values).

### Key Insights
- Distinct groups exist by loyalty balance, bonus miles, and flight activity.
- A large low-engagement group and smaller high-value frequent-flyer groups are visible.
- Action: run tiered loyalty campaigns and targeted reward strategies by segment.

## Q2. Crime Dataset
### Goal
Group states/regions by crime intensity patterns.

### Preprocessing
- Retained state names for interpretation.
- Used numeric crime indicators for clustering (Murder, Assault, UrbanPop, Rape).
- Imputed missing values and standardized features.

### Optimal Clusters
- Selected k = 2.

### Key Insights
- Cluster split reflects lower-crime vs higher-crime regions.
- Higher-crime segment shows elevated murder/assault/rape averages.
- Action: prioritize resources and intervention plans by cluster severity.

## Q3. Insurance Policy Dataset
### Goal
Segment policyholders by premium, claims, age, renewal cycle, and income.

### Preprocessing
- Used all numeric policyholder features.
- Imputed missing values and standardized features.

### Optimal Clusters
- Selected k = 2.

### Key Insights
- One segment shows higher premium and claim values (higher-value/higher-risk profiles).
- Another segment reflects relatively lower-risk and lower-premium customers.
- Action: customize retention, renewal, and risk-monitoring strategy by segment.

## Q4. Telecom Customer Dataset
### Goal
Cluster customers using mixed features and examine churn-related behavior.

### Preprocessing
- Dropped identifier-like and constant fields (for modeling).
- Numeric columns: median imputation + scaling.
- Categorical columns: mode imputation + one-hot encoding.

### Optimal Clusters
- Selected k = 2.

### Key Insights
- Segments differ strongly in tenure, monthly charge, total charges, referrals, and service bundles.
- Segment profiles help identify likely risk patterns even when explicit churn label is absent.
- Action: target high-risk segments with contract conversion offers, support bundles, and proactive retention campaigns.

## Q5. Auto Insurance Dataset
### Goal
Segment customers from mixed insurance data and identify behavior patterns.

### Preprocessing
- Dropped identifier and date columns from clustering features.
- Numeric columns: median imputation + scaling.
- Categorical columns: mode imputation + one-hot encoding.

### Optimal Clusters
- Selected k = 2.

### Key Insights
- Segments separate by income, premium, total claim amount, and policy/customer attributes.
- One segment shows lower income with higher premium/claim tendency.
- Action: apply risk-sensitive pricing and segment-wise campaign strategy.

## Summary of Optimal Cluster Counts
| Problem | Dataset | Optimal k |
|---|---|---|
| Q1 | Airlines | 6 |
| Q2 | Crime | 2 |
| Q3 | Insurance Policy | 2 |
| Q4 | Telecom | 2 |
| Q5 | Auto Insurance | 2 |

## Overall Conclusion
K-Means clustering successfully identified meaningful groups across all five datasets. The segmentation outcomes can support:

- Better customer targeting and personalization
- Risk-aware insurance and policy decisions
- Focused crime prevention planning
- Churn reduction and stronger retention strategy in telecom

## How To Reproduce
1. Open K-Means_Clustering_CA1.ipynb.
2. Run all cells from top to bottom.
3. Review elbow/silhouette charts, PCA plots, and cluster profile tables.

## Author Notes
This report is aligned with the implemented notebook pipeline and generated outputs currently present in the workspace.
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text,
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code,
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


kmeans_cells = [
    md_cell("# K-Means Clustering Assignment Solutions\n\nThis notebook solves all 5 problem statements from `7 K-Means Clustering (1).pdf`."),
    md_cell("## Imports"),
    code_cell(
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.cluster import KMeans, AgglomerativeClustering\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.metrics import silhouette_score, adjusted_rand_score\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler"
    ),
    md_cell("## Utility Functions"),
    code_cell(
        "def pick_kmeans_k(x, k_min=2, k_max=10, random_state=42):\n"
        "    ks = list(range(k_min, k_max + 1))\n"
        "    inertia, sil = [], []\n"
        "    for k in ks:\n"
        "        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)\n"
        "        labels = km.fit_predict(x)\n"
        "        inertia.append(float(km.inertia_))\n"
        "        sil.append(float(silhouette_score(x, labels)))\n"
        "    best_k = ks[int(np.argmax(sil))]\n"
        "    return ks, inertia, sil, best_k\n\n"
        "def cluster_summary(x, k):\n"
        "    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n"
        "    km_labels = km.fit_predict(x)\n"
        "    hc = AgglomerativeClustering(n_clusters=k, linkage='ward')\n"
        "    hc_labels = hc.fit_predict(x)\n"
        "    return {\n"
        "        'kmeans_silhouette': silhouette_score(x, km_labels),\n"
        "        'hierarchical_silhouette': silhouette_score(x, hc_labels),\n"
        "        'ari_kmeans_vs_hierarchical': adjusted_rand_score(km_labels, hc_labels),\n"
        "        'kmeans_cluster_sizes': pd.Series(km_labels).value_counts().sort_index().to_dict(),\n"
        "        'hierarchical_cluster_sizes': pd.Series(hc_labels).value_counts().sort_index().to_dict(),\n"
        "        'kmeans_labels': km_labels\n"
        "    }"
    ),
    md_cell("## Problem 1: Airlines Dataset"),
    code_cell(
        "air = pd.read_excel('EastWestAirlines (1) (1).xlsx', sheet_name='data')\n"
        "x_air = air.drop(columns=['ID#'])\n"
        "sc = StandardScaler()\n"
        "x_air_s = sc.fit_transform(x_air)\n"
        "ks, inertia, sil, best_k = pick_kmeans_k(x_air_s)\n"
        "summary_air = cluster_summary(x_air_s, best_k)\n"
        "print('Best K:', best_k)\n"
        "print('Silhouette:', summary_air['kmeans_silhouette'])\n"
        "print('Cluster sizes:', summary_air['kmeans_cluster_sizes'])\n"
        "air.assign(cluster=summary_air['kmeans_labels']).groupby('cluster').mean(numeric_only=True).round(2)"
    ),
    md_cell("## Problem 2: Crime Dataset"),
    code_cell(
        "crime = pd.read_excel('crime_data (1).xlsx')\n"
        "x_crime = crime.drop(columns=['Unnamed: 0'])\n"
        "x_crime_s = StandardScaler().fit_transform(x_crime)\n"
        "ks, inertia, sil, best_k = pick_kmeans_k(x_crime_s)\n"
        "summary_crime = cluster_summary(x_crime_s, best_k)\n"
        "print('Best K:', best_k)\n"
        "print('Silhouette:', summary_crime['kmeans_silhouette'])\n"
        "crime.assign(cluster=summary_crime['kmeans_labels']).groupby('cluster').mean(numeric_only=True).round(2)"
    ),
    md_cell("## Problem 3: Insurance Dataset"),
    code_cell(
        "ins = pd.read_excel('Insurance Dataset.xlsx')\n"
        "x_ins_s = StandardScaler().fit_transform(ins)\n"
        "ks, inertia, sil, best_k = pick_kmeans_k(x_ins_s)\n"
        "summary_ins = cluster_summary(x_ins_s, best_k)\n"
        "print('Best K:', best_k)\n"
        "print('Silhouette:', summary_ins['kmeans_silhouette'])\n"
        "ins.assign(cluster=summary_ins['kmeans_labels']).groupby('cluster').mean(numeric_only=True).round(2)"
    ),
    md_cell("## Problem 4: Telecom Mixed Dataset"),
    code_cell(
        "tel = pd.read_excel('Telco_customer_churn (1) (1).xlsx')\n"
        "tel = tel.drop(columns=['Customer ID', 'Count'])\n"
        "num_cols = tel.select_dtypes(include=['number']).columns.tolist()\n"
        "cat_cols = [c for c in tel.columns if c not in num_cols]\n"
        "pre = ColumnTransformer([\n"
        "    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),\n"
        "    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols),\n"
        "])\n"
        "x_tel = pre.fit_transform(tel)\n"
        "ks, inertia, sil, best_k = pick_kmeans_k(x_tel)\n"
        "summary_tel = cluster_summary(x_tel, best_k)\n"
        "print('Best K:', best_k)\n"
        "print('Silhouette:', summary_tel['kmeans_silhouette'])\n"
        "tmp = pd.read_excel('Telco_customer_churn (1) (1).xlsx')\n"
        "tmp['cluster'] = summary_tel['kmeans_labels']\n"
        "tmp.groupby('cluster')[['Tenure in Months','Monthly Charge','Total Charges','Total Revenue']].mean().round(2)"
    ),
    md_cell("## Problem 5: AutoInsurance Mixed Dataset"),
    code_cell(
        "auto = pd.read_excel('AutoInsurance (1).xlsx')\n"
        "auto['Effective To Date'] = pd.to_datetime(auto['Effective To Date'], errors='coerce').map(lambda x: x.toordinal() if pd.notna(x) else np.nan)\n"
        "auto = auto.drop(columns=['Customer'])\n"
        "num_cols = auto.select_dtypes(include=['number']).columns.tolist()\n"
        "cat_cols = [c for c in auto.columns if c not in num_cols]\n"
        "pre = ColumnTransformer([\n"
        "    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),\n"
        "    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols),\n"
        "])\n"
        "x_auto = pre.fit_transform(auto)\n"
        "ks, inertia, sil, best_k = pick_kmeans_k(x_auto)\n"
        "summary_auto = cluster_summary(x_auto, best_k)\n"
        "print('Best K:', best_k)\n"
        "print('Silhouette:', summary_auto['kmeans_silhouette'])\n"
        "tmp2 = pd.read_excel('AutoInsurance (1).xlsx')\n"
        "tmp2['cluster'] = summary_auto['kmeans_labels']\n"
        "tmp2.groupby('cluster')[['Income','Customer Lifetime Value','Total Claim Amount']].mean().round(2)"
    ),
]

pca_cells = [
    md_cell("# PCA Problem Statement Solution\n\nThis notebook solves the PCA + clustering problem from `8 PCA_Problem Statement (1).pdf` using `heart disease.xlsx`."),
    md_cell("## Imports"),
    code_cell(
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.cluster import KMeans, AgglomerativeClustering\n"
        "from sklearn.decomposition import PCA\n"
        "from sklearn.metrics import silhouette_score, adjusted_rand_score\n"
        "from sklearn.preprocessing import StandardScaler"
    ),
    md_cell("## Load Data and Prepare Features"),
    code_cell(
        "df = pd.read_excel('heart disease.xlsx')\n"
        "print('Shape:', df.shape)\n"
        "print('Missing:', int(df.isna().sum().sum()))\n"
        "print('Target distribution:', df['target'].value_counts().to_dict())\n"
        "X = df.drop(columns=['target'])\n"
        "X_scaled = StandardScaler().fit_transform(X)"
    ),
    md_cell("## Select Best K Using Silhouette"),
    code_cell(
        "ks = list(range(2, 11))\n"
        "inertia, sil = [], []\n"
        "for k in ks:\n"
        "    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n"
        "    labels = km.fit_predict(X_scaled)\n"
        "    inertia.append(float(km.inertia_))\n"
        "    sil.append(float(silhouette_score(X_scaled, labels)))\n"
        "best_k = ks[int(np.argmax(sil))]\n"
        "print('Best K:', best_k)\n"
        "pd.DataFrame({'k': ks, 'inertia': inertia, 'silhouette': sil})"
    ),
    md_cell("## Clustering on Original Data"),
    code_cell(
        "km_o = KMeans(n_clusters=best_k, random_state=42, n_init=10)\n"
        "labels_km_o = km_o.fit_predict(X_scaled)\n"
        "hc_o = AgglomerativeClustering(n_clusters=best_k, linkage='ward')\n"
        "labels_hc_o = hc_o.fit_predict(X_scaled)\n"
        "print('Original KMeans silhouette:', silhouette_score(X_scaled, labels_km_o))\n"
        "print('Original Hierarchical silhouette:', silhouette_score(X_scaled, labels_hc_o))"
    ),
    md_cell("## PCA (3 Components)"),
    code_cell(
        "pca = PCA(n_components=3, random_state=42)\n"
        "X_pca = pca.fit_transform(X_scaled)\n"
        "print('Explained variance ratio:', pca.explained_variance_ratio_)\n"
        "print('Cumulative (3 PCs):', pca.explained_variance_ratio_.sum())\n"
        "pca_df = pd.DataFrame(X_pca, columns=['PC1','PC2','PC3'])\n"
        "pca_df.to_csv('heart_pca_3_components.csv', index=False)\n"
        "pca_df.head()"
    ),
    md_cell("## Clustering on PCA Data and Comparison"),
    code_cell(
        "km_p = KMeans(n_clusters=best_k, random_state=42, n_init=10)\n"
        "labels_km_p = km_p.fit_predict(X_pca)\n"
        "hc_p = AgglomerativeClustering(n_clusters=best_k, linkage='ward')\n"
        "labels_hc_p = hc_p.fit_predict(X_pca)\n"
        "print('PCA KMeans silhouette:', silhouette_score(X_pca, labels_km_p))\n"
        "print('PCA Hierarchical silhouette:', silhouette_score(X_pca, labels_hc_p))\n"
        "print('ARI KMeans original vs PCA:', adjusted_rand_score(labels_km_o, labels_km_p))\n"
        "print('ARI Hierarchical original vs PCA:', adjusted_rand_score(labels_hc_o, labels_hc_p))"
    ),
]

kmeans_nb = notebook(kmeans_cells)
pca_nb = notebook(pca_cells)

(BASE / "KMeans_Clustering_Solutions.ipynb").write_text(json.dumps(kmeans_nb, indent=2), encoding="utf-8")
(BASE / "PCA_Problem_Statement_Solution.ipynb").write_text(json.dumps(pca_nb, indent=2), encoding="utf-8")

print("Created notebooks:")
print("- KMeans_Clustering_Solutions.ipynb")
print("- PCA_Problem_Statement_Solution.ipynb")

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
OUT = BASE / "assignment_results.json"


def pick_kmeans_k(x, k_min=2, k_max=10, random_state=42):
    n = x.shape[0]
    k_max = min(k_max, max(k_min, n - 1))
    ks = list(range(k_min, k_max + 1))
    inertia = []
    sil = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(x)
        inertia.append(float(model.inertia_))
        sil.append(float(silhouette_score(x, labels)))
    best_k = int(ks[int(np.argmax(sil))])
    return {
        "k_values": ks,
        "inertia": [round(v, 3) for v in inertia],
        "silhouette": [round(v, 4) for v in sil],
        "best_k": best_k,
    }


def cluster_sizes(labels):
    return {str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().to_dict().items()}


def analyze_numeric_dataset(df, drop_cols=None, dataset_name="dataset"):
    drop_cols = drop_cols or []
    x = df.drop(columns=drop_cols, errors="ignore").copy()

    # Keep only numeric columns for this path.
    x = x.select_dtypes(include=[np.number])

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imp = imp.fit_transform(x)
    x_scaled = scaler.fit_transform(x_imp)

    scree = pick_kmeans_k(x_scaled)
    k = scree["best_k"]

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(x_scaled)

    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    hc_labels = hc.fit_predict(x_scaled)

    df_with = x.copy()
    df_with["cluster"] = km_labels
    profile = df_with.groupby("cluster").mean(numeric_only=True).round(3).to_dict()

    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "features_used": list(x.columns),
        "missing_values": int(df.isna().sum().sum()),
        "scree": scree,
        "kmeans_silhouette": round(float(silhouette_score(x_scaled, km_labels)), 4),
        "hierarchical_silhouette": round(float(silhouette_score(x_scaled, hc_labels)), 4),
        "ari_kmeans_vs_hierarchical": round(float(adjusted_rand_score(km_labels, hc_labels)), 4),
        "kmeans_cluster_sizes": cluster_sizes(km_labels),
        "hierarchical_cluster_sizes": cluster_sizes(hc_labels),
        "kmeans_cluster_profile_means": profile,
    }


def analyze_mixed_dataset(df, id_like_cols=None, date_cols=None):
    id_like_cols = id_like_cols or []
    date_cols = date_cols or []

    data = df.copy()
    for c in date_cols:
        if c in data.columns:
            data[c] = pd.to_datetime(data[c], errors="coerce")
            # Convert date to ordinal day count.
            data[c] = data[c].map(lambda x: x.toordinal() if pd.notna(x) else np.nan)

    data = data.drop(columns=id_like_cols, errors="ignore")

    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in data.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    x = pre.fit_transform(data)

    scree = pick_kmeans_k(x)
    k = scree["best_k"]

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(x)

    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    hc_labels = hc.fit_predict(x)

    result = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "missing_values": int(df.isna().sum().sum()),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "id_like_dropped": id_like_cols,
        "date_cols_encoded": date_cols,
        "transformed_dimension": int(x.shape[1]),
        "scree": scree,
        "kmeans_silhouette": round(float(silhouette_score(x, km_labels)), 4),
        "hierarchical_silhouette": round(float(silhouette_score(x, hc_labels)), 4),
        "ari_kmeans_vs_hierarchical": round(float(adjusted_rand_score(km_labels, hc_labels)), 4),
        "kmeans_cluster_sizes": cluster_sizes(km_labels),
        "hierarchical_cluster_sizes": cluster_sizes(hc_labels),
    }

    # Add a compact profile from original columns.
    numeric_profile_cols = [c for c in ["Tenure in Months", "Monthly Charge", "Total Charges", "Total Revenue", "Income", "Customer Lifetime Value", "Total Claim Amount"] if c in df.columns]
    if numeric_profile_cols:
        tmp = df[numeric_profile_cols].copy()
        tmp["cluster"] = km_labels
        result["kmeans_profile_selected_means"] = tmp.groupby("cluster").mean(numeric_only=True).round(3).to_dict()

    return result


def analyze_pca_heart(df):
    x = df.drop(columns=["target"], errors="ignore")

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(imp.fit_transform(x))

    scree = pick_kmeans_k(x_scaled)
    k = scree["best_k"]

    km_o = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_o_labels = km_o.fit_predict(x_scaled)

    hc_o = AgglomerativeClustering(n_clusters=k, linkage="ward")
    hc_o_labels = hc_o.fit_predict(x_scaled)

    pca = PCA(n_components=3, random_state=42)
    x_pca = pca.fit_transform(x_scaled)

    pca_df = pd.DataFrame(x_pca, columns=["PC1", "PC2", "PC3"])
    pca_df.to_csv(BASE / "heart_pca_3_components.csv", index=False)

    km_p = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_p_labels = km_p.fit_predict(x_pca)

    hc_p = AgglomerativeClustering(n_clusters=k, linkage="ward")
    hc_p_labels = hc_p.fit_predict(x_pca)

    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "missing_values": int(df.isna().sum().sum()),
        "target_distribution": {str(k2): int(v) for k2, v in df["target"].value_counts().to_dict().items()},
        "scree": scree,
        "pca_explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "pca_cumulative_variance_3": round(float(np.sum(pca.explained_variance_ratio_)), 4),
        "original_kmeans_silhouette": round(float(silhouette_score(x_scaled, km_o_labels)), 4),
        "original_hierarchical_silhouette": round(float(silhouette_score(x_scaled, hc_o_labels)), 4),
        "pca_kmeans_silhouette": round(float(silhouette_score(x_pca, km_p_labels)), 4),
        "pca_hierarchical_silhouette": round(float(silhouette_score(x_pca, hc_p_labels)), 4),
        "ari_kmeans_original_vs_pca": round(float(adjusted_rand_score(km_o_labels, km_p_labels)), 4),
        "ari_hierarchical_original_vs_pca": round(float(adjusted_rand_score(hc_o_labels, hc_p_labels)), 4),
        "cluster_sizes": {
            "kmeans_original": cluster_sizes(km_o_labels),
            "hierarchical_original": cluster_sizes(hc_o_labels),
            "kmeans_pca": cluster_sizes(km_p_labels),
            "hierarchical_pca": cluster_sizes(hc_p_labels),
        },
    }


def main():
    results = {
        "kmeans_pdf_problem_statements": {},
        "pca_pdf_problem_statement": {},
    }

    # 1) Airlines
    airlines = pd.read_excel(BASE / "EastWestAirlines (1) (1).xlsx", sheet_name="data")
    results["kmeans_pdf_problem_statements"]["1_airlines"] = analyze_numeric_dataset(
        airlines,
        drop_cols=["ID#"],
        dataset_name="EastWestAirlines",
    )

    # 2) Crime
    crime = pd.read_excel(BASE / "crime_data (1).xlsx")
    results["kmeans_pdf_problem_statements"]["2_crime"] = analyze_numeric_dataset(
        crime,
        drop_cols=["Unnamed: 0"],
        dataset_name="crime_data",
    )

    # 3) Insurance
    insurance = pd.read_excel(BASE / "Insurance Dataset.xlsx")
    results["kmeans_pdf_problem_statements"]["3_insurance"] = analyze_numeric_dataset(
        insurance,
        drop_cols=[],
        dataset_name="Insurance",
    )

    # 4) Telco mixed
    telco = pd.read_excel(BASE / "Telco_customer_churn (1) (1).xlsx")
    results["kmeans_pdf_problem_statements"]["4_telco"] = analyze_mixed_dataset(
        telco,
        id_like_cols=["Customer ID", "Count"],
        date_cols=[],
    )

    # 5) AutoInsurance mixed
    auto = pd.read_excel(BASE / "AutoInsurance (1).xlsx")
    results["kmeans_pdf_problem_statements"]["5_autoinsurance"] = analyze_mixed_dataset(
        auto,
        id_like_cols=["Customer"],
        date_cols=["Effective To Date"],
    )

    # PCA heart disease
    heart = pd.read_excel(BASE / "heart disease.xlsx")
    results["pca_pdf_problem_statement"]["heart_disease"] = analyze_pca_heart(heart)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {OUT}")
    for key, val in results["kmeans_pdf_problem_statements"].items():
        print(key, "best_k=", val["scree"]["best_k"])
    print("pca best_k=", results["pca_pdf_problem_statement"]["heart_disease"]["scree"]["best_k"])


if __name__ == "__main__":
    main()

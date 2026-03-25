import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "heart disease.xlsx"
OUT_JSON = BASE / "pca_assignment_results.json"


def safe_silhouette(x, labels):
    unique = np.unique(labels)
    if len(unique) < 2:
        return None
    return float(silhouette_score(x, labels))


def main():
    df = pd.read_excel(DATA_PATH)

    result = {}
    result["dataset"] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_names": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "duplicates": int(df.duplicated().sum()),
    }

    # Use target only for interpretation; avoid using it while clustering.
    x_df = df.drop(columns=["target"]).copy()

    result["eda"] = {
        "summary": x_df.describe().round(3).to_dict(),
        "target_distribution": {str(k): int(v) for k, v in df["target"].value_counts().to_dict().items()},
        "sex_distribution": {str(k): int(v) for k, v in df["sex"].value_counts().to_dict().items()},
        "correlation_with_target": df.corr(numeric_only=True)["target"].sort_values(ascending=False).round(3).to_dict(),
    }

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_df)

    k_values = list(range(2, 11))
    inertia_values = []
    silhouette_values = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(x_scaled)
        inertia_values.append(float(km.inertia_))
        silhouette_values.append(float(silhouette_score(x_scaled, labels)))

    best_k = int(k_values[int(np.argmax(silhouette_values))])

    km_orig = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_orig_labels = km_orig.fit_predict(x_scaled)

    hc_orig = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hc_orig_labels = hc_orig.fit_predict(x_scaled)

    pca = PCA(n_components=3, random_state=42)
    x_pca = pca.fit_transform(x_scaled)
    pca_df = pd.DataFrame(x_pca, columns=["PC1", "PC2", "PC3"])
    pca_df.to_csv(BASE / "heart_pca_3_components.csv", index=False)

    km_pca = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_pca_labels = km_pca.fit_predict(x_pca)

    hc_pca = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hc_pca_labels = hc_pca.fit_predict(x_pca)

    # Extra hierarchy metadata to describe dendrogram tendency.
    z = linkage(x_scaled, method="ward")
    last_merges = z[-10:, 2].round(3).tolist()

    result["modeling"] = {
        "kmeans_scree": {
            "k_values": k_values,
            "inertia": [round(v, 3) for v in inertia_values],
            "silhouette": [round(v, 4) for v in silhouette_values],
            "selected_k": best_k,
        },
        "pca": {
            "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "cumulative_variance_3": round(float(np.sum(pca.explained_variance_ratio_)), 4),
            "components_shape": list(x_pca.shape),
        },
        "cluster_quality": {
            "original_kmeans_silhouette": round(float(safe_silhouette(x_scaled, km_orig_labels)), 4),
            "original_hierarchical_silhouette": round(float(safe_silhouette(x_scaled, hc_orig_labels)), 4),
            "pca_kmeans_silhouette": round(float(safe_silhouette(x_pca, km_pca_labels)), 4),
            "pca_hierarchical_silhouette": round(float(safe_silhouette(x_pca, hc_pca_labels)), 4),
        },
        "cluster_comparison": {
            "ari_kmeans_original_vs_pca": round(float(adjusted_rand_score(km_orig_labels, km_pca_labels)), 4),
            "ari_hierarchical_original_vs_pca": round(float(adjusted_rand_score(hc_orig_labels, hc_pca_labels)), 4),
            "ari_kmeans_vs_hierarchical_original": round(float(adjusted_rand_score(km_orig_labels, hc_orig_labels)), 4),
            "ari_kmeans_vs_hierarchical_pca": round(float(adjusted_rand_score(km_pca_labels, hc_pca_labels)), 4),
            "ward_last_merge_distances": last_merges,
        },
        "cluster_sizes": {
            "original_kmeans": {str(k): int(v) for k, v in pd.Series(km_orig_labels).value_counts().sort_index().to_dict().items()},
            "original_hierarchical": {str(k): int(v) for k, v in pd.Series(hc_orig_labels).value_counts().sort_index().to_dict().items()},
            "pca_kmeans": {str(k): int(v) for k, v in pd.Series(km_pca_labels).value_counts().sort_index().to_dict().items()},
            "pca_hierarchical": {str(k): int(v) for k, v in pd.Series(hc_pca_labels).value_counts().sort_index().to_dict().items()},
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved results to: {OUT_JSON}")
    print(f"Selected k (silhouette max): {best_k}")
    print("PCA explained variance ratio:", [round(float(v), 4) for v in pca.explained_variance_ratio_])


if __name__ == "__main__":
    main()

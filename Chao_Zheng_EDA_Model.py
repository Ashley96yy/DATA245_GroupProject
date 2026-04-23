import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from IPython.display import display
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

file_name = "NYC_Building_Energy_and_Water_Data_Disclosure_shared_cleaned.csv"
df = pd.read_csv(file_name, low_memory=False)
print("Shape:", df.shape)
print("Number of columns:", len(df.columns))
df.head()

col_year = "Calendar Year"
col_energy = "Weather Normalized Site EUI (kBtu/ft²)"
col_water = "Water Use (All Water Sources) (kgal)"
col_gfa = "Property GFA - Calculated (Buildings and Parking) (ft²)"
col_type = "Primary Property Type - Self Selected"
col_yb = "Year Built"

target_year = 2024

def coerce_numeric(series):
    s = series.astype(str).str.strip()
    s = s.replace({
        "": np.nan,
        "nan": np.nan,
        "None": np.nan,
        "Not Available": np.nan
    })
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def winsorize(series, lower_q=0.01, upper_q=0.99):
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower, upper)

for col in [col_year, col_energy, col_water, col_gfa, col_yb]:
    df[col] = coerce_numeric(df[col])

print("Numeric conversion complete.")

df_2024 = df[df[col_year] == target_year].copy()

print("Records in 2024:", len(df_2024))
df_2024.head()

missing_summary = pd.DataFrame({
    "missing_count": df_2024[[col_energy, col_water, col_gfa, col_type, col_yb]].isna().sum(),
    "missing_rate": df_2024[[col_energy, col_water, col_gfa, col_type, col_yb]].isna().mean()
}).sort_values("missing_rate", ascending=False)

display(missing_summary)

plot_df = missing_summary["missing_rate"].sort_values()

plt.figure(figsize=(8, 5))
bars = plt.barh(plot_df.index, plot_df.values, color="skyblue")
plt.xlabel("Missing Rate")
plt.title("Missing Rate of Key Variables (2024)")
plt.xlim(0, 1)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.1%}", va="center")

plt.tight_layout()
plt.show()

work = df_2024.copy()

work = work[
    (work[col_gfa] > 0) &
    (work[col_energy] > 0) &
    (work[col_water] > 0)
].copy()

work["water_intensity"] = work[col_water] / work[col_gfa]
work = work[work["water_intensity"] > 0].copy()

print("Records after filtering:", len(work))
print("Retention rate:", round(len(work) / len(df_2024) * 100, 2), "%")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.histplot(work[col_energy].dropna(), bins=30, color="skyblue", ax=axes[0, 0])
axes[0, 0].set_title("Distribution of Raw Weather-Normalized Site EUI")

sns.boxplot(x=work[col_energy].dropna(), color="skyblue", ax=axes[0, 1])
axes[0, 1].set_title("Boxplot of Raw Weather-Normalized Site EUI")

sns.histplot(work["water_intensity"].dropna(), bins=30, color="skyblue", ax=axes[1, 0])
axes[1, 0].set_title("Distribution of Raw Water Intensity")

sns.boxplot(x=work["water_intensity"].dropna(), color="skyblue", ax=axes[1, 1])
axes[1, 1].set_title("Boxplot of Raw Water Intensity")

plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(work[col_energy], work["water_intensity"], alpha=0.25, s=10)
plt.xlabel("Weather-Normalized Site EUI")
plt.ylabel("Water Intensity")
plt.title("Raw Energy-Water Relationship")
plt.tight_layout()
plt.show()

work["energy_wins"] = winsorize(work[col_energy], 0.01, 0.99)
work["water_intensity_wins"] = winsorize(work["water_intensity"], 0.01, 0.99)
work["log_water_intensity"] = np.log1p(work["water_intensity_wins"])

work["property_type_clean"] = work[col_type].fillna("Unknown").astype(str)

work["size_bin"] = pd.qcut(
    work[col_gfa],
    q=4,
    labels=["Small", "Medium", "Large", "Very Large"],
    duplicates="drop"
)

work["vintage_bin"] = pd.cut(
    work[col_yb],
    bins=[-np.inf, 1939, 1979, 1999, np.inf],
    labels=["Pre-1940", "1940-1979", "1980-1999", "2000 and later"],
    include_lowest=True
)

work.head()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.histplot(work["energy_wins"], bins=30, color="skyblue", ax=axes[0, 0])
axes[0, 0].set_title("Distribution of Winsorized Energy")

sns.boxplot(x=work["energy_wins"], color="skyblue", ax=axes[0, 1])
axes[0, 1].set_title("Boxplot of Winsorized Energy")

sns.histplot(work["log_water_intensity"], bins=30, color="skyblue", ax=axes[1, 0])
axes[1, 0].set_title("Distribution of log(1 + Water Intensity)")

sns.boxplot(x=work["log_water_intensity"], color="skyblue", ax=axes[1, 1])
axes[1, 1].set_title("Boxplot of log(1 + Water Intensity)")

plt.tight_layout()
plt.show()

feature_df = work[["energy_wins", "log_water_intensity"]].dropna().copy()
work = work.loc[feature_df.index].copy()

scaler = StandardScaler()
X = scaler.fit_transform(feature_df)

print("Final clustering sample size:", len(work))

seeds = [7, 11, 21, 42, 88]
results = []

for k in range(2, 11):
    inertias = []
    silhouettes = []
    labels_by_seed = {}

    for seed in seeds:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = km.fit_predict(X)

        inertias.append(km.inertia_)
        silhouettes.append(
            silhouette_score(X, labels, sample_size=2000, random_state=seed)
        )
        labels_by_seed[seed] = labels

    aris = []
    seed_list = list(labels_by_seed.keys())
    for i in range(len(seed_list)):
        for j in range(i + 1, len(seed_list)):
            aris.append(
                adjusted_rand_score(labels_by_seed[seed_list[i]], labels_by_seed[seed_list[j]])
            )

    results.append({
        "k": k,
        "inertia_mean": np.mean(inertias),
        "silhouette_mean": np.mean(silhouettes),
        "stability_mean_ari": np.mean(aris)
    })

results_df = pd.DataFrame(results)
display(results_df)

plt.figure(figsize=(6, 4))
plt.plot(results_df["k"], results_df["inertia_mean"], marker="o")
plt.xlabel("k")
plt.ylabel("Mean inertia")
plt.title("Elbow Plot")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(results_df["k"], results_df["silhouette_mean"], marker="o")
plt.xlabel("k")
plt.ylabel("Mean silhouette score")
plt.title("Silhouette Scores")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(results_df["k"], results_df["stability_mean_ari"], marker="o")
plt.xlabel("k")
plt.ylabel("Mean ARI")
plt.title("Repeated-run Stability")
plt.tight_layout()
plt.show()

final_k = 3
print("Selected final k =", final_k)

final_km = KMeans(n_clusters=final_k, random_state=42, n_init=20)
work["cluster"] = final_km.fit_predict(X)

display(work["cluster"].value_counts().sort_index())

centers_original = scaler.inverse_transform(final_km.cluster_centers_)

centers_df = pd.DataFrame(
    centers_original,
    columns=["energy_wins_center", "log_water_center"]
)
centers_df["cluster"] = centers_df.index
centers_df["approx_water_intensity_center"] = np.expm1(centers_df["log_water_center"])

display(centers_df)

cluster_summary = (
    work.groupby("cluster")
    .agg(
        n=("cluster", "size"),
        energy_mean=("energy_wins", "mean"),
        energy_median=("energy_wins", "median"),
        water_mean=("water_intensity_wins", "mean"),
        water_median=("water_intensity_wins", "median"),
        gfa_median=(col_gfa, "median")
    )
    .reset_index()
)

cluster_summary = cluster_summary.merge(centers_df, on="cluster", how="left")
display(cluster_summary)

cluster_names = {
    0: "Moderate Energy - Higher Water",
    1: "Low Energy - Low Water",
    2: "High Energy - Moderate Water"
}

work["cluster_name"] = work["cluster"].map(cluster_names)
cluster_summary["cluster_name"] = cluster_summary["cluster"].map(cluster_names)

display(cluster_summary)

cluster_names = {
    0: "Moderate Energy - Higher Water",
    1: "Low Energy - Low Water",
    2: "High Energy - Moderate Water"
}

work["cluster_name"] = work["cluster"].map(cluster_names)
cluster_summary["cluster_name"] = cluster_summary["cluster"].map(cluster_names)

cluster_summary

# Save the final clustering artifact for quick demo.
energy_lower = work[col_energy].quantile(0.01)
energy_upper = work[col_energy].quantile(0.99)
water_lower = work["water_intensity"].quantile(0.01)
water_upper = work["water_intensity"].quantile(0.99)

cluster_artifact = {
    "model": final_km,
    "scaler": scaler,
    "final_k": final_k,
    "energy_lower": energy_lower,
    "energy_upper": energy_upper,
    "water_lower": water_lower,
    "water_upper": water_upper,
    "feature_names": ["energy_wins", "log_water_intensity"],
    "cluster_names": cluster_names,
}

model_path = "chao_kmeans_cluster_artifact.pkl"
joblib.dump(cluster_artifact, model_path, compress=3)

loaded_artifact = joblib.load(model_path)
size_mb = Path(model_path).stat().st_size / (1024 * 1024)

sample_row = work.iloc[[0]].copy()
sample_energy = sample_row[col_energy].clip(lower=energy_lower, upper=energy_upper)
sample_water_intensity = (sample_row[col_water] / sample_row[col_gfa]).clip(lower=water_lower, upper=water_upper)
sample_log_water = np.log1p(sample_water_intensity)

sample_features = pd.DataFrame({
    "energy_wins": sample_energy.values,
    "log_water_intensity": sample_log_water.values
})

sample_scaled = loaded_artifact["scaler"].transform(sample_features)
sample_cluster = loaded_artifact["model"].predict(sample_scaled)[0]

print("Reload success:", type(loaded_artifact))
print(f"File size: {size_mb:.2f} MB")
print("Saved keys:", list(loaded_artifact.keys()))
print("Demo cluster assignment:", sample_cluster)
print("Cluster name:", loaded_artifact["cluster_names"].get(sample_cluster, "Unknown"))

plt.figure(figsize=(7, 5))
sc = plt.scatter(
    work["energy_wins"],
    work["log_water_intensity"],
    c=work["cluster"],
    s=12,
    alpha=0.6
)
plt.xlabel("Weather-Normalized Site EUI (winsorized)")
plt.ylabel("log(1 + Water Intensity)")
plt.title("Final Cluster Solution")
plt.colorbar(sc, label="Cluster")
plt.tight_layout()
plt.show()

cluster_by_type_counts = pd.crosstab(work["cluster_name"], work["property_type_clean"])
cluster_by_type_percent = (
    pd.crosstab(work["cluster_name"], work["property_type_clean"], normalize="index") * 100
).round(2)

display(cluster_by_type_counts)
display(cluster_by_type_percent)

cluster_by_type_percent.T.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6),
    colormap="tab20"
)
plt.ylabel("Percentage")
plt.title("Property Type Composition Within Each Cluster")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

cluster_by_size_counts = pd.crosstab(work["cluster_name"], work["size_bin"])
cluster_by_size_percent = (
    pd.crosstab(work["cluster_name"], work["size_bin"], normalize="index") * 100
).round(2)

display(cluster_by_size_counts)
display(cluster_by_size_percent)

cluster_by_size_percent.T.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5),
    colormap="Set2"
)
plt.ylabel("Percentage")
plt.title("Size-Bin Composition Within Each Cluster")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

cluster_by_vintage_counts = pd.crosstab(work["cluster_name"], work["vintage_bin"])
cluster_by_vintage_percent = (
    pd.crosstab(work["cluster_name"], work["vintage_bin"], normalize="index") * 100
).round(2)

display(cluster_by_vintage_counts)
display(cluster_by_vintage_percent)

cluster_by_vintage_percent.T.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5),
    colormap="Set3"
)
plt.ylabel("Percentage")
plt.title("Vintage-Bin Composition Within Each Cluster")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

print("Summary:")
print("- Target year:", target_year)
print("- Records in 2024:", len(df_2024))
print("- Records after filtering:", len(work))
print("- Final number of clusters:", final_k)
print("\nCluster summary:")
display(cluster_summary)

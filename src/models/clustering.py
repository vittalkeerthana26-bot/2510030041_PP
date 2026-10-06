from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram

# 1. Setup paths relative to script location
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'src' / 'data' / 'raw_placement_data.csv'
OUT = ROOT / 'reports' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# 2. Features used for clustering
FEATURES = [
    'college_tier',
    'cgpa',
    'backlogs',
    'coding_skill_score',
    'communication_skill_score',
    'internships_count',
    'projects_count'
]

def load_data():
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()
    
    # Standardize tier column if present as string
    if 'college_tier' in df.columns and not pd.api.types.is_numeric_dtype(df['college_tier']):
        df['college_tier'] = df['college_tier'].astype(str).str.extract(r'(\d+)')
        
    df_subset = df[FEATURES].apply(pd.to_numeric, errors='coerce').dropna()
    return StandardScaler().fit_transform(df_subset)

def evaluate_clusters(X, labels):
    mask = labels != -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    n_clusters = len(set(labels_filtered))
    
    if n_clusters < 2:
        return np.nan, np.nan, n_clusters
        
    sample_sz = min(len(X_filtered), 2000)
    sil = silhouette_score(X_filtered, labels_filtered, sample_size=sample_sz, random_state=42)
    db = davies_bouldin_score(X_filtered, labels_filtered)
    
    return round(sil, 4), round(db, 4), n_clusters

def run():
    print("Loading placement data...")
    X = load_data()
    n_samples = len(X)
    print(f"Loaded {n_samples} records. Evaluating K-Means...")
    
    sample_sz = min(n_samples, 2000)
    ks = range(2, 11)
    inertias = []
    silhouettes = []
    
    for k in ks:
        km_model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km_model.fit_predict(X)
        inertias.append(km_model.inertia_)
        silhouettes.append(silhouette_score(X, labels, sample_size=sample_sz, random_state=42))
        
    optimal_k = list(ks)[int(np.argmax(silhouettes))]
    print(f"Optimal K selected: {optimal_k}")
    
    # 3. Generate Elbow plot
    plt.figure(figsize=(8, 5))
    plt.plot(list(ks), inertias, marker='o', color='#2b5c8f')
    plt.xlabel('K (Number of Clusters)')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve - Placement Dataset')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / 'clustering_kmeans_elbow.png', dpi=150)
    plt.close()
    
    # 4. Generate Silhouette Analysis plot
    plt.figure(figsize=(8, 5))
    plt.plot(list(ks), silhouettes, marker='o', color='#d95f02')
    plt.xlabel('K (Number of Clusters)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis - Placement Dataset')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / 'clustering_kmeans_silhouette.png', dpi=150)
    plt.close()
    
    print("\nRunning algorithm evaluations...")
    # K-Means model
    km_labels = KMeans(n_clusters=optimal_k, n_init=10, random_state=42).fit_predict(X)
    sil_km, db_km, c_km = evaluate_clusters(X, km_labels)
    
    # Agglomerative Hierarchical model
    agg_labels = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward').fit_predict(X)
    sil_agg, db_agg, c_agg = evaluate_clusters(X, agg_labels)
    
    # 5. Generate Dendrogram plot
    print("Generating dendrogram...")
    sub_sample = X[np.random.RandomState(42).choice(n_samples, min(n_samples, 1000), replace=False)]
    Z = linkage(sub_sample, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode='lastp', p=30)
    plt.title('Hierarchical Dendrogram - Placement Dataset')
    plt.xlabel('Cluster / Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(OUT / 'hierarchical_dendrogram.png', dpi=150)
    plt.close()
    
    # DBSCAN model
    print("Running DBSCAN...")
    db_model = DBSCAN(eps=0.8, min_samples=5).fit_predict(X)
    sil_db, db_db, c_db = evaluate_clusters(X, db_model)
    noise_count = int(np.sum(db_model == -1))
    
    # 6. Save Summary Output
    summary_df = pd.DataFrame([
        ['K-Means', c_km, sil_km, db_km, 0],
        ['Agglomerative', c_agg, sil_agg, db_agg, 0],
        ['DBSCAN', c_db, sil_db, db_db, noise_count]
    ], columns=['Algorithm', 'Clusters', 'Silhouette_Score', 'Davies_Bouldin_Index', 'Noise_Points'])
    
    print("\nClustering Comparison Results:")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUT / 'clustering_comparison.csv', index=False)
    print(f"\nCompleted successfully! Output files saved to: {OUT}")

if __name__ == '__main__':
    run()
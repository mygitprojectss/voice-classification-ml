from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

def train_cluster(X):

    model = KMeans(n_clusters=2, random_state=42)

    clusters = model.fit_predict(X)

    score = silhouette_score(X, clusters)

    joblib.dump(model, "models/cluster_model.pkl")

    return clusters, score
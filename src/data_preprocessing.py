import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(path):

    df = pd.read_csv(path)

    return df


def preprocess(df):

    X = df.drop("label", axis=1)
    y = df["label"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "models/scaler.pkl")

    return X_scaled, y
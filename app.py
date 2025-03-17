from flask import Flask, jsonify
from google.cloud import bigquery, storage
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def train_and_upload_model():
    """BigQueryからデータ取得 → モデル作成 → Cloud Storageへ保存"""

    # **① BigQueryからデータ取得**
    client = bigquery.Client()
    query = """
    SELECT annual_income, credit_score, loan_amount, past_defaults, investment_frequency, default_next_6_months
    FROM credit_risk_dataset.investors
    """
    df = client.query(query).to_dataframe()

    # **② モデル作成**
    X = df.drop(columns=["default_next_6_months"])
    y = df["default_next_6_months"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC Score: {auc:.3f}")

    # **③ Cloud Storage にアップロード**
    model_filename = "credit_risk_model.pkl"
    joblib.dump(model, model_filename)

    bucket_name = "credit-risk-bucket-test"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_filename)
    blob.upload_from_filename(model_filename)

    return jsonify({"status": "success", "auc_score": auc})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
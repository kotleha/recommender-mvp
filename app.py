from flask import Flask, request, jsonify
import os
import numpy as np
import xgboost as xgb

app = Flask(__name__)

MODEL_AUC = float(os.getenv("MODEL_AUC", "0.9666"))

model = None
if os.path.exists("xgb_model.json"):
    model = xgb.Booster()
    model.load_model("xgb_model.json")
elif os.path.exists("xgb_model.pkl"):
    import pickle
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify(AUC=MODEL_AUC)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error="Model not loaded"), 500

    data = request.get_json(silent=True) or {}
    try:
        view = float(data.get("view", 0))
        addtocart = float(data.get("addtocart", 0))
    except Exception:
        return jsonify(error="Bad input types"), 400

    try:
        dm = xgb.DMatrix(np.array([[view, addtocart]], dtype=float))
        y = float(model.predict(dm, validate_features=False)[0])
        return jsonify(purchase_probability=y)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=False)

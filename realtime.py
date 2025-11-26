import pandas as pd
import joblib
import datetime

model = joblib.load("fraud_detector.pkl")
scaler = joblib.load("scaler.pkl")

# Example input (Replace with live transaction)
transaction = {
    "Time": 10,
    "Amount": 200.75,
    "V1": -1.1,
    "V2": 0.83,
    "V3": -2.4,
    # ...
    "V28": 0.45
}

df = pd.DataFrame([transaction])

df["Amount"] = scaler.transform(df["Amount"].values.reshape(-1,1))

prediction = model.predict(df)[0]

result = "⚠️ FRAUD DETECTED!" if prediction == 1 else "✔️ Transaction is SAFE"
print("\nResult:", result)

# Logging
log = pd.DataFrame([{
    "timestamp": datetime.datetime.now(),
    "amount": transaction["Amount"],
    "status": result
}])

log.to_csv("logs/prediction_log.csv", mode="a", header=False, index=False)
print("Prediction logged successfully.")

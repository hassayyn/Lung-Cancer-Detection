

import numpy as np
import pandas as pd
import joblib

#  Load the trained Random Forest model
model_path = "/kaggle/input/symptom-based/scikitlearn/default/1/best_random_forest_model.pkl"  # Update with the actual path
rf_model = joblib.load(model_path)

#  Define feature names (ensure order matches training)
feature_names = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
    "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE", "ALLERGY",
    "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY", "CHEST PAIN"
]  # 15 input features

#  Function to Take User Input & Predict
def predict_lung_cancer():
    print("\n🩺 *Lung Cancer Prediction System* 🩺")

    user_data = []
    
    for feature in feature_names:
        if feature == "GENDER":
            value = input(f"Enter {feature} (M/F): ").strip().upper()
            while value not in ["M", "F"]:
                print("Invalid input. Please enter M or F.")
                value = input(f"Enter {feature} (M/F): ").strip().upper()
            user_data.append(1 if value == "M" else 0)
        else:
            while True:
                try:
                    value = float(input(f"Enter {feature}: ").strip())
                    user_data.append(value)
                    break
                except ValueError:
                    print("Invalid number. Please try again.")

    # Convert to DataFrame to match model input format
    user_input_df = pd.DataFrame([user_data], columns=feature_names)
    
    # Make Prediction
    pred_prob = rf_model.predict_proba(user_input_df)[0][1]
    prediction = "Cancer" if pred_prob >= 0.5 else "No Cancer"
    
    # Show Results
    print("\n🔍 *Prediction Result* 🔍")
    print(f"   *Prediction:* {prediction}")
    print(f"   *Confidence:* {pred_prob:.2f}")

# ✅ Run Prediction
predict_lung_cancer()
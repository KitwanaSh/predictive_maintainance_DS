import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix

# Load the model and saved data
results_df = pd.read_csv("results.csv")
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

importance_df = pd.read_csv("feature_importances.csv")
X_test = pd.read_csv("X_test.csv")

# Title and Introduction
st.title("Predictive Maintenance Dashboard")
st.write("""
This dashboard helps you predict machine failures and take proactive maintenance actions.
Use the options below to explore the model's performance and test predictions.
""")

# Side-by-Side Visualizations
st.subheader("Model Performance")
col1, col2 = st.columns(2)

# Confusion Matrix
with col1:
    st.write("**Confusion Matrix**")
    cm = confusion_matrix(results_df["y_test"], results_df["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Failure", "Failure"],
                yticklabels=["No Failure", "Failure"], ax=ax)
    st.pyplot(fig)

# Feature importance
with col2:
    st.write("**Feature Importance**")
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 7))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(10), palette="viridis", ax=ax)
    st.pyplot(fig)

# Interactive Testing Section
st.subheader("Test Machine Failure Prediction")
st.write("""
Enter the machine's sensor data below to predict whether it is likely to fail.
""")

# Input fieds fro sensor data
st.write("**Enter Sensor Data:**")
time = st.number_input("Time a machine is running (hour)", min_value=0, max_value=23, value=8)
lights = st.number_input("Lights Energy Consumption (Wh)", min_value=0.0, max_value=100.0, value=10.0)
windspreed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=14.0, value=3.666667)
temperature = st.number_input("Temperature in Teenager Room (°C)", min_value=-10.0, max_value=50.0, value=20.0)
humidity_p = st.number_input("Humidity in Parents Room (%)", min_value=29.166667, max_value=53.326667, value=40.0)
humidity_b = st.number_input("Humidity in Bathroom (%)", min_value=25.0, max_value=100.0, value= 49.090000)
rh_out = st.number_input("Outside Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
temp_laundry = st.number_input("Temperature in the Laundry Area (°C)", min_value=-10.0, max_value=50.0, value=20.0)


# Predict button
if st.button("Predict Failure"):
    # Create input data for the model
    input_data = pd.DataFrame({
        "hour": [time],             # Hour time
        "lights": [lights],         # Lights energy consumption
        "Windspeed": [windspreed],
        "T8": [temperature],        # Temperature in teenager room
        "RH_9": [humidity_p],       # Humidity in kitchen area
        "RH_out": [rh_out],         # Outside humidity
        "RH_5": [humidity_b],       # Humidity in the bathroom
        "T3": [temp_laundry]        # Temparature in the laundry
    })

    # Add missing columns with default values if necessary
    for col in X_test.columns:
        if col not in input_data.columns:
            input_data[col] = 0.0

    # Reorder columns to match the training data
    input_data = input_data[X_test.columns]

    # Make prediction
    prediction = rf_model.predict(input_data)[0]
    prediction_proba = rf_model.predict_proba(input_data)[0][1]

    # Display results
    if prediction == 1:
        st.error("🚨 **Failure Predicted!**")
        st.write(f"The model predicts a **{prediction_proba * 100:.2f}%** chance of failure.")
        st.write("**Actionable Insight:** Schedule maintenance immediately to avoid downtime.")
    else:
        st.success("✅ **No Failure Predicted**")
        st.write(f"The model predicts a **{prediction_proba * 100:.2f}%** chance of failure.")
        st.write("**Actionable Insight:** Continue monitoring the machine for any changes.")

# Storytelling Section
st.subheader("The Story Behind the Model")
st.write("""
### The Problem
Household appliances (eg: refigirators, washing machine, ...) can fail unexpectedly, leading to inconvinience and repair costs.
By analysing IoT sensor data (eg: temparature, humidity, time and power consumption), we can predict if an appliance is likey to fail and recommend mantainance.

### Our Solution
We developed a predictive maintenance model using machine learning to forecast machine failures based on sensor data (e.g., temperature, humidity, energy consumption). This allows for proactive maintenance, reducing downtime and costs.

### The Impact
- **Reduced Downtime:** By predicting failures before they happen, maintenance can be scheduled during planned downtime.
- **Cost Savings:** Proactive maintenance is cheaper than emergency repairs.
- **Improved Efficiency:** Machines operate more reliably, increasing overall productivity.
""")

# Footer
st.write("---")
st.write("Built with ❤️ using Streamlit By Kitwana Ezechiel")
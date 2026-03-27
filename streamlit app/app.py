# streamlit app
import streamlit as st
import pandas as pd
import sys
import os
from inference.Prediction import Predictor

# Add src folder to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_PATH)

# Streamlit Page Config
st.set_page_config(
    page_title="Medical Insurance Charges Predictor",
    layout="wide"
)

st.title("Annual Medical Insurance Charges Predictor")
st.write(
    """
    Enter your information below manually or upload a CSV file to predict insurance charges
    using a trained Random Forest model.
    """
)

# Load Predictor Model
predictor = Predictor(
    model_path=os.path.join(PROJECT_ROOT, "/home/selowa-mphadi/PycharmProjects/pythonProject/medical project/models/artifacts.pkl"),
    target_column="charges",
    debug=False
)

# Manual Input Section
st.header("Manual Input")
age = st.number_input("Age", min_value=0, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", options=["male", "female"])
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox("Region", options=["northwest", "northeast", "southwest", "southeast"])

manual_input_df = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex": sex,
    "smoker": smoker,
    "region": region
}])

# CSV Upload Section
st.header("Upload CSV")
uploaded_file = st.file_uploader("Upload a CSV with the same columns as training data", type=["csv"])
if uploaded_file:
    csv_df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(csv_df.head())
    input_df = csv_df
else:
    input_df = manual_input_df

# Generate Predictions
if st.button("Generate Predictions"):
    with st.spinner("Predicting..."):
        predictions_df = predictor.predict(input_df)

    st.success("Predictions Generated!")
    st.dataframe(predictions_df)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

@st.cache_resource
def load_model():
    return joblib.load("model_bundle.pkl")

bundle = load_model()

model = bundle["model"]
threshold = bundle["threshold"]
features = bundle["features"]

st.title("❤️ Heart Attack Risk Prediction")
st.caption("Machine Learning–based risk screening tool")
st.caption(
    "Prediction is based on lifestyle, medical history, and demographic factors. "
    "Feature importance reflects global model behavior."
)

st.sidebar.header("Patient Details")
st.sidebar.subheader("Lifestyle")

age_label_map = {
    "Age 18 to 24": "18-24",
    "Age 25 to 29": "25-29",
    "Age 30 to 34": "30-34",
    "Age 35 to 39": "35-39",
    "Age 40 to 44": "40-44",
    "Age 45 to 49": "45-49",
    "Age 50 to 54": "50-54",
    "Age 55 to 59": "55-59",
    "Age 60 to 64": "60-64",
    "Age 65 to 69": "65-69",
    "Age 70 to 74": "70-74",
    "Age 75 to 79": "75-79",
    "Age 80 or older": "80 or older"
}

age_label = st.sidebar.selectbox(
    "Age Group",
    list(age_label_map.keys())
)

age_value = age_label_map[age_label]

sleep = st.sidebar.number_input("Sleep Hours", 0.0, 24.0, 7.0)

smoker = st.sidebar.selectbox(
    "Smoking Status",
    ["Never smoked","Former smoker","Current smoker - now smokes every day","Current smoker - now smokes some days"]
)

physical_activity = st.sidebar.selectbox(
    "PhysicalActivities",
    ["No","Yes"]
)

general_health = st.sidebar.selectbox(
    "GeneralHealth",
    ["Excellent","Very good","Good","Fair","Poor"]
)


st.sidebar.subheader("Medical History")

bmi = st.sidebar.number_input("BMI", 10.0, 98.0, 25.0)

diabetic = st.sidebar.selectbox(
    "HadDiabetes",
    ["No","Yes","Yes, but only during pregnancy (female)","No, pre-diabetes or borderline diabetes"]
)

sex = st.sidebar.selectbox(
    "Sex",
    ["Female","Male"]
)

angina = st.sidebar.selectbox(
    "HadAngina",
    ["No","Yes"]
)

chestscan = st.sidebar.selectbox(
    "ChestScan",
    ["No","Yes"]

)

stroke = st.sidebar.selectbox(
    "HadStroke",
    ["No","Yes"]
)

input_dict = {}

numeric_cols = [
    "BMI",
    "SleepHours",
    "PhysicalHealthDays",
    "MentalHealthDays",
    "HeightInMeters",
    "WeightInKilograms"
]

for col in features:
    input_dict[col] = 0 if col in numeric_cols else "No"

input_dict.update({
    "AgeCategory": age_value,
    "BMI": bmi,
    "SleepHours": sleep,
    "SmokerStatus": smoker,
    "HadDiabetes": diabetic,
    "Sex": sex,
    "HadAngina": angina,
    "ChestScan": chestscan,
    "HadStroke": stroke,
    "PhysicalActivities": physical_activity,
    "GeneralHealth": general_health
})

input_df = pd.DataFrame([input_dict])
input_df = input_df[features]

missing = set(features) - set(input_df.columns)
if missing:
    st.error(f"Missing features: {missing}")
    st.stop()

try:
    prob = model.predict_proba(input_df)[0, 1]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

st.write(
    f"The model estimates a **{prob:.1%} probability** of heart attack risk "
    f"based on the provided inputs."
)

prediction = int(prob >= threshold)

st.info(
    f"The risk threshold is set at {threshold:.2f}. "
    "Patients above this value are flagged as higher risk."
)

st.subheader("Prediction Result")

if prediction == 1:
    st.error(f"⚠ High Risk of Heart Attack ({prob:.2%})")
else:
    st.success(f"✅ Low Risk ({prob:.2%})")



st.progress(min(prob, 1.0))
if prob < 0.3:
    st.success("🟢 Low Risk")
elif prob < 0.6:
    st.warning("🟡 Moderate Risk")
else:
    st.error("🔴 High Risk")


importance_df = pd.read_csv("top_feature_importance.csv")
st.bar_chart(importance_df.set_index("feature").head(10))

st.warning(
    "This tool is for educational purposes only and not a medical diagnosis."
)
# python -m streamlit run app.py  

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓"
)

st.title("🎓 Student Performance Prediction")
st.write("Predict student performance using Machine Learning.")

@st.cache_resource
def load_model():
    return joblib.load("student_performance_model.pkl")

try:
    model = load_model()
    st.success("Model loaded successfully!")

    studytime = st.selectbox("Study Time", [1, 2, 3, 4])
    failures = st.number_input("Past Failures", 0, 10, 0)
    absences = st.number_input("Absences", 0, 100, 0)
    G1 = st.number_input("First Period Grade (G1)", 0, 20, 10)
    G2 = st.number_input("Second Period Grade (G2)", 0, 20, 10)

    if st.button("Predict Final Grade"):
        data = pd.DataFrame([{
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "G1": G1,
            "G2": G2
        }])

        prediction = model.predict(data)[0]
        st.success(f"Predicted Final Grade: {prediction:.2f}/20")

except Exception as e:
    st.error(f"Error: {e}")

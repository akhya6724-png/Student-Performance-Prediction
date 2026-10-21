
import gradio as gr
import pandas as pd
import numpy as np
import joblib

model = joblib.load("student_performance_model.pkl")

def predict_grade(studytime, failures, absences, G1, G2):
    student = pd.DataFrame([{
        "studytime": int(studytime),
        "failures": int(failures),
        "absences": int(absences),
        "G1": int(G1),
        "G2": int(G2)
    }])

    prediction = model.predict(student)[0]
    prediction = np.clip(prediction, 0, 20)

    return f"Predicted Final Grade: {prediction:.2f} / 20"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Student Performance Predictor")
    gr.Markdown("Estimate a student's final grade using Machine Learning.")

    studytime = gr.Dropdown(
        choices=[1, 2, 3, 4], value=3, label="Study Time Category"
    )
    failures = gr.Dropdown(
        choices=[0, 1, 2, 3], value=0, label="Previous Failures"
    )
    absences = gr.Number(value=4, label="Absences", precision=0)
    G1 = gr.Slider(0, 20, value=12, step=1, label="First Period Grade (G1)")
    G2 = gr.Slider(0, 20, value=13, step=1, label="Second Period Grade (G2)")

    button = gr.Button("Predict Final Grade", variant="primary")
    output = gr.Textbox(label="Prediction Result")

    button.click(
        fn=predict_grade,
        inputs=[studytime, failures, absences, G1, G2],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)

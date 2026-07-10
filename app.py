import sys
import warnings

# Suppress the deprecation warning since we're intentionally targeting a Python 3.14 bug
warnings.filterwarnings("ignore", category=DeprecationWarning, module="asyncio")

if sys.platform == 'win32':
    import asyncio
    try:
        # Falls back cleanly if policies are completely removed in a later version
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load("student_model.pkl")

# UI Title
st.title("Student Performance Prediction")

# Collect the 5 EXACT features expected by the model
studytime = st.number_input("Study Time (e.g., hours per week)", min_value=0.0, step=1.0)
failures = st.number_input("Past Class Failures", min_value=0, max_value=4, step=1)
absences = st.number_input("Number of Absences", min_value=0, step=1)
g1_grade = st.number_input("G1 Grade (First Period Marks)", min_value=0.0, step=1.0)
g2_grade = st.number_input("G2 Grade (Second Period Marks)", min_value=0.0, step=1.0)

if st.button("Predict"):
    # Package inputs into a 2D array matching the exact training feature order
    data = np.array([
        [
            studytime,   # Feature 0
            failures,    # Feature 1
            absences,    # Feature 2
            g1_grade,    # Feature 3
            g2_grade     # Feature 4
        ]
    ])

    # Predict using the 5 features
    result = model.predict(data)

    # Output the final prediction (usually G3 score)
    st.success(f"Predicted Final Marks (G3): {result[0]:.2f}")
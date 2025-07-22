
import streamlit as st
import numpy as np
import pickle
import os
from pathlib import Path
import plotly.express as px

# ✅ Ensure model directory exists
Path("model").mkdir(parents=True, exist_ok=True)

# 📈 Train a simple model (only for demo purposes)
def train_model():
    X = np.array([[1, 50000], [2, 60000], [3, 70000], [4, 80000], [5, 90000]])
    y = np.array([55000, 62000, 70000, 82000, 94000])
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(X, y)
    with open("model/salary_model.pkl", "wb") as f:
        pickle.dump(model, f)

if not os.path.exists("model/salary_model.pkl"):
    train_model()

# 🔄 Load model
model = pickle.load(open("model/salary_model.pkl", "rb"))

# 🎨 Page config
st.set_page_config(page_title="Employee Salary Predictor", layout="wide", page_icon="💼")
st.markdown("<h1 style='text-align: center; color: green;'>💼 EMPLOYEE SALARY PREDICTION DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown("---")

# 📊 Layout with tabs
tab1, tab2, tab3 = st.tabs(["📌 Prediction", "📈 Salary Trends", "🔍 Insights"])

with tab1:
    st.header("📌 Enter Your Details")

    col1, col2 = st.columns(2)
    with col1:
        exp = st.slider("Years of Experience", 0, 30, 3)
    with col2:
        current_salary = st.number_input("Current Salary (INR)", 10000, 2000000, 50000, step=1000)

    if st.button("🔮 Predict Salary"):
        pred = model.predict([[exp, current_salary]])[0]
        st.success(f"🎯 Predicted Future Salary: ₹{int(pred):,}")
        growth = pred - current_salary
        pct = (growth / current_salary) * 100
        st.info(f"📈 Estimated Growth: ₹{int(growth):,} ({pct:.2f}%)")

        st.progress(min(int(pct), 100))

        # Simulated feature importance
        st.markdown("#### 🔍 Feature Importance")
        st.json({
            "Years of Experience": f"{(exp / (exp + current_salary)) * 100:.2f}%",
            "Current Salary": f"{(current_salary / (exp + current_salary)) * 100:.2f}%"
        })

with tab2:
    st.header("📈 Salary vs Experience")
    # Demo data
    exp_data = np.arange(0, 31, 1)
    sal_data = model.predict(np.column_stack((exp_data, np.full_like(exp_data, current_salary))))
    fig = px.line(x=exp_data, y=sal_data, labels={"x": "Years of Experience", "y": "Predicted Salary"},
                  title="📉 Predicted Salary Based on Experience")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🧠 Insights & Tips")
    st.markdown("""
    - 💬 *More experience* typically leads to *higher salary potential*.
    - 💼 *Current salary* provides a strong signal about future pay.
    - 🚀 Consider upskilling or certifications to boost your worth.
    - 🧾 Keep your resume and portfolio updated regularly.
    """)

# 📎 Custom CSS (if any assets present)
css_path = "assets/custom.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

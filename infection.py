import matplotlib.pyplot as plt
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap

# 加载随机森林模型
model = joblib.load('9-RF.pkl')

# 定义特征名称（根据你的数据调整）
feature_names = [
    "LYMPH%", "RBC", "RDW-CV", "HGB", "MONO#", "NEUT#", "HCT", "WBC", "NEUT%"]

# Streamlit 用户界
st.title("ACP severity App")

# 用户输入特征数据
LYMPH_percent = st.number_input("LYMPH%:", min_value=0.0, max_value=100.0, value=25.3)
RBC = st.number_input("RBC(10^12/L):", min_value=0.0, max_value=100.0, value=1.66)
RDW_CV = st.number_input("RDW-CV(%):", min_value=0.0, max_value=100.0, value=23.6)
HGB = st.number_input("HGB(g/L):", min_value=0.0, max_value=500.0, value=49.0)
MONO_num = st.number_input("MONO#(10^9/L):", min_value=0.0, max_value=100.0, value=0.68)
NEUT_num = st.number_input("NEUT#(10^9/L):", min_value=0.0, max_value=100.0, value=3.52)
HCT = st.number_input("HCT(%):", min_value=0.0, max_value=100.0, value=15.8)
WBC = st.number_input("WBC(10^9/L):", min_value=0.0, max_value=100.0, value=5.65)
NEUT_percent = st.number_input("NEUT%:", min_value=0.0, max_value=100.0, value=62.3)

# 将输入的数据转化为模型的输入格式
feature_values = [
    LYMPH_percent, RBC, RDW_CV, HGB, MONO_num, NEUT_num, HCT, WBC, NEUT_percent
]
features = np.array([feature_values])

# 当点击按钮时进行预测
if st.button("Predict"):
    # 进行预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (0: Mild, 1: Severe)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果提供建议
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of Severe ACP. "
            f"The model predicts that your probability of having Severe ACP is {probability:.1f}%. "
        )
    else:
        advice = (
            f"According to our model, you have a low risk of Mild ACP. "
            f"The model predicts that your probability of having Mild ACP is {probability:.1f}%. "
        )

    st.write(advice)

    # 计算并显示SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 根据预测结果生成并显示SHAP force plot
    if predicted_class == 1:
        shap.force_plot(explainer.expected_value[1], shap_values[:, :, 1],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer.expected_value[0], shap_values[:, :, 0],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    # 保存SHAP图并显示
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

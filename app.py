import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -----------------------------

# LOAD MODEL BUNDLE

# -----------------------------

bundle = joblib.load("breast_cancer_models_compressed.pkl")

scaler = bundle["scaler"]

# -----------------------------

# PAGE SETTINGS

# -----------------------------

st.set_page_config(
page_title="Breast Cancer Survival Prediction",
page_icon="??",
layout="wide"
)

# -----------------------------

# SIDEBAR

# -----------------------------

st.sidebar.title("?? Navigation")

page = st.sidebar.radio(
"Go To",
[
"Prediction",
"Compare Models"
]
)

# -----------------------------

# PREDICTION PAGE

# -----------------------------

if page == "Prediction":

    st.title("?? Breast Cancer Survival Prediction")

    st.markdown(
        """
Predict patient survival status using Machine Learning models.
"""
    )

    model_name = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "KNN",
            "SVM",
            "Naive Bayes",
            "Gradient Boosting",
            "XGBoost"
        ]
    )

    model = bundle[model_name]

    col1, col2 = st.columns(2)

    with col1:

        survival_months = st.number_input(
            "Survival Months",
            min_value=1,
            max_value=107,
            value=60
        )

        regional_nodes = st.number_input(
            "Regional Nodes Positive",
            min_value=1,
            max_value=46,
            value=5
        )

        tumor_size = st.number_input(
            "Tumor Size",
            min_value=1,
            max_value=140,
            value=30
        )

    with col2:

        estrogen_status = st.selectbox(
            "Estrogen Status",
            ["Positive", "Negative"]
        )

        progesterone_status = st.selectbox(
            "Progesterone Status",
            ["Positive", "Negative"]
        )

        a_stage = st.selectbox(
            "A Stage",
            ["Regional", "Distant"]
        )

    if st.button("?? Predict"):

        input_df = pd.DataFrame([[
            survival_months,
            regional_nodes,
            0 if estrogen_status == "Positive" else 1,
            0 if progesterone_status == "Positive" else 1,
            tumor_size,
            0 if a_stage == "Regional" else 1
        ]],
        columns=[
            'Survival Months',
            'Reginol Node Positive',
            'Estrogen Status',
            'Progesterone Status',
            'Tumor Size',
            'A Stage'
        ])

        scaled_input = scaler.transform(input_df)

        prediction = model.predict(
            scaled_input
        )[0]

        probabilities = model.predict_proba(
            scaled_input
        )[0]

        alive_prob = probabilities[0] * 100
        dead_prob = probabilities[1] * 100

        st.subheader("Prediction Result")

        if prediction == 0:

            st.success(
                "?? Predicted Status: Alive"
            )

        else:

            st.error(
                "?? Predicted Status: Dead"
            )

        m1, m2 = st.columns(2)

        m1.metric(
            "Alive Probability",
            f"{alive_prob:.2f}%"
        )

        m2.metric(
            "Dead Probability",
            f"{dead_prob:.2f}%"
        )

        chart_df = pd.DataFrame({
            "Status": [
                "Alive",
                "Dead"
            ],
            "Probability": [
                alive_prob,
                dead_prob
            ]
        })

        fig = px.pie(
            chart_df,
            values="Probability",
            names="Status",
            title="Prediction Probability"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Patient Information"
        )

        st.dataframe(
            input_df,
            use_container_width=True
        )


# -----------------------------

# MODEL COMPARISON PAGE

# -----------------------------

elif page == "Compare Models":

    st.title("?? Compare All Models")

    survival_months = st.number_input(
        "Survival Months",
        min_value=1,
        max_value=107,
        value=60
    )

    regional_nodes = st.number_input(
        "Regional Nodes Positive",
        min_value=1,
        max_value=46,
        value=5
    )

    tumor_size = st.number_input(
        "Tumor Size",
        min_value=1,
        max_value=140,
        value=30
    )

    estrogen_status = st.selectbox(
        "Estrogen Status",
        ["Positive", "Negative"]
    )

    progesterone_status = st.selectbox(
        "Progesterone Status",
        ["Positive", "Negative"]
    )

    a_stage = st.selectbox(
        "A Stage",
        ["Regional", "Distant"]
    )

    if st.button("Run All Models"):

        input_df = pd.DataFrame([[
            survival_months,
            regional_nodes,
            0 if estrogen_status == "Positive" else 1,
            0 if progesterone_status == "Positive" else 1,
            tumor_size,
            0 if a_stage == "Regional" else 1
        ]],
        columns=[
            'Survival Months',
            'Reginol Node Positive',
            'Estrogen Status',
            'Progesterone Status',
            'Tumor Size',
            'A Stage'
        ])

        scaled_input = scaler.transform(
            input_df
        )

        results = []

        model_names = [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "KNN",
            "SVM",
            "Naive Bayes",
            "Gradient Boosting",
            "XGBoost"
        ]

        for name in model_names:

            model = bundle[name]

            pred = model.predict(
                scaled_input
            )[0]

            results.append([
                name,
                "Alive" if pred == 0 else "Dead"
            ])

        results_df = pd.DataFrame(
            results,
            columns=[
                "Model",
                "Prediction"
            ]
        )

        st.dataframe(
            results_df,
            use_container_width=True
        )

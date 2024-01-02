import streamlit as st
import pandas as pd
import pickle
import time
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

df_clean = pd.read_csv("Datasets/dfclean.csv")
X = df_clean.drop("target", axis=1)
y = df_clean["target"]

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

model = pickle.load(open("Model/xgb_model.pkl", "rb"))

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
accuracy = round((accuracy * 100), 2)

df_final = X
df_final["target"] = y

st.set_page_config(page_title="Hungarian Heart Disease", page_icon=":heart:")

st.title("Hungarian Heart Disease")
st.write(
    f"**_Model's Accuracy_** :  :green[**{accuracy}**]% (:red[_Do not copy outright_])"
)
st.write("")

tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

with tab1:
    st.sidebar.header("**User Input** Sidebar")

    age = st.sidebar.number_input(
        label=":violet[**Age**]",
        min_value=df_final["age"].min(),
        max_value=df_final["age"].max(),
    )
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['age'].min()}**], :red[Max] value: :red[**{df_final['age'].max()}**]"
    )
    st.sidebar.write("")

    sex_sb = st.sidebar.selectbox(label=":violet[**Sex**]", options=["Male", "Female"])
    st.sidebar.write("")
    st.sidebar.write("")
    if sex_sb == "Male":
        sex = 1
    elif sex_sb == "Female":
        sex = 0

    cp_sb = st.sidebar.selectbox(
        label=":violet[**Chest pain type**]",
        options=[
            "Typical angina",
            "Atypical angina",
            "Non-anginal pain",
            "Asymptomatic",
        ],
    )
    st.sidebar.write("")
    st.sidebar.write("")
    if cp_sb == "Typical angina":
        cp = 1
    elif cp_sb == "Atypical angina":
        cp = 2
    elif cp_sb == "Non-anginal pain":
        cp = 3
    elif cp_sb == "Asymptomatic":
        cp = 4

    trestbps = st.sidebar.number_input(
        label=":violet[**Resting blood pressure** (in mm Hg on admission to the hospital)]",
        min_value=df_final["trestbps"].min(),
        max_value=df_final["trestbps"].max(),
    )
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['trestbps'].min()}**], :red[Max] value: :red[**{df_final['trestbps'].max()}**]"
    )
    st.sidebar.write("")

    chol = st.sidebar.number_input(
        label=":violet[**Serum cholestoral** (in mg/dl)]",
        min_value=df_final["chol"].min(),
        max_value=df_final["chol"].max(),
    )
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['chol'].min()}**], :red[Max] value: :red[**{df_final['chol'].max()}**]"
    )
    st.sidebar.write("")

    fbs_sb = st.sidebar.selectbox(
        label=":violet[**Fasting blood sugar > 120 mg/dl?**]", options=["False", "True"]
    )
    st.sidebar.write("")
    st.sidebar.write("")
    if fbs_sb == "False":
        fbs = 0
    elif fbs_sb == "True":
        fbs = 1

    restecg_sb = st.sidebar.selectbox(
        label=":violet[**Resting electrocardiographic results**]",
        options=[
            "Normal",
            "Having ST-T wave abnormality",
            "Showing left ventricular hypertrophy",
        ],
    )
    st.sidebar.write("")
    st.sidebar.write("")
    if restecg_sb == "Normal":
        restecg = 0
    elif restecg_sb == "Having ST-T wave abnormality":
        restecg = 1
    elif restecg_sb == "Showing left ventricular hypertrophy":
        restecg = 2

    thalach = st.sidebar.number_input(
        label=":violet[**Maximum heart rate achieved**]",
        min_value=df_final["thalach"].min(),
        max_value=df_final["thalach"].max(),
    )
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['thalach'].min()}**], :red[Max] value: :red[**{df_final['thalach'].max()}**]"
    )
    st.sidebar.write("")

    exang_sb = st.sidebar.selectbox(
        label=":violet[**Exercise induced angina?**]", options=["No", "Yes"]
    )
    st.sidebar.write("")
    st.sidebar.write("")
    if exang_sb == "No":
        exang = 0
    elif exang_sb == "Yes":
        exang = 1

    oldpeak = st.sidebar.number_input(
        label=":violet[**ST depression induced by exercise relative to rest**]",
        min_value=df_final["oldpeak"].min(),
        max_value=df_final["oldpeak"].max(),
    )
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['oldpeak'].min()}**], :red[Max] value: :red[**{df_final['oldpeak'].max()}**]"
    )
    st.sidebar.write("")

    data = {
        "Age": age,
        "Sex": sex_sb,
        "Chest pain type": cp_sb,
        "RPB": f"{trestbps} mm Hg",
        "Serum Cholestoral": f"{chol} mg/dl",
        "FBS > 120 mg/dl?": fbs_sb,
        "Resting ECG": restecg_sb,
        "Maximum heart rate": thalach,
        "Exercise induced angina?": exang_sb,
        "ST depression": oldpeak,
    }

    preview_df = pd.DataFrame(data, index=["input"])

    st.header("User Input as DataFrame")
    st.write("")
    st.dataframe(preview_df.iloc[:, :6])
    st.write("")
    st.dataframe(preview_df.iloc[:, 6:])
    st.write("")

    result = ":violet[-]"

    predict_btn = st.button("**Predict**", type="primary")

    st.write("")
    if predict_btn:
        inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
        prediction = model.predict(inputs)[0]

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        if prediction == 0:
            result = ":green[**Healthy**]"
        elif prediction == 1:
            result = ":orange[**Heart disease level 1**]"
        elif prediction == 2:
            result = ":orange[**Heart disease level 2**]"
        elif prediction == 3:
            result = ":red[**Heart disease level 3**]"
        elif prediction == 4:
            result = ":red[**Heart disease level 4**]"

    st.write("")
    st.write("")
    st.subheader("Prediction:")
    st.subheader(result)

with tab2:
    st.header("Predict multiple data:")

    sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode("utf-8")

    st.write("")
    st.download_button(
        "Download CSV Example",
        data=sample_csv,
        file_name="sample_heart_disease_parameters.csv",
        mime="text/csv",
    )

    st.write("")
    st.write("")
    file_uploaded = st.file_uploader("Upload a CSV file", type="csv")

    if file_uploaded:
        uploaded_df = pd.read_csv(file_uploaded)
        prediction_arr = model.predict(uploaded_df)

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 70):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)

        result_arr = []

        for prediction in prediction_arr:
            if prediction == 0:
                result = "Healthy"
            elif prediction == 1:
                result = "Heart disease level 1"
            elif prediction == 2:
                result = "Heart disease level 2"
            elif prediction == 3:
                result = "Heart disease level 3"
            elif prediction == 4:
                result = "Heart disease level 4"
            result_arr.append(result)

        uploaded_result = pd.DataFrame({"Prediction Result": result_arr})

        for i in range(70, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(uploaded_result)
        with col2:
            st.dataframe(uploaded_df)
# Mengimpor library yang diperlukan
import streamlit as st
import pandas as pd
import pickle
import time
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

# Membaca data bersih dari file CSV
df_clean = pd.read_csv("Datasets/dfclean.csv")
X = df_clean.drop("target", axis=1)
y = df_clean["target"]

# Melakukan oversampling menggunakan SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Memuat model dari file pickle
model = pickle.load(open("Model/oversample_xgb.pkl", "rb"))

# Memprediksi data yang sama dengan yang digunakan untuk training
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
accuracy = round((accuracy * 100), 2)

# Menyiapkan DataFrame untuk user input
df_final = X
df_final["target"] = y

# Mengatur konfigurasi halaman Streamlit
st.set_page_config(page_title="Hungarian Heart Disease", page_icon=":heart:")

# Menampilkan judul halaman
st.title("Hungarian Heart Disease")
# Menampilkan akurasi model
st.write(
    f"**_Model's Accuracy_** :  :green[**{accuracy}**]%"
)
st.write("")

# Membuat tab untuk single prediction dan multi-prediction
tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

# Bagian Single-predict
with tab1:
    st.sidebar.header("**User Input** Sidebar")

        # Menambahkan input untuk usia dengan batasan nilai minimum dan maksimum
    age = st.sidebar.number_input(
        label=":white[**Age**]",
        min_value=int(df_final["age"].min()),
        max_value=int(df_final["age"].max()),
        step=1 # Mengatur langkah menjadi 1 agar hanya menerima nilai integer
    )
        # Menampilkan informasi tentang batas nilai usia
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['age'].min()}**], :red[Max] value: :red[**{df_final['age'].max()}**]"
    )
    st.sidebar.write("")

    # Menambahkan pilihan jenis kelamin
    sex_sb = st.sidebar.selectbox(label=":white[**Sex**]", options=["Male", "Female"])
    st.sidebar.write("")
    # Mengkonversi jenis kelamin menjadi angka
    if sex_sb == "Male":
        sex = 1
    elif sex_sb == "Female":
        sex = 0

    # Menambahkan pilihan jenis nyeri dada
    cp_sb = st.sidebar.selectbox(
        label=":white[**Chest pain type**]",
        options=[
            "Typical angina",
            "Atypical angina",
            "Non-anginal pain",
            "Asymptomatic",
        ],
    )
    st.sidebar.write("")
    # Mengkonversi jenis nyeri dada menjadi angka
    if cp_sb == "Typical angina":
        cp = 1
    elif cp_sb == "Atypical angina":
        cp = 2
    elif cp_sb == "Non-anginal pain":
        cp = 3
    elif cp_sb == "Asymptomatic":
        cp = 4

    # Menambahkan input untuk tekanan darah istirahat
    trestbps = st.sidebar.number_input(
        label=":white[**Resting blood pressure** (in mm Hg on admission to the hospital)]",
        min_value=df_final["trestbps"].min(),
        max_value=df_final["trestbps"].max(),
    )
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['trestbps'].min()}**], :red[Max] value: :red[**{df_final['trestbps'].max()}**]"
    )
    st.sidebar.write("")

    # Menambahkan input untuk serum kolesterol
    chol = st.sidebar.number_input(
        label=":white[**Serum cholestoral** (in mg/dl)]",
        min_value=df_final["chol"].min(),
        max_value=df_final["chol"].max(),
    )
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['chol'].min()}**], :red[Max] value: :red[**{df_final['chol'].max()}**]"
    )
    st.sidebar.write("")

    # Menambahkan pilihan untuk kadar gula darah puasa
    fbs_sb = st.sidebar.selectbox(
        label=":white[**Fasting blood sugar > 120 mg/dl?**]", options=["False", "True"]
    )
    st.sidebar.write("")
    # Mengkonversi pilihan gula darah puasa menjadi angka
    if fbs_sb == "False":
        fbs = 0
    elif fbs_sb == "True":
        fbs = 1

    # Menambahkan pilihan untuk hasil elektrokardiogram istirahat
    restecg_sb = st.sidebar.selectbox(
        label=":white[**Resting electrocardiographic results**]",
        options=[
            "Normal",
            "Having ST-T wave abnormality",
            "Showing left ventricular hypertrophy",
        ],
    )
    st.sidebar.write("")
    # Mengkonversi hasil elektrokardiogram istirahat menjadi angka
    if restecg_sb == "Normal":
        restecg = 0
    elif restecg_sb == "Having ST-T wave abnormality":
        restecg = 1
    elif restecg_sb == "Showing left ventricular hypertrophy":
        restecg = 2

    
    # Menambahkan input untuk detak jantung maksimal
    thalach = st.sidebar.number_input(
        label=":white[**Maximum heart rate achieved**]",
        min_value=df_final["thalach"].min(),
        max_value=df_final["thalach"].max(),
    )
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['thalach'].min()}**], :red[Max] value: :red[**{df_final['thalach'].max()}**]"
    )
    st.sidebar.write("")

    # Menambahkan pilihan untuk angina yang diinduksi oleh olahraga
    exang_sb = st.sidebar.selectbox(
        label=":white[**Exercise induced angina?**]", options=["No", "Yes"]
    )
    st.sidebar.write("")
    # Mengkonversi pilihan angina yang diinduksi oleh olahraga menjadi angka
    if exang_sb == "No":
        exang = 0
    elif exang_sb == "Yes":
        exang = 1

    # Menambahkan input untuk depresi ST
    oldpeak = st.sidebar.number_input(
        label=":white[**ST depression induced by exercise relative to rest**]",
        min_value=df_final["oldpeak"].min(),
        max_value=df_final["oldpeak"].max(),
    )
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['oldpeak'].min()}**], :red[Max] value: :red[**{df_final['oldpeak'].max()}**]"
    )
    st.sidebar.write("")

    # Membuat DataFrame untuk input pengguna
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

    # Menampilkan DataFrame hasil input pengguna
    st.header("User Input as DataFrame")
    st.write("")
    # st.dataframe(preview_df.iloc[:, :6])
    # st.dataframe(preview_df.iloc[:, 6:])
    st.dataframe(preview_df)

    # Menambahkan tombol prediksi
    predict_btn = st.button("**Predict**", type="primary")

    # Melakukan prediksi saat tombol ditekan
    st.write("")
    if predict_btn:
        inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
        prediction = model.predict(inputs)[0]

        # Menampilkan bar progress selama prediksi berlangsung
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

        # Menampilkan hasil prediksi dan deskripsi hasilnya
        if prediction == 0:
            result = ":green[**Healthy**]"
            desc = 'Ini menunjukkan bahwa seseorang dinyatakan sebagai individu yang sehat dari segi kesehatan jantung. Biasanya, ini berarti bahwa hasil pemeriksaan atau model prediktif menunjukkan bahwa risiko penyakit jantung pada individu tersebut rendah atau tidak ada.'
        elif prediction == 1:
            result = ":orange[**Heart disease level 1**]"
            desc = 'Ini mengindikasikan bahwa seseorang memiliki tingkat ringan dari penyakit jantung. Meskipun mungkin ada beberapa indikasi atau faktor risiko, tingkat ini biasanya dianggap sebagai awal dari perkembangan penyakit jantung.'
        elif prediction == 2:
            result = ":orange[**Heart disease level 2**]"
            desc = 'Tingkat ini menunjukkan tingkat penyakit jantung yang lebih lanjut atau lebih serius daripada tingkat 1. Gejala dan risiko komplikasi bisa menjadi lebih signifikan.'
        elif prediction == 3:
            result = ":red[**Heart disease level 3**]"
            desc = 'Ini mencerminkan penyakit jantung pada tingkat yang cukup serius, dengan gejala dan risiko komplikasi yang lebih parah. Pengelolaan dan perawatan medis yang intensif mungkin diperlukan.'
        elif prediction == 4:
            result = ":red[**Heart disease level 4**]"
            desc = 'Ini adalah tingkat penyakit jantung yang paling parah. Pada tingkat ini, seseorang mungkin menghadapi risiko yang sangat tinggi terhadap masalah kesehatan jantung dan mungkin memerlukan perawatan medis yang sangat intensif.'

        # Menampilkan hasil prediksi dan deskripsi
        st.write("")
        st.subheader("Prediction:")
        st.subheader(result)
        st.write(desc)

# Bagian Multi-predict
with tab2:
    st.header("Predict multiple data:")

    # Membuat sample file CSV
    sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode("utf-8")

    st.write("")
    # Menambahkan tombol untuk mengunduh file CSV sample
    st.download_button(
        "Download CSV Example",
        data=sample_csv,
        file_name="sample_heart_disease_parameters.csv",
        mime="text/csv",
    )

    st.write("")
    # Menambahkan input untuk mengunggah file CSV
    file_uploaded = st.file_uploader("Upload a CSV file", type="csv")

    if file_uploaded:
        # Membaca file CSV yang diunggah
        uploaded_df = pd.read_csv(file_uploaded)
        # Memprediksi hasil untuk data yang diunggah
        prediction_arr = model.predict(uploaded_df)

        # Menampilkan bar progress selama prediksi berlangsung
        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 70):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)

        # Menyiapkan hasil prediksi dalam bentuk DataFrame
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

        # Menampilkan hasil prediksi dan data yang diunggah
        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(uploaded_result)
        with col2:
            st.dataframe(uploaded_df)
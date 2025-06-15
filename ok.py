import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import joblib
import os
from PIL import UnidentifiedImageError

st.set_page_config(
    page_title="Klasifikasi Stroke",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

model_data = np.load("mlp_manual_model.npz")
weights_input_hidden = model_data["weights_input_hidden"]
bias_hidden = model_data["bias_hidden"]
weights_hidden_output = model_data["weights_hidden_output"]
bias_output = model_data["bias_output"]

label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

def halaman_beranda():
    """Menampilkan halaman Beranda Awal."""

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: #fff;
        }
        .css-18e3th9 { color: white; }
        .css-1d391kg { color: white; }

        div.stAlert > div > p {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Selamat Datang di PrediStrok")
    st.subheader("Klasifikasi Risiko Serangan Stroke Berdasarkan Data Kesehatan Anda")
    st.markdown("--- ")
    st.header("Deskripsi Proyek")
    st.write(
        "Proyek ini adalah proyek yang dibangun untuk mengklasifikasi kemungkinan "
        "seseorang terkena stroke berdasarkan beberapa faktor risiko kesehatan dan gaya hidup. "
        "Stroke adalah kondisi medis serius yang terjadi ketika pasokan darah ke bagian otak "
        "terganggu atau berkurang, mencegah jaringan otak mendapatkan oksigen dan nutrisi, "
        "yang menyebabkan sel-sel otak mulai mati dalam hitungan menit."
    )
    st.write(
        "Dengan mengidentifikasi faktor risiko utama, sistem ini bertujuan untuk memberikan "
        "indikasi awal potensi risiko stroke, memungkinkan pengguna untuk mengambil tindakan "
        "pencegahan atau mencari konsultasi medis lebih lanjut."
    )

    st.header("Tujuan Proyek")
    st.markdown("""
    Tujuan dari proyek ini adalah untuk membangun sebuah sistem klasifikasi stroke menggunakan model MLP dengan Backpropagation. Sistem ini dirancang untuk membantu tenaga medis dan masyarakat umum dalam mengidentifikasi risiko terjadinya stroke secara dini berdasarkan data kesehatan individu. Model diharapkan mampu mengklasifikasikan kemungkinan stroke dengan tingkat akurasi tinggi berdasarkan atribut-atribut kesehatan seperti usia, tekanan darah, kadar glukosa, riwayat penyakit, dan gaya hidup.
    dan ada juga poin poin tujuan dari proyek ini adalah:
    - Mengembangkan model klasifikasi yang akurat untuk risiko stroke menggunakan teknik machine learning.  
    - Menyediakan antarmuka pengguna yang mudah digunakan (aplikasi Streamlit ini) untuk memasukkan data pasien dan mendapatkan hasil klasifikasi.  
    - Meningkatkan kesadaran tentang faktor-faktor risiko stroke.  
    - Menampilkan evaluasi kinerja model yang digunakan.
    """)

    st.markdown(
        """
        <p style="color: white; font-weight: bold; font-style: italic; font-size: 1.2rem;">
            ℹ️ Silakan pilih menu lain di navigasi sebelah kiri untuk melanjutkan.
        </p>
        """,
        unsafe_allow_html=True
    )

def halaman_dataset():
    """Menampilkan halaman Data Set."""
    st.markdown(
        """
        <style>
        .judul-dataset {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: white !important;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            display: inline-block;
            margin-bottom: 1rem;
            border-bottom: 4px solid #2c5364;
            font-weight: 700;
        }
    
        .subjudul {
            color: #0f2027;
            font-weight: 600;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
        }
        .subjudul::before {
            content: "ℹ️";
            margin-right: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<h1 class="judul-dataset">Informasi Data Set</h1>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="subjudul">Penjelasan Data Set</h2>', unsafe_allow_html=True)
    st.write(
        "Data set yang digunakan dalam proyek ini berisi informasi demografis, kesehatan, "
        "dan gaya hidup dari sejumlah individu. Setiap baris data mewakili satu individu, "
        "dan kolom-kolomnya mencakup berbagai atribut yang relevan untuk klasifikasi stroke."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("Fitur-fitur utama dalam data set ini meliputi:")
    st.markdown("""
    <style>
        .fitur-item {
            background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
            color: white;
            padding: 8px 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 14px;
            list-style-type: none;
            display: flex;
            align-items: center;
        }
        .fitur-item::before {
            content: "•";
            color: white;
            font-weight: bold;
            display: inline-block;
            width: 1em;
            margin-right: 8px;
            font-size: 18px;
            line-height: 0;
        }
        .label {
            display: inline-block;
            width: 120px;  /* Lebar label tetap */
            font-weight: bold;
            padding-right: 8px; /* Spasi antara label dan isi */
            flex-shrink: 0;
        }
        .fitur-text {
            flex: 1; /* biar teks isi mengisi sisa lebar */
        }
        .fitur-container {
            max-width: 700px;
            padding-left: 0;
            margin-bottom: 1rem;
        }
    </style>
    <div class="fitur-container">
        <div class="fitur-item"><span class="label">id</span><span class="fitur-text">Nomor identifikasi unik untuk setiap pasien.</span></div>
        <div class="fitur-item"><span class="label">gender</span><span class="fitur-text">Jenis kelamin pasien ('Male', 'Female', atau 'Other').</span></div>
        <div class="fitur-item"><span class="label">age</span><span class="fitur-text">Usia pasien dalam tahun.</span></div>
        <div class="fitur-item"><span class="label">hypertension</span><span class="fitur-text">Apakah pasien memiliki hipertensi (1: Ya, 0: Tidak).</span></div>
        <div class="fitur-item"><span class="label">heart_disease</span><span class="fitur-text">Apakah pasien memiliki penyakit jantung (1: Ya, 0: Tidak).</span></div>
        <div class="fitur-item"><span class="label">ever_married</span><span class="fitur-text">Status pernikahan pasien ('Yes' atau 'No').</span></div>
        <div class="fitur-item"><span class="label">work_type</span><span class="fitur-text">Jenis pekerjaan pasien ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked').</span></div>
        <div class="fitur-item"><span class="label">Residence_type</span><span class="fitur-text">Tipe tempat tinggal pasien ('Urban' atau 'Rural').</span></div>
        <div class="fitur-item"><span class="label">avg_glucose_level</span><span class="fitur-text">Rata-rata kadar glukosa dalam darah pasien.</span></div>
        <div class="fitur-item"><span class="label">bmi</span><span class="fitur-text">Indeks Massa Tubuh (Body Mass Index) pasien.</span></div>
        <div class="fitur-item"><span class="label">smoking_status</span><span class="fitur-text">Status merokok pasien ('formerly smoked', 'never smoked', 'smokes', 'Unknown').</span></div>
        <div class="fitur-item"><span class="label">stroke</span><span class="fitur-text">Variabel target, apakah pasien mengalami stroke (1: Ya, 0: Tidak).</span></div>
    </div>
    """, unsafe_allow_html=True)


    st.header("Sumber Data Set")
    st.write(
        "Data set ini umumnya berasal dari sumber data kesehatan publik atau studi epidemiologi. "
        "Salah satu sumber populer untuk data set semacam ini adalah Kaggle."
    )
    st.markdown("""
    Contoh sumber data set (perlu diverifikasi sumber pastinya):  
    [Stroke Prediction Dataset di Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
    """)
    st.warning("Pastikan untuk selalu memeriksa lisensi dan etika penggunaan data set.")


def hasil_klasifikasi():
    """Menampilkan HasiL Klasifikasi Pada Data Test"""
    st.markdown(
        """
        <style>
        .judul-dataset {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: white !important;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            display: inline-block;
            margin-bottom: 1rem;
            border-bottom: 4px solid #2c5364;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h1 class="judul-dataset">Klasifikasi Risiko Stroke Pada Data Test</h1>', unsafe_allow_html=True)
    st.markdown("--- ")

    df = pd.read_csv("hasil_prediksi_dan_fitur.csv")

    def highlight_benar_salah(row):
        warna = "background-color: lightgreen" if row['benar/salah'] == "benar" else "background-color: salmon"
        return [warna] * len(row)
    
    rows_per_page = 10
    total_pages = int(np.ceil(len(df) / rows_per_page))
    page = st.number_input("Halaman", min_value=1, max_value=total_pages, step=1)

    start = (page - 1) * rows_per_page
    end = start + rows_per_page
    df_page = df.iloc[start:end]

    st.write(f"Menampilkan baris {start + 1} sampai {min(end, len(df))} dari {len(df)}")
    st.dataframe(df_page.style.apply(highlight_benar_salah, axis=1))

    st.markdown("---")
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(df['stroke'], df['hasil_klasifikasi'])

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Stroke", "Stroke"], yticklabels=["No Stroke", "Stroke"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


def halaman_evaluasi():
    """Menampilkan halaman Evaluasi Model."""
    st.title("Evaluasi Kinerja Model")
    st.markdown("--- ")
    st.write(
        "Bagian ini menampilkan metrik evaluasi dari model machine learning yang digunakan "
        "untuk prediksi stroke. Evaluasi ini penting untuk memahami seberapa baik kinerja model."
    )

    # --- Metrik Evaluasi (Placeholder) ---
    accuracy = 0.6663
    training_time = 2.74 
    testing_time =  0.01 
    precision = 0.1099
    recall = 0.8200
    f1_score = 0.1939
    specificity = 0.6584

    st.subheader("Metrik Utama")
    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi (Accuracy)", f"{accuracy:.2%}")
    col2.metric("Presisi (Precision)", f"{precision:.2f}")
    col3.metric("Recall (Sensitivity)", f"{recall:.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1-Score", f"{f1_score:.2f}")
    col5.metric("Spesifisitas (Specificity)", f"{specificity:.2f}")

    st.subheader("Waktu Komputasi")
    col7, col8 = st.columns(2)
    col7.metric("Waktu Training", f"{training_time:.2f} detik")
    col8.metric("Waktu Testing", f"{testing_time:.2f} detik")

    st.markdown("--- ")
    st.subheader("Penjelasan Metrik")
    st.markdown(
        """
        *   **Akurasi:** Persentase total prediksi yang benar (baik positif maupun negatif) dari keseluruhan data.
        *   **Presisi:** Dari semua prediksi positif yang dibuat model, berapa persen yang benar-benar positif.
        *   **Recall (Sensitivity):** Dari semua kasus positif yang sebenarnya, berapa persen yang berhasil diprediksi dengan benar oleh model.
        *   **F1-Measure:** Rata-rata harmonik dari Presisi dan Recall, memberikan skor tunggal yang menyeimbangkan keduanya.
        *   **Spesifisitas:** Dari semua kasus negatif yang sebenarnya, berapa persen yang berhasil diprediksi dengan benar oleh model.
        *   **Waktu Training:** Waktu yang dibutuhkan untuk melatih model menggunakan data training.
        *   **Waktu Testing:** Waktu yang dibutuhkan model untuk membuat prediksi pada data testing.
        """
    )


def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_mlp_manual(X):
    Zin = np.dot(X, weights_input_hidden) + bias_hidden
    A_hidden = relu(Zin)
    Yin = np.dot(A_hidden, weights_hidden_output) + bias_output
    A_output = sigmoid(Yin)
    return (A_output > 0.5).astype(int).flatten()[0], A_output.flatten()[0]

def klasifikasi_manual():
    """Menampilkan halaman klasifikasi."""
    st.title("Klasifikasi Risiko Stroke")
    st.markdown("--- ")
    st.write("Masukkan data pasien di bawah ini untuk melakukan klasifikasi.")

    with st.form("prediction_form"):
        st.subheader("Informasi Pasien")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Jenis Kelamin", options=label_encoders['gender'].classes_)
            age = st.number_input("Usia (tahun)", min_value=0, max_value=120, value=50, step=1)
            hypertension = st.radio("Memiliki Hipertensi?", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak", horizontal=True)
            heart_disease = st.radio("Memiliki Penyakit Jantung?", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak", horizontal=True)
            ever_married = st.selectbox("Status Pernikahan", options=label_encoders['ever_married'].classes_)

        with col2:
            work_type = st.selectbox("Jenis Pekerjaan", options=label_encoders['work_type'].classes_)
            Residence_type = st.selectbox("Tipe Tempat Tinggal", options=label_encoders['Residence_type'].classes_)
            avg_glucose_level = st.number_input("Rata-rata Kadar Glukosa Darah", min_value=0.0, value=100.0, step=0.1)
            bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, value=25.0, step=0.1)
            smoking_status = st.selectbox("Status Merokok", options=label_encoders['smoking_status'].classes_)

        submitted = st.form_submit_button("Klasifikasikan Sekarang")

    if submitted:
        st.markdown("--- ")
        st.subheader("Hasil klasifikasi")

        # Encode input data
        input_data = pd.DataFrame({
            'gender': [label_encoders['gender'].transform(['Male'])[0]], 
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [label_encoders['ever_married'].transform([ever_married])[0]],
            'work_type': [label_encoders['work_type'].transform([work_type])[0]],
            'Residence_type': [label_encoders['Residence_type'].transform([Residence_type])[0]],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [label_encoders['smoking_status'].transform([smoking_status])[0]],
        })

        st.write("Data yang Dimasukkan:")
        st.dataframe(input_data)

        # Imputasi & scaling
        input_data[["bmi"]] = imputer.transform(input_data[["bmi"]])
        input_data[["age", "avg_glucose_level", "bmi"]] = scaler.transform(
            input_data[["age", "avg_glucose_level", "bmi"]])

        # Prediksi
        prediction, prob = predict_mlp_manual(input_data.to_numpy())

        # Tampilkan hasil
        if prediction == 1:
            st.error(f"Hasil klasifikasi: **Berisiko Terkena Stroke** (Probabilitas: {prob:.2f})")
            st.warning("Penting: Hasil klasifikasi ini bersifat indikatif. Segera konsultasikan dengan dokter.")
        else:
            st.success(f"Hasil klasifikasi: **Risiko Rendah Terkena Stroke** (Probabilitas: {prob:.2f})")
            st.info("Tetap jaga gaya hidup sehat.")


def halaman_tentang_kami():
    st.markdown("""
        <style>
        /* Warna latar belakang biru di bagian atas */
        .header-biru {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            padding: 3rem 1rem 2rem 1rem;
            text-align: center;
            color: white;
            border-radius: 0 0 30px 30px;
            margin-bottom: 2rem;
        }


        .profil-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }

        .profil-container img {
            border-radius: 50%;
            margin-bottom: 10px;
        }

        .profil-nama {
            font-weight: 700;
            font-size: 1.2rem;
            margin: 0.5rem 0 0.2rem 0;
            color: #1a1a1a;
        }

        .profil-nim {
            font-size: 0.95rem;
            color: #555;
        }

        .profil-deskripsi {
            font-size: 0.9rem;
            color: #444;
            margin-top: 0.5rem;
        }
        </style>

        <div class="header-biru">
            <h1>Tentang Kami</h1>
        </div>
    """, unsafe_allow_html=True)

    st.write("Tim pengembang aplikasi klasifikasi stroke ini terdiri dari:")

    anggota1 = {
        "nama": "M. Alfa Reza Gobel",
        "nim": "23031554207",
        "deskripsi": "Mahasiswa  dengan minat di bidang Data Science dan Analisis Data. Aktif mengembangkan keterampilan dalam pengolahan, visualisasi, dan interpretasi data untuk mendukung pengambilan keputusan berbasis data.",
        "foto": "Alfareza.jpg"
    }
    anggota2 = {
        "nama": "Akhmad Dany",
        "nim": "23031554234",
        "deskripsi": "Saya Akhmad Dany, seorang profesional di bidang Sains Data yang memiliki ketertarikan pada analisis data, machine learning, dan visualisasi informasi untuk mendukung pengambilan keputusan berbasis data.",
        "foto": "Akhmad Dany.jpg"
    }
    anggota3 = {
        "nama": "M. Aqsa Firdaus",
        "nim": "23031554087",
        "deskripsi": "Mahasiswa dengan ketertarikan pada analisis data dan keterampilan dalam menggunakan tools seperti Python, PostgreSQL dan Excel. Antusias dalam mengembangkan keterampilan baru dan siap berkolaborasi dalam tim.",
        "foto": "Moh Aqsa Firdaus.jpg"
    }

    tim = [anggota1, anggota2, anggota3]

    cols = st.columns(len(tim))
    for i, anggota in enumerate(tim):
        with cols[i]:
            st.markdown('<div class="profil-container">', unsafe_allow_html=True)

            foto_path = anggota.get("foto", "")
            try:
                if foto_path and os.path.exists(foto_path):
                    st.image(foto_path, width=150)
                else:
                    st.image("https://via.placeholder.com/150", width=150)
            except UnidentifiedImageError:
                st.image("https://via.placeholder.com/150", width=150)
                st.warning(f"Gagal menampilkan gambar: {foto_path}")

            st.markdown(f"""
                <div class="profil-nama">{anggota["nama"]}</div>
                <div class="profil-nim"><strong>NIM:</strong> {anggota["nim"]}</div>
                <div class="profil-deskripsi">{anggota["deskripsi"]}</div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


# ========================================================================================================
st.markdown(
    """
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: white;
        padding: 1rem;
    }

    /* Judul Navigasi Menu */
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h3 {
        color: black !important;
        font-weight: bold !important;
        font-size: 2.5rem !important;
        margin-bottom: 1.5rem;
    }

    /* Kontainer radio */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    /* Label navigasi */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 1rem;
        font-weight: 600;
        border: 1px solid transparent;
        width: 100%;
        box-sizing: border-box;
        transition: all 0.2s ease;
        cursor: pointer;
    }

    /* Hover */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label:hover {
        background-color: #000000;
        color: white !important;
    }

    /* Aktif */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[aria-checked="true"] {
        background-color: #1a73e8 !important;
        color: white !important;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Judul sidebar: 1 baris, hitam, besar
st.sidebar.markdown("## Navigasi Menu")

# Radio menu
pilihan_menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["Beranda awal", "Data Set", "Hasil Klasifikasi", "Evaluasi Model", "Klasifikasi Manual", "Tentang Kami"]
)

# Routing ke halaman
if pilihan_menu == "Beranda awal":
    halaman_beranda()
elif pilihan_menu == "Data Set":
    halaman_dataset()
elif pilihan_menu == "Hasil Klasifikasi":
    hasil_klasifikasi()
elif pilihan_menu == "Evaluasi Model":
    halaman_evaluasi()
elif pilihan_menu == "Klasifikasi Manual":
    klasifikasi_manual()
elif pilihan_menu == "Tentang Kami":
    halaman_tentang_kami()



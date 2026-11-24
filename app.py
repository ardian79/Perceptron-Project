import pickle
import pandas as pd
import streamlit as st

# Konfigurasi halaman web
st.set_page_config(
    page_title="Klasifikasi Kualitas Kopi Arabika",
    layout="centered"
)

# 1 Muat model Perceptron dan Scaler dari folder models
@st.cache_resource
def load_assets():
    with open('models/perceptron_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# 2 Tampilkan judul dan deskripsi aplikasi
st.title("Klasifikasi Kualitas Kopi Arabika")
st.write(
    "Aplikasi ini menggunakan algoritma Single Layer Perceptron untuk "
    "mengklasifikasikan kualitas kopi Arabika menjadi Premium atau Standar "
    "berdasarkan 6 parameter uji cita rasa skala 0 sampai 10."
)

st.divider()

# 3 Buat form input 6 fitur menggunakan slider dibagi 2 kolom agar rapi
st.subheader("Masukkan Skor Uji Cita Rasa")
st.caption("Rentang nilai setiap parameter 0.00 sampai 10.00 sesuai standar uji cita rasa CQI.")

col1, col2 = st.columns(2)

with col1:
    aroma = st.number_input("Aroma atau Keharuman 0 sampai 10", min_value=0.0, max_value=10.0, value=7.60, step=0.01, format="%.2f", help="Skor aroma kopi maksimal 10.00")
    flavor = st.number_input("Flavor atau Cita Rasa 0 sampai 10", min_value=0.0, max_value=10.0, value=7.60, step=0.01, format="%.2f", help="Skor cita rasa utama maksimal 10.00")
    aftertaste = st.number_input("Aftertaste atau Rasa Tertinggal 0 sampai 10", min_value=0.0, max_value=10.0, value=7.50, step=0.01, format="%.2f", help="Skor rasa setelah diminum maksimal 10.00")

with col2:
    acidity = st.number_input("Acidity atau Keasaman 0 sampai 10", min_value=0.0, max_value=10.0, value=7.60, step=0.01, format="%.2f", help="Skor tingkat keasaman maksimal 10.00")
    body = st.number_input("Body atau Kekentalan 0 sampai 10", min_value=0.0, max_value=10.0, value=7.50, step=0.01, format="%.2f", help="Skor tekstur kekentalan maksimal 10.00")
    balance = st.number_input("Balance atau Keseimbangan 0 sampai 10", min_value=0.0, max_value=10.0, value=7.50, step=0.01, format="%.2f", help="Skor keseimbangan rasa maksimal 10.00")

# 4 Tombol prediksi
if st.button("Prediksi Kualitas Kopi", type="primary", use_container_width=True):
    # Susun input menjadi DataFrame sesuai urutan kolom saat training
    data_input = pd.DataFrame(
        [[aroma, flavor, aftertaste, acidity, body, balance]],
        columns=['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
    )

    # Normalisasi input menggunakan StandardScaler yang sudah disimpan
    data_scaled = scaler.transform(data_input)

    # Prediksi kelas 0 Standar atau 1 Premium dan hitung Net Input
    prediksi = model.predict(data_scaled)[0]
    net_input = model.decision_function(data_scaled)[0]

    # Tampilkan hasil klasifikasi
    if prediksi == 1:
        st.success("### Hasil Klasifikasi: Kopi Kualitas PREMIUM Kelas 1")
        st.write("Biji kopi ini diprediksi memiliki skor total kualitas tinggi minimal 82.5.")
    else:
        st.warning("### Hasil Klasifikasi: Kopi Kualitas STANDAR Kelas 0")
        st.write("Biji kopi ini diprediksi memiliki skor total kualitas standar di bawah 82.5.")

# 5 Tampilkan detail perhitungan Perceptron sebagai nilai tambah untuk demo ke dosen
    with st.expander("Lihat Detail Perhitungan Perceptron Net Input"):
        st.write(f"**Nilai Net Input v = sum xi wi + b:** `{net_input:.4f}`")
        st.write(
            "Sesuai aturan Step Function dengan Threshold 0: "
            "jika nilai v lebih besar sama dengan 0 maka hasil 1 Premium, "
            "jika nilai v lebih kecil dari 0 maka hasil 0 Standar."
        )
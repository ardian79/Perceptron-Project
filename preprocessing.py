import pandas as pd

# Script untuk membersihkan data kopi arabika
# Memilih fitur, menghapus data kosong, dan membuat label kualitas

# 1 Baca dataset awal
df = pd.read_csv('arabica_data_cleaned.csv')
print(f"Jumlah data awal: {len(df)} baris")

# 2 Pilih kolom fitur rasa dan skor total
kolom_dipakai = [
    'Aroma', 'Flavor', 'Aftertaste',
    'Acidity', 'Body', 'Balance',
    'Total.Cup.Points'
]
df = df[kolom_dipakai].copy()

# 3 Hapus data kosong dan data bernilai nol
df = df.dropna()
df = df[(df > 0).all(axis=1)]

# 4 Ubah skor desimal menjadi dua kelas
# Nilai 82.5 atau lebih menjadi 1 Premium
# Nilai di bawah 82.5 menjadi 0 Standar
# Nilai 82.5 didapatkan dari median pada Total.Cup.Points
batas_nilai = 82.5
df['Kualitas'] = (df['Total.Cup.Points'] >= batas_nilai).astype(int)

# 5 Buang kolom skor total agar AI tidak curang
# Simpan data yang sudah bersih ke file baru
df_final = df[['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Kualitas']]
df_final.to_csv('data_arabica_clean.csv', index=False)

# 6 Tampilkan hasil ringkasan di terminal
print(f"Jumlah data bersih: {len(df_final)} baris")
print("\nPembagian Kelas 1 Premium dan 0 Standar:")
print(df_final['Kualitas'].value_counts())
print("\nSelesai! File disimpan sebagai data_arabica_clean.csv")
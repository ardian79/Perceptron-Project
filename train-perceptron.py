import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# 1 Baca dataset bersih hasil preprocessing
df = pd.read_csv('data_arabica_clean.csv')

# Tentukan kolom fitur dan kolom target
fitur = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
X = df[fitur]
y = df['Kualitas']

# 2 Bagi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3 Samakan skala data agar model stabil
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4 Latih model Perceptron
model = Perceptron(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 5 Uji model dan hitung akurasi
y_pred = model.predict(X_test_scaled)
akurasi = accuracy_score(y_test, y_pred)

print("=" * 45)
print(f"Akurasi Perceptron: {akurasi * 100:.2f}%")
print("=" * 45)

# Tampilkan laporan klasifikasi
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Standar 0', 'Premium 1']))

# Tampilkan bobot fitur dan bias
print("Bobot tiap fitur:")
for nama, bobot in zip(fitur, model.coef_[0]):
    print(f"  {nama:10s}: {bobot:.4f}")
print(f"Bias: {model.intercept_[0]:.4f}")

# 6 Buat grafik confusion matrix untuk laporan
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Standar 0', 'Premium 1'])
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix Perceptron Akurasi {akurasi * 100:.2f}%')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("\nGrafik disimpan sebagai confusion_matrix.png")

# 7 Simpan model dan scaler untuk website
os.makedirs('models', exist_ok=True)
with open('models/perceptron_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model dan Scaler berhasil disimpan di folder models")
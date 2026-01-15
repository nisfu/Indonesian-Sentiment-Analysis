import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# 1. Dataset Sederhana (Contoh)
data = {
    'text': [
        'Produk ini sangat bagus dan bermanfaat', 
        'Saya sangat kecewa, barang rusak saat sampai',
        'Luar biasa, pengiriman cepat sekali',
        'Kualitas buruk, tidak sesuai gambar',
        'Sangat puas dengan pelayanannya',
        'Penjual tidak ramah, respon sangat lama',
        'Meskipun fasilitas yang ada saat ini sudah memadai, beberapa area masih bisa ditingkatkan.',
        'Produk sangat bagus, bahan gak tipis, gak nerawang, rekomended banget buat dijual lagi',
        'Kualitas nya terbilang baik, lentur dan nyaman di pakai untuk aktifitas kerja indoor maupun outdoor.',
        'Fast respon, pengiriman jg cepet bgt, packing rapi.',
        'Memiliki warna indah dan menarik dengan desain yang sangat menarik dan cantik.',
        'Pengalaman belaja yang sangat mengecewakan, barangnya kualitas jelek',
        'Barang yang dikirim tidak sesuai dengan pesanan',
        'Pelayanan cukup lama dan memakan waktu pelanggan.',
        'Sudah PO lama, tapi barang yang datang tidak sesuai pesanan.',
        'Seller tidak ramah, barang yang dikirim tidak sesuai.',
        'Jam pelayanan tidak sesuai dengan yang tertera.',
        'produk banyak barang tiruan.',
        'Terimakasih pesanan nya sudah sampai di rumah dan sudah digunakan juga semoga awet dan cocok ya ketika di pakai.',
        'Barang sudah sampai dengan cepat dan selamat, kualitasnya juga bagus banget.',
        'Produk yang dijual berkualitas baik terimakasih pelapak yang menjual produk yang berkualitas baik.',
        'Paket pesanan belum sampai.'
    ],
    'sentiment': ['positif', 'negatif', 'positif', 'negatif', 'positif', 'negatif',
                  'positif', 'positif', 'positif', 'positif', 'positif', 'negatif',
                  'negatif', 'negatif', 'negatif', 'negatif', 'negatif', 'negatif',
                  'positif', 'positif', 'positif', 'negatif']
}

df = pd.DataFrame(data)

# 2. Mengubah teks menjadi angka (Vectorization)
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['text'])
y = df['sentiment']

# 3. Training Model (Gunakan Naive Bayes - Jagoannya Teks)
model = MultinomialNB()
model.fit(X, y)

# 4. Simpan Model dan Vectorizer (PENTING: Harus simpan dua-duanya)
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Sukses! Model dan Vectorizer telah dibuat.")
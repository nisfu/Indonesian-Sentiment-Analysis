import streamlit as st
import pickle
import os

# Set judul halaman
st.set_page_config(page_title="Analisis Sentimen App", page_icon="😊")

# --- LOAD MODEL & VECTORIZER ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'sentiment_model.pkl')
vectorizer_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        tfidf = pickle.load(f)
else:
    st.error("File model atau vectorizer tidak ditemukan! Jalankan train_sentiment.py dulu.")
    st.stop()

# --- TAMPILAN DASHBOARD ---
st.title("🤖 Pendeteksi Sentimen Ulasan")
st.write("Masukkan ulasan pelanggan di bawah ini untuk mengetahui apakah sentimennya **Positif** atau **Negatif**.")

# Input teks dari user
user_input = st.text_area("Tulis ulasan di sini:", placeholder="Contoh: Barang bagus, pengiriman cepat!")

if st.button("Analisis Sentimen"):
    if user_input.strip() != "":
        # 1. Transform teks input menggunakan TF-IDF yang sudah dilatih
        input_vector = tfidf.transform([user_input])
        
        # 2. Prediksi
        prediction = model.predict(input_vector)[0]
        prediction_proba = model.predict_proba(input_vector)
        
        # 3. Tampilkan Hasil
        st.write("---")
        if prediction == 'positif':
            st.success(f"Hasil Analisis: **SENTIMEN POSITIF** 😊")
        else:
            st.error(f"Hasil Analisis: **SENTIMEN NEGATIF** 😡")
            
        # Menampilkan probabilitas (keyakinan AI)
        st.write(f"Tingkat keyakinan AI: {max(prediction_proba[0])*100:.2f}%")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")

st.sidebar.info("Proyek ini menggunakan algoritma Naive Bayes dan TF-IDF Vectorizer untuk mengklasifikasikan teks Bahasa Indonesia.")
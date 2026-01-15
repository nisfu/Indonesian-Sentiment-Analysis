# 🎭 Indonesian Sentiment Analysis Dashboard
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

Aplikasi berbasis Machine Learning untuk mendeteksi sentimen (Positif atau Negatif) dari ulasan pelanggan dalam Bahasa Indonesia secara real-time.

---

## 🌟 Live Demo
🔗 **[KLIK DI SINI UNTUK MENCOBA APLIKASI](https://indonesian-sentiment-analysis.streamlit.app/)**

---

## 📌 Project Overview
Proyek ini dibuat untuk membantu bisnis memahami kepuasan pelanggan secara cepat. Dengan menggunakan teknik **Natural Language Processing (NLP)**, aplikasi ini mampu membedakan ulasan yang bersifat mendukung (positif) maupun keluhan (negatif).

### 🛠️ Alur Kerja (Workflow)
1. **Data Collection**: Simulasi data ulasan pelanggan Bahasa Indonesia.
2. **Text Vectorization**: Mengubah teks menjadi angka menggunakan **TF-IDF Vectorizer**.
3. **Machine Learning Model**: Menggunakan algoritma **Multinomial Naive Bayes** (sangat efektif untuk klasifikasi teks).
4. **Deployment**: Interface interaktif menggunakan **Streamlit Cloud**.

---

## 📊 Key Insights (Wawasan Teknis)
* **TF-IDF Importance**: Model ini tidak hanya menghitung frekuensi kata, tetapi juga memberikan bobot lebih tinggi pada kata-kata yang unik dan bermakna.
* **Naive Bayes Efficiency**: Dipilih karena memiliki performa tinggi pada dataset teks kecil dan waktu proses yang sangat instan (low latency).
* **Indonesian Nuance**: Model dilatih untuk mengenali kata-kata khas ulasan marketplace di Indonesia.

---

## 🚀 Cara Menjalankan di Lokal
1. **Clone Repository**
   ```bash
   git clone [https://github.com/USERNAME_KAMU/Indonesian-Sentiment-Analysis.git](https://github.com/USERNAME_KAMU/Indonesian-Sentiment-Analysis.git)
2. **Install Library
   ```bash
    pip install -r requirements.txt
4. **Run APP
    ```bash
   streamlit run app_sentiment.py

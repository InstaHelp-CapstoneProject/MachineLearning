# ML Capstone Speech-to-Text and Classification Emergency Case

# InstaHelp ML Model Deployment

This notebook demonstrates how to load and use the trained machine learning model for classifying emergency cases.

### Purpose:
- Classify emergency cases into `high`, `medium`, or `low` levels based on input text.

### Steps:
```
1. Install required dependencies.
2. Load the trained model.
3. Use the model to classify example cases.
```

###  Install Dependencies
```
!pip install tensorflow numpy
```

### Load the trained model
```
import tensorflow as tf
import numpy as np

model.save('model_name.h5')

print("Model loaded successfully!")
```

### Define Prediction Function
```
def predict_emergency_case(text):
    # Preprocessing teks
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus tanda baca dan karakter non-alfabet
    tokens = word_tokenize(text)  # Tokenisasi
    tokens = [word for word in tokens if word not in stop_words]  # Menghapus stopwords
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    
    # Mengonversi token menjadi urutan indeks
    sequence = tokenizer.texts_to_sequences([tokens])
    
    # Melakukan padding pada urutan
    padded = pad_sequences(sequence, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
    
    # Melakukan prediksi
    prediction = model.predict(padded)
    
    # Mengambil indeks label dengan probabilitas tertinggi
    predicted_index = np.argmax(prediction, axis=1)
    
    # Mengonversi indeks kembali ke label asli
    predicted_label = label_encoder.inverse_transform(predicted_index)
    
    return predicted_label[0]
```

### Test Prediction
```
# Contoh teks untuk diuji
sample_texts = [
    "Dilokasi ini ada tawuran, dan 2 orang luka luka.",
    "Saya merasa pusing dan ingin berkonsultasi dengan dokter.",
    "Ingin tahu informasi jadwal ambulans terdekat."
]

# Melakukan prediksi pada setiap contoh teks
for text in sample_texts:
    label = predict_emergency_case(text)
    print(f"Text: {text}\nPredicted Label: {label}\n")
```

## Clone Repository
```
Classification_case/
├── Classification_case.ipynb  # Notebook berisi panduan dan kode
├── model_name.h5              # Model ML yang telah dilatih
├── README.md                  # Dokumentasi proyek
```

import librosa
import os
import soundfile as sf  # Menggunakan pustaka soundfile untuk menulis file audio

# Direktori audio
input_dir = r"C:\\Users\\amali\\OneDrive\\Documents\\Capstone voice\\cv-corpus-19.0-2024-09-13\\id\\clips"
output_dir = r"C:\Users\amali\OneDrive\Documents\Capstone voice\normalized_clips"
os.makedirs(output_dir, exist_ok=True)

# Mengecek file di input_dir
print(f"Menulis ke folder: {output_dir}")
print(f"File yang ada di {input_dir}: {os.listdir(input_dir)}")

for filename in os.listdir(input_dir):
    if filename.endswith(".mp3"):  # Ganti dengan format .mp3
        print(f"Membaca file: {filename}")
        audio_path = os.path.join(input_dir, filename)
        y, sr = librosa.load(audio_path, sr=None)
        y_normalized = librosa.util.normalize(y)
        output_path = os.path.join(output_dir, filename)
        
        # Menulis file audio yang sudah dinormalisasi
        sf.write(output_path, y_normalized, sr)
        
        # Memberi pesan jika file sudah ditulis
        print(f"File ditulis: {output_path}")

print("Normalisasi selesai!")

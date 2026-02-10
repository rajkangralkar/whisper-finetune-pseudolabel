# generate_transcripts.py

import os
import whisper
import pandas as pd

AUDIO_DIR = "audio"
OUTPUT_CSV = "transcripts.csv"

model = whisper.load_model("base")

data = []

print("🔍 Transcribing audio files...")
for file in os.listdir(AUDIO_DIR):
    if file.endswith((".wav", ".mp3")):
        path = os.path.join(AUDIO_DIR, file)
        result = model.transcribe(path, language="en")
        text = result["text"].strip()

        data.append({
            "audio": path,
            "transcript": text
        })

        print(f"✅ {file} → {text}")

df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)

print(f"📄 Saved transcripts to {OUTPUT_CSV}")

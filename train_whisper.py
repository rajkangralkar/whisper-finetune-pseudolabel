# train_whisper.py

import pandas as pd
import torch
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Seq2SeqTrainer,
    DataCollatorSpeechSeq2SeqWithPadding
)

MODEL_NAME = "openai/whisper-base"
CSV_PATH = "transcripts.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Load processor & model
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# Force English transcription
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="en", task="transcribe"
)
model.config.suppress_tokens = []

def preprocess(batch):
    audio = batch["audio"]

    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    )

    with processor.as_target_processor():
        labels = processor(batch["transcript"]).input_ids

    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = labels
    return batch

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    return_tensors="pt"
)

training_args = TrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=50,
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

print("🚀 Starting fine-tuning...")
trainer.train()

model.save_pretrained("./whisper-finetuned")
processor.save_pretrained("./whisper-finetuned")

print("🎉 Fine-tuning complete!")

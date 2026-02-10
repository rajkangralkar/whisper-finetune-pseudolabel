# Whisper Fine-Tuning with Pseudo-Labeling 🎙️🤖

This repository contains an **experimental pipeline for fine-tuning OpenAI’s Whisper ASR model** using **unlabeled audio data** via **pseudo-labeling**.

The project demonstrates how to:
- Automatically generate transcripts using a pretrained Whisper model
- Use those transcripts to fine-tune Whisper with HuggingFace Transformers
- Build an end-to-end ASR experimentation workflow

⚠️ This project is intended for **learning and experimentation**, not production deployment.

---

## 📌 Project Motivation

High-quality labeled speech data is expensive and hard to obtain.  
This project explores a common research technique called **pseudo-labeling**, where a pretrained model is used to generate labels for unlabeled data, which are then reused for fine-tuning.

This approach is useful for:
- Accent adaptation
- Speaker-specific fine-tuning
- Domain-specific ASR experiments

---

## 🧠 Pipeline Overview

1. **Input:** Raw audio files (`.wav` / `.mp3`)
2. **Auto-Transcription:** Generate transcripts using OpenAI Whisper
3. **Dataset Creation:** Store audio–text pairs in CSV format
4. **Fine-Tuning:** Train Whisper using HuggingFace’s Seq2SeqTrainer
5. **Output:** Fine-tuned Whisper model

---

## 📁 Repository Structure


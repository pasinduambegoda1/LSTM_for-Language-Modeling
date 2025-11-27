# LSTM_for-Language-Modeling Using IMDB Dataset

## 📌 Overview

This project implements a **Neural Language Model** using a custom-built **Peephole LSTM layer** from scratch in TensorFlow/Keras.  
Unlike standard LSTMs, peephole connections allow the internal gates (input, forget, and output gates) to access the **cell state directly**, improving the model’s ability to learn long-term dependencies.

The model is trained on the **IMDB movie reviews dataset**, using **GloVe word embeddings** for rich semantic representation. The final trained model is capable of generating text by predicting the next most probable word in a sequence.

---

## 🚀 Key Features

- Custom implementation of a **Peephole LSTM layer** (no built-in Keras LSTM)
- Uses **GloVe (50d)** pre-trained embeddings  
- Generates n-gram training sequences up to length 15  
- Complete neural architecture including:
  - Embedding Layer (GloVe vectors)
  - Peephole LSTM (stacked)
  - Batch Normalization
  - Dense ReLU layer
  - Softmax output layer
- Trains using categorical cross-entropy over a vocabulary of 10,000+ words  
- Includes **text generation** using the trained model  
- Fully compatible with **Google Colab** (GPU recommended)

---

## 📂 Project Structure


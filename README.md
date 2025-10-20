Hate Speech Detection using Deep Learning Models
This repository contains the source code and data for a comparative study on hate speech detection. We implement and evaluate three distinct deep learning architectures: a Bi-directional LSTM (Bi-LSTM), a 1D Convolutional Neural Network (TextCNN), and a fine-tuned XLM-RoBERTa model.

The primary goal is to classify text entries as either 'Hatespeech' or 'Neutral', with a focus on addressing class imbalance through techniques like weighted loss and Focal Loss.

1. Repository Structure
The core files are organized as follows:

.
├── Bilstm_model_training_testing.py        # PyTorch implementation of the Bi-LSTM model.
├── cnn_model_training_testing.py           # PyTorch implementation of the 1D CNN (TextCNN) model.
├── huggingface_model_trainging_testing.py  # Fine-tuning script for XLM-RoBERTa using the Hugging Face Trainer.
├── preprocessing.py                        # Script for data cleaning and preparation.
├── hatespeech_vs_neutral.xlsx - Sheet1.csv # Assumed original raw dataset (or similar CSV name).
├── cleaned_dataset.csv                     # Processed dataset file (output of preprocessing.py).
├── bilstm_model.pt                         # Saved model weights for Bi-LSTM.
├── textcnn_model.pt                        # Saved model weights for TextCNN.
├── vocab.pkl                               # Vocabulary object saved from PyTorch training.
├── label_encoder.pkl                       # LabelEncoder object saved from PyTorch training.
└── README.md                               # This file.
2. Setup and Prerequisites
Environment
The project is built using Python 3.8+ and relies heavily on the PyTorch and Hugging Face ecosystems.

Installation
Clone the repository:

Install Dependencies: A requirements.txt file listing all necessary packages is required. The key dependencies are:
# Content of requirements.txt
pandas
numpy
scikit-learn
torch
torchtext
nltk
transformers
datasets
accelerate # Required for Hugging Face Trainer

Install them using pip:

3. Execution Workflow
The project should be executed in three main stages: Preprocessing, Classical Model Training (Bi-LSTM & CNN), and Transformer Model Fine-Tuning.

Step 1: Data Preprocessing
The preprocessing.py script is the first step, preparing the raw data for model consumption.

Key Preprocessing Steps:

Cleaning: Removal of URLs, emojis, mentions (@user), and hashtags (#tag).

Normalization: Stripping extra whitespace and standardizing the text.

Output: Creates the cleaned_dataset.csv file from the input data.

Step 2: Training PyTorch Models (Bi-LSTM and TextCNN)
These scripts train neural networks using word embeddings learned during training. They utilize Weighted Cross-Entropy Loss to address class imbalance.

Bi-LSTM Training
The model uses a Bi-directional LSTM layer followed by a linear classifier.

1D CNN (TextCNN) TrainingThe TextCNN model uses convolutional filters of varying sizes (typically $3, 4, 5$) to capture n-gram features, followed by max-pooling.

Output: Both scripts print the Loss, Validation Accuracy, and a final Test Classification Report for the model, saving the model artifacts (.pt, .pkl files).

Step 3: Fine-Tuning XLM-RoBERTa (Hugging Face)
The huggingface_model_trainging_testing.py script fine-tunes the powerful multilingual xlm-roberta-base model.
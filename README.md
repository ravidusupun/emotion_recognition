Hate Speech Detection using Deep Learning

This repository presents a comparative study of deep learning models—Bi-LSTM, 1D CNN (TextCNN), and fine-tuned XLM-RoBERTa—for binary text classification: 'Hatespeech' vs. 'Neutral'. The models are specifically configured to address the inherent class imbalance in the dataset.1. Project Overview and StructureThe core components of the project are the data preprocessing script, the training scripts for the three models, and the output artifacts.

├── Bilstm_model_training_testing.py        # PyTorch: Bi-LSTM Model Training & Evaluation
├── cnn_model_training_testing.py           # PyTorch: 1D CNN (TextCNN) Model Training & Evaluation
├── huggingface_model_trainging_testing.py  # Hugging Face: XLM-RoBERTa Fine-Tuning Script
├── preprocessing.py                        # Data Cleaning and Preparation Script
|
├── hatespeech_vs_neutral.xlsx - Sheet1.csv # Raw Input Dataset
├── cleaned_dataset.csv                     # Cleaned Dataset (Output of preprocessing.py)
|
├── bilstm_model.pt                         # Saved Bi-LSTM Model Weights
├── textcnn_model.pt                        # Saved TextCNN Model Weights
├── vocab.pkl                               # Vocabulary object for PyTorch models
├── label_encoder.pkl                       # LabelEncoder object
└── README.md                               

2. Environment and Setup2.1 PrerequisitesEnsure you have Python 3.8+ installed.2.2 InstallationClone the Repository:Bashgit clone 

Install Dependencies:The project relies on PyTorch, Hugging Face Transformers, and standard machine learning libraries. Create a requirements.txt file and install the necessary packages.Bash# Content for requirements.txt
pandas
numpy
scikit-learn
torch
torchtext
nltk
transformers
datasets
accelerate
Bashpip install -r requirements.txt

3. Execution StepsFollow these steps sequentially to reproduce the experiments.Step 1: Data Preprocessing and CleaningThe preprocessing.py script takes the raw data and performs essential cleaning steps before training.Cleaning StepDetailsURL RemovalRemoves all HTTP/HTTPS links.Punctuation/Emoji RemovalRemoves emojis, unicode symbols, and common punctuation.Tag RemovalRemoves user mentions (@user) and hashtags (#tag).Execute:Bashpython preprocessing.py
Output: Generates the cleaned_dataset.csv file.

Step 2: Training PyTorch Models (Bi-LSTM & TextCNN)These models use Word Embeddings learned from the training data and Weighted Cross-Entropy Loss to manage class imbalance.A. Bi-LSTM ModelBashpython Bilstm_model_training_testing.py
B. 1D CNN (TextCNN) ModelBashpython cnn_model_training_testing.py
Output: Both scripts print the Epoch-wise Training Loss and Validation Report, followed by the Final Test Classification Report. They save their respective model weights and utility files.

Step 3: Fine-Tuning XLM-RoBERTaThe script fine-tunes the xlm-roberta-base model. It leverages the Hugging Face Trainer with a custom implementation of Focal Loss ($\gamma=2.0$), which is highly effective for imbalanced classification.Bashpython huggingface_model_trainging_testing.py
Output: Prints training logs and the Final Test Classification Report. The best performing checkpoint (based on Macro F1-score) is saved to the ./xlmr_hatespeech_best directory.
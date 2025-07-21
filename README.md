# DeepFake Classification using CNN

This project implements a Convolutional Neural Network (CNN) for binary image classification: detecting whether an input face image is real or DeepFake-generated. It was developed as a machine learning solution for a DeepFake detection competition (e.g., Kaggle-style challenge).

##  Problem

DeepFake technology has evolved rapidly and poses significant threats by generating hyper-realistic fake media. The goal of this project is to build a deep learning model capable of distinguishing between real and fake face images with high accuracy.

##  Model Overview

- **Architecture**: Custom CNN (Convolutional Neural Network) designed from scratch.
- **Framework**: TensorFlow / Keras
- **Input**: RGB face images
- **Output**: Binary classification (`real` or `deepfake`)
- **Evaluation Metric**: Accuracy / ROC-AUC (based on competition)

##  Project Structure

proiect_cu_cnn/
│
├── CNN.py # CNN architecture and training pipeline
├── proiect_cnn.py # Main script / orchestrator
├── plot_urile_and_stats.py # Visualization of training metrics
├── best_deepfake_model.h5 # Saved model (HDF5 format)
├── best_deepfake_model.keras # Saved model (Keras format)
├── kaggle_submission.csv # Submission example
├── .idea/ # PyCharm project settings
├── .venv/ # (Optional) Virtual environment


## Results

- Trained on a labeled dataset of real vs fake images.
- Achieved **~XX% accuracy** on validation set (fill in your value).
- Successfully submitted predictions for evaluation.

##  How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/username/DeepFake-Classification.git
   cd DeepFake-Classification
   
(Optional) Create and activate a virtual environment:


python -m venv .venv
.\.venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Train the model:

python CNN.py


Generate predictions and plots:

python proiect_cnn.py
python plot_urile_and_stats.py

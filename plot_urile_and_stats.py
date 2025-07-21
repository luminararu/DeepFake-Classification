from tensorflow.keras.models import load_model

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json



model = load_model('best_deepfake_model.keras')

# Extragem etichetele reale și predicțiile
y_true = []
y_pred = []

for batch in val_dataset:
    images, labels = batch
    preds = model.predict(images)

    y_true.extend(np.argmax(labels.numpy(), axis=1))  # dacă e one-hot
    y_pred.extend(np.argmax(preds, axis=1))

# Matricea de confuzie
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])  # modifică etichetele dacă e nevoie

plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix on Validation Set")
plt.show()


with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

with open('training_history.json', 'r') as f:
    history_dict = json.load(f)



def plot_training_history(history_dict):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Training Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_dict['accuracy'], label='Training Accuracy')
    plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history_dict)
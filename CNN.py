import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall

import pandas as pd
import numpy as np
import os
from PIL import Image


class DeepfakeDetector:
    def __init__(self, img_height=224, img_width=224, num_classes=5):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model = None
        self.label_map = None

    def build_simplified_model(self):

        model = keras.Sequential([
            # Data augmentation
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
            # Eliminare RandomContrast și RandomBrightness

            # Normalizare
            layers.Rescaling(1. / 255),

            # Primul bloc
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.2),

            # Al doilea bloc
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.3),

            # Al treilea bloc
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.4),

            # Global Average Pooling
            layers.GlobalAveragePooling2D(),

            # Un  Dense layer
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),

            # Output
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model = model
        return model

    def load_csv_data(self, csv_path, images_dir, is_test=False):

        df = pd.read_csv(csv_path)

        print(f"Path-ul CSV: {csv_path}")
        print(f"Primele rânduri ale DataFrame-ului:\n{df.head()}")
        print(f"Coloanele DataFrame-ului: {df.columns.tolist()}")
        print(f"Numărul de coloane: {len(df.columns)}")

        # Adaptează numele coloanelor în funcție de structura CSV-ului
        if 'filename' in df.columns:
            image_col = 'filename'
        elif 'image_id' in df.columns:
            image_col = 'image_id'
        else:
            image_col = df.columns[0]

        if is_test:
            label_col = None
            print(f"Set de test - doar imaginile vor fi procesate")
            print(f"Numărul de imagini de test: {len(df)}")
        else:
            if 'label' in df.columns:
                label_col = 'label'
            elif 'class' in df.columns:
                label_col = 'class'
            elif len(df.columns) > 1:
                label_col = df.columns[1]
            else:
                raise ValueError("Nu s-a găsit coloana cu label-uri pentru datele de antrenament/validare")

            print(f"\n=== VERIFICARE DETALII LABEL-URI ===")
            print(f"Coloana cu label-uri: {label_col}")
            print(f"Numărul de clase: {len(df[label_col].unique())}")
            print(f"Label-uri unice: {df[label_col].unique()}")
            print(f"Distribuția claselor:")
            print(df[label_col].value_counts())
            print(f"Tip de date label: {df[label_col].dtype}")

        return df, image_col, label_col

    def create_training_dataset(self, df, image_col, label_col, images_dir, batch_size=32, shuffle=True, augment=False):
        def load_and_preprocess_image(image_path, label):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, [self.img_height, self.img_width])


            #image = image / 255.0

            return image, label

        image_paths = [os.path.join(images_dir, img+'.png') for img in df[image_col]]

        # Convertire label-uri în format numeric
        if df[label_col].dtype == 'object':
            unique_labels = sorted(df[label_col].unique())
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            labels = [self.label_map[label] for label in df[label_col]]
            print(f"Mapare label-uri: {self.label_map}")
        else:
            labels = df[label_col].tolist()

        # Convertire în one-hot encoding
        labels = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)

        # Creare dataset-ul
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def create_test_dataset(self, df, image_col, images_dir, batch_size=32):

        def load_and_preprocess_test_image(image_path):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, [self.img_height, self.img_width])
            #image = image / 255.0  # Normalizare
            return image

        # Creare căi complete către imagini
        image_paths = [os.path.join(images_dir, img+'.png') for img in df[image_col]]

        # Creare dataset fără label-uri
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(load_and_preprocess_test_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def load_train_validation_data(self, train_csv, val_csv, train_images_dir, val_images_dir, batch_size=32):
        #Încar datele de antrenament și validare

        print("=== Încărcare date de antrenament ===")
        train_df, train_img_col, train_label_col = self.load_csv_data(train_csv, train_images_dir, is_test=False)
        train_dataset = self.create_training_dataset(
            train_df, train_img_col, train_label_col,
            train_images_dir, batch_size, shuffle=True, augment=True
        )

        print("\n=== Încărcare date de validare ===")
        val_df, val_img_col, val_label_col = self.load_csv_data(val_csv, val_images_dir, is_test=False)
        val_dataset = self.create_training_dataset(
            val_df, val_img_col, val_label_col,
            val_images_dir, batch_size, shuffle=False, augment=False
        )

        print(f"\nDataset-uri create:")
        print(f"- Antrenament: {len(train_df)} imagini")
        print(f"- Validare: {len(val_df)} imagini")

        return train_dataset, val_dataset

    def compile_model(self, learning_rate=0.001):
        #Compilarea modelului cu parametri optimizați

        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )

        return self.model

    def get_callbacks(self, model_save_path='best_deepfake_model.keras'):  # Schimbă extensia
        #Callback-uri pentru antrenament optimizat

        callbacks = [
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),

            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),

            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                #save_format='keras',
                verbose=1
            )
        ]

        return callbacks

    def train_model(self, train_dataset, val_dataset, epochs=50, model_save_path='best_deepfake_model.keras'):
        #Antrenarea modelului

        callbacks = self.get_callbacks(model_save_path)

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict_test_set(self, test_csv, test_images_dir, batch_size=32):

        print("\n=== Încărcare date de test ===")
        test_df, test_img_col, _ = self.load_csv_data(test_csv, test_images_dir, is_test=True)

        # Verificare  modelului daca este încărcat
        if self.model is None:
            raise ValueError("Modelul nu este încărcat! Antrenează modelul sau încarcă unul salvat.")

        test_dataset = self.create_test_dataset(test_df, test_img_col, test_images_dir, batch_size)

        print(f"Se fac predicții pentru {len(test_df)} imagini de test...")

        # Face predicțiile
        predictions = self.model.predict(test_dataset, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)

        # Convertirea numerelor înapoi în label-uri originale dacă există mapare
        if self.label_map:
            # Inversare maparii
            reverse_label_map = {v: k for k, v in self.label_map.items()}
            predicted_labels = [reverse_label_map[pred] for pred in predicted_classes]
            print(f"Folosind maparea inversă: {reverse_label_map}")
        else:
            predicted_labels = predicted_classes

        # Creare DataFrame pentru submisie
        submission_df = pd.DataFrame({
            test_img_col: test_df[test_img_col],
            'label': predicted_labels
        })

        return submission_df, predictions

    def save_submission(self, submission_df, filename='submission.csv'):

        submission_df.to_csv(filename, index=False)
        print(f"\nFișierul de submisie a fost salvat ca: {filename}")
        print(f"Format: {submission_df.shape}")
        print("Primele predicții:")
        print(submission_df.head())

    def load_model(self, model_path):
        """Încarcă un model salvat"""
        self.model = keras.models.load_model(model_path)
        print(f"Model încărcat din: {model_path}")


def main():
    # Căile către fișierele CSV
    train_csv = 'C:/Users/ionut/Downloads/deepfake-classification-unibuc (1)/train.csv'
    val_csv = 'C:/Users/ionut/Downloads/deepfake-classification-unibuc (1)/validation.csv'
    test_csv = 'C:/Users/ionut/Downloads/deepfake-classification-unibuc (1)/test.csv'

    # Căile către directoarele de imagini
    train_images_dir = r'C:\Users\ionut\Downloads\deepfake-classification-unibuc (1)\train'
    val_images_dir = r'C:\Users\ionut\Downloads\deepfake-classification-unibuc (1)\validation'
    test_images_dir = r'C:\Users\ionut\Downloads\deepfake-classification-unibuc (1)\test'

    # Inițializare detector
    detector = DeepfakeDetector(img_height=224, img_width=224, num_classes=5)

    print("=== CONSTRUIRE MODEL ===")
    model = detector.build_simplified_model()

    print("=== COMPILARE MODEL ===")
    detector.compile_model(learning_rate=0.001)

    print("\n=== ÎNCĂRCARE DATE ===")
    train_dataset, val_dataset = detector.load_train_validation_data(
        train_csv=train_csv,
        val_csv=val_csv,
        train_images_dir=train_images_dir,
        val_images_dir=val_images_dir,
        batch_size=32
    )


    print("\n=== ANTRENARE MODEL ===")
    history = detector.train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=50,
        model_save_path='best_deepfake_model.keras'
    )

    print("\n=== PREDICȚII PE SETUL DE TEST ===")
    #predicții pe test set
    submission_df, predictions = detector.predict_test_set(
        test_csv=test_csv,
        test_images_dir=test_images_dir,
        batch_size=32
    )

    # Salvare fișier de submisie
    detector.save_submission(submission_df, 'kaggle_submission.csv')



if __name__ == "__main__":
    main()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image


class DeepfakeDetector:
    def __init__(self, img_height=224, img_width=224, num_classes=5):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model = None

    def build_improved_cnn_model(self):
        """Arhitectură CNN îmbunătățită pentru detectarea deepfake"""

        model = keras.Sequential([
            # Data augmentation integrat în model
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.RandomBrightness(0.1),

            # Normalizare
            layers.Rescaling(1. / 255),

            # Primul bloc - extragere caracteristici de bază
            layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                          input_shape=(self.img_height, self.img_width, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            # Al doilea bloc - caracteristici complexe
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            # Al treilea bloc - pattern-uri avansate
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.3),

            # Al patrulea bloc - caracteristici high-level
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.3),

            # Al cincilea bloc - reprezentări abstracte
            layers.Conv2D(1024, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),

            # Layere fully connected cu regularizare
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model = model
        return model

    def load_csv_data(self, csv_path, images_dir):
        """Încarcă datele din fișierul CSV cu label-uri"""

        # Citirea CSV-ului
        df = pd.read_csv(csv_path)

        image_col = df.columns[0]
        # Check if 'label' column exists
        if 'label' in df.columns:
            label_col = 'label'
        else:
            label_col = None # No label column for test data

        print(f"Path-ul CSV: {csv_path}")
        print(f"Primele rânduri ale DataFrame-ului:\n{df.head()}")
        print(f"Coloanele DataFrame-ului: {df.columns.tolist()}")  # Afișează toate numele coloanelor ca listă
        print(f"Numărul de coloane: {len(df.columns)}")

        # Adaptează numele coloanelor în funcție de structura CSV-ului
        if 'filename' in df.columns and 'label' in df.columns:
            image_col = 'filename'
            label_col = 'label'
        elif 'image' in df.columns and 'class' in df.columns:
            image_col = 'image'
            label_col = 'class'
        else:
            # Presupunem că prima coloană e imaginea, a doua e label-ul
            image_col = df.columns[0]
            label_col = df.columns[1]

        return df, image_col, label_col

    def create_image_dataset(self, df, image_col, label_col, images_dir, batch_size=32, shuffle=True, augment=False):
        """Creează un dataset TensorFlow din DataFrame cu imagini și label-uri"""

        def load_and_preprocess_image(image_path, label):
            # Citește și procesează imaginea
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, [self.img_height, self.img_width])

            if not augment:  # Normalizare doar pentru validation/test
                image = image / 255.0

            return image, label

        # Creează căile complete către imagini
        image_paths = [os.path.join(images_dir, img) for img in df[image_col]]

        # Convertește label-urile în format numeric dacă e necesar
        if df[label_col].dtype == 'object':
            # Mapare label-uri text la numere
            unique_labels = sorted(df[label_col].unique())
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            labels = [label_map[label] for label in df[label_col]]
            print(f"Mapare label-uri: {label_map}")
        else:
            labels = df[label_col].tolist()

        # Convertește label-urile în one-hot encoding
        labels = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)

        # Creează dataset-ul TensorFlow
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(load_and_preprocess_image,
                              num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def load_train_validation_test_data(self, train_csv, val_csv, test_csv,
                                        train_images_dir, val_images_dir, test_images_dir,
                                        batch_size=32):
        """Încarcă toate cele trei seturi de date"""

        print("=== Încărcare date de antrenament ===")
        train_df, train_img_col, train_label_col = self.load_csv_data(train_csv, train_images_dir)
        train_dataset = self.create_image_dataset(
            train_df, train_img_col, train_label_col,
            train_images_dir, batch_size, shuffle=True, augment=True
        )

        print("\n=== Încărcare date de validare ===")
        val_df, val_img_col, val_label_col = self.load_csv_data(val_csv, val_images_dir)
        val_dataset = self.create_image_dataset(
            val_df, val_img_col, val_label_col,
            val_images_dir, batch_size, shuffle=False, augment=False
        )

        print("\n=== Încărcare date de test ===")
        test_df, test_img_col, test_label_col = self.load_csv_data(test_csv, test_images_dir)
        test_dataset = self.create_image_dataset(
            test_df, test_img_col, test_label_col,
            test_images_dir, batch_size, shuffle=False, augment=False
        )

        print(f"\nDataset-uri create:")
        print(f"- Antrenament: {len(train_df)} imagini")
        print(f"- Validare: {len(val_df)} imagini")
        print(f"- Test: {len(test_df)} imagini")

        return train_dataset, val_dataset, test_dataset

    def compile_model(self, learning_rate=0.001):
        """Compilarea modelului cu parametri optimizați"""

        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return self.model

    def get_callbacks(self, model_save_path='best_deepfake_model.h5'):
        """Callback-uri pentru antrenament optimizat"""

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
                verbose=1
            )
        ]

        return callbacks

    def train_model(self, train_dataset, val_dataset, epochs=50,
                    model_save_path='best_deepfake_model.h5'):
        """Antrenează modelul"""

        callbacks = self.get_callbacks(model_save_path)

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def evaluate_model(self, test_dataset):
        """Evaluează modelul pe setul de test"""

        results = self.model.evaluate(test_dataset, verbose=1)

        print(f"\nRezultate pe setul de test:")
        print(f"Loss: {results[0]:.4f}")
        print(f"Accuracy: {results[1]:.4f}")
        print(f"Precision: {results[2]:.4f}")
        print(f"Recall: {results[3]:.4f}")

        return results

    def predict_on_test(self, test_dataset):
        """Face predicții pe setul de test"""

        predictions = self.model.predict(test_dataset)
        predicted_classes = np.argmax(predictions, axis=1)

        return predictions, predicted_classes


# Exemplu de utilizare
def main():

    # Căile către fișierele CSV
    # În fișierul proiect_cnn.py, unde sunt definite căile

    train_csv = 'C:/Users/ionut/Downloads/deepfake-classification-unibuc (1)/train.csv'
    val_csv = 'C:/Users/ionut/Downloads/deepfake-classification-unibuc (1)/validation.csv'
    test_csv = 'C:/Users/ionut/Downloads/deepfake-classification-unibuc (1)/test.csv'

    # --- Definirea căilor pentru directoarele de imagini ---
    train_images_dir = r'C:\Users\ionut\Downloads\deepfake-classification-unibuc (1)\train'
    val_images_dir = r'C:\Users\ionut\Downloads\deepfake-classification-unibuc (1)\validation' # Atenție la numele folderului de validare!
    test_images_dir =r'C:\Users\ionut\Downloads\deepfake-classification-unibuc (1)\test'



    # Inițializează detectorul - verifică câte clase ai în dataset
    detector = DeepfakeDetector(img_height=224, img_width=224, num_classes=5)  # sau 5 dacă ai 5 clase

    print("=== CONSTRUIRE MODEL ===")
    model = detector.build_improved_cnn_model()

    print("=== COMPILARE MODEL ===")
    # Compilează modelul
    detector.compile_model(learning_rate=0.001)

    # Afișează arhitectura modelului
    #model.summary()

    print("\n=== ÎNCĂRCARE DATE ===")
    # Încarcă toate seturile de date
    train_dataset, val_dataset, test_dataset = detector.load_train_validation_test_data(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        train_images_dir=train_images_dir,
        val_images_dir=val_images_dir,
        test_images_dir=test_images_dir,
        batch_size=32
    )

    print("\n=== ANTRENARE MODEL ===")
    # Antrenează modelul
    history = detector.train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=50,  # Poți reduce la 20-30 pentru teste rapide
        model_save_path='best_deepfake_model.h5'
    )

if __name__ == "__main__":
    main()
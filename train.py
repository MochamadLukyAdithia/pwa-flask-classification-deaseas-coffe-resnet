# train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import os

# ==== Hyperparameter ====
learning_rate = 1e-4
batch_size = 32
epochs = 50
dataset_dir = 'dataset'  
save_dir = 'saved_model'
os.makedirs(save_dir, exist_ok=True)

# ==== Data Augmentasi & Split ====
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ==== Model: ResNet50 + Head ====
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze awal

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)

num_classes = train_generator.num_classes
output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# ==== Kompilasi ====
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ==== Training ====
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# ==== Save model dalam 3 format ====
# 1. Format Keras v3 (direkomendasikan)
model.save(os.path.join(save_dir, "model.keras"))

# 2. Format HDF5 (lama, banyak dipakai)
model.save(os.path.join(save_dir, "model.h5"))

# 3. Format TensorFlow SavedModel (untuk TF Serving / TFLite)
model.export(save_dir)

print(f"\n✅ Model berhasil disimpan di folder: {save_dir}")

# ==== Evaluasi (Confusion Matrix & Classification Report) ====
def evaluate_with_confusion_matrix(model, generator):
    y_prob = model.predict(generator)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = generator.classes

    cm = confusion_matrix(y_true, y_pred)
    print("\n📊 Confusion Matrix:\n", cm)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Accuracy: {acc:.4f}")

    class_names = list(generator.class_indices.keys())
    print("\n📄 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

evaluate_with_confusion_matrix(model, validation_generator)

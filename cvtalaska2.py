# -*- coding: utf-8 -*-
pip install tensorflow

# Install Kaggle
!pip install kaggle

# Upload kaggle.json
from google.colab import files
files.upload()  # Upload kaggle.json

# Set up Kaggle API
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download ALASKA2 dataset
!kaggle competitions download -c alaska2-image-steganalysis -p /content/alaska2

# Unzip dataset
!unzip /content/alaska2/alaska2-image-steganalysis.zip -d /content/alaska2_data

# Organize data
import os
import shutil

base_dir = '/content/alaska2_data'
dest_dir = '/content/organized_alaska2'
os.makedirs(os.path.join(dest_dir, 'manipulated'), exist_ok=True)
os.makedirs(os.path.join(dest_dir, 'cover'), exist_ok=True)

# Copy Cover images
for img in os.listdir(os.path.join(base_dir, 'Cover')):
    shutil.copy(os.path.join(base_dir, 'Cover', img), os.path.join(dest_dir, 'cover', img))

# Copy manipulated images
for folder in ['JMiPOD', 'JUNIWARD', 'UERD']:
    for img in os.listdir(os.path.join(base_dir, folder)):
        shutil.copy(os.path.join(base_dir, folder, img), os.path.join(dest_dir, 'manipulated', img))

print("Data organized in /content/organized_alaska2")

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow_datasets as tfds
import numpy as np
import os
import keras
# Import preprocessing layers directly from layers
from tensorflow.keras import layers, models
from tensorflow.keras import preprocessing as layers_preprocessing # Alias to avoid confusion

# Parámetros
IMG_SIZE = 224  # Tamaño de entrada para el modelo
PATCH_SIZE = 4
NUM_CLASSES = 2  # Clasificación binaria: manipulada vs no manipulada
EMBED_DIM = 64  # Dimensión de los embeddings
NUM_HEADS = 4  # Número de cabezas en la atención multi-cabeza
MLP_DIM = 128  # Dimensión del MLP en el bloque Transformer
BATCH_SIZE = 32
EPOCHS = 10

# Carga y preprocesamiento de ALASKA2
def load_alaska2_data():
    # Nota: ALASKA2 no está en TensorFlow Datasets, debes descargarla manualmente
    # Suponemos que las imágenes están organizadas en carpetas: 'manipulated' y 'cover'
    data_dir = '/content/organized_alaska2'  # Cambia esto por la ruta real
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    # Normalización y aumento de datos - Use the imported layers_preprocessing alias
    data_augmentation = tf.keras.Sequential([
        #layers_preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

# Capa convolucional para generar tokens
class ConvEmbedding(layers.Layer):
    def __init__(self, embed_dim, kernel_size=7, stride=4):
        super(ConvEmbedding, self).__init__()
        self.conv = layers.Conv2D(embed_dim, kernel_size, strides=stride, padding='same')
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = self.conv(x)  # [batch, h', w', embed_dim]
        # Use dynamic batch size tf.shape(x)[0] instead of static shape.
        b = tf.shape(x)[0]
        _, h, w, c = x.shape # h, w, c can still be extracted from static shape
        x = tf.reshape(x, [b, h * w, c])  # [batch, num_tokens, embed_dim]
        x = self.norm(x)
        return x

# Bloque Transformer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = models.Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout)
        ])

    def call(self, x, training=False):
        x_norm = self.norm1(x)
        attn_output = self.attn(x_norm, x_norm, training=training)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x), training=training)
        return x

# Custom layer to add the classification token
class AddClsToken(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        # The classification token will be a trainable parameter
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer='zeros',
            trainable=True,
            name='cls_token'
        )

    def call(self, x):
        # Get the batch size from the input tensor
        batch_size = tf.shape(x)[0]
        # Expand the cls_token to match the batch size
        cls_token = tf.tile(self.cls_token, [batch_size, 1, 1])
        # Concatenate the cls_token with the input tokens
        x = tf.concat([cls_token, x], axis=1)
        return x

# Modelo CvT
def create_cvt_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = ConvEmbedding(embed_dim=EMBED_DIM)(inputs)

    # Añadir token de clasificación usando la nueva capa
    x = AddClsToken(embed_dim=EMBED_DIM)(x)

    # Bloques Transformer
    x = TransformerBlock(EMBED_DIM, NUM_HEADS, MLP_DIM)(x, training=True)

    # Clasificación
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = x[:, 0, :]  # Tomar el cls_token
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return models.Model(inputs, outputs)

# Crear y compilar el modelo
model = create_cvt_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Cargar datos
train_ds, val_ds = load_alaska2_data()

# Entrenar el modelo
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
)

# Evaluar el modelo
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

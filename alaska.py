import tensorflow as tf
import os
from swint_se import ReshapeLayer, PatchEmbedding, PatchMerging, SwinTBlock
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import time

# ======================
# Selección de GPUs 1–4
# ======================
gpus = tf.config.list_physical_devices('GPU')
visible_gpus = [i for i in [ 1, 2, 6, 7] if i < len(gpus)]
tf.config.set_visible_devices([gpus[i] for i in visible_gpus], 'GPU')

# ======================
# Parámetros
# ======================
BATCH_SIZE = 128
IMG_SIZE = (256, 256)
EPOCHS = 400
IMAGE_SIZE = 256
srm_weights = np.load('filters/SRM_Kernels.npy') 
biasSRM = np.ones(30)

# ======================
# Cargar datasets
# ======================
X_train = np.load('/alaska2/database/WOW/X_train.npy')
y_train = np.load('/alaska2/database/WOW/y_train.npy')
X_val = np.load('/alaska2/database/WOW/X_valid.npy')
y_val = np.load('/alaska2/database/WOW/y_valid.npy')

# ======================
# Estrategia multi-GPU
# ======================
strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{i}" for i in visible_gpus])
print(f"GPUs activas: {strategy.num_replicas_in_sync}")

# ======================
# Definición de activación personalizada
# ======================
def Tanh3(x):
    return K.tanh(x)*3

# ======================
# Modelo CNN + Swin Transformer
# ======================
with strategy.scope():
    input_shape = tf.keras.Input(shape=(256, 256, 1), name="input_1")

    layers_ty = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=False, activation=Tanh3, use_bias=True)(input_shape)
    layers_tn = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=True, activation=Tanh3, use_bias=True)(input_shape)

    layers = tf.keras.layers.add([layers_ty, layers_tn])
    layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    # L1-L2
    
    
    layers = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.0001), bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers1)
    layers = tf.nn.leaky_relu(layers, alpha=0.2)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    image_size = layers.shape[1]
    layers = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(1, 1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.0001), bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)



    # Swin Transformer
    projection = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        strides=(2, 2),
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        bias_regularizer=tf.keras.regularizers.l2(0.0001)
    )(layers)

    _, h, w, c = projection.shape
    projected_patches = ReshapeLayer((-1, h * w, c))(projection)

    encoded_patches = PatchEmbedding(IMAGE_SIZE=image_size,PATCH_SIZE=2,PROJECTION_DIM=c)(projected_patches)

    layers2 = SwinTBlock(IMAGE_SIZE=image_size, PATCH_SIZE=2, PROJECTION_DIM=64, depth=3, NUM_HEADS=2, NUM_MLP=128, WINDOW_SIZE=4, DROPOUT_RATE=0.1, LAYER_NORM_EPS=1e-5)(encoded_patches)
    layers = PatchMerging(IMAGE_SIZE=image_size,PATCH_SIZE=2,PROJECTION_DIM=64)(layers2)

    layers3 = SwinTBlock(IMAGE_SIZE=image_size, PATCH_SIZE=4, PROJECTION_DIM=128, depth=2, NUM_HEADS=4, NUM_MLP=256, WINDOW_SIZE=4, DROPOUT_RATE=0.1, LAYER_NORM_EPS=1e-5)(layers)
    layers = PatchMerging(IMAGE_SIZE=image_size,PATCH_SIZE=4,PROJECTION_DIM=128)(layers3)

    layers4 = SwinTBlock(IMAGE_SIZE=image_size, PATCH_SIZE=8, PROJECTION_DIM=256, depth=2, NUM_HEADS=8, NUM_MLP=512, WINDOW_SIZE=8, DROPOUT_RATE=0.1, LAYER_NORM_EPS=1e-5)(layers)
    layers = PatchMerging(IMAGE_SIZE=image_size, PATCH_SIZE=8, PROJECTION_DIM=256)(layers4)


    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    layers = tf.keras.layers.GlobalAvgPool1D()(layers)

    layers = tf.keras.layers.Dense(128,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Dense(64,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Dense(32,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)

    predictions = tf.keras.layers.Dense(2, activation="softmax", name="output_1", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    model = tf.keras.Model(inputs=input_shape, outputs=predictions)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=5e-3, momentum=0.9)
        ,loss='categorical_crossentropy', metrics=['accuracy'])

# ======================
# Callbacks y entrenamiento
# ======================
start_time = time.time()
history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
)
end_time = time.time()

# ======================
# Guardar modelo completo
# ======================
model.save("modelo_entrenado.h5")

# ======================
# Tiempo total
# ======================
total_time = end_time - start_time
print(f"\n Tiempo total de entrenamiento: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")

# ======================
# Gráfica entrenamiento
# ======================
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.title("Precisión durante el entrenamiento")
plt.ylabel("Precisión")
plt.xlabel("Época")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(history.history["loss"], label="Entrenamiento")
plt.plot(history.history["val_loss"], label="Validación")
plt.title("Pérdida durante el entrenamiento")
plt.ylabel("Pérdida")
plt.xlabel("Época")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
print(" Gráfica de entrenamiento guardada como 'training_plot.png'")
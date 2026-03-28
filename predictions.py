import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess
import urllib.request

# ---------------- DOWNLOAD MODEL SAFELY ----------------
MODEL_URL = "https://huggingface.co/udaychowdhary/skin-disease-model/resolve/main/CBAM_ResNet50_model_final.h5"
MODEL_PATH = "res.h5"

def download_model():
    try:
        if not os.path.exists(MODEL_PATH):
            print("⬇️ Downloading model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("✅ Download complete")

        size = os.path.getsize(MODEL_PATH)
        if size < 100_000_000:
            print("❌ Corrupted model detected. Re-downloading...")
            os.remove(MODEL_PATH)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    except Exception as e:
        print("❌ Download error:", e)
        raise RuntimeError("Model download failed")

download_model()

# ---------------- CLASS NAMES ----------------
classes = [
    "Actinic Keratosis",
    "Dermatofibroma",
    "Nevus",
    "Pigmented Benign Keratosis",
    "Seborrheic Keratosis",
    "Vascular Lesion"
]
num_classes = len(classes)

# ---------------- CBAM ----------------
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_dense_1 = layers.Dense(channel // ratio, activation='relu')
    shared_dense_2 = layers.Dense(channel)

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_2(shared_dense_1(avg_pool))

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_2(shared_dense_1(max_pool))

    attention = layers.Activation('sigmoid')(avg_pool + max_pool)
    return layers.Multiply()([input_feature, attention])


def spatial_attention(input_feature):
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)

    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    attention = layers.Conv2D(
        filters=1,
        kernel_size=7,
        padding='same',
        activation='sigmoid'
    )(concat)

    return layers.Multiply()([input_feature, attention])


def cbam_block(x):
    x = channel_attention(x)
    x = spatial_attention(x)
    return x


# ---------------- MODEL ----------------
from tensorflow.keras.applications import ResNet50

def build_resnet_cbam(num_classes):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = cbam_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(num_classes, activation="softmax")(x)

    return Model(base_model.input, output)


# ---------------- LOAD MODEL ----------------
print("🧠 Loading model...")
resnet_model = build_resnet_cbam(num_classes)
resnet_model.load_weights(MODEL_PATH)
print("✅ Model loaded successfully")


# ---------------- PREPROCESS ----------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Invalid image path")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32")

    img = res_preprocess(img)
    img = np.expand_dims(img, axis=0)

    return img


# ---------------- PREDICTION ----------------
def predict_skin_disease(img_path):
    img = preprocess_image(img_path)

    preds = resnet_model.predict(img, verbose=0)

    predicted_class = int(np.argmax(preds))
    confidence = float(np.max(preds) * 100)

    return predicted_class, round(confidence, 2)
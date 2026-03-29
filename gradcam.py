import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


def generate_mobilenetv3_gradcam(img_path, model, output_path):

    # -------------------------------------------------
    # 1. Load image
    # -------------------------------------------------
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Invalid image path")

    original_img = img.copy()

    # -------------------------------------------------
    # 2. Illumination correction (CLAHE)
    # -------------------------------------------------
    lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    lab = cv2.merge((l, a, b))
    original_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # -------------------------------------------------
    # 3. Preprocess
    # -------------------------------------------------
    img = cv2.resize(original_img, (224, 224))
    img = preprocess_input(img.astype("float32"))
    img_array = np.expand_dims(img, axis=0)

    # -------------------------------------------------
    # 4. 🔥 AUTO LAYER SELECTION (FIX FOR RENDER)
    # -------------------------------------------------
    layers = []

    for layer in reversed(model.layers):
        if "project" in layer.name:
            layers.append(layer.name)
        if len(layers) == 3:
            break

    layers = layers[::-1]  # maintain order

    print("Using GradCAM layers:", layers)

    heatmaps = []
    target_size = (224, 224)

    # -------------------------------------------------
    # 5. Grad-CAM per layer
    # -------------------------------------------------
    for layer_name in layers:

        grad_model = Model(
            inputs=model.input,
            outputs=[
                model.get_layer(layer_name).output,
                model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        # Normalize
        heatmap = tf.nn.relu(heatmap)
        heatmap /= tf.reduce_max(heatmap) + 1e-8
        heatmap = heatmap.numpy()

        # Resize
        heatmap = cv2.resize(heatmap, target_size)

        heatmaps.append(heatmap)

    # -------------------------------------------------
    # 6. Combine heatmaps
    # -------------------------------------------------
    heatmap = np.max(np.array(heatmaps), axis=0)

    # -------------------------------------------------
    # 7. Remove noise
    # -------------------------------------------------
    heatmap[heatmap < 0.25] = 0
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    heatmap = np.power(heatmap, 0.9)

    # -------------------------------------------------
    # 8. Resize to original
    # -------------------------------------------------
    heatmap = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0])
    )

    # -------------------------------------------------
    # 9. Lesion mask
    # -------------------------------------------------
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = cv2.bitwise_not(mask)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    mask = mask / 255.0

    heatmap = heatmap * mask

    # -------------------------------------------------
    # 10. Fill internal gaps
    # -------------------------------------------------
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    heatmap[mask > 0.5] = np.maximum(
        heatmap[mask > 0.5],
        0.3 * mask[mask > 0.5]
    )

    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)

    # -------------------------------------------------
    # 11. Convert to color
    # -------------------------------------------------
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # -------------------------------------------------
    # 12. Overlay
    # -------------------------------------------------
    result = cv2.addWeighted(original_img, 0.7, heatmap_color, 0.3, 0)

    # -------------------------------------------------
    # 13. Save output
    # -------------------------------------------------
    cv2.imwrite(output_path, result)

    return output_path
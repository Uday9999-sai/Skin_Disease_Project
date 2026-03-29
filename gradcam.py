import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


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
    original_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # -------------------------------------------------
    # 3. Preprocess
    # -------------------------------------------------
    img_resized = cv2.resize(original_img, (224, 224))
    img_array = preprocess_input(img_resized.astype(np.float32))[None, ...]

    # -------------------------------------------------
    # 4. Multi-layer selection (SAME LOGIC)
    # -------------------------------------------------
    layers = []
    for layer in reversed(model.layers):
        if "project" in layer.name:
            layers.append(layer.name)
        if len(layers) == 3:
            break
    layers = layers[::-1]

    heatmaps = []
    target_size = (224, 224)

    # -------------------------------------------------
    # 5. Grad-CAM per layer (SAFE VERSION)
    # -------------------------------------------------
    for layer_name in layers:

        try:
            grad_model = Model(
                model.input,
                [model.get_layer(layer_name).output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array, training=False)
                loss = predictions[:, tf.argmax(predictions[0])]

            grads = tape.gradient(loss, conv_outputs)

            # 🔥 SAFETY CHECK
            if grads is None:
                print(f"⚠️ Skipping layer {layer_name} (grads None)")
                continue

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]

            heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

            heatmap = tf.maximum(heatmap, 0)

            max_val = tf.reduce_max(heatmap)

            # 🔥 AVOID DIVIDE BY ZERO
            if max_val == 0:
                print(f"⚠️ Skipping layer {layer_name} (zero heatmap)")
                continue

            heatmap /= max_val
            heatmap = heatmap.numpy()

            # 🔥 REMOVE NaNs
            heatmap = np.nan_to_num(heatmap)

            heatmap = cv2.resize(heatmap, target_size)

            heatmaps.append(heatmap)

        except Exception as e:
            print(f"🚨 Layer {layer_name} failed:", str(e))
            continue

    # -------------------------------------------------
    # 6. Combine heatmaps (SAFE)
    # -------------------------------------------------
    if len(heatmaps) == 0:
        print("🚨 All layers failed → fallback")
        cv2.imwrite(output_path, original_img)
        return output_path

    heatmap = np.max(np.stack(heatmaps, axis=0), axis=0)

    # -------------------------------------------------
    # 7. Noise removal
    # -------------------------------------------------
    heatmap = np.where(heatmap < 0.25, 0, heatmap)
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    heatmap = np.power(heatmap, 0.9)

    # -------------------------------------------------
    # 8. Resize to original
    # -------------------------------------------------
    h, w = original_img.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))

    # -------------------------------------------------
    # 9. Lesion mask
    # -------------------------------------------------
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.bitwise_not(mask)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    mask = mask.astype(np.float32) / 255.0

    heatmap *= mask

    # -------------------------------------------------
    # 10. Fill gaps
    # -------------------------------------------------
    heatmap /= (np.max(heatmap) + 1e-8)

    heatmap = np.where(mask > 0.5,
                       np.maximum(heatmap, 0.3 * mask),
                       heatmap)

    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)

    # 🔥 FINAL SAFETY
    heatmap = np.nan_to_num(heatmap)

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
    # 13. Save
    # -------------------------------------------------
    cv2.imwrite(output_path, result)

    return output_path
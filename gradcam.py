import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


def generate_gradcam(img_path, model, output_path):

    # -------------------------------------------------
    # 1. Load image
    # -------------------------------------------------
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Invalid image path")

    original_img = img.copy()

    # -------------------------------------------------
    # 2. Illumination correction (keep this 👍)
    # -------------------------------------------------
    lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    lab = cv2.merge((l, a, b))
    original_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # -------------------------------------------------
    # 3. Preprocess (MobileNet)
    # -------------------------------------------------
    img = cv2.resize(original_img, (224, 224))
    img = img.astype("float32")
    img = preprocess_input(img)
    img_array = np.expand_dims(img, axis=0)

    # -------------------------------------------------
    # 4. Find last Conv layer (MobileNet safe)
    # -------------------------------------------------
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        raise ValueError("No Conv layer found for GradCAM")

    # -------------------------------------------------
    # 5. Create GradCAM model
    # -------------------------------------------------
    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    # -------------------------------------------------
    # 6. GradCAM++ computation
    # -------------------------------------------------
    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, feature_maps)

    feature_maps = feature_maps[0]
    grads = grads[0]

    grads_power_2 = grads ** 2
    grads_power_3 = grads ** 3

    sum_grads = tf.reduce_sum(feature_maps * grads_power_3, axis=(0, 1))

    eps = 1e-8
    alpha = grads_power_2 / (2 * grads_power_2 + sum_grads + eps)

    positive_gradients = tf.nn.relu(grads)
    weights = tf.reduce_sum(alpha * positive_gradients, axis=(0, 1))

    heatmap = tf.reduce_sum(weights * feature_maps, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()

    # -------------------------------------------------
    # 7. Remove weak activations (tune if needed)
    # -------------------------------------------------
    heatmap[heatmap < 0.3] = 0

    # -------------------------------------------------
    # 8. Resize to original
    # -------------------------------------------------
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # -------------------------------------------------
    # 9. Apply colormap
    # -------------------------------------------------
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # -------------------------------------------------
    # 10. Overlay
    # -------------------------------------------------
    superimposed_img = cv2.addWeighted(original_img, 0.7, heatmap_color, 0.3, 0)

    # -------------------------------------------------
    # 11. Save
    # -------------------------------------------------
    cv2.imwrite(output_path, superimposed_img)

    return output_path
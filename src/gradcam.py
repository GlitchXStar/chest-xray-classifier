import tensorflow as tf
import numpy as np
import cv2

from src.config import IMG_SIZE


# ============================================================
# GRAD-CAM MODEL BUILDER
# ============================================================

def build_grad_model(model):
    """
    Finds the DenseNet sub-model inside the wrapper model,
    auto-detects the last Conv2D layer, and builds both the
    sub_densenet model and a standalone head_model with cloned weights.

    Returns: (sub_densenet, head_model) or (None, None) on failure.
    """
    densenet = None
    for layer in model.layers:
        if 'densenet' in layer.name.lower():
            densenet = layer
            print(f"[GradCAM] Found DenseNet sub-model: '{layer.name}'")
            break

    if densenet is None:
        print("[GradCAM] WARNING: No DenseNet sub-model found. Grad-CAM disabled.")
        return None, None

    last_conv = None
    for layer in reversed(densenet.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            print(f"[GradCAM] Auto-detected last conv layer: '{last_conv}'")
            break

    if last_conv is None:
        print("[GradCAM] WARNING: No Conv2D layer found in DenseNet. Grad-CAM disabled.")
        return None, None

    try:
        sub_densenet = tf.keras.Model(
            inputs=densenet.input,
            outputs=[
                densenet.get_layer(last_conv).output,
                densenet.output
            ]
        )

        feat_shape = densenet.output_shape[1:]
        head_input = tf.keras.Input(shape=feat_shape, name='head_input')
        x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(head_input)
        x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
        x = tf.keras.layers.Dropout(0.5, name='drop1')(x)
        x = tf.keras.layers.Dense(512, activation='relu', name='fc1')(x)
        x = tf.keras.layers.BatchNormalization(name='fc_bn')(x)
        x = tf.keras.layers.Dropout(0.3, name='drop2')(x)
        preds = tf.keras.layers.Dense(14, activation='sigmoid', name='predictions')(x)

        head_model = tf.keras.Model(inputs=head_input, outputs=preds)
        for l in head_model.layers:
            if l.name in ['gap', 'head_bn', 'drop1', 'fc1', 'fc_bn', 'drop2', 'predictions']:
                l.set_weights(model.get_layer(l.name).get_weights())

        print(f"[GradCAM] Grad-CAM model components built successfully.")
        return sub_densenet, head_model
    except Exception as e:
        print(f"[GradCAM] WARNING: Could not build grad_model components: {e}")
        return None, None


# ============================================================
# HEAD LAYER VALIDATOR
# ============================================================

def get_head_layers(model):
    """
    Checks that all expected head layer names exist in the model.
    Returns a dict of {name: name} for found layers.
    """
    expected = ['gap', 'head_bn', 'drop1', 'fc1', 'fc_bn', 'drop2', 'predictions']
    found = {}

    for name in expected:
        try:
            model.get_layer(name)
            found[name] = name
        except Exception:
            pass

    return found


# ============================================================
# GRAD-CAM COMPUTATION
# ============================================================

def get_gradcam(sub_densenet, head_model, img_tensor, class_idx):
    """
    Computes the Grad-CAM heatmap for a given class index.

    Args:
        sub_densenet: Sub-model outputting (conv_features, densenet_output).
        head_model:   Standalone classification head model.
        img_tensor:   Preprocessed image tensor of shape (1, H, W, 3).
        class_idx:    Index of the class to visualize.

    Returns:
        heatmap: np.ndarray of shape (IMG_SIZE, IMG_SIZE), float32, range [0, 1].
                 Returns None if computation fails.
    """
    if sub_densenet is None or head_model is None:
        print("[GradCAM] Skipped: sub_densenet or head_model not available.")
        return None

    try:
        with tf.GradientTape() as tape:
            conv_features, base_features = sub_densenet(img_tensor, training=False)
            tape.watch(conv_features)
            predictions = head_model(base_features, training=False)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_features)

        if grads is None:
            print("[GradCAM] Gradients are None — graph not connected. Skipping.")
            return None

        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_np = conv_features[0].numpy()
        pool_np = pooled.numpy()

        heatmap = np.dot(conv_np, pool_np)
        heatmap = np.maximum(heatmap, 0)

        vmax = heatmap.max()
        if vmax < 1e-8:
            heatmap = np.ones_like(heatmap) * 0.5
        else:
            heatmap /= vmax

        heatmap = tf.image.resize(
            heatmap[..., np.newaxis],
            [IMG_SIZE, IMG_SIZE]
        ).numpy().squeeze()

        return heatmap.astype(np.float32)

    except Exception as e:
        print(f"[GradCAM] Error during computation: {e}")
        return None


# ============================================================
# OVERLAY
# ============================================================

def overlay_gradcam(image, heatmap, alpha=0.4):
    """
    Blends the original image with the Grad-CAM heatmap.

    Args:
        image:   Original image as np.ndarray (H, W, 3), uint8.
        heatmap: Normalized heatmap from get_gradcam(), float32 [0, 1].
        alpha:   Heatmap blend weight (default 0.4).

    Returns:
        overlay: np.ndarray (IMG_SIZE, IMG_SIZE, 3), uint8.
    """
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    img_display = tf.image.resize(
        image, [IMG_SIZE, IMG_SIZE]
    ).numpy().astype(np.uint8)

    overlay = cv2.addWeighted(
        img_display,    1 - alpha,
        heatmap_colored, alpha,
        0
    )
    return overlay
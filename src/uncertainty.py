import tensorflow as tf
import numpy as np
import cv2
from src.config import IMG_SIZE, MC_DROPOUT_PASSES

def perform_mc_dropout(sub_densenet, head_model, img_tensor, class_idx, passes=MC_DROPOUT_PASSES):
    """
    Performs Monte Carlo Dropout inference by running multiple stochastic forward passes.
    Returns:
       mc_cams: list of normalized Grad-CAM heatmaps for the selected class
       mc_probs: list of probabilities for the selected class
    """
    mc_cams = []
    mc_probs = []

    for _ in range(passes):
        with tf.GradientTape() as tape:
            # DenseNet remains completely deterministic
            base_features = sub_densenet(img_tensor, training=False)
            tape.watch(base_features)
            
            # Explicitly execute head_model layers to ensure BatchNormalization remains frozen (training=False)
            # while making sure Dropout layers are stochastic (training=True)
            x = head_model.get_layer('gap')(base_features)
            x = head_model.get_layer('head_bn')(x, training=False)
            x = head_model.get_layer('drop1')(x, training=True) # STOCHASTIC
            x = head_model.get_layer('fc1')(x)
            x = head_model.get_layer('fc_bn')(x, training=False)
            x = head_model.get_layer('drop2')(x, training=True) # STOCHASTIC
            predictions = head_model.get_layer('predictions')(x)
            
            loss = predictions[:, class_idx]
            
        prob = predictions[0, class_idx].numpy()
        mc_probs.append(prob)

        # Compute Grad-CAM for this pass
        grads = tape.gradient(loss, base_features)
        if grads is None:
            # Fallback if disconnected
            heatmap = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32) * 0.5
        else:
            pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_np = base_features[0].numpy()
            pool_np = pooled.numpy()

            heatmap = np.dot(conv_np, pool_np)
            heatmap = np.maximum(heatmap, 0)
            vmax = heatmap.max()
            if not np.isfinite(vmax) or vmax < 1e-8:
                heatmap = np.ones_like(heatmap) * 0.5
            else:
                heatmap /= vmax

            heatmap = tf.image.resize(
                heatmap[..., np.newaxis],
                [IMG_SIZE, IMG_SIZE]
            ).numpy().squeeze()

        mc_cams.append(heatmap.astype(np.float32))

    return mc_cams, mc_probs

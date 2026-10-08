import os
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

from src.config import (
    CLASSES, THRESHOLDS, IMG_SIZE, 
    CONFIDENCE_THRESHOLD, RELIABILITY_THRESHOLD,
    RELIABILITY_WEIGHT_ENTROPY, RELIABILITY_WEIGHT_VARIANCE
)
from src.losses import AsymmetricLoss
from src.uncertainty import perform_mc_dropout
from src.reliability import (
    calculate_expected_saliency,
    calculate_saliency_entropy,
    calculate_saliency_variance,
    calculate_reliability_score,
    make_abstention_decision
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.set_visible_devices([], 'GPU')

print("[SYSTEM] Loading model...")
MODEL_PATH = 'outputs/weights/best_p2.keras'

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'AsymmetricLoss': AsymmetricLoss},
    compile=False
)

print(f"\n✓ Loaded model: {MODEL_PATH}")
print(f"✓ Input shape: {model.input_shape}")
print(f"✓ Output shape: {model.output_shape}")

# ============================================================
# GRAD-CAM SETUP
# ============================================================
densenet = None
for layer in model.layers:
    if 'densenet' in layer.name.lower():
        densenet = layer
        break

sub_densenet = None
head_model = None

if densenet is not None:
    try:
        sub_densenet = densenet
        dn_out = densenet.output[0] if isinstance(densenet.output, list) else densenet.output
        feat_shape = dn_out.shape[1:]
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
    except Exception as e:
        print(f"[SYSTEM] WARNING: Could not build Grad-CAM model parts: {e}")

# ============================================================
# INFERENCE LOGIC
# ============================================================

def preprocess_image(image):
    img = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.densenet.preprocess_input(img)
    return tf.expand_dims(img, 0)

def predict_base(image):
    print("\n[REQUEST] POST /api/predict")
    if image is None:
        return {"Error": 1.0}, gr.update(choices=[], value=None)

    img_tensor = preprocess_image(image)
    preds = model(img_tensor, training=False).numpy()[0]

    results = {}

    for i, cls in enumerate(CLASSES):
        prob = float(preds[i])
        results[cls] = prob

    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
    # Just standard ordered classes for dropdown
    ordered_classes = [k.replace('✅ ', '') for k in results.keys()]
    
    print("[PREDICTION] Generated initial inference predictions.")
    return results, gr.update(choices=ordered_classes, value=ordered_classes[0])

def colorize_map(heatmap):
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

def explain_disease(image, disease_name):
    print(f"\n[REQUEST] POST /api/analyze - Disease: {disease_name}")
    if image is None or not disease_name:
        return (None, None, None, "", "", "", "", "ABSTAIN", "No image or disease selected.")

    if sub_densenet is None or head_model is None:
        return (None, None, None, "", "", "", "", "ABSTAIN", "Grad-CAM model components missing.")
        
    class_idx = CLASSES.index(disease_name)
    print(f"[GRADCAM] Target Disease: {disease_name} (Index: {class_idx})")
    img_tensor = preprocess_image(image)
    
    print(f"[EXPLAIN] Running MC Dropout for saliency generation.")
    mc_cams, mc_probs = perform_mc_dropout(sub_densenet, head_model, img_tensor, class_idx)
    
    # We use empirical mean of MC passes for expected prob
    expected_prob = float(np.mean(mc_probs))
    
    # Saliency Processing
    expected_cam = calculate_expected_saliency(mc_cams)
    entropy = calculate_saliency_entropy(expected_cam)
    variance, var_map = calculate_saliency_variance(mc_cams)
    
    reliability = calculate_reliability_score(
        entropy, variance, 
        w_entropy=RELIABILITY_WEIGHT_ENTROPY, 
        w_variance=RELIABILITY_WEIGHT_VARIANCE
    )
    
    decision, reason = make_abstention_decision(
        expected_prob, reliability, 
        CONFIDENCE_THRESHOLD, RELIABILITY_THRESHOLD
    )
    
    print(f"[RELIABILITY] Saliency entropy: {entropy:.4f}, Saliency variance: {variance:.4f}")
    print(f"[RELIABILITY] Score: {reliability:.4f}")
    print(f"[EXPLAIN] Decision: {decision}")

    # UI Asset gen
    img_display = tf.image.resize(image, [IMG_SIZE, IMG_SIZE]).numpy().astype(np.uint8)
    
    overlay_cam = cv2.addWeighted(img_display, 0.6, colorize_map(expected_cam), 0.4, 0)
    
    # Normalize var_map visually for UI just to show spatial uncertainty highlights
    vmax = var_map.max()
    vis_var_map = var_map / vmax if vmax > 1e-8 else var_map
    overlay_var = cv2.addWeighted(img_display, 0.6, colorize_map(vis_var_map), 0.4, 0)
    
    res_conf = f"{expected_prob:.2%}"
    res_ent = f"{entropy:.4f}"
    res_var = f"{variance:.4f}"
    res_rel = f"{reliability:.2%}"
    
    dec_visual = f"🟢 ACCEPT" if decision == "ACCEPT" else f"🔴 ABSTAIN"
    
    return (
        img_display, 
        overlay_cam, 
        overlay_var,
        res_conf, res_ent, res_var, res_rel,
        dec_visual, reason
    )

# ============================================================
# UI
# ============================================================
with gr.Blocks(title="Explainable Chest X-Ray Analysis") as demo:
    gr.Markdown("# 🫁 Explainable Multi-Label Chest X-Ray Disease Detection")
    gr.Markdown("**Research Prototype:** Combining Bayesian Saliency Entropy and Reliability for Explanation-Aware Abstention.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Chest X-Ray", type="numpy")
            analyze_btn = gr.Button("1. Fast Analyze", variant="primary")
        with gr.Column():
            label_output = gr.Label(label="Disease Predictions (✅ = above threshold)", num_top_classes=14)
            
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            disease_dropdown = gr.Dropdown(choices=CLASSES, label="Target Disease")
            explain_btn = gr.Button("2. Explain & Evaluate Reliability", variant="secondary")
    
    with gr.Row():
        orig_img_out = gr.Image(label="Original X-Ray")
        expected_cam_out = gr.Image(label="Expected Grad-CAM")
        uncertainty_map_out = gr.Image(label="Saliency Uncertainty Map")
        
    with gr.Row():
        conf_out = gr.Textbox(label="Prediction Confidence")
        rel_out = gr.Textbox(label="Explanation Reliability")
        entropy_out = gr.Textbox(label="Saliency Entropy")
        var_out = gr.Textbox(label="Saliency Variance")
        
    with gr.Row():
        decision_out = gr.Textbox(label="Decision")
        reason_out = gr.Textbox(label="Reason")
        
    analyze_btn.click(
        fn=predict_base,
        inputs=image_input,
        outputs=[label_output, disease_dropdown]
    )
    
    explain_btn.click(
        fn=explain_disease,
        inputs=[image_input, disease_dropdown],
        outputs=[
            orig_img_out, expected_cam_out, uncertainty_map_out,
            conf_out, entropy_out, var_out, rel_out,
            decision_out, reason_out
        ]
    )
    
    gr.Markdown("---")
    gr.Markdown("⚠️ **For research purposes only. Not a certified medical device. This system is not intended for clinical diagnosis or treatment.**")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865, share=False)
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import csv
import numpy as np
import tensorflow as tf

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

def main():
    print("=" * 60)
    print("RELIABILITY & UNCERTAINTY EVALUATION")
    print("=" * 60)
    
    # 1. Load Model
    print("\n[1] Loading Model...")
    try:
        model = tf.keras.models.load_model(
            'outputs/weights/best_model_phaseC.keras',
            custom_objects={'AsymmetricLoss': AsymmetricLoss},
            compile=False
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Extract sub-models for Grad-CAM
    densenet = next((l for l in model.layers if 'densenet' in l.name.lower()), None)
    if not densenet:
        print("DenseNet not found.")
        return
        
    LAST_CONV = next((l.name for l in reversed(densenet.layers) if isinstance(l, tf.keras.layers.Conv2D)), None)
    
    dn_out = densenet.output[0] if isinstance(densenet.output, list) else densenet.output
    lc_out = densenet.get_layer(LAST_CONV).output
    lc_out = lc_out[0] if isinstance(lc_out, list) else lc_out
    
    sub_densenet = tf.keras.Model(
        inputs=densenet.input,
        outputs=[lc_out, dn_out]
    )
    
    # Reconstruct custom head
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
            
    print("[1] Model decomposition complete.")

    # 2. Gather Test Images
    img_paths = glob.glob('assets/samples/*.*')
    if not img_paths:
        print("No sample images found in assets/samples/")
        # Try finding some image elsewhere just for testing
        return

    results_data = []

    # 3. Evaluate each image
    print(f"\n[2] Evaluating {len(img_paths)} samples...")
    for idx, path in enumerate(img_paths):
        filename = os.path.basename(path)
        
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.densenet.preprocess_input(img)
        img_tensor = tf.expand_dims(img, 0)
        
        # Standard predict
        base_preds = model(img_tensor, training=False).numpy()[0]
        top_idx = int(np.argmax(base_preds))
        top_class = CLASSES[top_idx]
        
        # MC Dropout logic
        mc_cams, mc_probs = perform_mc_dropout(sub_densenet, head_model, img_tensor, top_idx)
        
        expected_prob = float(np.mean(mc_probs))
        expected_cam = calculate_expected_saliency(mc_cams)
        entropy = calculate_saliency_entropy(expected_cam)
        variance, _ = calculate_saliency_variance(mc_cams)
        
        reliability = calculate_reliability_score(
            entropy, variance, 
            w_entropy=RELIABILITY_WEIGHT_ENTROPY, 
            w_variance=RELIABILITY_WEIGHT_VARIANCE
        )
        
        decision, reason = make_abstention_decision(
            expected_prob, reliability, 
            CONFIDENCE_THRESHOLD, RELIABILITY_THRESHOLD
        )
        
        results_data.append({
            'filename': filename,
            'top_class': top_class,
            'confidence': expected_prob,
            'entropy': entropy,
            'variance': variance,
            'reliability_score': reliability,
            'decision': decision,
            'reason': reason
        })
        
        print(f" -> {filename} | {top_class} | Conf: {expected_prob:.2f} | Rel: {reliability:.2f} | {decision}")

    # 4. Save to CSV
    csv_path = 'outputs/reliability_evaluation.csv'
    os.makedirs('outputs', exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'top_class', 'confidence', 'entropy', 'variance', 'reliability_score', 'decision', 'reason']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results_data:
            writer.writerow(row)
            
    print(f"\n[3] Saved evaluation results to {csv_path}")
    
    # 5. Risk-Coverage Summary
    total = len(results_data)
    accepted = sum(1 for r in results_data if r['decision'] == 'ACCEPT')
    abstained = total - accepted
    coverage = accepted / total if total > 0 else 0
    
    print("\n" + "=" * 40)
    print("RISK-COVERAGE SUMMARY")
    print("=" * 40)
    print(f"Total Samples Tested: {total}")
    print(f"Accepted Predictions (Coverage): {accepted} ({coverage:.1%})")
    print(f"Abstained Predictions: {abstained} ({(1-coverage):.1%})")
    print("=" * 40)

if __name__ == "__main__":
    main()

import numpy as np

def calculate_expected_saliency(mc_cams):
    """Calculates the empirical mean across stochastic Grad-CAM passes."""
    cams = np.stack(mc_cams, axis=0) # Shape: (passes, H, W)
    mean_cam = np.mean(cams, axis=0)
    
    # Normalize to [0,1]
    vmax = mean_cam.max()
    if not np.isfinite(vmax) or vmax < 1e-8:
        mean_cam = np.ones_like(mean_cam) * 0.5
    else:
        mean_cam /= vmax
    return mean_cam

def calculate_saliency_entropy(expected_cam):
    """
    Calculates the normalized Shannon entropy of the expected saliency map.
    Interprets the map as a spatial 2D probability distribution.
    """
    epsilon = 1e-12
    # Flatten array
    flat_cam = expected_cam.flatten()
    
    # Normalize to sum = 1 to form a valid prob distribution
    total_sum = np.sum(flat_cam)
    if total_sum > epsilon:
        p_i = flat_cam / total_sum
    else:
        # Uniform distribution if map is empty
        p_i = np.ones_like(flat_cam) / len(flat_cam)
        
    p_i = np.clip(p_i, epsilon, 1.0)
    entropy = -np.sum(p_i * np.log(p_i))
    
    # Max possible entropy = log(N)
    max_entropy = np.log(len(flat_cam))
    normalized_entropy = entropy / max_entropy
    
    return float(np.clip(normalized_entropy, 0.0, 1.0))

def calculate_saliency_variance(mc_cams):
    """
    Calculates the point-wise variance map and scalar mean variance.
    Max theoretically possible variance for [0, 1] values is 0.25.
    Returns:
       normalized_mean_var: scalar in [0,1]
       variance_map: np.ndarray shape (H,W)
    """
    cams = np.stack(mc_cams, axis=0)
    variance_map = np.nanvar(cams, axis=0) # Sample variance across dimension 0
    
    mean_variance = np.nanmean(variance_map)
    if np.isnan(mean_variance) or not np.isfinite(mean_variance):
        mean_variance = 0.0
        variance_map = np.zeros_like(variance_map)
    
    # Normalize with theoretical max of p*(1-p) for range [0,1], which is 0.25
    normalized_mean_var = mean_variance / 0.25 
    
    return float(np.clip(normalized_mean_var, 0.0, 1.0)), variance_map

def calculate_reliability_score(entropy, variance, w_entropy=0.5, w_variance=0.5):
    """
    Combines spatial entropy and variance into a single Explanation Reliability Score.
    """
    if np.isnan(entropy) or np.isnan(variance):
        return 0.0
    score = (w_entropy * (1.0 - entropy)) + (w_variance * (1.0 - variance))
    return float(np.clip(score, 0.0, 1.0))

def make_abstention_decision(confidence, reliability, conf_thresh, rel_thresh):
    """
    Gate logic for ACCEPT / ABSTAIN behavior.
    Returns: (decision_str, reason_str)
    """
    if confidence < conf_thresh:
        return "ABSTAIN", f"Low prediction confidence ({confidence:.2f} < {conf_thresh:.2f})."
    
    if reliability < rel_thresh:
        return "ABSTAIN", f"High prediction confidence, but unreliable spatial explanation (Reliability {reliability:.2f} < {rel_thresh:.2f})."
        
    return "ACCEPT", "Prediction and explanation meet reliability criteria."

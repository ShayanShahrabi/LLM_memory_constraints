"""
Utility functions for sequence generation and attention analysis
"""

import random
import numpy as np
import torch
from config import FOCUS_WINDOW, EPS

def generate_sequence(length, task_type="random_letters"):
    """Generate test sequences"""
    if task_type == "random_letters":
        seq = [chr(65 + random.randint(0, 25)) for _ in range(length)]
        return " ".join(seq), {"task_type": "random_letters"}

def robust_attention_entropy(attn_weights):
    """Robust entropy calculation that handles all cases"""
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()

    # Handle different shapes
    if attn_weights.ndim > 1:
        attn_weights = attn_weights.flatten()

    # Remove any NaN or Inf values
    attn_weights = attn_weights[np.isfinite(attn_weights)]

    if len(attn_weights) == 0:
        return 0.0

    # Normalize to probability distribution
    total = np.sum(attn_weights)
    if total <= EPS:
        return 0.0

    probs = attn_weights / total

    # Remove zeros to avoid log(0)
    probs = probs[probs > EPS]

    if len(probs) == 0:
        return 0.0

    # Calculate entropy
    entropy = -np.sum(probs * np.log(probs + EPS))
    return entropy

def calculate_attention_focus(attn_weights, focus_window=FOCUS_WINDOW):
    """Calculate attention focus on recent tokens"""
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()

    seq_len = attn_weights.shape[-1]
    if seq_len <= focus_window:
        return 1.0

    # Average attention to recent tokens across all positions
    recent_attention = attn_weights[..., -focus_window:].mean()
    return recent_attention
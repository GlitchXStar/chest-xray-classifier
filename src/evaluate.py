import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os

from src.config import CLASSES, CHEXNET_AUCS


def get_predictions(model, dataset):
    all_preds  = []
    all_labels = []
    for images, labels in dataset:
        preds = model(images, training=False)
        all_preds.append(preds.numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)


def compute_auc_scores(all_labels, all_preds):
    auc_scores = {}
    for i, cls in enumerate(CLASSES):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        except Exception:
            auc = float('nan')
        auc_scores[cls] = auc
    return auc_scores


def print_auc_table(auc_scores):
    mean_auc = np.nanmean(list(auc_scores.values()))
    print("\n" + "=" * 50)
    print(f"{'Disease':<22} {'Your AUC':>10} {'CheXNet':>10}")
    print("=" * 50)
    for cls in CLASSES:
        print(f"{cls:<22} {auc_scores[cls]:>10.4f} {CHEXNET_AUCS[cls]:>10.4f}")
    print("=" * 50)
    print(f"{'Mean AUC':<22} {mean_auc:>10.4f} {'0.8414':>10}")
    print("=" * 50)
    return mean_auc


def plot_auc_comparison(auc_scores, save_dir):
    your_aucs    = [auc_scores[cls] for cls in CLASSES]
    chexnet_aucs = [CHEXNET_AUCS[cls] for cls in CLASSES]
    mean_auc     = np.nanmean(your_aucs)

    x     = np.arange(len(CLASSES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, your_aucs,    width, label='Your Model',
           color='steelblue',  alpha=0.85)
    ax.bar(x + width/2, chexnet_aucs, width, label='CheXNet',
           color='darkorange', alpha=0.85)
    ax.axhline(y=mean_auc, color='steelblue',  linestyle='--', alpha=0.5,
               label=f'Your Mean: {mean_auc:.4f}')
    ax.axhline(y=0.8414,   color='darkorange', linestyle='--', alpha=0.5,
               label='CheXNet Mean: 0.8414')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_title('Per-Class AUC: Your Model vs CheXNet', fontsize=14)
    ax.set_ylabel('AUC Score')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/auc_comparison.png', dpi=150)
    plt.show()
    print("AUC comparison chart saved!")


def plot_roc_curves(all_labels, all_preds, auc_scores, save_dir):
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_preds[:, i])
        axes[i].plot(fpr, tpr, color='steelblue', lw=2,
                     label=f'AUC = {auc_scores[cls]:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[i].set_title(cls, fontsize=11)
        axes[i].set_xlabel('FPR', fontsize=9)
        axes[i].set_ylabel('TPR', fontsize=9)
        axes[i].legend(fontsize=9)
        axes[i].grid(alpha=0.3)

    axes[14].set_visible(False)
    plt.suptitle('ROC Curves — All 14 Diseases', fontsize=16, y=1.02)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("ROC curves saved!")


def plot_training_history(history_a, history_b, history_c, save_dir):
    """
    Plots combined training history across all 3 phases (A, B, C).
    Each history is a dict with keys: train_loss, val_loss, val_auc.
    """
    def chain(key):
        return history_a[key] + history_b[key] + history_c[key]

    all_train_loss = chain('train_loss')
    all_val_loss   = chain('val_loss')
    all_val_auc    = chain('val_auc')
    total_epochs   = len(all_train_loss)
    all_epochs     = list(range(1, total_epochs + 1))

    split_a = len(history_a['train_loss'])
    split_b = split_a + len(history_b['train_loss'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(all_epochs, all_train_loss, 'b-o', label='Train Loss', markersize=4)
    axes[0].plot(all_epochs, all_val_loss,   'r-o', label='Val Loss',   markersize=4)
    axes[0].axvline(x=split_a + 0.5, color='gray',   linestyle='--',
                    alpha=0.7, label='Phase A → B')
    axes[0].axvline(x=split_b + 0.5, color='purple', linestyle='--',
                    alpha=0.7, label='Phase B → C')
    axes[0].set_title('Training & Validation Loss', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(all_epochs, all_val_auc, 'g-o', label='Val AUC', markersize=4)
    axes[1].axvline(x=split_a + 0.5, color='gray',   linestyle='--',
                    alpha=0.7, label='Phase A → B')
    axes[1].axvline(x=split_b + 0.5, color='purple', linestyle='--',
                    alpha=0.7, label='Phase B → C')
    axes[1].axhline(y=max(all_val_auc), color='green', linestyle=':',
                    alpha=0.5, label=f'Best AUC: {max(all_val_auc):.4f}')
    axes[1].set_title('Validation AUC over Epochs', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(
        'Training History — DenseNet121 + ASL on NIH ChestX-ray14',
        fontsize=14
    )
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/training_history.png', dpi=150)
    plt.show()
    print("Training history saved!")
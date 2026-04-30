# utils/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def find_optimal_threshold(y_true, y_pred):
    """Simple threshold search for micro F1"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_t = 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (y_pred > t).astype(int), average='micro')
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def compute_per_class_metrics(y_true, y_pred_probs, labels):
    aucs = []
    for i, label in enumerate(labels):
        if y_true[:, i].sum() > 0:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
                aucs.append(auc)
            except:
                aucs.append(np.nan)
        else:
            aucs.append(np.nan)
    
    return dict(zip(labels, aucs))


def plot_per_class_auc(auc_dict, save_path=None):
    auc_df = pd.DataFrame(list(auc_dict.items()), columns=['Disease', 'AUC'])
    auc_df = auc_df.sort_values('AUC', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=auc_df, x='AUC', y='Disease', palette='viridis')
    plt.title('Per-Class AUC on Test Set')
    plt.xlabel('AUC')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()




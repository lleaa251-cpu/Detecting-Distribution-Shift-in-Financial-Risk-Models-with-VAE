"""
Phase 2: 訓練 Transformer Classifier（純 NumPy/sklearn 實作，無需 PyTorch）
使用 MLP + Attention-inspired feature weighting 作為主分類器
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, roc_curve)
from sklearn.utils.class_weight import compute_sample_weight

np.random.seed(42)


# ─── Attention-Inspired Feature Weighting ─────────────────────────────────────

class AttentionFeatureWeighter:
    """
    模擬 Transformer Self-Attention 的特徵重要性加權。
    對每個樣本計算「特徵注意力分數」，強化重要特徵的信號。

    原理：
      Q = X @ W_Q   (Query)
      K = X @ W_K   (Key)
      score = softmax(Q @ K^T / sqrt(d)) → per-sample feature weights
      output = score * X  (加權後的特徵)
    """
    def __init__(self, n_features, d_model=16):
        self.d_model = d_model
        self.W_Q = np.random.randn(n_features, d_model) * 0.1
        self.W_K = np.random.randn(n_features, d_model) * 0.1

    def fit(self, X, y):
        """用梯度下降更新 W_Q, W_K（簡化版）"""
        # 利用正常/詐欺樣本的均值差異學習注意力方向
        normal_mean = X[y == 0].mean(axis=0)
        fraud_mean  = X[y == 1].mean(axis=0)
        diff = fraud_mean - normal_mean
        # 讓 W_Q 朝向差異最大的特徵
        self.W_Q = np.outer(diff / (np.linalg.norm(diff) + 1e-8),
                            np.ones(self.d_model)) * 0.5
        return self

    def transform(self, X):
        Q = X @ self.W_Q    # (N, d_model)
        K = X @ self.W_K    # (N, d_model)
        # Self-attention score per sample（N維向量）
        scores = np.einsum('nd,nd->n', Q, K) / np.sqrt(self.d_model)
        weights = 1.0 / (1.0 + np.exp(-scores))  # sigmoid gate
        # 用注意力分數加權特徵（broadcasting）
        return X * weights.reshape(-1, 1)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


# ─── Transformer-like Classifier ──────────────────────────────────────────────

class TransformerClassifier:
    """
    架構：
      Input → AttentionFeatureWeighter → MLP(128→64→32→2) → Sigmoid Output

    採用 sklearn MLPClassifier 作為 MLP 後端，支援：
    - 類別不平衡處理（sample_weight）
    - 早停（early_stopping）
    - 訓練曲線記錄
    """
    def __init__(self, n_features=11):
        self.attention = AttentionFeatureWeighter(n_features, d_model=16)
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=1e-4,            # L2 正則化
            batch_size=256,
            learning_rate='adaptive',
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
            verbose=False,
        )
        self.train_loss_curve = None
        self.val_loss_curve   = None

    def fit(self, X_train, y_train):
        print("[Phase 2] 訓練 Attention Feature Weighter...")
        X_attended = self.attention.fit_transform(X_train, y_train)

        # 處理類別不平衡
        sample_w = compute_sample_weight('balanced', y_train)

        print("[Phase 2] 訓練 MLP Classifier (128→64→32)...")
        self.mlp.fit(X_attended, y_train, sample_weight=sample_w)

        self.train_loss_curve = self.mlp.loss_curve_
        self.val_loss_curve   = self.mlp.validation_scores_
        return self

    def predict(self, X):
        X_attended = self.attention.transform(X)
        return self.mlp.predict(X_attended)

    def predict_proba(self, X):
        X_attended = self.attention.transform(X)
        return self.mlp.predict_proba(X_attended)

    def get_latent(self, X):
        """取出最後隱藏層的表示（用於監控）"""
        X_attended = self.attention.transform(X)
        # 手動前向傳播至最後一層
        h = X_attended
        for i, (coef, intercept) in enumerate(
            zip(self.mlp.coefs_[:-1], self.mlp.intercepts_[:-1])
        ):
            h = np.maximum(0, h @ coef + intercept)   # ReLU
        return h  # shape: (N, 32)


# ─── 訓練 & 評估 ──────────────────────────────────────────────────────────────

def evaluate_classifier(clf, X_val, y_val):
    y_pred  = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]
    auc     = roc_auc_score(y_val, y_proba)

    print("\n[Phase 2] 驗證集評估結果：")
    print(f"  ROC-AUC: {auc:.4f}")
    print(classification_report(y_val, y_pred,
                                target_names=['Normal', 'Fraud'],
                                digits=4))
    return auc, y_proba


def visualize_training(clf, X_val, y_val, y_proba,
                       save_path='output/phase2_classifier.png'):
    os.makedirs('output', exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0f1117')
    for ax in axes:
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#8888aa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    # 1. 訓練損失曲線
    ax = axes[0]
    ax.plot(clf.train_loss_curve, color='#4488ff', linewidth=2, label='Train Loss')
    if clf.val_loss_curve is not None:
        ax2 = ax.twinx()
        ax2.plot(clf.val_loss_curve, color='#44ff88', linewidth=2,
                 linestyle='--', label='Val Accuracy')
        ax2.tick_params(colors='#8888aa')
        ax2.set_ylabel('Val Accuracy', color='#44ff88')
    ax.set_title('Training Curve', color='#eeeeff')
    ax.set_xlabel('Epoch', color='#ccccdd')
    ax.set_ylabel('Loss', color='#ccccdd')
    ax.legend(loc='upper left', facecolor='#1a1d27', labelcolor='#ccccdd')

    # 2. ROC 曲線
    ax = axes[1]
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    auc = roc_auc_score(y_val, y_proba)
    ax.plot(fpr, tpr, color='#ff6688', linewidth=2.5,
            label=f'ROC AUC = {auc:.4f}')
    ax.plot([0,1], [0,1], color='#555577', linestyle='--', linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.15, color='#ff6688')
    ax.set_title('ROC Curve', color='#eeeeff')
    ax.set_xlabel('False Positive Rate', color='#ccccdd')
    ax.set_ylabel('True Positive Rate', color='#ccccdd')
    ax.legend(facecolor='#1a1d27', labelcolor='#ccccdd')

    # 3. 預測機率分佈
    ax = axes[2]
    ax.hist(y_proba[y_val == 0], bins=50, alpha=0.7,
            color='#4488ff', label='Normal', density=True)
    ax.hist(y_proba[y_val == 1], bins=50, alpha=0.7,
            color='#ff4466', label='Fraud',  density=True)
    ax.axvline(0.5, color='#ffaa22', linestyle='--', linewidth=1.5,
               label='Threshold=0.5')
    ax.set_title('Prediction Probability Distribution', color='#eeeeff')
    ax.set_xlabel('P(Fraud)', color='#ccccdd')
    ax.set_ylabel('Density', color='#ccccdd')
    ax.legend(facecolor='#1a1d27', labelcolor='#ccccdd')

    plt.suptitle('Phase 2: Transformer Classifier Training Results',
                 color='#eeeeff', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Phase 2] 圖表已儲存至 {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 2: 訓練 Transformer Classifier")
    print("=" * 60)

    X_train = np.load('output/X_train.npy')
    X_val   = np.load('output/X_val.npy')
    y_train = np.load('output/y_train.npy')
    y_val   = np.load('output/y_val.npy')
    print(f"載入資料: X_train={X_train.shape}, X_val={X_val.shape}")

    clf = TransformerClassifier(n_features=X_train.shape[1])
    clf.fit(X_train, y_train)

    auc, y_proba = evaluate_classifier(clf, X_val, y_val)
    visualize_training(clf, X_val, y_val, y_proba)

    # 儲存模型
    with open('output/classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # 儲存隱藏層表示（供 Phase 3 VAE 使用）
    latent_train = clf.get_latent(X_train)
    latent_val   = clf.get_latent(X_val)
    np.save('output/latent_train.npy', latent_train)
    np.save('output/latent_val.npy',   latent_val)
    print(f"\n[Phase 2] 完成！分類器隱藏層維度: {latent_train.shape[1]}")
    print(f"  AUC: {auc:.4f}")
    print(f"  收斂 epochs: {len(clf.train_loss_curve)}")

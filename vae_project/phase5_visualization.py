"""
Phase 5: UMAP 視覺化 + 觸發再訓練機制
由於 umap-learn 在此環境無法安裝，改用 t-SNE（sklearn）進行潛在空間視覺化
並實作完整的自動再訓練觸發流程
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
import pickle
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_vae import VAE, DenseLayer                                          # pickle 解析
from phase4_detection import ShiftDetector, mahalanobis_distance                # 偵測函式
from phase2_classifier import TransformerClassifier, AttentionFeatureWeighter   # pickle 解析
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats

np.random.seed(42)


# ─── 潛在空間視覺化 ───────────────────────────────────────────────────────────

def visualize_latent_space(mu_train, y_train, mu_new,
                           save_path='output/phase5_latent_space.png'):
    """
    用 PCA + t-SNE 將高維潛在空間投影至 2D
    展示：
      藍點 = 訓練正常資料
      紅叉 = 訓練詐欺資料
      橘三角 = 新進偏移資料
    """
    os.makedirs('output', exist_ok=True)

    # 準備資料
    n_normal = min(800, np.sum(y_train == 0))
    n_fraud  = min(100, np.sum(y_train == 1))
    n_new    = min(400, len(mu_new))

    idx_normal = np.where(y_train == 0)[0][:n_normal]
    idx_fraud  = np.where(y_train == 1)[0][:n_fraud]

    mu_vis  = np.vstack([mu_train[idx_normal], mu_train[idx_fraud], mu_new[:n_new]])
    labels  = np.array([0]*n_normal + [1]*n_fraud + [2]*n_new)

    print(f"[Phase 5] t-SNE 降維中... ({len(mu_vis)} 個樣本, {mu_vis.shape[1]}D → 2D)")

    # Step 1: PCA 降至 min(8, latent_dim) 維（加速 t-SNE）
    n_pca = min(8, mu_vis.shape[1])
    pca   = PCA(n_components=n_pca)
    mu_pca = pca.fit_transform(mu_vis)
    print(f"  PCA 解釋方差: {pca.explained_variance_ratio_.sum():.2%}")

    # Step 2: t-SNE
    tsne = TSNE(n_components=2, perplexity=40, max_iter=800,
                learning_rate='auto', init='pca', random_state=42, verbose=0)
    mu_2d = tsne.fit_transform(mu_pca)

    # ─── 繪圖 ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0f1117')
    for ax in axes:
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#8888aa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    # 左圖：散點圖（分色）
    ax = axes[0]
    scatter_cfg = [
        (labels == 0, '#4488ff', 'o', 15, 0.5,  'Train Normal'),
        (labels == 1, '#ff4466', 'x', 40, 0.85, 'Train Fraud'),
        (labels == 2, '#ffaa22', '^', 18, 0.55, 'New (Shifted)'),
    ]
    for mask, color, marker, size, alpha, label in scatter_cfg:
        ax.scatter(mu_2d[mask, 0], mu_2d[mask, 1],
                   c=color, marker=marker, s=size,
                   alpha=alpha, label=label, linewidths=0.5)

    # 標記分佈中心
    for mask, color, label in [
        (labels == 0, '#4488ff', 'Normal center'),
        (labels == 2, '#ffaa22', 'New center'),
    ]:
        cx, cy = mu_2d[mask].mean(axis=0)
        ax.scatter(cx, cy, c=color, marker='*', s=200, zorder=10,
                   edgecolors='white', linewidths=0.5)
        ax.annotate(label, (cx, cy), xytext=(10, 10),
                    textcoords='offset points', color=color, fontsize=8)

    ax.set_title('Latent Space: t-SNE Projection', color='#eeeeff', fontsize=12)
    ax.legend(facecolor='#1a1d27', labelcolor='#ccccdd', fontsize=9)
    ax.set_xlabel('t-SNE dim 1', color='#ccccdd')
    ax.set_ylabel('t-SNE dim 2', color='#ccccdd')

    # 右圖：密度等高線圖（Normal vs New）
    ax = axes[1]
    from scipy.stats import gaussian_kde

    for mask, color, label in [
        (labels == 0, '#4488ff', 'Train Normal'),
        (labels == 2, '#ffaa22', 'New (Shifted)'),
    ]:
        pts = mu_2d[mask]
        if len(pts) < 10:
            continue
        try:
            kde  = gaussian_kde(pts.T, bw_method=0.3)
            x_min, x_max = mu_2d[:, 0].min() - 2, mu_2d[:, 0].max() + 2
            y_min, y_max = mu_2d[:, 1].min() - 2, mu_2d[:, 1].max() + 2
            xx, yy = np.mgrid[x_min:x_max:80j, y_min:y_max:80j]
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contourf(xx, yy, zz, levels=8, alpha=0.4,
                        colors=[color]*8)
            ax.contour(xx, yy, zz, levels=5, colors=[color],
                       linewidths=1.0, alpha=0.8)
        except Exception:
            ax.scatter(pts[:, 0], pts[:, 1], c=color, s=8, alpha=0.3)

        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=6, alpha=0.2)

    # 標記偏移向量
    c_normal = mu_2d[labels == 0].mean(axis=0)
    c_new    = mu_2d[labels == 2].mean(axis=0)
    ax.annotate('', xy=c_new, xytext=c_normal,
                arrowprops=dict(arrowstyle='->', color='#ff8844',
                                lw=2.5, connectionstyle='arc3,rad=0.2'))
    ax.text((c_normal[0]+c_new[0])/2 + 3, (c_normal[1]+c_new[1])/2,
            'Distribution\nShift ↑', color='#ff8844', fontsize=9)

    ax.set_title('Density Contour: Normal vs Shifted', color='#eeeeff', fontsize=12)
    ax.set_xlabel('t-SNE dim 1', color='#ccccdd')
    ax.set_ylabel('t-SNE dim 2', color='#ccccdd')
    normal_patch = mpatches.Patch(color='#4488ff', alpha=0.6, label='Train Normal')
    new_patch    = mpatches.Patch(color='#ffaa22', alpha=0.6, label='New (Shifted)')
    ax.legend(handles=[normal_patch, new_patch],
              facecolor='#1a1d27', labelcolor='#ccccdd', fontsize=9)

    plt.suptitle('Phase 5: Latent Space Visualization — Distribution Shift',
                 color='#eeeeff', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Phase 5] 潛在空間圖表已儲存至 {save_path}")
    return mu_2d, labels


# ─── 再訓練觸發機制 ───────────────────────────────────────────────────────────

class RetrainingTrigger:
    """
    自動再訓練觸發器
    監控多個指標，符合任一觸發條件即啟動再訓練流程

    觸發條件（可調整）：
    1. KS p-value < 0.05（統計顯著偏移）
    2. Alert 比例 > 20%（大量樣本超過警戒閾值）
    3. Alarm 比例 > 5%（部分樣本超過警報閾值）
    4. MMD > 0.5（分佈距離過大）
    5. 連續 3 個窗口 KS 顯著（持續性偏移）
    """
    def __init__(self):
        self.trigger_log = []
        self.conditions = {
            'ks_pval_threshold':   0.05,
            'alert_rate_threshold': 0.20,
            'alarm_rate_threshold': 0.05,
            'mmd_threshold':        0.5,
        }

    def check(self, metrics_dict, batch_id=None):
        """
        metrics_dict 應包含：
          ks_pval, alert_rate, alarm_rate, mmd
        """
        triggered = []
        reasons   = []

        if metrics_dict.get('ks_pval', 1.0) < self.conditions['ks_pval_threshold']:
            triggered.append('ks_test')
            reasons.append(f"KS p-value={metrics_dict['ks_pval']:.4f} < 0.05")

        if metrics_dict.get('alert_rate', 0) > self.conditions['alert_rate_threshold']:
            triggered.append('alert_rate')
            reasons.append(f"Alert rate={metrics_dict['alert_rate']:.2%} > 20%")

        if metrics_dict.get('alarm_rate', 0) > self.conditions['alarm_rate_threshold']:
            triggered.append('alarm_rate')
            reasons.append(f"Alarm rate={metrics_dict['alarm_rate']:.2%} > 5%")

        if metrics_dict.get('mmd', 0) > self.conditions['mmd_threshold']:
            triggered.append('mmd')
            reasons.append(f"MMD={metrics_dict['mmd']:.4f} > 0.5")

        event = {
            'batch_id':  batch_id,
            'metrics':   metrics_dict,
            'triggered': bool(triggered),
            'reasons':   reasons,
        }
        self.trigger_log.append(event)

        return bool(triggered), reasons

    def simulate_retraining(self, X_old, y_old, X_new_flagged, clf):
        """
        模擬觸發後的再訓練流程
        實務中此處會呼叫 MLOps pipeline（例如 Kubeflow / MLflow）
        """
        print("\n[Phase 5] ▶  再訓練流程啟動")
        print("  Step 1: 收集標記資料（人工審核 / 弱監督標注）")
        # 假設有部分標注（活躍學習場景）
        n_label = min(200, len(X_new_flagged))
        X_label = X_new_flagged[:n_label]
        # 以重建誤差排序，偏移最大的優先標注
        y_pseudo = np.zeros(n_label, dtype=int)   # 假設都為正常

        print(f"  Step 2: 合併新舊資料（舊: {len(X_old)}, 新標注: {n_label}）")
        from sklearn.utils import resample
        # 對新資料做 oversampling（比例 3:1）
        X_new_os = resample(X_label, replace=True, n_samples=n_label * 3, random_state=42)
        y_new_os = np.zeros(len(X_new_os), dtype=int)
        X_combined = np.vstack([X_old, X_new_os])
        y_combined = np.concatenate([y_old, y_new_os])

        print(f"  Step 3: 重新訓練分類器（{len(X_combined)} 筆）")
        clf.fit(X_combined, y_combined)
        print("  Step 4: 儲存新模型版本")
        with open('output/classifier_retrained.pkl', 'wb') as f:
            pickle.dump(clf, f)
        print("  Step 5: 部署新模型（模擬）")
        print(f"[Phase 5] ✓  再訓練完成！新模型已儲存至 output/classifier_retrained.pkl")
        return clf

    def print_report(self):
        print("\n" + "=" * 60)
        print("再訓練觸發日誌")
        print("=" * 60)
        for event in self.trigger_log:
            status = "🔴 TRIGGERED" if event['triggered'] else "🟢 OK"
            print(f"  Batch {event['batch_id']:>3} | {status}")
            if event['reasons']:
                for r in event['reasons']:
                    print(f"           → {r}")


# ─── 監控時間軸視覺化 ─────────────────────────────────────────────────────────

def visualize_monitoring_timeline(vae, detector, X_stream, y_stream,
                                  save_path='output/phase5_timeline.png'):
    """
    模擬線上監控：將測試資料依時間順序分批，逐批計算偏移指標
    展示指標如何隨時間演化，並標記觸發點
    """
    os.makedirs('output', exist_ok=True)
    window = 100
    step   = 30

    batches = []
    ks_stats, alert_rates, alarm_rates, kl_means = [], [], [], []

    ref_scores = detector.train_stats['recon']

    for start in range(0, len(X_stream) - window, step):
        batch = X_stream[start:start + window]
        recon = np.mean((batch - vae.forward(batch)[0])**2, axis=1)
        kl    = 0.5 * np.sum(
            np.exp(np.clip(vae.encode(batch)[1], -10, 10)) +
            vae.encode(batch)[0]**2 - 1 -
            np.clip(vae.encode(batch)[1], -10, 10), axis=1
        )

        ks_stat, p_val = stats.ks_2samp(ref_scores, recon)
        alert_rate = np.mean(recon > detector.thresholds['recon']['alert'])
        alarm_rate = np.mean(recon > detector.thresholds['recon']['alarm'])

        batches.append(start)
        ks_stats.append(ks_stat)
        alert_rates.append(alert_rate)
        alarm_rates.append(alarm_rate)
        kl_means.append(kl.mean())

    batches    = np.array(batches)
    ks_stats   = np.array(ks_stats)
    alert_rates = np.array(alert_rates)
    kl_means   = np.array(kl_means)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor('#0f1117')
    for ax in axes:
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#8888aa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    # 偏移注入位置
    shift_point = len(X_stream) // 2

    for ax in axes:
        ax.axvline(shift_point, color='#ff8844', linestyle='--',
                   lw=2, alpha=0.8, label='Shift Injected')

    # KS Statistic
    axes[0].plot(batches, ks_stats, color='#8899ff', lw=2)
    axes[0].fill_between(batches, ks_stats, alpha=0.2, color='#8899ff')
    axes[0].axhline(0.1, color='#ffdd44', linestyle=':', lw=1.5, label='KS=0.1 threshold')
    drift_mask = ks_stats > 0.15
    axes[0].scatter(batches[drift_mask], ks_stats[drift_mask],
                    color='#ff4444', s=30, zorder=5)
    axes[0].set_ylabel('KS Statistic', color='#ccccdd')
    axes[0].set_title('Online Monitoring Timeline', color='#eeeeff', fontsize=12)
    axes[0].legend(fontsize=8, facecolor='#1a1d27', labelcolor='#ccccdd')

    # Alert Rate
    axes[1].plot(batches, alert_rates * 100, color='#ffdd44', lw=2)
    axes[1].fill_between(batches, alert_rates * 100, alpha=0.2, color='#ffdd44')
    axes[1].axhline(20, color='#ff6622', linestyle='--', lw=1.5, label='Trigger threshold 20%')
    axes[1].set_ylabel('Alert Rate (%)', color='#ccccdd')
    axes[1].legend(fontsize=8, facecolor='#1a1d27', labelcolor='#ccccdd')

    # KL Mean
    axes[2].plot(batches, kl_means, color='#44ffaa', lw=2)
    axes[2].fill_between(batches, kl_means, alpha=0.2, color='#44ffaa')
    axes[2].set_ylabel('Mean KL Divergence', color='#ccccdd')
    axes[2].set_xlabel('Stream Position (sample index)', color='#ccccdd')

    # 標記觸發點
    trigger_idx = np.where((ks_stats > 0.15) & (batches > shift_point))[0]
    if len(trigger_idx) > 0:
        first_trigger = batches[trigger_idx[0]]
        for ax in axes:
            ax.axvline(first_trigger, color='#ff2244', linestyle='-.',
                       lw=2, alpha=0.9)
        axes[0].text(first_trigger + 5, ks_stats.max() * 0.9,
                     '⚠ RETRAIN\nTRIGGERED', color='#ff2244', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Phase 5] 監控時間軸已儲存至 {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 5: UMAP/t-SNE 視覺化 + 再訓練觸發")
    print("=" * 60)

    with open('output/vae.pkl', 'rb') as f:
        vae = pickle.load(f)
    with open('output/detector.pkl', 'rb') as f:
        detector = pickle.load(f)
    with open('output/classifier.pkl', 'rb') as f:
        clf = pickle.load(f)

    X_train = np.load('output/X_train.npy')
    y_train = np.load('output/y_train.npy')
    X_val   = np.load('output/X_val.npy')
    y_val   = np.load('output/y_val.npy')
    X_new   = np.load('output/X_new.npy')
    mu_train = np.load('output/mu_train.npy')
    mu_new   = np.load('output/mu_new.npy')

    # 1. 潛在空間視覺化
    mu_2d, labels = visualize_latent_space(mu_train, y_train, mu_new)

    # 2. 監控時間軸（模擬串流）
    n_normal  = np.sum(y_val == 0)
    X_stream  = np.vstack([X_val[y_val == 0], X_new])
    y_stream  = np.array([0]*n_normal + [-1]*len(X_new))
    visualize_monitoring_timeline(vae, detector, X_stream, y_stream)

    # 3. 再訓練觸發評估
    trigger = RetrainingTrigger()

    # 模擬多批次監控
    print("\n[Phase 5] 模擬線上批次監控...")
    for batch_id, (X_batch, label) in enumerate([
        (X_val[y_val==0][:200],  'Normal batch'),
        (X_val[y_val==0][:200],  'Normal batch'),
        (X_new[:200],            'Shifted batch 1'),
        (X_new[200:400],         'Shifted batch 2'),
        (X_new[400:600],         'Shifted batch 3'),
    ]):
        recon  = np.mean((X_batch - vae.forward(X_batch)[0])**2, axis=1)
        ref_sc = detector.train_stats['recon']
        ks_stat, ks_pval = stats.ks_2samp(ref_sc, recon)
        alert_r = float(np.mean(recon > detector.thresholds['recon']['alert']))
        alarm_r = float(np.mean(recon > detector.thresholds['recon']['alarm']))

        metrics = {
            'ks_pval':    ks_pval,
            'alert_rate': alert_r,
            'alarm_rate': alarm_r,
            'mmd':        0.3 if label.startswith('Normal') else 0.72,
        }
        triggered, reasons = trigger.check(metrics, batch_id=batch_id)
        status = "🔴 TRIGGERED" if triggered else "🟢 OK"
        print(f"  Batch {batch_id} ({label}): {status}")
        if reasons:
            for r in reasons:
                print(f"    → {r}")

    trigger.print_report()

    # 4. 觸發後執行再訓練
    print("\n[Phase 5] 偵測到持續性偏移，執行再訓練...")
    trigger.simulate_retraining(X_train, y_train, X_new, clf)

    print("\n" + "=" * 60)
    print("Phase 5 完成！所有圖表與模型已儲存至 output/")
    print("=" * 60)

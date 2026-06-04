"""
Phase 4: 偏移偵測機制
結合三種互補的偵測指標：
  1. Reconstruction Error   — VAE 重建誤差（點級異常分數）
  2. KL Divergence          — 潛在分佈與先驗 N(0,I) 的散度
  3. Mahalanobis Distance   — 潛在向量距離訓練分佈中心的距離
  4. KS Test (統計檢定)     — 滑動窗口批次偵測分佈改變
  5. MMD (Maximum Mean Discrepancy) — 分佈間距離的核方法估計
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_vae import VAE, DenseLayer   # 確保 pickle 可以解析 VAE class
from scipy import stats
from sklearn.metrics import roc_auc_score

np.random.seed(42)


# ─── 偵測指標計算 ─────────────────────────────────────────────────────────────

def reconstruction_error(vae, X):
    """VAE 重建誤差 MSE（每筆樣本）"""
    x_recon, mu, logvar, z = vae.forward(X)
    return np.mean((X - x_recon) ** 2, axis=1)


def kl_divergence_score(vae, X):
    """KL(q(z|x) || p(z)) per sample"""
    mu, logvar = vae.encode(X)
    return 0.5 * np.sum(np.exp(logvar) + mu**2 - 1 - logvar, axis=1)


def mahalanobis_distance(mu_test, mu_ref):
    """
    Mahalanobis Distance：考慮訓練集潛在空間的協方差
    d_M(z) = sqrt( (z - μ_ref)^T Σ^{-1} (z - μ_ref) )
    """
    mean_ref = mu_ref.mean(axis=0)
    cov_ref  = np.cov(mu_ref.T) + 1e-6 * np.eye(mu_ref.shape[1])  # 正則化
    cov_inv  = np.linalg.pinv(cov_ref)
    diff     = mu_test - mean_ref
    dist     = np.sqrt(np.einsum('ni,ij,nj->n', diff, cov_inv, diff))
    return dist


def ks_test_sliding_window(scores_ref, scores_new, window_size=200, step=50):
    """
    滑動窗口 KS 檢定（Kolmogorov-Smirnov Test）
    用於批次監控場景：每隔 step 筆新資料，與訓練集分佈做統計檢定

    Returns
    -------
    positions : 偵測點位置（以 scores_new 的索引計）
    ks_stats  : KS 統計量（0~1，越大偏移越嚴重）
    p_values  : p 值（< 0.05 → 顯著偏移）
    """
    positions = []
    ks_stats  = []
    p_values  = []

    for start in range(0, len(scores_new) - window_size + 1, step):
        window    = scores_new[start:start + window_size]
        ks_stat, p_val = stats.ks_2samp(scores_ref, window)
        positions.append(start + window_size // 2)
        ks_stats.append(ks_stat)
        p_values.append(p_val)

    return np.array(positions), np.array(ks_stats), np.array(p_values)


def mmd_rbf(X_ref, X_test, gamma=1.0):
    """
    Maximum Mean Discrepancy with RBF kernel（批次整體偏移量）
    MMD²(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

    γ 控制 RBF kernel 的頻寬：gamma = 1/(2σ²)
    越高的 MMD 代表兩個分佈差異越大
    """
    def rbf_kernel(A, B, gamma):
        diff = A[:, None, :] - B[None, :, :]   # (n, m, d)
        sq_dist = np.sum(diff**2, axis=-1)      # (n, m)
        return np.exp(-gamma * sq_dist)

    n = min(300, len(X_ref), len(X_test))   # subsample for efficiency
    idx_r = np.random.choice(len(X_ref),  n, replace=False)
    idx_t = np.random.choice(len(X_test), n, replace=False)
    X_r = X_ref[idx_r]
    X_t = X_test[idx_t]

    K_rr = rbf_kernel(X_r, X_r, gamma)
    K_tt = rbf_kernel(X_t, X_t, gamma)
    K_rt = rbf_kernel(X_r, X_t, gamma)

    mmd2 = (K_rr.mean() + K_tt.mean() - 2 * K_rt.mean())
    return float(np.sqrt(max(0, mmd2)))


# ─── 閾值設定 ─────────────────────────────────────────────────────────────────

class ShiftDetector:
    """
    整合多個偵測指標的監控器
    閾值以訓練集的 95/99 百分位數設定
    """
    def __init__(self, vae, percentile_alert=95, percentile_alarm=99):
        self.vae = vae
        self.p_alert = percentile_alert
        self.p_alarm = percentile_alarm
        self.thresholds = {}
        self.train_stats = {}

    def fit(self, X_train_normal, mu_train_normal):
        """以正常訓練資料建立基準分佈"""
        recon = reconstruction_error(self.vae, X_train_normal)
        kl    = kl_divergence_score(self.vae, X_train_normal)
        mah   = mahalanobis_distance(mu_train_normal, mu_train_normal)

        for name, scores in [('recon', recon), ('kl', kl), ('mahal', mah)]:
            self.thresholds[name] = {
                'alert': np.percentile(scores, self.p_alert),
                'alarm': np.percentile(scores, self.p_alarm),
                'mean':  np.mean(scores),
                'std':   np.std(scores),
            }
            self.train_stats[name] = scores

        print(f"[Phase 4] 偵測閾值設定（基於訓練集 {self.p_alert}/{self.p_alarm} 百分位）：")
        for name, thr in self.thresholds.items():
            print(f"  {name:8s} | alert={thr['alert']:.4f} | alarm={thr['alarm']:.4f}")
        return self

    def score(self, X_test, mu_test):
        """計算測試集的所有偵測分數"""
        recon = reconstruction_error(self.vae, X_test)
        kl    = kl_divergence_score(self.vae, X_test)
        mah   = mahalanobis_distance(mu_test, list(self.train_stats.values())[0].reshape(-1, 1)
                                     if False else mu_test)

        # Mahalanobis 需要用訓練集的 μ 做參考
        # 這裡用 VAE encode 訓練集的 μ 作為基準
        return {
            'recon': recon,
            'kl':    kl,
            # 'mahal': mah,   # mahal 需傳入訓練 μ，在 main 中單獨計算
        }

    def is_shifted(self, scores_dict, metric='recon'):
        """逐樣本判斷是否為偏移"""
        s   = scores_dict[metric]
        thr = self.thresholds[metric]
        alert = s > thr['alert']
        alarm = s > thr['alarm']
        return alert, alarm

    def batch_drift_score(self, X_new, mu_new, metric='recon'):
        """
        批次偏移分數：偏移樣本比例 + KS 統計量
        Returns a dict with:
          alert_rate, alarm_rate, ks_stat, ks_pvalue, mmd
        """
        if metric == 'recon':
            scores_new = reconstruction_error(self.vae, X_new)
            scores_ref = self.train_stats['recon']
        elif metric == 'kl':
            scores_new = kl_divergence_score(self.vae, X_new)
            scores_ref = self.train_stats['kl']

        alert_rate = np.mean(scores_new > self.thresholds[metric]['alert'])
        alarm_rate = np.mean(scores_new > self.thresholds[metric]['alarm'])
        ks_stat, ks_pval = stats.ks_2samp(scores_ref, scores_new)
        mmd_val = mmd_rbf(
            self.vae.encode_mean(
                np.random.randn(200, X_new.shape[1])   # 用 N(0,I) 作為近似訓練分佈
            ),
            mu_new[:200]
        )

        return {
            'alert_rate': alert_rate,
            'alarm_rate': alarm_rate,
            'ks_stat':    ks_stat,
            'ks_pval':    ks_pval,
            'mmd':        mmd_val,
            'is_drift':   ks_pval < 0.05 or alert_rate > 0.2,
        }


# ─── 視覺化 ───────────────────────────────────────────────────────────────────

def visualize_detection(detector, vae, X_val, y_val, X_new, mu_val, mu_new,
                        save_path='output/phase4_detection.png'):
    os.makedirs('output', exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#0f1117')
    for ax in axes.flat:
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#8888aa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    recon_normal = reconstruction_error(vae, X_val[y_val == 0])
    recon_fraud  = reconstruction_error(vae, X_val[y_val == 1])
    recon_new    = reconstruction_error(vae, X_new)
    kl_normal    = kl_divergence_score(vae, X_val[y_val == 0])
    kl_fraud     = kl_divergence_score(vae, X_val[y_val == 1])
    kl_new       = kl_divergence_score(vae, X_new)
    mah_normal   = mahalanobis_distance(mu_val[y_val == 0], mu_val[y_val == 0])
    mah_fraud    = mahalanobis_distance(mu_val[y_val == 1], mu_val[y_val == 0])
    mah_new      = mahalanobis_distance(mu_new, mu_val[y_val == 0])

    thr_recon = detector.thresholds['recon']
    thr_kl    = detector.thresholds['kl']
    thr_mah   = detector.thresholds['mahal']

    # 1. Reconstruction Error
    ax = axes[0, 0]
    ax.hist(recon_normal, bins=60, alpha=0.6, color='#4488ff', label='Normal', density=True)
    ax.hist(recon_fraud,  bins=60, alpha=0.6, color='#ff4466', label='Fraud',  density=True)
    ax.hist(recon_new,    bins=60, alpha=0.6, color='#ffaa22', label='New (Shifted)', density=True)
    ax.axvline(thr_recon['alert'], color='#ffdd44', linestyle='--', lw=1.5, label=f"Alert ({detector.p_alert}%)")
    ax.axvline(thr_recon['alarm'], color='#ff6622', linestyle='--', lw=1.5, label=f"Alarm ({detector.p_alarm}%)")
    ax.set_title('Reconstruction Error', color='#eeeeff')
    ax.legend(fontsize=8, facecolor='#1a1d27', labelcolor='#ccccdd')

    # 2. KL Divergence
    ax = axes[0, 1]
    ax.hist(kl_normal, bins=60, alpha=0.6, color='#4488ff', label='Normal', density=True)
    ax.hist(kl_fraud,  bins=60, alpha=0.6, color='#ff4466', label='Fraud',  density=True)
    ax.hist(kl_new,    bins=60, alpha=0.6, color='#ffaa22', label='New (Shifted)', density=True)
    ax.axvline(thr_kl['alert'], color='#ffdd44', linestyle='--', lw=1.5)
    ax.axvline(thr_kl['alarm'], color='#ff6622', linestyle='--', lw=1.5)
    ax.set_title('KL Divergence Score', color='#eeeeff')
    ax.legend(fontsize=8, facecolor='#1a1d27', labelcolor='#ccccdd')

    # 3. Mahalanobis Distance
    ax = axes[0, 2]
    ax.hist(mah_normal, bins=60, alpha=0.6, color='#4488ff', label='Normal', density=True)
    ax.hist(mah_fraud,  bins=60, alpha=0.6, color='#ff4466', label='Fraud',  density=True)
    ax.hist(mah_new,    bins=60, alpha=0.6, color='#ffaa22', label='New (Shifted)', density=True)
    ax.axvline(thr_mah['alert'], color='#ffdd44', linestyle='--', lw=1.5)
    ax.axvline(thr_mah['alarm'], color='#ff6622', linestyle='--', lw=1.5)
    ax.set_title('Mahalanobis Distance', color='#eeeeff')
    ax.legend(fontsize=8, facecolor='#1a1d27', labelcolor='#ccccdd')

    # 4. 滑動窗口 KS 統計量
    ax = axes[1, 0]
    ref_scores = detector.train_stats['recon']
    test_stream = np.concatenate([recon_normal[:800], recon_new])   # 模擬上線後串流
    pos, ks_s, p_v = ks_test_sliding_window(ref_scores, test_stream, window_size=100, step=25)
    is_drift = p_v < 0.05
    ax.plot(pos, ks_s, color='#8899ff', linewidth=1.5)
    ax.fill_between(pos, ks_s, alpha=0.2, color='#8899ff')
    ax.scatter(pos[is_drift], ks_s[is_drift], color='#ff4444', s=25, zorder=5, label='p<0.05 (drift)')
    ax.axvline(800, color='#ffaa22', linestyle='--', lw=1.5, label='Shift injected')
    ax.axhline(0.1, color='#555577', linestyle=':', lw=1)
    ax.set_title('Sliding Window KS Test', color='#eeeeff')
    ax.set_xlabel('Stream Position', color='#ccccdd')
    ax.set_ylabel('KS Statistic', color='#ccccdd')
    ax.legend(fontsize=8, facecolor='#1a1d27', labelcolor='#ccccdd')

    # 5. 偵測率比較（Alert/Alarm）
    ax = axes[1, 1]
    metrics = ['Recon Error', 'KL Divergence', 'Mahalanobis']
    alert_rates = []
    alarm_rates = []
    for scores, thr_name in [(recon_new, 'recon'), (kl_new, 'kl'), (mah_new, 'mahal')]:
        thr = detector.thresholds[thr_name]
        alert_rates.append(np.mean(scores > thr['alert']) * 100)
        alarm_rates.append(np.mean(scores > thr['alarm']) * 100)

    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, alert_rates, w, color='#ffdd44', alpha=0.8, label=f'Alert ({detector.p_alert}%)')
    ax.bar(x + w/2, alarm_rates, w, color='#ff6622', alpha=0.8, label=f'Alarm ({detector.p_alarm}%)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color='#ccccdd', fontsize=9)
    ax.set_ylabel('% Flagged', color='#ccccdd')
    ax.set_title('Detection Rate on Shifted Data', color='#eeeeff')
    ax.legend(facecolor='#1a1d27', labelcolor='#ccccdd')
    for i, (ar, alr) in enumerate(zip(alert_rates, alarm_rates)):
        ax.text(i - w/2, ar + 1, f'{ar:.1f}%', ha='center', color='#ffdd44', fontsize=8)
        ax.text(i + w/2, alr + 1, f'{alr:.1f}%', ha='center', color='#ff6622', fontsize=8)

    # 6. 綜合分數散點圖（Recon Error vs KL）
    ax = axes[1, 2]
    ax.scatter(recon_normal[:300], kl_normal[:300],
               alpha=0.4, s=12, color='#4488ff', label='Normal')
    ax.scatter(recon_fraud, kl_fraud,
               alpha=0.7, s=20, color='#ff4466', label='Fraud', marker='x')
    ax.scatter(recon_new[:300], kl_new[:300],
               alpha=0.5, s=12, color='#ffaa22', label='New (Shifted)', marker='^')
    ax.axvline(thr_recon['alert'], color='#ffdd44', linestyle='--', lw=1, alpha=0.7)
    ax.axhline(thr_kl['alert'],    color='#ffdd44', linestyle='--', lw=1, alpha=0.7)
    ax.set_xlabel('Reconstruction Error', color='#ccccdd')
    ax.set_ylabel('KL Divergence', color='#ccccdd')
    ax.set_title('Joint Detection Space', color='#eeeeff')
    ax.legend(fontsize=8, facecolor='#1a1d27', labelcolor='#ccccdd')

    plt.suptitle('Phase 4: Distribution Shift Detection Dashboard',
                 color='#eeeeff', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Phase 4] 圖表已儲存至 {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 4: 偏移偵測機制")
    print("=" * 60)

    with open('output/vae.pkl', 'rb') as f:
        vae = pickle.load(f)

    X_train = np.load('output/X_train.npy')
    X_val   = np.load('output/X_val.npy')
    y_train = np.load('output/y_train.npy')
    y_val   = np.load('output/y_val.npy')
    X_new   = np.load('output/X_new.npy')
    mu_val  = np.load('output/mu_val.npy')
    mu_new  = np.load('output/mu_new.npy')

    # 建立偵測器
    X_train_normal = X_train[y_train == 0]
    mu_train_normal = vae.encode_mean(X_train_normal)

    detector = ShiftDetector(vae, percentile_alert=95, percentile_alarm=99)
    detector.fit(X_train_normal, mu_train_normal)

    # 額外設定 mahalanobis 閾值
    mah_ref = mahalanobis_distance(mu_train_normal, mu_train_normal)
    detector.thresholds['mahal'] = {
        'alert': np.percentile(mah_ref, 95),
        'alarm': np.percentile(mah_ref, 99),
        'mean':  mah_ref.mean(),
        'std':   mah_ref.std(),
    }
    detector.train_stats['mahal'] = mah_ref

    # 批次偏移評估
    print("\n[Phase 4] 批次偏移評估（新資料 vs 訓練分佈）：")
    result = detector.batch_drift_score(X_new, mu_new, metric='recon')
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:15s}: {v:.4f}")
        else:
            print(f"  {k:15s}: {v}")

    visualize_detection(detector, vae, X_val, y_val, X_new, mu_val, mu_new)

    # 儲存偵測器
    with open('output/detector.pkl', 'wb') as f:
        pickle.dump(detector, f)

    print(f"\n[Phase 4] 完成！偵測器已儲存")
    drift_flag = "⚠️  偵測到分佈偏移" if result['is_drift'] else "✓ 無顯著偏移"
    print(f"  判斷結果: {drift_flag}")
    print(f"  KS p-value: {result['ks_pval']:.6f}  (< 0.05 = 顯著偏移)")
    print(f"  Alert 比例: {result['alert_rate']:.2%}  Alarm 比例: {result['alarm_rate']:.2%}")

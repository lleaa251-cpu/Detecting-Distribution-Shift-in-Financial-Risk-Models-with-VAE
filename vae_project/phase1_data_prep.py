"""
Phase 1: 資料準備 + 人工注入分佈偏移
使用合成金融詐欺資料集（模擬 Kaggle Credit Card Fraud Detection）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans']
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)

# ─── 1. 生成合成金融資料 ───────────────────────────────────────────────────────

def generate_financial_data(n_samples=10000, fraud_rate=0.02):
    """
    模擬信用卡交易資料
    正常交易：低金額、特定時段、低風險特徵
    詐欺交易：高金額、異常時段、高風險特徵
    """
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud

    # 正常交易
    normal = np.column_stack([
        np.random.normal(0,   1,   (n_normal, 8)),   # V1-V8: PCA 特徵
        np.random.normal(50,  30,  (n_normal, 1)),   # Amount: 低金額
        np.random.uniform(0,  24,  (n_normal, 1)),   # Hour: 正常時段
        np.random.normal(0.3, 0.1, (n_normal, 1)),   # RiskScore
    ])

    # 詐欺交易（分佈明顯不同）
    fraud = np.column_stack([
        np.random.normal(2,   1.5, (n_fraud, 8)),    # V1-V8: 偏移的 PCA 特徵
        np.random.normal(400, 200, (n_fraud, 1)),    # Amount: 高金額
        np.random.choice([1,2,3,4], size=n_fraud).reshape(-1,1),  # Hour: 異常時段(凌晨)
        np.random.normal(0.8, 0.15, (n_fraud, 1)),  # RiskScore
    ])

    X = np.vstack([normal, fraud])
    y = np.array([0]*n_normal + [1]*n_fraud)

    cols = [f'V{i}' for i in range(1, 9)] + ['Amount', 'Hour', 'RiskScore']
    df = pd.DataFrame(X, columns=cols)
    df['Class'] = y
    return df.sample(frac=1).reset_index(drop=True)


# ─── 2. 人工注入分佈偏移 ──────────────────────────────────────────────────────

def inject_distribution_shift(df_train, shift_type='gradual', n_new=2000):
    """
    模擬三種常見的線上資料偏移：

    covariate : 特徵分佈改變（例如：新用戶族群、消費行為改變）
    concept   : 標籤邊界改變（模型認為正常的行為開始成為詐欺）
    gradual   : 漸進式偏移（最常見，季節性或市場變化）
    """
    # 取訓練集的特徵統計
    feature_cols = [c for c in df_train.columns if c != 'Class']
    mu    = df_train[feature_cols].mean().values
    sigma = df_train[feature_cols].std().values

    if shift_type == 'covariate':
        # 特徵整體平移 + 縮放（新客群）
        shift_factor = np.random.uniform(1.5, 3.0, len(feature_cols))
        X_new = np.random.normal(mu * shift_factor, sigma, (n_new, len(feature_cols)))
        label = "Covariate Shift (新客群特徵偏移)"

    elif shift_type == 'concept':
        # 特徵不變，但 Amount / RiskScore 欄位意義改變
        X_new = np.random.normal(mu, sigma, (n_new, len(feature_cols)))
        idx_amount = feature_cols.index('Amount')
        idx_risk   = feature_cols.index('RiskScore')
        # 正常金額範圍擴大（過去高金額 = 詐欺，現在不一定）
        X_new[:, idx_amount] = np.random.normal(250, 150, n_new)
        X_new[:, idx_risk]   = np.random.normal(0.55, 0.2, n_new)
        label = "Concept Drift (詐欺模式改變)"

    else:  # gradual
        # 每個 batch 漸進式偏移
        batches = []
        for i in range(5):
            drift = 0.3 * (i + 1)
            b = np.random.normal(mu + drift, sigma, (n_new // 5, len(feature_cols)))
            batches.append(b)
        X_new = np.vstack(batches)
        label = "Gradual Drift (漸進式偏移)"

    df_new = pd.DataFrame(X_new, columns=feature_cols)
    df_new['Class'] = -1   # -1 代表「偏移後的未知樣本」
    df_new['ShiftType'] = shift_type
    print(f"[Phase 1] 注入偏移類型: {label}，共 {len(df_new)} 筆新樣本")
    return df_new


# ─── 3. 前處理 ────────────────────────────────────────────────────────────────

def preprocess(df_train, df_new=None):
    feature_cols = [c for c in df_train.columns if c not in ['Class', 'ShiftType']]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[feature_cols])
    y_train = df_train['Class'].values

    X_train, X_val, y_train_s, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    result = {
        'X_train': X_train,
        'X_val':   X_val,
        'y_train': y_train_s,
        'y_val':   y_val,
        'scaler':  scaler,
        'feature_cols': feature_cols,
    }

    if df_new is not None:
        X_new_scaled = scaler.transform(df_new[feature_cols])
        result['X_new'] = X_new_scaled
        result['shift_type'] = df_new['ShiftType'].iloc[0]

    return result


# ─── 4. 視覺化 ────────────────────────────────────────────────────────────────

def visualize_data(df, df_new, save_path='output/phase1_distribution.png'):
    os.makedirs('output', exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor('#0f1117')
    for ax in axes.flat:
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#8888aa')
        ax.xaxis.label.set_color('#ccccdd')
        ax.yaxis.label.set_color('#ccccdd')

    features_to_plot = ['V1', 'Amount', 'RiskScore', 'Hour', 'V2', 'V3']
    colors_train_normal = '#4488ff'
    colors_train_fraud  = '#ff4466'
    colors_new          = '#ffaa22'

    normal_mask = df['Class'] == 0
    fraud_mask  = df['Class'] == 1

    for ax, feat in zip(axes.flat, features_to_plot):
        bins = 50
        ax.hist(df.loc[normal_mask, feat], bins=bins, alpha=0.6,
                color=colors_train_normal, label='Train Normal', density=True)
        ax.hist(df.loc[fraud_mask, feat],  bins=bins, alpha=0.6,
                color=colors_train_fraud,  label='Train Fraud',  density=True)
        ax.hist(df_new[feat],              bins=bins, alpha=0.55,
                color=colors_new,          label='New (Shifted)', density=True)
        ax.set_title(feat, color='#eeeeff', fontsize=11)
        ax.legend(fontsize=7, facecolor='#1a1d27', labelcolor='#ccccdd')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    plt.suptitle('Phase 1: Feature Distribution — Train vs Shifted New Data',
                 color='#eeeeff', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Phase 1] 圖表已儲存至 {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 1: 資料準備 + 分佈偏移注入")
    print("=" * 60)

    df_train = generate_financial_data(n_samples=10000, fraud_rate=0.02)
    print(f"訓練資料: {len(df_train)} 筆  |  詐欺率: {df_train['Class'].mean():.2%}")

    df_new = inject_distribution_shift(df_train, shift_type='gradual', n_new=2000)

    data = preprocess(df_train, df_new)
    print(f"Train set: {data['X_train'].shape}  |  Val set: {data['X_val'].shape}")
    print(f"New (shifted) set: {data['X_new'].shape}")

    visualize_data(df_train, df_new)

    # 儲存資料供後續 Phase 使用
    os.makedirs('output', exist_ok=True)
    np.save('output/X_train.npy',    data['X_train'])
    np.save('output/X_val.npy',      data['X_val'])
    np.save('output/y_train.npy',    data['y_train'])
    np.save('output/y_val.npy',      data['y_val'])
    np.save('output/X_new.npy',      data['X_new'])
    import pickle
    with open('output/scaler.pkl', 'wb') as f:
        pickle.dump(data['scaler'], f)

    print("\n[Phase 1] 完成！資料已儲存至 output/")
    print(f"  特徵維度: {data['X_train'].shape[1]}")
    print(f"  類別比例 (train): 正常={np.mean(data['y_train']==0):.2%}, 詐欺={np.mean(data['y_train']==1):.2%}")

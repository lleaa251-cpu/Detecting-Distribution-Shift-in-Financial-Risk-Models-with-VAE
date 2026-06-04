# VAE Distribution Drift Monitor
## 基於變分自編碼器的金融風控模型資料漂移即時監控系統

---

## 快速開始

```bash
# 安裝依賴
pip install numpy pandas matplotlib scikit-learn scipy

# 執行完整流程（5個階段）
python run_all.py

# 只執行特定階段
python run_all.py --phase 1 2 3
python run_all.py --phase 4
```

---

## 專題架構

```
vae_project/
│
├── run_all.py              # 主執行腳本（統一入口）
│
├── phase1_data_prep.py     # 資料準備 + 分佈偏移注入
├── phase2_classifier.py    # Transformer Classifier（Attention + MLP）
├── phase3_vae.py           # VAE 監控器（純 NumPy 實作）
├── phase4_detection.py     # 偏移偵測（KS Test / KL / Mahalanobis / MMD）
├── phase5_visualization.py # t-SNE 視覺化 + 再訓練觸發機制
│
├── requirements.txt        # 依賴套件
└── output/                 # 自動生成
    ├── phase1_distribution.png
    ├── phase2_classifier.png
    ├── phase3_vae.png
    ├── phase4_detection.png
    ├── phase5_latent_space.png
    ├── phase5_timeline.png
    ├── X_train.npy / X_val.npy / X_new.npy
    ├── classifier.pkl / vae.pkl / detector.pkl
    └── classifier_retrained.pkl
```

---

## 各 Phase 說明

### Phase 1 — 資料準備
- 生成 10,000 筆合成金融交易資料（含 2% 詐欺）
- 支援三種偏移類型注入：
  - `covariate`：特徵分佈改變（新客群）
  - `concept`：標籤邊界改變（詐欺模式演化）
  - `gradual`：漸進式偏移（預設）

### Phase 2 — Transformer Classifier
- `AttentionFeatureWeighter`：學習特徵的注意力權重
- `MLPClassifier(128→64→32)`：主分類器
- 處理類別不平衡（sample_weight balanced）
- 輸出：ROC-AUC、訓練曲線、機率分佈

### Phase 3 — VAE 監控器
- 純 NumPy 手刻 VAE（含完整反向傳播 + Adam 優化器）
- 架構：`11→64→32→[μ(8), logσ²(8)]` + `8→32→64→11`
- β-VAE：β=2.0 強化 KL 約束，潛在空間更接近 N(0,I)
- 僅使用正常樣本訓練

### Phase 4 — 偏移偵測
| 指標 | 原理 | 適用場景 |
|------|------|----------|
| Reconstruction Error | VAE 重建誤差 MSE | 點級異常偵測 |
| KL Divergence | KL(q(z\|x) \|\| p(z)) | 潛在分佈偏離程度 |
| Mahalanobis Distance | 考慮協方差的距離 | 多維分佈偏移 |
| KS Test (滑動窗口) | 分佈形狀統計檢定 | 批次監控 |
| MMD (RBF kernel) | 核方法分佈距離 | 整體偏移量化 |

### Phase 5 — 視覺化 + 再訓練
- t-SNE 降維（PCA → t-SNE 二段式加速）
- 密度等高線圖（KDE 估計）
- 線上監控時間軸（KS / Alert Rate / KL 三面板）
- `RetrainingTrigger`：多條件觸發，自動合併新資料再訓練

---

## 技術亮點

1. **純 NumPy VAE**：完整實作前向/反向傳播、Adam 優化器，無需 PyTorch
2. **β-VAE**：β > 1 增強解耦，更利於分佈偏移偵測
3. **重參數化技巧**：`z = μ + σ * ε`，可微分的採樣過程
4. **多維偵測**：單一指標易受噪聲影響，組合指標更穩健
5. **滑動窗口 KS**：適合串流資料的線上監控
6. **自動再訓練**：觸發後合併新舊資料，支援持續學習

---

## 依賴

```
numpy >= 1.24
pandas >= 1.5
matplotlib >= 3.6
scikit-learn >= 1.2
scipy >= 1.9
```

不需要 PyTorch / TensorFlow！

<br>架構圖網站：file:///D:/%E9%87%8D%E8%A6%81%E8%B3%87%E6%96%99%E5%8D%80/Desktop/vae%20project%20page.html

"""
Phase 3: 訓練 VAE（Variational Autoencoder）監控器
純 NumPy 實作，學習「正常輸入」的潛在空間分佈 z ~ N(μ, σ²)

架構：
  Encoder: input(11) → 64 → 32 → [μ(8), log_σ²(8)]
  Reparameterization: z = μ + σ * ε,  ε ~ N(0,I)
  Decoder: z(8) → 32 → 64 → reconstruction(11)

Loss = Reconstruction Loss (MSE) + β * KL Divergence
     = ||x - x̂||² + β * Σ(σ² + μ² - 1 - log σ²) / 2
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

np.random.seed(42)

# ─── Activation Functions ─────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ─── Dense Layer ──────────────────────────────────────────────────────────────

class DenseLayer:
    def __init__(self, in_dim, out_dim, activation='relu'):
        # He initialization
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
        self.activation = activation
        # 梯度
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # Adam 動量
        self.mW = np.zeros_like(self.W);  self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b);  self.vb = np.zeros_like(self.b)
        # 快取
        self.x_cache = None
        self.z_cache = None

    def forward(self, x):
        self.x_cache = x
        z = x @ self.W + self.b
        self.z_cache = z
        if self.activation == 'relu':
            return relu(z)
        elif self.activation == 'linear':
            return z
        elif self.activation == 'sigmoid':
            return sigmoid(z)
        return z

    def backward(self, grad_out):
        if self.activation == 'relu':
            grad_act = grad_out * relu_grad(self.z_cache)
        else:
            grad_act = grad_out  # linear

        self.dW = self.x_cache.T @ grad_act
        self.db = grad_act.sum(axis=0)
        grad_x  = grad_act @ self.W.T
        return grad_x

    def adam_update(self, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        for param, grad, m, v in [
            (self.W, self.dW, self.mW, self.vW),
            (self.b, self.db, self.mb, self.vb),
        ]:
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ─── VAE ──────────────────────────────────────────────────────────────────────

class VAE:
    """
    純 NumPy VAE 實作
    僅使用正常樣本訓練，學習正常資料的潛在空間分佈

    Attributes
    ----------
    latent_dim : int
        潛在空間維度（z 的維度）
    beta : float
        β-VAE 係數，控制 KL 散度的權重
        β > 1 鼓勵更解耦的潛在空間，更利於異常偵測
    """
    def __init__(self, input_dim=11, hidden_dim=64, latent_dim=8, beta=1.0):
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.beta       = beta

        # Encoder
        self.enc1 = DenseLayer(input_dim,   hidden_dim, activation='relu')
        self.enc2 = DenseLayer(hidden_dim,  hidden_dim // 2, activation='relu')
        self.mu_layer     = DenseLayer(hidden_dim // 2, latent_dim, activation='linear')
        self.logvar_layer = DenseLayer(hidden_dim // 2, latent_dim, activation='linear')

        # Decoder
        self.dec1 = DenseLayer(latent_dim,  hidden_dim // 2, activation='relu')
        self.dec2 = DenseLayer(hidden_dim // 2, hidden_dim, activation='relu')
        self.dec3 = DenseLayer(hidden_dim,  input_dim, activation='linear')

        self.history = {'loss': [], 'recon_loss': [], 'kl_loss': []}

    @property
    def all_layers(self):
        return [self.enc1, self.enc2, self.mu_layer, self.logvar_layer,
                self.dec1, self.dec2, self.dec3]

    def encode(self, x):
        h = self.enc1.forward(x)
        h = self.enc2.forward(h)
        mu     = self.mu_layer.forward(h)
        logvar = self.logvar_layer.forward(h)
        logvar = np.clip(logvar, -10, 10)   # 數值穩定
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重參數化技巧：z = μ + σ * ε,  ε ~ N(0, I)"""
        sigma = np.exp(0.5 * logvar)
        eps   = np.random.randn(*mu.shape)
        return mu + sigma * eps

    def decode(self, z):
        h = self.dec1.forward(z)
        h = self.dec2.forward(h)
        return self.dec3.forward(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_recon    = self.decode(z)
        return x_recon, mu, logvar, z

    def loss(self, x, x_recon, mu, logvar):
        """
        ELBO Loss = Reconstruction Loss + β * KL Divergence

        KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        """
        batch_size  = x.shape[0]
        recon_loss  = np.mean(np.sum((x - x_recon)**2, axis=1))
        kl_loss     = -0.5 * np.mean(np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1))
        total_loss  = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def backward(self, x, x_recon, mu, logvar, z, lr, t):
        """手動反向傳播"""
        batch_size = x.shape[0]

        # ─── Reconstruction gradient (dL/dx_recon) ───────────────────────────
        grad_recon = 2 * (x_recon - x) / (batch_size * self.input_dim)

        # ─── Decoder backward ────────────────────────────────────────────────
        g = self.dec3.backward(grad_recon)
        g = self.dec2.backward(g)
        grad_z = self.dec1.backward(g)

        # ─── KL gradient (dKL/dmu, dKL/dlogvar) ─────────────────────────────
        grad_mu_kl     = self.beta * mu     / batch_size
        grad_logvar_kl = self.beta * 0.5 * (np.exp(logvar) - 1) / batch_size

        # ─── Reparameterization gradient: dz → dmu, dlogvar ──────────────────
        sigma  = np.exp(0.5 * logvar)
        eps    = (z - mu) / (sigma + 1e-8)
        grad_mu     = grad_z + grad_mu_kl
        grad_sigma  = grad_z * eps
        grad_logvar = grad_sigma * 0.5 * sigma + grad_logvar_kl

        # ─── Encoder head backward ────────────────────────────────────────────
        # mu 和 logvar 共享 enc1/enc2 的梯度，需要疊加
        g_mu = self.mu_layer.backward(grad_mu)
        g_lv = self.logvar_layer.backward(grad_logvar)
        g    = g_mu + g_lv
        g    = self.enc2.backward(g)
        self.enc1.backward(g)

        # ─── Adam updates ─────────────────────────────────────────────────────
        for layer in self.all_layers:
            layer.adam_update(lr, t)

    def train(self, X_normal, epochs=100, batch_size=256, lr=1e-3,
              print_every=10):
        n = X_normal.shape[0]
        t = 0

        for epoch in range(1, epochs + 1):
            # Shuffle
            idx = np.random.permutation(n)
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0

            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                x_batch   = X_normal[batch_idx]
                t += 1

                x_recon, mu, logvar, z = self.forward(x_batch)
                loss, recon_loss, kl_loss = self.loss(x_batch, x_recon, mu, logvar)
                self.backward(x_batch, x_recon, mu, logvar, z, lr, t)

                epoch_loss  += loss
                epoch_recon += recon_loss
                epoch_kl    += kl_loss

            n_batches = max(1, n // batch_size)
            self.history['loss'].append(epoch_loss / n_batches)
            self.history['recon_loss'].append(epoch_recon / n_batches)
            self.history['kl_loss'].append(epoch_kl / n_batches)

            if epoch % print_every == 0:
                print(f"  Epoch {epoch:3d}/{epochs} | "
                      f"Loss={self.history['loss'][-1]:.4f} | "
                      f"Recon={self.history['recon_loss'][-1]:.4f} | "
                      f"KL={self.history['kl_loss'][-1]:.4f}")

        return self

    def reconstruction_error(self, X):
        """計算每筆樣本的重建誤差（偵測指標之一）"""
        x_recon, mu, logvar, z = self.forward(X)
        return np.mean((X - x_recon)**2, axis=1)

    def encode_mean(self, X):
        """取得 μ（潛在空間均值，用於視覺化與統計檢定）"""
        mu, _ = self.encode(X)
        return mu


# ─── 視覺化 ───────────────────────────────────────────────────────────────────

def visualize_vae(vae, X_train_normal, X_val, y_val,
                  save_path='output/phase3_vae.png'):
    os.makedirs('output', exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor('#0f1117')
    for ax in axes.flat:
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#8888aa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    epochs = range(1, len(vae.history['loss']) + 1)

    # 1. Total Loss
    axes[0,0].plot(epochs, vae.history['loss'], color='#4488ff', linewidth=2)
    axes[0,0].set_title('Total ELBO Loss', color='#eeeeff')
    axes[0,0].set_xlabel('Epoch', color='#ccccdd')

    # 2. Recon + KL
    axes[0,1].plot(epochs, vae.history['recon_loss'], color='#44ff88', linewidth=2, label='Recon')
    axes[0,1].plot(epochs, vae.history['kl_loss'],    color='#ffaa22', linewidth=2, label='KL')
    axes[0,1].set_title('Reconstruction vs KL Loss', color='#eeeeff')
    axes[0,1].legend(facecolor='#1a1d27', labelcolor='#ccccdd')

    # 3. Latent Space μ 分佈（前2維）
    mu_normal = vae.encode_mean(X_val[y_val == 0])
    mu_fraud  = vae.encode_mean(X_val[y_val == 1])
    axes[0,2].scatter(mu_normal[:200, 0], mu_normal[:200, 1],
                      alpha=0.5, s=15, color='#4488ff', label='Normal')
    axes[0,2].scatter(mu_fraud[:, 0], mu_fraud[:, 1],
                      alpha=0.7, s=25, color='#ff4466', label='Fraud', marker='x')
    axes[0,2].set_title('Latent Space μ (dim 0 vs 1)', color='#eeeeff')
    axes[0,2].legend(facecolor='#1a1d27', labelcolor='#ccccdd')

    # 4. 重建誤差分佈
    recon_normal = vae.reconstruction_error(X_val[y_val == 0])
    recon_fraud  = vae.reconstruction_error(X_val[y_val == 1])
    axes[1,0].hist(recon_normal, bins=50, alpha=0.7, color='#4488ff',
                   label='Normal', density=True)
    axes[1,0].hist(recon_fraud,  bins=50, alpha=0.7, color='#ff4466',
                   label='Fraud',  density=True)
    axes[1,0].set_title('Reconstruction Error Distribution', color='#eeeeff')
    axes[1,0].set_xlabel('MSE', color='#ccccdd')
    axes[1,0].legend(facecolor='#1a1d27', labelcolor='#ccccdd')

    # 5. 重建誤差 CDF
    from scipy import stats
    x_range = np.linspace(0, max(recon_normal.max(), recon_fraud.max()), 200)
    axes[1,1].plot(np.sort(recon_normal),
                   np.linspace(0, 1, len(recon_normal)),
                   color='#4488ff', linewidth=2, label='Normal')
    axes[1,1].plot(np.sort(recon_fraud),
                   np.linspace(0, 1, len(recon_fraud)),
                   color='#ff4466', linewidth=2, label='Fraud')
    axes[1,1].set_title('Reconstruction Error CDF', color='#eeeeff')
    axes[1,1].set_xlabel('MSE', color='#ccccdd')
    axes[1,1].legend(facecolor='#1a1d27', labelcolor='#ccccdd')

    # 6. KL Divergence per sample（正常 vs 詐欺）
    def kl_per_sample(vae, X):
        mu, logvar = vae.encode(X)
        return 0.5 * np.sum(np.exp(logvar) + mu**2 - 1 - logvar, axis=1)

    kl_n = kl_per_sample(vae, X_val[y_val == 0])
    kl_f = kl_per_sample(vae, X_val[y_val == 1])
    axes[1,2].hist(kl_n, bins=50, alpha=0.7, color='#4488ff', label='Normal', density=True)
    axes[1,2].hist(kl_f, bins=50, alpha=0.7, color='#ff4466', label='Fraud',  density=True)
    axes[1,2].set_title('KL Divergence per Sample', color='#eeeeff')
    axes[1,2].set_xlabel('KL(q||p)', color='#ccccdd')
    axes[1,2].legend(facecolor='#1a1d27', labelcolor='#ccccdd')

    plt.suptitle('Phase 3: VAE Latent Space Learning',
                 color='#eeeeff', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Phase 3] 圖表已儲存至 {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 3: 訓練 VAE 監控器")
    print("=" * 60)

    X_train = np.load('output/X_train.npy')
    X_val   = np.load('output/X_val.npy')
    y_train = np.load('output/y_train.npy')
    y_val   = np.load('output/y_val.npy')

    # 只用正常樣本訓練 VAE
    X_train_normal = X_train[y_train == 0]
    print(f"VAE 訓練資料（僅正常樣本）: {X_train_normal.shape}")
    print(f"輸入維度: {X_train_normal.shape[1]}, 潛在維度: 8")

    vae = VAE(
        input_dim  = X_train_normal.shape[1],
        hidden_dim = 64,
        latent_dim = 8,
        beta       = 2.0    # β > 1 增強 KL 約束，潛在空間更接近 N(0,I)
    )

    print("\n開始訓練 VAE...")
    vae.train(X_train_normal, epochs=100, batch_size=256, lr=3e-4, print_every=20)

    # 評估重建誤差區分能力
    recon_normal = vae.reconstruction_error(X_val[y_val == 0])
    recon_fraud  = vae.reconstruction_error(X_val[y_val == 1])
    print(f"\n重建誤差統計：")
    print(f"  Normal: mean={recon_normal.mean():.4f}, std={recon_normal.std():.4f}")
    print(f"  Fraud:  mean={recon_fraud.mean():.4f},  std={recon_fraud.std():.4f}")

    visualize_vae(vae, X_train_normal, X_val, y_val)

    # 儲存 VAE
    with open('output/vae.pkl', 'wb') as f:
        pickle.dump(vae, f)

    # 儲存潛在表示
    mu_train = vae.encode_mean(X_train)
    mu_val   = vae.encode_mean(X_val)
    np.save('output/mu_train.npy', mu_train)
    np.save('output/mu_val.npy',   mu_val)

    X_new = np.load('output/X_new.npy')
    mu_new = vae.encode_mean(X_new)
    np.save('output/mu_new.npy', mu_new)

    print(f"\n[Phase 3] 完成！潛在空間維度: {vae.latent_dim}")

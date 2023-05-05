import numpy as np
import matplotlib.pyplot as plt

# 初期値
phi0 = 1.0
phi1 = 1.5

# 時間間隔とシミュレーション時間
dt = 0.01
t_end = 100

# シミュレーションに必要なステップ数
n_steps = int(t_end / dt)

# 初期化
phi = np.zeros((n_steps, 2))
phi[0] = [phi0, phi1]

# ルンゲクッタ法によるシミュレーション
for i in range(n_steps-1):
    t = i * dt
    k1 = dt * np.array([
        11 + np.sin(phi[i, 1] - phi[i, 0]),
        10 + np.sin(phi[i, 0] - phi[i, 1])
    ])
    k2 = dt * np.array([
        11 + np.sin(phi[i, 1] - phi[i, 0] + k1[0]/2),
        10 + np.sin(phi[i, 0] - phi[i, 1] + k1[1]/2)
    ])
    k3 = dt * np.array([
        11 + np.sin(phi[i, 1] - phi[i, 0] + k2[0]/2),
        10 + np.sin(phi[i, 0] - phi[i, 1] + k2[1]/2)
    ])
    k4 = dt * np.array([
        11 + np.sin(phi[i, 1] - phi[i, 0] + k3[0]),
        10 + np.sin(phi[i, 0] - phi[i, 1] + k3[1])
    ])
    phi[i+1] = phi[i] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

# プロット
plt.plot(phi[:,0], label=r"$\phi_0$")
plt.plot(phi[:,1], label=r"$\phi_1$")
plt.legend()
plt.show()

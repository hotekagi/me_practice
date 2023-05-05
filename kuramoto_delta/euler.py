import numpy as np
import matplotlib.pyplot as plt

# 初期値
psi = 1.0

# 時間間隔とシミュレーション時間
dt = 0.01
t_end = 10

# シミュレーションに必要なステップ数
n_steps = int(t_end / dt)

# 初期化
psi_arr = np.zeros(n_steps)
psi_arr[0] = psi

# オイラー法によるシミュレーション
for i in range(n_steps-1):
    psi_arr[i+1] = psi_arr[i] + dt * (1 - 2*np.sin(psi_arr[i]))

# プロット
plt.plot(psi_arr)
plt.xlabel("time")
plt.ylabel(r"$\psi$")

# plt.axhline(0.7, color="k", linestyle="--")
plt.show()

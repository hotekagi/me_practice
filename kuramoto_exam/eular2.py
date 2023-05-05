import numpy as np
import matplotlib.pyplot as plt

# 相互作用強度
K = 10

def get_Omegas(omega0, omega1=10):
    """
    入力：\omega_0, \omega_1
    出力：(\Omega_0 - \Omega_1, \omega_0 - \omega_1)
    """
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

    # オイラー法によるシミュレーション
    for i in range(n_steps-1):
        dphi = np.array([
            omega0 + K * np.sin(phi[i, 1] - phi[i, 0]),
            omega1 + K * np.sin(phi[i, 0] - phi[i, 1])
        ])
        phi[i+1] = phi[i] + dt * dphi

    Omega0 = (phi[n_steps-2, 0] - phi[n_steps // 2, 0]) / (n_steps-2 - n_steps // 2) / dt
    Omega1 = (phi[n_steps-2, 1] - phi[n_steps // 2, 1]) / (n_steps-2 - n_steps // 2) / dt
    print(Omega0 - Omega1)

    return Omega0 - Omega1, omega0 - omega1


# \omega_0を動かす
omega0_arr = np.linspace(-40, 60, 500)

dOmega_domega_arr = np.zeros((len(omega0_arr), 2))

for i, omega0 in enumerate(omega0_arr):
    dOmega_domega_arr[i] = get_Omegas(omega0)

plt.scatter(dOmega_domega_arr[:, 1], dOmega_domega_arr[:, 0], marker=".")

plt.ylabel(r"$\Omega_0 - \Omega_1$")
plt.xlabel(r"$\omega_0 - \omega_1$")
plt.axvline(0, linestyle="--")
plt.axhline(0, linestyle="--")
plt.savefig(f"mod_t_omegas_K{K}_wide.png")

plt.show()

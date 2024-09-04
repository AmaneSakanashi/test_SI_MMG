import numpy as np
import random


# x = [x, u, y, v, psi, r]^T
# u = [delta(t), n(t), nB(t)]^T
# tf:最終時刻
# tau : 無次元化した時間
# x(0) = x_0 : 初期条件のベクトル
# x(1) = x_1 : 最終条件のベクトル
# p_i : 船を長方形で囲んだ場合の頂点
# y_i : p_iのy座標
# y_birth : 桟橋のy座標 = 0
# w_1 : 調整可能な重み = 5.0 * 10^4
# w_2 : 調整可能な重み = 1.0 * 10^5
# |x_i(1) - x_1_i| ≤ x_tol_i for i = 1,2,...,6
# x_tol = [0.01m, 0.01m/s, 0.01m, 0.01m/s, 0.005rad, 0.005rad/s]
# tau_s = t_s / tf : プロペラの回転方向変換の頻度を制限するための時間
# n(tau) ← |n(tau)| for tau ≤ tau_s
# n(tau) ← -|n(tau)| for tau_s ≤ tau
# X = [tf, ts, delta_1, ... ,delta_m, n_1, ...,n_m, n_B_1, ...,n_B_m] : 最適化のための未知変数ベクトル

def random_action_genarator():
    delta_p = random.random() * (delta_max - delta_min) + delta_min
    delta_s = 0
    n = random.random() * (n_max - n_min) + n_min
    n_B = random.random() * (n_B_max - n_B_min) + n_B_min
    u = np.array([
        delta_p * np.pi/180,
        delta_s*np.pi/180,
        n,
        n_B
    ])
    # print(f"action", u)

    return u

def random_action_generator_esso():
    delta = random.random() * (delta_max - delta_min) + delta_min
    # delta = -35
    n = random.random() * (n_max - n_min) + n_min
    # n = -10
    u = np.array([
        delta,
        n
    ])
    
    return u


delta_min = -35
delta_max = 35
# delta_range = [delta_min, delta_max] 
n_min = -20
n_max = 20
# n_range =[n_min, n_max]
n_B_min = 0
n_B_max = 20
# n_B_range = [n_B_min, n_B_max]
    

# action = random_action_genarator()
# print(f"action", action)

        
# x_1 = [1, 1, 1, 1, 1, 1]
# x_tol = [0.01, 0.01, 0.01, 0.01, 0.005, 0.005]
    
# J_2(x_hat, x_1, x_tol, SIM_TIME, w_2=10, C)



import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

def solve_damped_spring_motion(t,f_data,initial_state):
    # ==========================================
    # 1. パラメータ設定
    # ==========================================
    m = 0.1       # 質量 [kg]
    k = 90.0      # ばね定数 [N/m]
    c = 1.8       # ★追加: 粘性減衰係数 [N*s/m] (値を大きくすると早く止まります)
    
    omega0_sq = k / m  # k/m
    damping_term = c / m # c/m

    # 補間関数の作成
    f_interpolated = interp1d(t, f_data, kind='linear', fill_value="extrapolate")

    # ==========================================
    # 3. 運動方程式の定義 (減衰あり)
    # ==========================================
    def model(state, t):
        u, v = state # u: 変位, v: 速度
        forcing = f_interpolated(t)
        
        dudt = v
        # ★変更: 減衰項 (- damping_term * v) を追加
        # dv/dt = (k/m)*(f - u) - (c/m)*v
        dvdt = omega0_sq * (forcing - u) - damping_term * v
        
        return [dudt, dvdt]

    # ==========================================
    # 4. 数値計算の実行
    # ==========================================

    solution = odeint(model, initial_state, t)
    u_t = solution[:, 0]
    return u_t

def resonate(t,u_data):
    # ==========================================
    # System 2: 共振系（入力 u(t) -> 出力 x(t)）
    # ==========================================
    # 設定: omega_1 を System 1 の固有振動数に近づけて "共振" させてみます
    # System 1 の固有振動数
    omega_1 = 50  
    zeta_1 = 0.15   # 減衰比 (小さいほどよく響く)
    
    # System 2 の運動方程式係数
    # x'' + 2*zeta*w1 * x' + w1^2 * x = w1^2 * u_in
    w1_sq = omega_1**2
    damping_coef2 = 2 * zeta_1 * omega_1

    # ★重要: System 1 の出力 u_data を System 2 の入力関数にする
    u_func = interp1d(t, u_data, kind='linear', fill_value="extrapolate")

    def model_sys2(state, t):
        x, v = state
        input_val = u_func(t) # System 1 の計算結果を入力として取得
        
        dxdt = v
        dvdt = w1_sq * (input_val - x) - damping_coef2 * v
        return [dxdt, dvdt]

    # System 2 を解く
    sol2 = odeint(model_sys2, [0, 0], t)
    x_data = sol2[:, 0] # これが最終的な出力
    return x_data

def maximize(x,a=1):
    return 2*np.arctan(x*a)/np.pi

def add_dumping(x,maxdiff=20):
    t = np.arange(len(x))*0.005
    initial_state = [0.0, 0.0]
    u_t = solve_damped_spring_motion(t,x,initial_state)
    u_t = resonate(t,u_t)
    for i in range(len(u_t)):
        if x[i] == 0:
            u_t[i] = 0
    y = u_t - x
    ymax = np.max(np.abs(y))
    if ymax > maxdiff:
        y = maximize(y/ymax,2.0)*maxdiff

    return x+y

if __name__ == "__main__":
    import matplotlib.pyplot as plt

        # 時間設定 (0秒から20秒まで)
    t = np.linspace(0, 20, 1000)

    # ==========================================
    # 2. 入力信号 f(t) の作成
    # ==========================================
    # 今回は挙動が見やすいよう「ステップ入力」にします
    # (1秒後に支点を 1.0m 持ち上げて、そのまま固定)
    f_data = np.zeros_like(t)
    f_data[0:100] = 50
    f_data[100:200] = 100
    f_data[200:300] = 150
    f_data[300:400] = 100
    f_data[400:500] = 0
    f_data[500:600] = 100
    f_data[600:700] = 110
    f_data[700:800] = 80
    u_t = add_dumping(f_data)
    # ==========================================
    # 5. 結果の可視化
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # 支点の動き
    plt.plot(t, f_data, 'r--', label='Support (Input) $f(t)$', alpha=0.6)
    
    # おもりの動き
    plt.plot(t, u_t, 'b-', label='Mass (Output) $u(t)$', linewidth=2)
    
    plt.title(f'Damped Vibration')
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

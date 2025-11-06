import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import math
from tqdm import tqdm
import time

# 系統參數設定
class SystemParameters:
    def __init__(self):
        self.BS_COUNT = 37           # 基地台數量
        self.RADIUS = 500            # 基地台間距(m)
        self.UE_COUNT = 100          # 使用者數量
        self.LAMBDA = 0.1            # 封包到達率(封包/ms)

        # 噪聲：使用噪聲功率密度 (dBm/Hz) 與頻寬 (Hz) 來計算總噪聲
        self.NOISE_POWER_DENSITY = -174.0  # dBm/Hz (thermal noise density)
        self.NOISE_FIGURE = 5.0            # dB，接收端雜訊因子 (可調)
        self.BANDWIDTH = 10e6              # Hz，頻寬 (預設 10 MHz)

        self.BS_POWER = 43         # 基地台發射功率(dBm)
        self.PATH_LOSS_EXP = 3.5   # 路徑損耗指數

        # 新增：封包相關參數
        self.PACKET_SIZE_BITS = 1500 * 8  # 1500 bytes = 12000 bits
        self.TIMESLOT_DURATION_SEC = 0.005  # 1 ms per timeslot
        
        # SINR 門檻 (dB)
        self.SINR_THRESHOLD_DB = 0.0
        
        # Shannon capacity 效率因子（實際系統約 0.6~0.8，考慮編碼、控制開銷等）
        self.SPECTRAL_EFFICIENCY_FACTOR = 0.7

    def noise_power_dbm(self) -> float:
        """
        計算總噪聲功率 (dBm):
        noise (dBm) = noise_density (dBm/Hz) + 10*log10(BANDWIDTH) + noise_figure (dB)
        """
        return self.NOISE_POWER_DENSITY + 10.0 * math.log10(self.BANDWIDTH) + self.NOISE_FIGURE

    def sinr_to_capacity_bps(self, sinr_db: float) -> float:
        """
        使用 Shannon capacity 將 SINR(dB) 轉為位元率(bps)
        C = B * log2(1 + SNR) * efficiency_factor
        """
        sinr_linear = 10.0 ** (sinr_db / 10.0)
        capacity = self.BANDWIDTH * math.log2(1.0 + sinr_linear) * self.SPECTRAL_EFFICIENCY_FACTOR
        return capacity
    
    def capacity_to_max_packets(self, capacity_bps: float) -> int:
        """
        將位元率轉為每時隙最大可處理封包數
        max_packets = floor(capacity * timeslot_duration / packet_size)
        """
        bits_per_slot = capacity_bps * self.TIMESLOT_DURATION_SEC
        max_packets = int(bits_per_slot / self.PACKET_SIZE_BITS)
        return max(1, max_packets)  # 至少能處理 1 個封包
    
# --- coordinate transforms (axial -> cartesian) ---
def axial_to_xy(q: int, r: int, radius: float) -> tuple[float, float]:
    """
    pointy-top 六邊形坐標轉換。
    radius: 期望相鄰中心在垂直方向的距離 (例如 500 m)
    """
    # 六邊形邊長 s，滿足 s*sqrt(3) = radius
    s = radius / math.sqrt(3.0)

    x = s * 1.5 * q
    y = s * math.sqrt(3.0) * (r + 0.5 * q)
    return x, y


# === 四層六邊形基地台佈局 ===
def generate_hex_grid(layers: int | None = None) -> np.ndarray:
    """
    產生以 (0,0) 為中心的對稱六邊形基地台佈局。
    想要「四層」 → axial 半徑 R = 3。
    """
    params = SystemParameters()

    # 固定用 4 層（R=3），如果你要之後改成可調就改這裡
    if layers is None:
        layers = 3   # center + 3 個 ring

    R = layers
    centers: list[tuple[float, float]] = []

    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            s_coord = -q - r
            # 這個條件保證點落在「六角形形狀」裡
            if max(abs(q), abs(r), abs(s_coord)) <= R:
                x, y = axial_to_xy(q, r, params.RADIUS)
                centers.append((x, y))

    # 確保只回傳 BS_COUNT 個（37 個）基地台位置
    centers = centers[: params.BS_COUNT]

    return np.array(centers)

# 使用者位置生成
def generate_ue_positions(params):
    area_radius = 1500  # 覆蓋區域半徑
    angles = np.random.uniform(0, 2*np.pi, params.UE_COUNT)
    # 使用 sqrt 以達成面積上均勻分布
    radii = area_radius * np.sqrt(np.random.uniform(0, 1, params.UE_COUNT))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

# SINR計算 (clamp distances to avoid log10(0))
def calculate_sinr(ue_pos, bs_positions, active_bs, params):
    distances = np.sqrt(np.sum((bs_positions - ue_pos)**2, axis=1))
    distances = np.maximum(distances, 1e-3)  # avoid log10(0)
    path_loss = 10 * params.PATH_LOSS_EXP * np.log10(distances)
    received_power = params.BS_POWER - path_loss

    # 只考慮開啟的基地台
    received_power[~active_bs] = -np.inf

    # 若所有 BS 都關閉，回傳無服務
    if np.all(np.isneginf(received_power)):
        return -np.inf, -1

    # 找最大接收功率的 BS（serving）
    max_power_idx = int(np.argmax(received_power))

    # 線性域計算（dBm -> mW），-inf 會對應到 0
    linear_p = 10.0 ** (received_power / 10.0)
    signal = linear_p[max_power_idx]
    # 干擾為總和扣除主訊號（mW）
    interference = np.sum(linear_p) - signal

    # 噪聲（dBm -> mW）
    noise_dbm = params.noise_power_dbm()
    noise = 10.0 ** (noise_dbm / 10.0)

    # 加小常數避免除以 0 或 log10(0)
    eps = 1e-30
    sinr_linear = signal / (interference + noise + eps)
    sinr = 10.0 * np.log10(sinr_linear + eps)

    return sinr, max_power_idx

# 能耗模型
def calculate_power_consumption(active_bs, traffic_load):
    P_sleep = 10.0           # 休眠功耗 (W)
    P_active_base = 50.0     # 活躍但空載時基本功耗 (W)
    P_active_load_coeff = 100.0  # 負載係數 (W per normalized load)
    # traffic_load 預期為 [0,1] 的 normalized load
    active_power = P_active_base + P_active_load_coeff * traffic_load
    total_power = np.sum(active_bs * active_power) + np.sum((~active_bs) * P_sleep)
    return total_power

# Visualize Hexagonal Base Station Layout (helper)
def visualize_bs_layout(bs_positions, annotate=False):
    plt.figure(figsize=(8, 8))
    bs_positions = np.asarray(bs_positions)
    plt.scatter(bs_positions[:, 0], bs_positions[:, 1], c='blue', marker='o', label='Base Stations')
    if annotate:
        for i, (x, y) in enumerate(bs_positions):
            plt.text(x, y, str(i), fontsize=8, ha='center', va='center')
    plt.title('Hexagonal Base Station Layout')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.axhline(0, color='grey', lw=0.5, ls='--')
    plt.axvline(0, color='grey', lw=0.5, ls='--')
    plt.grid(True, ls=':')
    plt.axis('equal')
    plt.legend()
    plt.show()



def _compute_grid_signal_and_sinr(params, bs_positions, active_bs, area_radius=2000, grid_res=200):
    """
    回傳格點 (X,Y)、best_rssi_db (dBm)、sinr_db (dB)。
    area_radius: 覆蓋半徑 (m)
    grid_res: 每邊格點數 (解析度)，值越大越細但越慢
    """
    # 建格點
    xs = np.linspace(-area_radius, area_radius, grid_res)
    ys = np.linspace(-area_radius, area_radius, grid_res)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack((X.ravel(), Y.ravel()))

    # 計算每個格點對每個 BS 的距離 -> received power (dBm)
    bs_pos = np.asarray(bs_positions)
    diff = pts[:, None, :] - bs_pos[None, :, :]         # (Npoints, Nbs, 2)
    dists = np.sqrt(np.sum(diff**2, axis=2))           # (Npoints, Nbs)
    dists = np.maximum(dists, 1e-3)
    path_loss = 10.0 * params.PATH_LOSS_EXP * np.log10(dists)
    recv_power = params.BS_POWER - path_loss           # dBm

    # 關閉的 BS 設 -inf
    recv_power[:, ~active_bs] = -np.inf

    # best RSSI (dBm)
    best_rssi_db = np.max(recv_power, axis=1)          # (Npoints,)

    # 計算 SINR：信號=最大，干擾=其餘線性和，噪聲使用 params.noise_power_dbm()
    # 先把 -inf 轉成一個非常小值在線性域會為 0
    linear = 10.0**(recv_power/10.0)                    # mW
    signal = np.max(linear, axis=1)
    total = np.sum(linear, axis=1)
    interference = total - signal
    noise_mw = 10.0**(params.noise_power_dbm()/10.0)
    sinr_linear = signal / (interference + noise_mw + 1e-30)
    sinr_db = 10.0 * np.log10(sinr_linear + 1e-30)

    return X, Y, best_rssi_db.reshape(X.shape), sinr_db.reshape(X.shape)


def visualize_signal_and_ues(params, bs_positions, ue_positions, active_bs=None,
                             area_radius=2000, grid_res=200, show_sinr=True):
    """
    畫出區域內的 RSSI（或 SINR）熱圖，並疊加 BS 與 UE 位置。
    - params: SystemParameters instance
    - active_bs: 布林陣列，長度為 params.BS_COUNT；若 None 則視為全部啟用
    - grid_res: 解析度 (建議 100~300)
    """
    if active_bs is None:
        active_bs = np.ones(params.BS_COUNT, dtype=bool)

    X, Y, rssi_db, sinr_db = _compute_grid_signal_and_sinr(params, bs_positions, active_bs,
                                                           area_radius=area_radius, grid_res=grid_res)

    plt.figure(figsize=(12, 5))
    # RSSI 圖
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(rssi_db[::-1, :], extent=[-area_radius, area_radius, -area_radius, area_radius],
                     cmap='viridis', aspect='equal')
    plt.colorbar(im1, label='Best RSSI (dBm)')
    plt.scatter(bs_positions[:, 0], bs_positions[:, 1], c='red', marker='^', edgecolor='k', label='BS')
    plt.scatter(ue_positions[:, 0], ue_positions[:, 1], c='black', s=8, alpha=0.6, label='UE')
    plt.title('Best RSSI map and UE distribution')
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.legend(loc='upper right')

    # SINR 圖
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(sinr_db[::-1, :], extent=[-area_radius, area_radius, -area_radius, area_radius],
                     cmap='plasma', aspect='equal', vmin=-30, vmax=30)
    plt.colorbar(im2, label='SINR (dB)')
    plt.scatter(bs_positions[:, 0], bs_positions[:, 1], c='red', marker='^', edgecolor='k', label='BS')
    plt.scatter(ue_positions[:, 0], ue_positions[:, 1], c='black', s=8, alpha=0.6, label='UE')
    plt.title('SINR map and UE distribution')
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def visualize_ue_distribution(ue_positions, area_radius=1500):
    """
    單純畫 UE 分佈（散佈圖）與覆蓋圓（可選）。
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(ue_positions[:, 0], ue_positions[:, 1], c='black', s=12, alpha=0.7)
    circle = plt.Circle((0, 0), area_radius, color='gray', fill=False, ls='--')
    plt.gca().add_patch(circle)
    plt.title('UE Distribution')
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True, ls=':')
    plt.show()


# 主模擬函數
def simulate(params, bs_positions, ue_positions, active_bs, simulation_time=1000):
    """
    每個時隙先蒐集各 UE 的需求與所屬 BS，再依每個 BS 的容量決定實際可處理（allocated）封包數。
    成功封包需同時 (1) 被分配處理位置 (allocated) 且 (2) UE SINR > SINR_THRESHOLD_DB。
    """
    N_bs = params.BS_COUNT
    N_ue = params.UE_COUNT

    traffic_load = np.zeros(N_bs)  # 累積每個 BS 實際處理的封包數（用於能耗）
    successful_transmissions = 0
    total_attempts = 0

    
    for t in range(simulation_time):
        # 生成封包(Poisson分布)
        packets = poisson.rvs(params.LAMBDA, size=N_ue)
        total_attempts += packets.sum()

        # 先找出每個 UE 的 serving BS 與 SINR
        ue_serving = -np.ones(N_ue, dtype=int)
        ue_sinr = np.full(N_ue, -np.inf)
        for i, ue_pos in enumerate(ue_positions):
            if packets[i] > 0:
                sinr, serving = calculate_sinr(ue_pos, bs_positions, active_bs, params)
                ue_serving[i] = serving
                ue_sinr[i] = sinr

        # 對每個 BS 聚合需求並依容量分配
        for bs in range(N_bs):
            if not active_bs[bs]:
                continue
            idxs = np.where((ue_serving == bs) & (packets > 0))[0]
            if idxs.size == 0:
                continue

            # 計算此 BS 的總可用容量（使用所有連接 UE 的平均 SINR）
            avg_sinr = np.mean(ue_sinr[idxs])
            if avg_sinr > params.SINR_THRESHOLD_DB:
                bs_capacity_bps = params.sinr_to_capacity_bps(avg_sinr)
                bs_max_packets = params.capacity_to_max_packets(bs_capacity_bps)
            else:
                bs_max_packets = 0

            requested = int(packets[idxs].sum())
            
            # 依容量分配
            if requested <= bs_max_packets:
                # 未超載：所有需求都能處理
                handled_per_ue = packets[idxs].astype(int)
            else:
                # 超載：按比例分配
                if requested > 0:
                    frac_alloc = packets[idxs] * (bs_max_packets / float(requested))
                    base_alloc = np.floor(frac_alloc).astype(int)
                    rem = bs_max_packets - base_alloc.sum()
                    if rem > 0:
                        fracs = frac_alloc - base_alloc
                        order = np.argsort(-fracs)
                        for j in order[:int(rem)]:
                            base_alloc[j] += 1
                    handled_per_ue = base_alloc
                else:
                    handled_per_ue = np.zeros_like(packets[idxs], dtype=int)


            # 被分配到的封包如果 UE SINR 達標則視為成功
            for k, ue_idx in enumerate(idxs):
                alloc = int(handled_per_ue[k])
                if alloc <= 0:
                    continue
                if ue_sinr[ue_idx] > params.SINR_THRESHOLD_DB:
                    successful_transmissions += alloc

            # 累積實際處理量（用於能耗）
            traffic_load[bs] += handled_per_ue.sum()
    
    # 計算性能指標
    success_rate = successful_transmissions / total_attempts if total_attempts > 0 else 0

    # 把 traffic_load 轉為每個 BS 的平均封包率（per timestep）
    avg_load_per_bs = traffic_load / float(simulation_time)

    # 預期最大平均負載（用來正規化到 [0,1]）：假設均勻分配
    expected_total_packets_per_timestep = params.LAMBDA * params.UE_COUNT
    expected_per_bs = max(1e-9, expected_total_packets_per_timestep / params.BS_COUNT)
    normalized_load = np.minimum(avg_load_per_bs / expected_per_bs, 1.0)

    power_consumption = calculate_power_consumption(active_bs, normalized_load)
    
    return success_rate, power_consumption

# 主程式
def main():
    params = SystemParameters()
    bs_positions = generate_hex_grid()
    
    print("初始化模擬環境...")
    ue_positions = generate_ue_positions(params)
    
    print("繪製基地台佈局...")
    visualize_bs_layout(bs_positions, annotate=True)
    
    print("繪製使用者分佈...")
    visualize_ue_distribution(ue_positions)
    
    print("計算並繪製訊號覆蓋圖...")
    visualize_signal_and_ues(params, bs_positions, ue_positions, area_radius=2000, grid_res=200)

    # 測試不同的休眠策略（多次實驗取平均）
    strategies = []
    rng = np.random.default_rng(12345)
    n_experiments = 10  # 每個 ratio 重複次數
    active_ratios = np.arange(0.1, 1.1, 0.1)

    # prepare order by distance-from-center (ring-based)
    order_by_center = np.argsort(np.sum(bs_positions**2, axis=1))

    print(f"\n開始模擬 {len(active_ratios)} 個啟用比例，每個重複 {n_experiments} 次...")
    start_time = time.time()

    # 使用 tqdm 顯示外層迴圈進度
    for active_ratio in tqdm(active_ratios, desc="模擬進度"):
        # 收集這個 ratio 的多次實驗結果
        exp_success_rates = []
        exp_powers = []
        
        for _ in range(n_experiments):
            # 使用 ring-based 啟用：選取最近的 n_active 個 BS 確保覆蓋連續性
            n_active = max(1, int(round(active_ratio * params.BS_COUNT)))
            active_bs = np.zeros(params.BS_COUNT, dtype=bool)
            active_bs[order_by_center[:n_active]] = True
            
            # 每次實驗重新生成 UE 位置
            ue_positions = generate_ue_positions(params)
            
            success_rate, power = simulate(params, bs_positions, ue_positions, active_bs)
            exp_success_rates.append(success_rate)
            exp_powers.append(power)
        
        # 計算平均值和標準差
        avg_success = np.mean(exp_success_rates)
        std_success = np.std(exp_success_rates)
        avg_power = np.mean(exp_powers)
        std_power = np.std(exp_powers)


        strategies.append({
            'active_ratio': active_ratio,
            'success_rate': avg_success,
            'success_std': std_success,
            'power': avg_power,
            'power_std': std_power
        })
    total_time = time.time() - start_time
    print(f"\n模擬完成! 總耗時: {total_time:.2f} 秒")
    print("繪製結果...")
    # 結果視覺化（含誤差範圍）
    plt.figure(figsize=(12, 5))
    active_ratios_plot = [s['active_ratio'] for s in strategies]
    success_rates = [s['success_rate'] for s in strategies]
    success_stds = [s['success_std'] for s in strategies]
    powers = [s['power'] for s in strategies]
    power_stds = [s['power_std'] for s in strategies]
    
    plt.subplot(1, 2, 1)
    plt.errorbar(active_ratios_plot, success_rates, yerr=success_stds, 
                fmt='b-o', capsize=3, alpha=0.7, label='Mean ± Std')
    plt.fill_between(active_ratios_plot, 
                    np.array(success_rates) - np.array(success_stds),
                    np.array(success_rates) + np.array(success_stds),
                    color='b', alpha=0.2)
    plt.xlabel('Active BS Ratio')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs Active BS Ratio\n(with Shannon Capacity)')
    plt.grid(True, ls=':')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.errorbar(active_ratios_plot, powers, yerr=power_stds,
                fmt='r-o', capsize=3, alpha=0.7, label='Mean ± Std')
    plt.fill_between(active_ratios_plot,
                    np.array(powers) - np.array(power_stds),
                    np.array(powers) + np.array(power_stds),
                    color='r', alpha=0.2)
    plt.xlabel('Active BS Ratio')
    plt.ylabel('Power Consumption (W)')
    plt.title('Power Consumption vs Active BS Ratio')
    plt.grid(True, ls=':')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
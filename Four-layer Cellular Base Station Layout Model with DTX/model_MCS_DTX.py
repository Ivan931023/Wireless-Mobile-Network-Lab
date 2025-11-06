import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from tqdm import tqdm
import math
import time

# 系統參數設定
class SystemParameters:
    def __init__(self):
        self.BS_COUNT = 37           # 基地台數量
        self.RADIUS = 500            # 基地台間距(m)
        self.UE_COUNT = 100          # 使用者數量
        self.LAMBDA = 0.05            # 封包到達率(封包/ms)
        self.CAPACITY_MODE = 'share'     # 'avg' | 'sum' | 'share'
        self.SHARE_OVER = 'all'         # 'busy' | 'all'  ← 新增：分享集合

        # --- DTX 參數 ---
        self.ENABLE_DTX = True          # 開啟 DTX
        self.DTX_MIN_OFF_SLOTS = 1      # 最少連續關閉時隙
        self.DTX_MIN_ON_SLOTS  = 1      # 最少連續開啟時隙
        self.DTX_MODE = 'idle'          # 'idle'：僅在無請求時關閉


        # 噪聲：使用噪聲功率密度 (dBm/Hz) 與頻寬 (Hz) 來計算總噪聲
        self.NOISE_POWER_DENSITY = -174.0  # dBm/Hz (thermal noise density)
        self.NOISE_FIGURE = 5.0            # dB，接收端雜訊因子 (可調)
        self.BANDWIDTH = 10e6              # Hz，頻寬 (預設 10 MHz)

        self.BS_POWER = 43         # 基地台發射功率(dBm)
        self.PATH_LOSS_EXP = 3.5   # 路徑損耗指數

        # 新增：封包相關參數
        self.PACKET_SIZE_BITS = 1500 * 8  # 1500 bytes = 12000 bits
        self.TIMESLOT_DURATION_SEC = 0.001  # 1 ms per timeslot
        
        # SINR 門檻 (dB) - MCS 0 的最低門檻
        self.SINR_THRESHOLD_DB = -6.0
        
        # MCS 表：(SINR門檻 dB, 頻譜效率 bits/s/Hz, 調變方式)
        # 基於 LTE MCS 表簡化版本
        self.MCS_TABLE = [
            (-6.0, 0.15, 'QPSK'),    # MCS 0
            (-4.0, 0.23, 'QPSK'),    # MCS 1
            (-2.0, 0.38, 'QPSK'),    # MCS 2
            (0.0, 0.60, 'QPSK'),     # MCS 3
            (2.0, 0.88, 'QPSK'),     # MCS 4
            (4.0, 1.18, 'QPSK'),     # MCS 5
            (6.0, 1.48, 'QPSK'),     # MCS 6
            (8.0, 1.91, '16-QAM'),   # MCS 7
            (10.0, 2.41, '16-QAM'),  # MCS 8
            (11.0, 2.57, '16-QAM'),  # MCS 9
            (12.0, 2.73, '16-QAM'),  # MCS 10
            (13.0, 3.03, '64-QAM'),  # MCS 11
            (14.0, 3.32, '64-QAM'),  # MCS 12
            (15.0, 3.61, '64-QAM'),  # MCS 13
            (16.0, 3.90, '64-QAM'),  # MCS 14
            (17.0, 4.21, '64-QAM'),  # MCS 15
            (18.0, 4.52, '64-QAM'),  # MCS 16
            (19.0, 4.82, '64-QAM'),  # MCS 17
            (20.0, 5.12, '64-QAM'),  # MCS 18
            (21.0, 5.33, '64-QAM'),  # MCS 19
            (22.0, 5.55, '64-QAM'),  # MCS 20
            (24.0, 5.89, '256-QAM'), # MCS 21 (5G)
            (26.0, 6.54, '256-QAM'), # MCS 22 (5G)
            (28.0, 7.16, '256-QAM'), # MCS 23 (5G)
        ]

    def noise_power_dbm(self) -> float:
        """
        計算總噪聲功率 (dBm)：
        noise (dBm) = noise_density (dBm/Hz) + 10*log10(BANDWIDTH) + noise_figure (dB)
        """
        return self.NOISE_POWER_DENSITY + 10.0 * math.log10(self.BANDWIDTH) + self.NOISE_FIGURE
    
    def sinr_to_mcs(self, sinr_db: float) -> tuple[int, float, str]:
        """
        根據 SINR 選擇最佳 MCS
        回傳：(MCS index, 頻譜效率 bits/s/Hz, 調變方式)
        """
        if sinr_db < self.MCS_TABLE[0][0]:
            # SINR 太低，無法通訊
            return -1, 0.0, 'NONE'
        
        # 找最高但不超過當前 SINR 的 MCS
        selected_mcs = 0
        for idx, (threshold, se, mod) in enumerate(self.MCS_TABLE):
            if sinr_db >= threshold:
                selected_mcs = idx
            else:
                break
        
        mcs_info = self.MCS_TABLE[selected_mcs]
        return selected_mcs, mcs_info[1], mcs_info[2]
    
    def sinr_to_capacity_bps(self, sinr_db: float) -> float:
        """
        使用 MCS 表計算容量（更真實）
        C = B × 頻譜效率
        """
        mcs_idx, spectral_eff, mod = self.sinr_to_mcs(sinr_db)
        if mcs_idx < 0:
            return 0.0
        capacity = self.BANDWIDTH * spectral_eff
        return capacity
    
    def capacity_to_max_packets(self, capacity_bps: float) -> int:
        """
        將位元率轉為每時隙最大可處理封包數
        max_packets = floor(capacity * timeslot_duration / packet_size)
        """
        if capacity_bps <= 0:
            return 0
        bits_per_slot = capacity_bps * self.TIMESLOT_DURATION_SEC
        max_packets = int(bits_per_slot / self.PACKET_SIZE_BITS)
        return max(0, max_packets)

# --- coordinate transforms (axial -> cartesian) ---
def axial_to_xy(q: int, r: int, radius: float) -> tuple[float, float]:
    """
    將六邊形 axial 座標 (q, r) 轉為笛卡爾座標 (x, y)。
    radius 為相鄰中心間的垂直距離，對應 pointy-top 六邊形。
    """
    # 對 pointy-top：相鄰中心垂直距離 = sqrt(3) * side_length
    s = radius / math.sqrt(3.0)
    x = 1.5 * s * q
    y = math.sqrt(3.0) * s * (r + 0.5 * q)
    return (x, y)

# === 四層六邊形基地台佈局 ===
def generate_hex_grid(layers: int | None = None) -> np.ndarray:
    """
    產生以 (0,0) 為中心的對稱六邊形基地台佈局。
    想要「四層」 → axial 半徑 R = 3。
    """
    params = SystemParameters()

    # 固定用 4 層（R=3），如果你要之後改成可調就改這裡
    if layers is None:
        layers = 3

    R = layers
    centers: list[tuple[float, float]] = []

    for q in range(-R, R + 1):
        r1 = max(-R, -q - R)
        r2 = min(R, -q + R)
        for r in range(r1, r2 + 1):
            centers.append(axial_to_xy(q, r, params.RADIUS))

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

def calculate_sinr_fixed_serving(ue_pos, bs_positions, serving_idx, active_tx_bs, params):
    """
    給定 UE 的 serving BS、不發射的 BS（DTX）遮罩，計算該 UE 的 SINR。
    - serving_idx: 以原本的 cell selection 決定（不因 DTX 改變）
    - active_tx_bs: 本時隙正在發射的 BS 布林陣列
    """
    if serving_idx < 0:
        return -np.inf

    # 距離與路損
    d = np.sqrt(np.sum((bs_positions - ue_pos)**2, axis=1))
    d = np.maximum(d, 1e-3)
    path_loss = 10.0 * params.PATH_LOSS_EXP * np.log10(d)
    recv_dbm = params.BS_POWER - path_loss

    # 線性功率 (mW)
    p_mw = 10.0 ** (recv_dbm / 10.0)

    # 僅計入「正在發射」的干擾者
    tx_mask = active_tx_bs.copy()
    tx_mask[serving_idx] = True  # serving 一定在（有請求才會開啟）
    signal = p_mw[serving_idx]
    interference = np.sum(p_mw[tx_mask]) - signal

    noise_mw = 10.0 ** (params.noise_power_dbm() / 10.0)
    sinr_lin = signal / (interference + noise_mw + 1e-30)
    return 10.0 * np.log10(sinr_lin + 1e-30)

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
    plt.scatter(bs_positions[:, 0], bs_positions[:, 1], c='red', marker='^', s=100, edgecolor='k')
    if annotate:
        for i, (x, y) in enumerate(bs_positions):
            plt.text(x, y, str(i), fontsize=8, ha='center', va='bottom')
    plt.title('Base Station Layout (Hexagonal)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True, ls=':')
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
    diff = pts[:, None, :] - bs_pos[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    dists = np.maximum(dists, 1e-3)
    path_loss = 10.0 * params.PATH_LOSS_EXP * np.log10(dists)
    recv_power = params.BS_POWER - path_loss

    # 關閉的 BS 設 -inf
    recv_power[:, ~active_bs] = -np.inf

    # best RSSI (dBm)
    best_rssi_db = np.max(recv_power, axis=1)

    # 計算 SINR
    linear = 10.0**(recv_power/10.0)
    signal = np.max(linear, axis=1)
    total = np.sum(linear, axis=1)
    interference = total - signal
    noise_mw = 10.0**(params.noise_power_dbm()/10.0)
    sinr_linear = signal / (interference + noise_mw + 1e-30)
    sinr_db = 10.0 * np.log10(sinr_linear + 1e-30)

    return X, Y, best_rssi_db.reshape(X.shape), sinr_db.reshape(X.shape)

def calculate_power_consumption_with_dtx(active_bs, normalized_load, tx_duty):
    """
    三級功耗模型：
    - P_sleep: 深度睡眠（硬體關機）= 10W
    - P_idle: 輕度睡眠（DTX 關閉發射）= 30W
    - P_active: 基礎 + 負載 = 50 + 100×load W
    """
    P_sleep = 10.0
    P_idle = 30.0
    P_active_base = 50.0
    P_active_load_coeff = 100.0

    power = np.zeros(len(active_bs))
    
    for i in range(len(active_bs)):
        if not active_bs[i]:
            power[i] = P_sleep
        else:
            # 發射時功率
            p_tx = P_active_base + P_active_load_coeff * normalized_load[i]
            # 按占空比加權
            power[i] = tx_duty[i] * p_tx + (1 - tx_duty[i]) * P_idle
    
    return np.sum(power)

def visualize_signal_and_ues(params, bs_positions, ue_positions, active_bs=None,
                             area_radius=2000, grid_res=200, show_sinr=True):
    """
    畫出區域內的 RSSI（或 SINR）熱圖，並疊加 BS 與 UE 位置。
    """
    if active_bs is None:
        active_bs = np.ones(params.BS_COUNT, dtype=bool)

    X, Y, rssi_db, sinr_db = _compute_grid_signal_and_sinr(params, bs_positions, active_bs,
                                                           area_radius=area_radius, grid_res=grid_res)

    plt.figure(figsize=(12, 5))
    # RSSI 圖
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(rssi_db[::-1, :], extent=[-area_radius, area_radius, -area_radius, area_radius],
                     cmap='viridis', aspect='equal', vmin=-100, vmax=-40)
    plt.colorbar(im1, label='Best RSSI (dBm)')
    plt.scatter(bs_positions[:, 0], bs_positions[:, 1], c='red', marker='^', edgecolor='k', s=80, label='BS')
    plt.scatter(ue_positions[:, 0], ue_positions[:, 1], c='black', s=8, alpha=0.6, label='UE')
    plt.title('Best RSSI map and UE distribution')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend(loc='upper right')

    # SINR 圖
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(sinr_db[::-1, :], extent=[-area_radius, area_radius, -area_radius, area_radius],
                     cmap='plasma', aspect='equal', vmin=-30, vmax=30)
    plt.colorbar(im2, label='SINR (dB)')
    plt.scatter(bs_positions[:, 0], bs_positions[:, 1], c='red', marker='^', edgecolor='k', s=80, label='BS')
    plt.scatter(ue_positions[:, 0], ue_positions[:, 1], c='black', s=8, alpha=0.6, label='UE')
    plt.title('SINR map and UE distribution')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
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
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True, ls=':')
    plt.show()

# 主模擬函數
def simulate(params, bs_positions, ue_positions, active_bs, simulation_time=1000):
    """
    改進版模擬：使用 MCS 表計算每個 BS 的可達位元率，
    再換算成可處理封包數。考慮容量限制與 SINR 門檻。
    """
    """
    修正版模擬：
    - 先對所有 UE 計算 serving 與 SINR（不只限有封包的）
    - 每個 BS 的容量用「頻寬分享」模式：C_bs = B × mean(SE of contenders)
    - busy(有請求) 與 all(所有活躍 BS) 雙軌診斷
    """
    N_bs = params.BS_COUNT
    N_ue = params.UE_COUNT

    traffic_load = np.zeros(N_bs)
    successful_transmissions = 0
    total_attempts = 0

    # 診斷累計（雙軌）
    busy_req_tot = busy_cap_tot = busy_over_tot = busy_cnt = 0
    all_req_tot = all_cap_pot_tot = all_cnt = 0

    # 新增：UE 數量累計
    total_served_ues_sum = 0   # 所有時隙所有 BS 的連線 UE 總數
    total_busy_ues_sum = 0     # 所有時隙所有 BS 的忙碌 UE 總數
    active_bs_slots = 0        # 活躍 BS×時隙總數
    
    # 統計 MCS 使用情況
    mcs_usage = np.zeros(len(params.MCS_TABLE) + 1)  # +1 for no connection

    # --- DTX 狀態（每 BS）---
    bs_tx_state = active_bs.copy()                 # 本時隙是否發射
    on_streak  = np.zeros(N_bs, dtype=int)         # 連續 on 計數
    off_streak = np.zeros(N_bs, dtype=int)         # 連續 off 計數
    tx_on_count = np.zeros(N_bs, dtype=int)        # 累計 on 次數（算占空比）

    for t in range(simulation_time):
        packets = poisson.rvs(params.LAMBDA, size=N_ue)
        total_attempts += packets.sum()

        # 步驟 1：所有 UE 先計算 serving 與 SINR（不管有無封包）
        ue_serving = -np.ones(N_ue, dtype=int)
        ue_sinr = np.full(N_ue, -np.inf)

        for i, ue_pos in enumerate(ue_positions):
            sinr, serving = calculate_sinr(ue_pos, bs_positions, active_bs, params)
            ue_serving[i] = serving
            ue_sinr[i] = sinr

        # 步驟 2：只對有封包的 UE 記錄 MCS 使用
        for i in range(N_ue):
            if packets[i] > 0:
                if ue_serving[i] >= 0 and ue_sinr[i] > params.SINR_THRESHOLD_DB:
                    mcs_idx, _, _ = params.sinr_to_mcs(ue_sinr[i])
                    mcs_usage[mcs_idx] += packets[i]
                else:
                    mcs_usage[-1] += packets[i]

        # === DTX 決策：依「是否有請求」開關每個 BS 的發射 ===
        if params.ENABLE_DTX:
            # 每個 BS 當前時隙的請求量（以原 serving 關係計）
            requested_by_bs = np.zeros(N_bs, dtype=int)
            for bs in range(N_bs):
                if not active_bs[bs]:
                    continue
                idxs_req_bs = np.where((ue_serving == bs) & (packets > 0))[0]
                requested_by_bs[bs] = int(np.sum(packets[idxs_req_bs]))

            desired_tx = (requested_by_bs > 0)  # idle-DTX：無請求就關閉

            # 遲滯：最少連續 on/off 時隙
            for bs in range(N_bs):
                if not active_bs[bs]:
                    bs_tx_state[bs] = False
                    on_streak[bs] = off_streak[bs] = 0
                    continue

                if bs_tx_state[bs]:
                    if desired_tx[bs]:
                        on_streak[bs] += 1
                        off_streak[bs] = 0
                    else:
                        off_streak[bs] += 1
                        if off_streak[bs] >= params.DTX_MIN_OFF_SLOTS:
                            bs_tx_state[bs] = False
                            on_streak[bs] = 0
                else:
                    if desired_tx[bs]:
                        on_streak[bs] += 1
                        if on_streak[bs] >= params.DTX_MIN_ON_SLOTS:
                            bs_tx_state[bs] = True
                            off_streak[bs] = 0
                    else:
                        on_streak[bs] = 0
                        # 保持關閉            
        # 本時隙的「正在發射」遮罩
            active_tx_bs = active_bs & bs_tx_state
            tx_on_count += active_tx_bs.astype(int)

            # 以固定 serving、關閉干擾者後的 SINR（覆蓋 ue_sinr 供後續容量計算）
            ue_sinr_tx = np.full(N_ue, -np.inf)
            for i, ue_pos in enumerate(ue_positions):
                ue_serv = ue_serving[i]
                if ue_serv < 0:
                    continue
                ue_sinr_tx[i] = calculate_sinr_fixed_serving(
                    ue_pos, bs_positions, ue_serv, active_tx_bs, params
                )
        else:
            # 不用 DTX 時，沿用原本的 ue_sinr
            ue_sinr_tx = ue_sinr.copy()

        # 步驟 2.5：統計本時隙每個活躍 BS 的連線/忙碌 UE 數
        for bs in range(N_bs):
            if not active_bs[bs]:
                continue
            served = np.sum(ue_serving == bs)
            busy = np.sum((ue_serving == bs) & (packets > 0))
            
            total_served_ues_sum += served
            total_busy_ues_sum += busy
            active_bs_slots += 1

        
        # 步驟 3：對每個 BS 計算容量、統計與分配
        for bs in range(N_bs):
            if not active_bs[bs]:
                continue

            # 1) 本 BS 的 UE 集合
            all_served = np.where(ue_serving == bs)[0]
            idxs_req = np.where((ue_serving == bs) & (packets > 0))[0]
            requested = int(packets[idxs_req].sum()) if idxs_req.size > 0 else 0

            # 2) 潛在容量（ALL，使用 ue_sinr_tx）
            if all_served.size > 0:
                se_all = []
                for ue_idx in all_served:
                    if ue_sinr_tx[ue_idx] > params.SINR_THRESHOLD_DB:
                        _, se, _ = params.sinr_to_mcs(ue_sinr_tx[ue_idx])
                        se_all.append(se)
                avg_se_all = float(np.mean(se_all)) if se_all else 0.0
            else:
                avg_se_all = 0.0

            cap_bps_potential = params.BANDWIDTH * avg_se_all
            cap_pkts_potential = params.capacity_to_max_packets(cap_bps_potential)

            # DTX：若本時隙不發射，潛在容量視為 0（不供吞吐也不產生干擾）
            if params.ENABLE_DTX and not bs_tx_state[bs]:
                cap_pkts_potential = 0

            # 3) 實際分配用的容量（BUSY，用 ue_sinr_tx）
            mode = params.CAPACITY_MODE
            if mode == 'avg':
                if idxs_req.size > 0:
                    avg_sinr = float(np.mean(ue_sinr_tx[idxs_req]))
                    if avg_sinr > params.SINR_THRESHOLD_DB:
                        _, se_avg, _ = params.sinr_to_mcs(avg_sinr)
                        cap_bps_used = params.BANDWIDTH * se_avg
                    else:
                        cap_bps_used = 0.0
                else:
                    cap_bps_used = 0.0
            elif mode == 'sum':
                cap_bps_used = 0.0
                for ue_idx in idxs_req:
                    cap_bps_used += params.sinr_to_capacity_bps(ue_sinr_tx[ue_idx])
            else:  # 'share'
                if getattr(params, 'SHARE_OVER', 'busy') == 'all':
                    share_set = all_served
                else:
                    share_set = idxs_req
                se_share = []
                for ue_idx in share_set:
                    if ue_sinr_tx[ue_idx] > params.SINR_THRESHOLD_DB:
                        _, se, _ = params.sinr_to_mcs(ue_sinr_tx[ue_idx])
                        se_share.append(se)
                avg_se_share = float(np.mean(se_share)) if se_share else 0.0
                cap_bps_used = params.BANDWIDTH * avg_se_share

            bs_max_packets = params.capacity_to_max_packets(cap_bps_used)

            # 4) 利用率（所有活躍 BS 都統計）
            if bs_max_packets <= 0:
                util = 1.0 if requested > 0 else 0.0
            else:
                util = min(1.0, requested / float(bs_max_packets))
            traffic_load[bs] += util

            # 5) all 軌統計（所有活躍 BS×時隙都納入）
            all_req_tot += requested
            all_cap_pot_tot += cap_pkts_potential
            all_cnt += 1

            # 6) busy 軌統計與比例分配（只有有請求時）
            if idxs_req.size > 0:
                busy_req_tot += requested
                busy_cap_tot += bs_max_packets
                busy_over_tot += int(requested > bs_max_packets)
                busy_cnt += 1

                # 分配
                if requested <= bs_max_packets:
                    handled_per_ue = packets[idxs_req].astype(int)
                else:
                    frac_alloc = packets[idxs_req] * (bs_max_packets / float(requested))
                    base_alloc = np.floor(frac_alloc).astype(int)
                    rem = bs_max_packets - base_alloc.sum()
                    if rem > 0:
                        fracs = frac_alloc - base_alloc
                        order = np.argsort(-fracs)
                        for j in order[:int(rem)]:
                            base_alloc[j] += 1
                    handled_per_ue = base_alloc

                # 成功封包（SINR 門檻）
                for k, ue_idx in enumerate(idxs_req):
                    alloc = int(handled_per_ue[k])
                    if alloc > 0 and ue_sinr_tx[ue_idx] > params.SINR_THRESHOLD_DB:
                        successful_transmissions += alloc

    success_rate = successful_transmissions / total_attempts if total_attempts > 0 else 0.0
    normalized_load = traffic_load / float(simulation_time)

    # DTX 占空比
    if params.ENABLE_DTX:
        tx_duty = tx_on_count / float(simulation_time)
        power_consumption = calculate_power_consumption_with_dtx(active_bs, normalized_load, tx_duty)
    else:
        power_consumption = calculate_power_consumption(active_bs, normalized_load)
    
    
    # normalize MCS usage
    mcs_dist = mcs_usage / total_attempts if total_attempts > 0 else mcs_usage
    
    diag = {
        # 舊鍵維持：busy-only
        "avg_requested_per_bs_per_slot": (busy_req_tot / busy_cnt) if busy_cnt > 0 else 0.0,
        "avg_capacity_per_bs_per_slot": (busy_cap_tot / busy_cnt) if busy_cnt > 0 else 0.0,
        "overload_ratio_per_bs": (busy_over_tot / busy_cnt) if busy_cnt > 0 else 0.0,
        # 新增：all-BS
        "avg_requested_all_bs_per_slot": all_req_tot / max(1, all_cnt),
        "avg_capacity_all_bs_per_slot": all_cap_pot_tot / max(1, all_cnt),
        "busy_bs_ratio": busy_cnt / max(1, all_cnt),
        # 修正：UE 數量平均（跨所有時隙）
        "avg_ues_per_bs": total_served_ues_sum / max(1, active_bs_slots),
        "avg_busy_ues_per_bs": total_busy_ues_sum / max(1, busy_cnt),
    }


    return success_rate, power_consumption, mcs_dist, diag


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
    visualize_signal_and_ues(params, bs_positions, ue_positions, area_radius=2000, grid_res=150)

    strategies = []
    rng = np.random.default_rng(12345)
    n_experiments = 10
    active_ratios = np.arange(0.1, 1.05, 0.1)
    # active_ratios = [1.0]

    # ring-based 啟用策略：由近到遠
    order_by_center = np.argsort(np.sum(bs_positions**2, axis=1))

    print(f"\n開始模擬 {len(active_ratios)} 個啟用比例，每個重複 {n_experiments} 次...")
    start_time = time.time()

    for active_ratio in tqdm(active_ratios, desc="模擬進度"):
        exp_success_rates = []
        exp_powers = []
        exp_mcs_dists = []

        n_active = max(1, int(round(active_ratio * params.BS_COUNT)))
        # active_bs_mask = np.zeros(params.BS_COUNT, dtype=bool)
        # active_bs_mask[order_by_center[:n_active]] = True

        for _ in range(n_experiments):
            # ring-based 啟用
            # active_bs = active_bs_mask  # simulate() 內部不會改這個

            # 純 DTX 改為：每次實驗隨機挑 n_active 台 BS 啟用
            idx = rng.choice(params.BS_COUNT, size=n_active, replace=False)
            active_bs = np.zeros(params.BS_COUNT, dtype=bool)
            active_bs[idx] = True

            # 每次實驗重新生成 UE 位置
            ue_positions = generate_ue_positions(params)

            success_rate, power,  mcs_dist, diag = simulate(params, bs_positions, ue_positions, active_bs)

            print(f"[λ={params.LAMBDA:.3f}, ratio={active_ratio:.2f}] "
                  f"req_busy={diag['avg_requested_per_bs_per_slot']:.2f}, "
                  f"cap_busy={diag['avg_capacity_per_bs_per_slot']:.2f}, "
                  f"avg_busy_ues={diag['avg_busy_ues_per_bs']:.1f}, "                # ← 加這行
                  f"over_busy%={diag['overload_ratio_per_bs']*100:.1f}, "
                  f"req_all={diag['avg_requested_all_bs_per_slot']:.2f}, "
                  f"cap_all={diag['avg_capacity_all_bs_per_slot']:.2f}, "
                  f"avg_ues={diag['avg_ues_per_bs']:.1f}, "                # ← 加這行
                  f"busy%={diag['busy_bs_ratio']*100:.1f}")
            
            exp_success_rates.append(success_rate)
            exp_powers.append(power)
            exp_mcs_dists.append(mcs_dist)

        avg_success = np.mean(exp_success_rates)
        std_success = np.std(exp_success_rates)
        avg_power = np.mean(exp_powers)
        std_power = np.std(exp_powers)
        avg_mcs_dist = np.mean(exp_mcs_dists, axis=0)

        strategies.append({
            'active_ratio': active_ratio,
            'success_rate': avg_success,
            'success_std': std_success,
            'power': avg_power,
            'power_std': std_power,
            'mcs_dist': avg_mcs_dist
        })
    
    total_time = time.time() - start_time
    print(f"\n模擬完成！總耗時: {total_time:.2f} 秒")
    

    print("繪製結果...")
    # 結果視覺化（含誤差範圍）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    active_ratios_plot = [s['active_ratio'] for s in strategies]
    success_rates = [s['success_rate'] for s in strategies]
    success_stds = [s['success_std'] for s in strategies]
    powers = [s['power'] for s in strategies]
    power_stds = [s['power_std'] for s in strategies]
    
    # Success Rate
    axes[0, 0].errorbar(active_ratios_plot, success_rates, yerr=success_stds, 
                fmt='b-o', capsize=3, alpha=0.7, label='Mean ± Std')
    axes[0, 0].fill_between(active_ratios_plot, 
                    np.array(success_rates) - np.array(success_stds),
                    np.array(success_rates) + np.array(success_stds),
                    color='b', alpha=0.2)
    axes[0, 0].set_xlabel('Active BS Ratio')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title(f'Success Rate vs Active BS Ratio\n(Mode: {params.CAPACITY_MODE})')
    axes[0, 0].grid(True, ls=':')
    axes[0, 0].legend()
    
    # Power Consumption
    axes[0, 1].errorbar(active_ratios_plot, powers, yerr=power_stds,
                fmt='r-o', capsize=3, alpha=0.7, label='Mean ± Std')
    axes[0, 1].fill_between(active_ratios_plot,
                    np.array(powers) - np.array(power_stds),
                    np.array(powers) + np.array(power_stds),
                    color='r', alpha=0.2)
    axes[0, 1].set_xlabel('Active BS Ratio')
    axes[0, 1].set_ylabel('Power Consumption (W)')
    axes[0, 1].set_title('Power Consumption vs Active BS Ratio')
    axes[0, 1].grid(True, ls=':')
    axes[0, 1].legend()
    
    # MCS Distribution (stacked bar for one example ratio)
    example_ratio_idx = len(strategies) // 2  # 中間的 ratio
    mcs_dist_full = strategies[example_ratio_idx]['mcs_dist']
    
    # 分離有效 MCS 與無連接
    mcs_values = mcs_dist_full[:-1]
    no_connection = mcs_dist_full[-1]
    
    # 繪圖
    x_pos = list(range(len(params.MCS_TABLE)))
    axes[1, 0].bar(x_pos, mcs_values, color='skyblue', edgecolor='k', label='Valid MCS')
    axes[1, 0].bar([len(x_pos)], [no_connection], color='red', edgecolor='k', label='No Connection')
    axes[1, 0].set_xlabel('MCS Index')
    axes[1, 0].set_ylabel('Usage Ratio')
    axes[1, 0].set_title(f'MCS Distribution (Active Ratio = {strategies[example_ratio_idx]["active_ratio"]:.1f})\n' + 
                        f'Sum = {mcs_dist_full.sum():.4f}')
    axes[1, 0].grid(True, ls=':', axis='y')
    axes[1, 0].legend()
    
    # 設置 x 軸標籤
    all_labels = [str(i) for i in range(len(params.MCS_TABLE))] + ['NC']
    axes[1, 0].set_xticks(range(len(all_labels)))
    axes[1, 0].set_xticklabels(all_labels, rotation=0, ha='center')
    
    # Energy Efficiency (Success Rate / Power)
    energy_eff = [s['success_rate'] / (s['power'] + 1e-9) for s in strategies]
    axes[1, 1].plot(active_ratios_plot, energy_eff, 'g-o', linewidth=2)
    axes[1, 1].set_xlabel('Active BS Ratio')
    axes[1, 1].set_ylabel('Energy Efficiency (Success Rate / Watt)')
    axes[1, 1].set_title('Energy Efficiency vs Active BS Ratio')
    axes[1, 1].grid(True, ls=':')
    
    plt.tight_layout()
    plt.show()
    
    # 輸出統計資訊
    print("\n=== 模擬結果摘要 ===")
    print(f"{'Active Ratio':<15} {'Success Rate':<15} {'Power (W)':<15} {'Energy Eff':<15} {'No Conn %':<15}")
    print("-" * 75)
    for s in strategies:
        ee = s['success_rate'] / (s['power'] + 1e-9)
        no_conn_pct = s['mcs_dist'][-1] * 100
        mcs_sum = s['mcs_dist'].sum()
        print(f"{s['active_ratio']:<15.2f} {s['success_rate']:<15.3f} {s['power']:<15.2f} {ee:<15.5f} {no_conn_pct:<15.1f}")
        
        # 驗證總和（應為 1.0）
        if abs(mcs_sum - 1.0) > 1e-6:
            print(f"  ⚠️  警告：MCS 分布總和 = {mcs_sum:.6f}（應為 1.0）")

    # 決策分析
    MIN_SUCCESS_RATE = 0.90
    print(f"\n=== 決策分析（最低成功率要求：{MIN_SUCCESS_RATE:.0%}）===")
    
    feasible = [s for s in strategies if s['success_rate'] >= MIN_SUCCESS_RATE]
    if feasible:
        best = max(feasible, key=lambda s: s['success_rate'] / (s['power'] + 1e-9))
        print(f"最佳操作點：")
        print(f"  Active Ratio: {best['active_ratio']:.1%}")
        print(f"  Success Rate: {best['success_rate']:.2%}")
        print(f"  Power: {best['power']:.1f} W")
        print(f"  Energy Efficiency: {best['success_rate']/(best['power']+1e-9):.6f}")
        
        full_active = strategies[-1]
        power_saving = (1 - best['power']/full_active['power']) * 100
        print(f"\n與全開策略相比：")
        print(f"  節省功耗: {power_saving:.1f}%")
        print(f"  成功率差距: {(best['success_rate']-full_active['success_rate'])*100:.1f}%")
    else:
        print("警告：無任何策略滿足 QoS 要求！")

if __name__ == "__main__":
    main()
import math
import matplotlib.pyplot as plt

# 參數設定 -------------------------------------------------
R = 3          # 六角形版圖的"半徑" (層數)：0~±R
s = 250.0/math.sqrt(3)*2      # 六邊形邊長，選 200 讓 Δx = 3/2*s = 300

# 軸座標(axial) -> 平面座標(pointy-top)
def axial_to_xy(q, r, s):
    x = s * (1.5 * q)                       # = s * (3/2 * q)
    y = s * (math.sqrt(3) * (r + q / 2.0))
    return x, y

# 給定中心，算出六邊形六個頂點座標（for 畫 Polygon 用）
def hex_corners(xc, yc, s):
    corners = []
    for k in range(6):
        angle = math.radians(60 * k - 60)  # pointy-top 版
        x = xc + s * math.cos(angle)
        y = yc + s * math.sin(angle)
        corners.append((x, y))
    return corners

# 產生 radius R 的「六角形形狀」蜂巢所有格子的 axial 座標 -------
axial_cells = []
for q in range(-R, R + 1):
    for r in range(-R, R + 1):
        s_coord = -q - r
        if max(abs(q), abs(r), abs(s_coord)) <= R:
            axial_cells.append((q, r))

# 把所有中心點轉成 (x, y)，順便存起來 ----------------------------
centers = []
for (q, r) in axial_cells:
    x, y = axial_to_xy(q, r, s)
    centers.append((x, y))

# 印出所有中心點座標（你可以拿去用） -----------------------------
print("Centers (x, y):")
for (x, y) in centers:
    print(f"({x:.1f}, {y:.1f})")

# 畫圖 -----------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))

# 畫每一個六邊形
for (x, y) in centers:
    pts = hex_corners(x, y, s)
    xs = [p[0] for p in pts] + [pts[0][0]]
    ys = [p[1] for p in pts] + [pts[0][1]]
    ax.plot(xs, ys, color='black', linewidth=2)
    # 填一點顏色看起來像蜂巢
    ax.fill(xs, ys, color='#f3d86b', alpha=0.9)

# 把所有中心點標出來（紅點）
cx = [c[0] for c in centers]
cy = [c[1] for c in centers]
ax.scatter(cx, cy, color='red', s=50, zorder=5)

# 強制等比例顯示
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Hex grid with all centers marked')

plt.show()

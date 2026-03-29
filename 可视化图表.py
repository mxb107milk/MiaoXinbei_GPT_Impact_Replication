import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# ========== 新图1：雷达图（原图2） ==========
radar_data = pd.DataFrame({
    "領域名稱": ["新媒體傳播", "文旅產業", "教育大數據", "電子商務", "人力資源管理"],
    "產出增幅": [152.98, 154.24, 150.27, 140.22, 140.83],
    "新穎度提升": [8.72, 7.95, 4.87, 6.34, 3.12],
    "合作率提升": [24.17, 26.38, 22.45, 20.17, 18.62]
})
labels = radar_data.columns[1:].tolist()
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
for idx, row in radar_data.iterrows():
    values = row[1:].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=row["領域名稱"])
    ax.fill(angles, values, alpha=0.15)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_title("图1 五大领域GPT影响多维度雷达图", fontsize=13, fontweight="bold", y=1.1)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.savefig("图1_雷达图.png")
plt.close()

# ========== 新图2：热力图（原图3） ==========
heatmap_data = radar_data.set_index("領域名稱").iloc[:, 1:]
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdBu_r", linewidths=0.5, ax=ax)
ax.set_xlabel("核心指標", fontsize=11)
ax.set_ylabel("研究領域", fontsize=11)
ax.set_title("图2 五大领域核心指标变化幅度热力图", fontsize=13, fontweight="bold")
plt.savefig("图2_热力图.png")
plt.close()

# ========== 新图3：教育大数据领域（原图4） ==========
edu_data = pd.DataFrame({
    "研究主題": ["線上學習研究", "教學效果實證研究", "教育理論研究"],
    "產出增幅(%)": [187.62, 208.63, 112.37],
    "GPT前週期(月)": [6.82, 7.15, 5.98],
    "GPT後週期(月)": [4.21, 4.53, 5.12]
})
fig, ax1 = plt.subplots(figsize=(10, 6))
bars = ax1.barh(edu_data["研究主題"], edu_data["產出增幅(%)"], color="#1f77b4", alpha=0.7)
ax1.bar_label(bars, fmt="%.2f%%")
ax1.set_xlabel("年均產出增幅(%)")
ax1.set_ylabel("研究主題")
ax1.grid(axis="x", linestyle="--", alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(edu_data["研究主題"], edu_data["GPT前週期(月)"], marker="o", color="#ff7f0e", linestyle="--", label="GPT前週期")
ax2.plot(edu_data["研究主題"], edu_data["GPT後週期(月)"], marker="s", color="#ff7f0e", linestyle="-", label="GPT後週期")
ax2.set_ylabel("平均發表週期(月)")
ax2.tick_params(axis="y", labelcolor="#ff7f0e")
plt.title("图3 教育大數據領域線上學習相關研究 GPT 前後核心指標變化圖", fontweight="bold")
fig.legend(loc="upper right")
plt.savefig("图3_教育大数据.png")
plt.close()

# ========== 新图4：电子商务领域（原图5） ==========
ecom_data = pd.DataFrame({
    "研究維度": ["消費者決策行為", "用戶體驗研究", "電子商務供應鏈"],
    "GPT前產出": [126, 118, 97],
    "GPT後產出": [343, 312, 189],
    "產出增幅(%)": [172.54, 164.41, 94.85]
})
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(ecom_data))
ax.bar(x - 0.2, ecom_data["GPT前產出"], 0.4, label="GPT前", color="#2ca02c", alpha=0.8)
ax.bar(x + 0.2, ecom_data["GPT後產出"], 0.4, label="GPT後", color="#d62728", alpha=0.8)
ax.set_xlabel("研究維度")
ax.set_ylabel("年均論文產出量（篇）")
ax.set_title("图4 電子商務領域用戶消費行為相關研究 GPT 前後產出變化圖", fontweight="bold")
ax.set_xticks(x, ecom_data["研究維度"])
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.3)
for i, row in ecom_data.iterrows():
    ax.text(i, max(row["GPT前產出"], row["GPT後產出"]) + 15,
            f"增幅：{row['產出增幅(%)']:.2f}%", ha="center", fontsize=8)
ax.bar_label(ax.containers[0], fmt="%d")
ax.bar_label(ax.containers[1], fmt="%d")
plt.savefig("图4_电子商务.png")
plt.close()

# ========== 新图5：新媒体传播引用量增幅（原图6） ==========
cite_data = pd.DataFrame({
    "領域": ["新媒體傳播", "文旅產業", "教育大數據"],
    "篇均引用量增幅(%)": [87.62, 82.37, 78.45]
})
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(cite_data["領域"], cite_data["篇均引用量增幅(%)"], color=["#d62728", "#ff7f0e", "#1f77b4"])
ax.set_ylabel("篇均引用量增幅(%)")
ax.set_title("图5 新媒體傳播領域篇均引用量增幅對比圖", fontweight="bold")
ax.bar_label(bars, fmt="%.2f%%")
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.savefig("图5_新媒体传播引用增幅.png")
plt.close()

# ========== 新图6：文旅产业趋势（原图7） ==========
tour_data = pd.DataFrame({
    "年份": [2019, 2020, 2021, 2022, 2023, 2024],
    "年度總產出": [1876, 2043, 2218, 2357, 3842, 4219],
    "文本挖掘佔比(%)": [28.74, 31.26, 33.85, 35.42, 48.76, 57.62]
})
fig, ax1 = plt.subplots(figsize=(10, 6))
line1 = ax1.plot(tour_data["年份"], tour_data["年度總產出"], marker="o", color="#9467bd", linewidth=2, label="年度總產出")
ax1.set_xlabel("發表年份")
ax1.set_ylabel("年度總產出量（篇）", color="#9467bd")
ax1.tick_params(axis="y", labelcolor="#9467bd")
ax1.set_xticks(tour_data["年份"])
ax1.grid(axis="y", linestyle="--", alpha=0.3)
ax2 = ax1.twinx()
line2 = ax2.plot(tour_data["年份"], tour_data["文本挖掘佔比(%)"], marker="s", color="#ff7f0e", linewidth=2, linestyle="--", label="文本挖掘論文佔比")
ax2.set_ylabel("文本挖掘論文佔比(%)", color="#ff7f0e")
ax2.tick_params(axis="y", labelcolor="#ff7f0e")
ax2.axvline(x=2022.8, color="red", linestyle=":", linewidth=2, label="GPT發布時間")
lines = line1 + line2
labels = [l.get_label() for l in lines]
plt.legend(lines, labels, loc="upper left")
plt.title("图6 文旅產業領域年度產出與文本挖掘論文佔比變化趨勢圖", fontweight="bold")
plt.savefig("图6_文旅产业趋势.png")
plt.close()

# ========== 新图7：人力资源管理领域（原图8） ==========
hr_data = pd.DataFrame({
    "研究類別": ["員工離職風險研究", "領域其他研究"],
    "GPT前產出": [89, 126],
    "GPT後產出": [240, 278],
    "產出增幅(%)": [168.74, 140.83],
    "GPT前合作率(%)": [32.47, 28.63],
    "GPT後合作率(%)": [48.72, 41.25]
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(hr_data))
bars1 = ax1.bar(x - 0.2, hr_data["GPT前產出"], 0.4, label="GPT前", color="#1f77b4")
bars2 = ax1.bar(x + 0.2, hr_data["GPT後產出"], 0.4, label="GPT後", color="#ff7f0e")
ax1.set_title("員工離職風險研究與領域整體產出對比")
ax1.set_xlabel("研究類別")
ax1.set_ylabel("年均產出（篇）")
ax1.set_xticks(x, hr_data["研究類別"])
ax1.legend()
ax1.grid(axis="y", linestyle="--", alpha=0.3)
ax1.bar_label(bars1, fmt="%d")
ax1.bar_label(bars2, fmt="%d")
bars3 = ax2.bar(x - 0.2, hr_data["GPT前合作率(%)"], 0.4, label="GPT前", color="#2ca02c")
bars4 = ax2.bar(x + 0.2, hr_data["GPT後合作率(%)"], 0.4, label="GPT後", color="#d62728")
ax2.set_title("跨機構合作率對比")
ax2.set_xlabel("研究類別")
ax2.set_ylabel("跨機構合作率(%)")
ax2.set_xticks(x, hr_data["研究類別"])
ax2.legend()
ax2.grid(axis="y", linestyle="--", alpha=0.3)
ax2.bar_label(bars3, fmt="%.2f%%")
ax2.bar_label(bars4, fmt="%.2f%%")
plt.suptitle("图7 人力資源管理領域員工離職風險相關研究 GPT 前後變化圖", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("图7_人力资源管理.png")
plt.close()

# ========== 新图8：新媒体传播短视频研究增幅（原图9） ==========
short_video_data = pd.DataFrame({
    "領域": ["新媒體傳播", "文旅產業", "教育大數據"],
    "短視頻相關研究增幅(%)": [189.62, 172.34, 150.28]
})
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(short_video_data["領域"], short_video_data["短視頻相關研究增幅(%)"], color=["#d62728", "#ff7f0e", "#1f77b4"])
ax.set_ylabel("年均產出增幅(%)")
ax.set_title("图8 新媒體傳播領域短視頻傳播相關研究 GPT 前後產出增幅對比圖", fontweight="bold")
ax.bar_label(bars, fmt="%.2f%%")
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.savefig("图8_新媒体传播短视频增幅.png")
plt.close()

print("所有图表生成完毕：图1-图8")
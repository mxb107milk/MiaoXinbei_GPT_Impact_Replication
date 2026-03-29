import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

# ========== 1. 读取数据 ==========
df = pd.read_excel("五大領域_標準化數據_完整.xlsx")

# ========== 2. 数据预处理 ==========
df['發表時間'] = pd.to_datetime(df['發表時間'])
df['month_num'] = (df['發表時間'].dt.year - 2019) * 12 + df['發表時間'].dt.month
df['is_gpt'] = (df['發表時間'] >= '2022-11-01').astype(int)

# 面板数据聚合（按领域+月度）
panel = df.groupby(['領域名稱', 'month_num']).agg({
    '關鍵詞新穎度': 'mean',
    '摘要原創度': 'mean',
    '跨機構合作': 'mean',
    '跨國合作': 'mean',
    '參考文獻數量': 'mean',
    '總引用量': 'mean',
    '作者數量': 'mean',
    'is_gpt': 'first'
}).reset_index()

# 过滤：每个领域-月度至少有2篇论文
paper_counts = df.groupby(['領域名稱', 'month_num']).size().reset_index(name='n_papers')
panel = panel.merge(paper_counts, on=['領域名稱', 'month_num'])
panel = panel[panel['n_papers'] >= 2].copy()
print(f"过滤后剩余记录数: {len(panel)}")

# 添加时间趋势项
panel['trend'] = panel.groupby('領域名稱')['month_num'].rank(method='first').astype(int)
panel = panel.set_index(['領域名稱', 'month_num'])

# ========== 3. 回归函数 ==========
def run_reg(y_var, data):
    try:
        X = data[['is_gpt', 'trend']]
        X = sm.add_constant(X)
        if data[y_var].std() == 0:
            print(f"警告：{y_var} 无变异，跳过")
            return None
        model = PanelOLS(data[y_var], X, entity_effects=True, time_effects=False)
        return model.fit(cov_type='clustered', cluster_entity='領域名稱')
    except Exception as e:
        print(f"回归 {y_var} 失败: {e}")
        return None

# ========== 4. 回归分析（无发表周期） ==========
vars_list = ['關鍵詞新穎度', '摘要原創度', '跨機構合作', '跨國合作', '參考文獻數量', '總引用量', '作者數量']
results = {}
for v in vars_list:
    res = run_reg(v, panel)
    if res is not None:
        results[v] = {
            '系数': res.params['is_gpt'],
            '标准误': res.std_errors['is_gpt'],
            'p值': res.pvalues['is_gpt']
        }
    else:
        results[v] = {'系数': np.nan, '标准误': np.nan, 'p值': np.nan}

result_df = pd.DataFrame(results).T
result_df['显著性'] = result_df['p值'].apply(
    lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    if pd.notna(p) else ''
)
result_df = result_df.round(4)
print("\n=== 双向固定效应回归结果 ===\n")
print(result_df)

# ========== 5. 保存回归结果表格图片 ==========
def save_table_image(df, title, filename):
    fig, ax = plt.subplots(figsize=(10, max(3, len(df)*0.3)))
    ax.axis('off')
    ax.axis('tight')
    table_data = [df.columns.tolist()] + df.values.tolist()
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (i, j) in table.get_celld().keys():
        if i == 0:
            table[(i, j)].set_facecolor('#e6e6e6')
            table[(i, j)].set_text_props(weight='bold')
    plt.title(title, fontsize=12, fontweight='bold', pad=20)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存：{filename}")

save_table_image(result_df, "表X 双向固定效应回归结果（无发表周期）", "回归结果表格_无周期.png")

# ========== 6. 生成系数条形图 ==========
plot_df = result_df.dropna(subset=['系数']).copy()
if not plot_df.empty:
    plot_df['CI_low'] = plot_df['系数'] - 1.96 * plot_df['标准误']
    plot_df['CI_high'] = plot_df['系数'] + 1.96 * plot_df['标准误']
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(plot_df))
    bars = ax.barh(y_pos, plot_df['系数'],
                   xerr=(plot_df['系数']-plot_df['CI_low'], plot_df['CI_high']-plot_df['系数']),
                   capsize=5, color='steelblue', alpha=0.7, error_kw={'ecolor': 'black'})
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df.index)
    ax.set_xlabel('迴歸係數')
    ax.set_title('雙向固定效應迴歸結果（無發表週期）', fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    for i, (coef, star) in enumerate(zip(plot_df['系数'], plot_df['显著性'])):
        if star:
            ha = 'left' if coef >= 0 else 'right'
            ax.text(coef + (0.01 if coef >= 0 else -0.01), i, star,
                    va='center', ha=ha, fontsize=12, fontweight='bold')
    ax.text(0.98, 0.02, '*** p<0.01', transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('回归系数图_无周期.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("回归系数图已保存为：回归系数图_无周期.png")
else:
    print("没有有效数据用于绘制系数图")
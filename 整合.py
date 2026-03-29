import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# 读取数据
df = pd.read_excel("五大領域_標準化數據_含摘要.xlsx")

# 参数设置：每篇论文只与最近 L 篇历史论文比较
L = 100   # 可根据实际情况调整，越大越精确但越慢

# ========== 计算关键词新颖度 ==========
def novelty_score(keywords_str, past_keywords_set):
    if pd.isna(keywords_str) or keywords_str == "":
        return 0
    current_set = set(keywords_str.split(','))
    if len(current_set) == 0:
        return 0
    if len(past_keywords_set) == 0:
        return 100
    jaccard = len(current_set & past_keywords_set) / len(current_set | past_keywords_set)
    return (1 - jaccard) * 100

df = df.sort_values(['領域名稱', '發表時間'])
df['關鍵詞新穎度'] = 0
for field in df['領域名稱'].unique():
    print(f"计算领域 {field} 的关键词新颖度...")
    field_df = df[df['領域名稱'] == field].copy()
    past_keywords = set()
    for idx, row in field_df.iterrows():
        keywords_str = row['關鍵詞']
        nov = novelty_score(keywords_str, past_keywords)
        df.at[idx, '關鍵詞新穎度'] = nov
        if not pd.isna(keywords_str) and keywords_str.strip() != "":
            past_keywords.update(keywords_str.split(','))

# ========== 优化版摘要原创度计算（只与最近 L 篇比较）==========
def originality_score_fast(text, past_texts):
    if not text or pd.isna(text) or text.strip() == "":
        return 0
    if len(past_texts) == 0:
        return 100
    # 只取最近 L 篇
    recent = past_texts[-L:] if len(past_texts) > L else past_texts
    try:
        vectorizer = TfidfVectorizer().fit([text] + recent)
        vec = vectorizer.transform([text] + recent)
        sim = cosine_similarity(vec[0:1], vec[1:])[0]
        max_sim = np.max(sim) if len(sim) > 0 else 0
        return (1 - max_sim) * 100
    except Exception as e:
        print(f"警告：计算摘要原创度失败，返回 0。错误：{e}")
        return 0

df['摘要原創度'] = 0
for field in df['領域名稱'].unique():
    print(f"计算领域 {field} 的摘要原创度（仅与最近{L}篇比较）...")
    field_df = df[df['領域名稱'] == field].sort_values('發表時間')
    past_abstracts = []
    total = len(field_df)
    for i, (idx, row) in enumerate(field_df.iterrows()):
        if i % 500 == 0:
            print(f"  处理进度: {i}/{total}")
        abstract = row.get('摘要', '')
        if pd.isna(abstract):
            abstract = ""
        orig = originality_score_fast(abstract, past_abstracts)
        df.at[idx, '摘要原創度'] = orig
        if abstract and abstract.strip() != "":
            past_abstracts.append(abstract)

# 保存新数据
df.to_excel("五大領域_標準化數據_完整.xlsx", index=False)
print("已计算并保存含新颖度和原创度的数据文件：五大領域_標準化數據_完整.xlsx")
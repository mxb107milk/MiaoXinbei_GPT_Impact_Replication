import requests
import pandas as pd
import time
from datetime import datetime

MY_EMAIL = "D22090103123@cityu.edu.mo"

FIELD_CONFIG = {
    "教育大數據": {
        "search_keywords": "educational big data OR online learning OR smart education",
        "field_name": "教育大數據"
    },
    "新媒體傳播": {
        "search_keywords": "new media OR short video OR online communication",
        "field_name": "新媒體傳播"
    },
    "文旅產業": {
        "search_keywords": "cultural tourism OR tourism destination OR text mining",
        "field_name": "文旅產業"
    },
    "電子商務": {
        "search_keywords": "e-commerce OR consumer behavior OR user experience",
        "field_name": "電子商務"
    },
    "人力資源管理": {
        "search_keywords": "human resource management OR employee turnover",
        "field_name": "人力資源管理"
    }
}

API_BASE_URL = "https://api.openalex.org/works"
START_YEAR = 2019
END_YEAR = 2024
PER_PAGE = 200
MAX_PAGES_PER_YEAR = 5
REQUEST_INTERVAL = 0.5

def extract_paper_info(paper_raw_data, field_name_cn):
    try:
        # 提取摘要倒排索引并转换为纯文本（简化版）
        abstract_inv = paper_raw_data.get("abstract_inverted_index", {})
        # 按索引顺序还原摘要文本（更准确）
        if abstract_inv:
            # 创建一个字典 {位置: 词}
            pos_words = {}
            for word, positions in abstract_inv.items():
                for pos in positions:
                    pos_words[pos] = word
            # 按位置排序拼接
            sorted_words = [pos_words[i] for i in sorted(pos_words.keys())]
            abstract_text = " ".join(sorted_words)
        else:
            abstract_text = ""

        institutions = []
        countries = []
        for auth in paper_raw_data.get("authorships", []):
            for inst in auth.get("institutions", []):
                institutions.append(inst.get("display_name", ""))
                countries.append(inst.get("country_code", ""))

        pub_date = paper_raw_data.get("publication_date", "")
        created_date = paper_raw_data.get("created_date", "")
        publish_cycle = 0
        if pub_date and created_date:
            try:
                pub_dt = datetime.strptime(pub_date, "%Y-%m-%d")
                create_dt = datetime.strptime(created_date, "%Y-%m-%d")
                if pub_dt >= create_dt:
                    publish_cycle = round((pub_dt - create_dt).days / 30, 2)
            except:
                pass

        return {
            "領域名稱": field_name_cn,
            "論文標題": paper_raw_data.get("title", ""),
            "DOI": paper_raw_data.get("doi", ""),
            "發表時間": pub_date,
            "投稿時間": created_date,
            "發表週期（月）": publish_cycle,
            "作者數量": len(paper_raw_data.get("authorships", [])),
            "跨機構合作": 1 if len(set(institutions)) >= 2 else 0,
            "跨國合作": 1 if len(set(countries)) >= 2 else 0,
            "關鍵詞": ",".join([kw["display_name"] for kw in paper_raw_data.get("keywords", [])]) or "",
            "摘要": abstract_text,
            "參考文獻數量": paper_raw_data.get("referenced_works_count", 0),
            "總引用量": paper_raw_data.get("cited_by_count", 0),
            "是否GPT後": 1 if pub_date >= "2022-11-01" else 0
        }
    except Exception as e:
        print(f"跳过异常论文：{str(e)}")
        return None

def collect_all_fields():
    all_papers = []
    print("开始采集五大领域数据（2019-2024），包含摘要...")
    print(f"邮箱：{MY_EMAIL}")

    for field_cn, config in FIELD_CONFIG.items():
        keywords = config["search_keywords"]
        print(f"\n{'='*50}\n正在采集：{field_cn}")

        for year in range(START_YEAR, END_YEAR + 1):
            cursor = "*"
            page = 0
            while page < MAX_PAGES_PER_YEAR:
                try:
                    params = {
                        "search": keywords,
                        "filter": f"publication_year:{year},type:article",
                        "per_page": PER_PAGE,
                        "cursor": cursor,
                        "mailto": MY_EMAIL
                    }
                    response = requests.get(API_BASE_URL, params=params, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                    results = data.get("results", [])

                    if not results:
                        break

                    for paper in results:
                        info = extract_paper_info(paper, field_cn)
                        if info:
                            all_papers.append(info)

                    print(f"{field_cn} {year}年 第{page+1}页 完成 | 累计{len(all_papers)}条")
                    cursor = data["meta"].get("next_cursor")
                    if not cursor:
                        break
                    page += 1
                    time.sleep(REQUEST_INTERVAL)

                except Exception as e:
                    print(f"错误：{e}")
                    page += 1
                    time.sleep(REQUEST_INTERVAL)

    df = pd.DataFrame(all_papers)
    df.to_excel("五大領域_標準化數據_含摘要.xlsx", index=False)
    print(f"\n采集完成！总样本量：{len(df)}条")
    print("保存为：五大領域_標準化數據_含摘要.xlsx")
    return df

if __name__ == "__main__":
    collect_all_fields()
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# 共享配置
from config import get_data_dir


# =========================
# 1. 基础配置
# =========================

# 输出到 data/ 下的时间戳子文件夹，避免覆盖旧数据
_RUN_ID = datetime.now().strftime("run_%Y%m%d_%H%M%S")
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / _RUN_ID

VIDEO_NUM = 100000      # 10万条视频
USER_NUM = 10000        # 1万个用户
MIN_WATCH = 50          # 每个用户最少观看数量
MAX_WATCH = 120         # 每个用户最多观看数量

random.seed(2026)
np.random.seed(2026)

# =========================
# 1.1 幂律分布配置 — 模拟真实视频热度
# =========================
# 真实短视频平台中，播放量遵循 Zipf 分布：
#   前 1% 的热门视频占据 ~90% 的播放量
#   使用 numpy.random.zipf(alpha) 实现
#   alpha=2.0 产生明显的长尾效应

ZIPF_ALPHA = 2.0  # Zipf 分布参数，越大头部越集中


# 视频类别
CATEGORIES = [
    "搞笑", "游戏", "美食", "学习", "运动",
    "音乐", "动漫", "宠物", "科技", "生活"
]

# 每个类别对应的标签
CATEGORY_TAGS = {
    "搞笑": ["段子", "整活", "搞笑日常", "反转", "沙雕"],
    "游戏": ["王者荣耀", "英雄联盟", "原神", "和平精英", "游戏解说"],
    "美食": ["家常菜", "探店", "甜品", "火锅", "烧烤"],
    "学习": ["编程", "英语", "考研", "数学", "学习方法"],
    "运动": ["篮球", "足球", "健身", "跑步", "羽毛球"],
    "音乐": ["翻唱", "流行音乐", "钢琴", "吉他", "说唱"],
    "动漫": ["二次元", "国漫", "日漫", "混剪", "动漫解说"],
    "宠物": ["猫咪", "狗狗", "萌宠", "治愈", "宠物日常"],
    "科技": ["人工智能", "手机", "数码", "机器人", "科技资讯"],
    "生活": ["vlog", "校园", "旅行", "穿搭", "日常生活"]
}


def ensure_data_dir():
    """创建 data 文件夹"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


# =========================
# 2. 生成视频数据
# =========================

def generate_videos():
    """生成视频数据。

    每个视频被赋予一个 Zipf 分布的"基础热度权重"，
    模拟真实场景中的幂律效应 —— 少数视频极热门，大量视频冷门。
    """
    videos = []

    # 用 Zipf 分布给每个视频分配基础热度权重
    # 将 10 万个视频按 video_id 排序，随机打乱再分配 Zipf 权重，
    # 保证热度权重与 video_id 无关
    video_order = list(range(1, VIDEO_NUM + 1))
    random.shuffle(video_order)

    # 生成 Zipf 热度权重：值越大代表该视频越容易被观看
    zipf_values = np.random.zipf(ZIPF_ALPHA, size=VIDEO_NUM)
    # zipf(a, N) 产生的值范围很大，映射到 (0, 1] 作为概率权重
    video_heat_weight = {}
    max_zipf = zipf_values.max()
    for i, vid in enumerate(video_order):
        video_heat_weight[vid] = zipf_values[i] / max_zipf

    for video_id in range(1, VIDEO_NUM + 1):
        category = random.choice(CATEGORIES)
        tags = random.sample(CATEGORY_TAGS[category], k=3)

        title = f"{category}视频_{video_id}_{tags[0]}"
        duration = random.randint(10, 180)  # 视频时长，单位秒
        author_id = random.randint(1, 5000)

        # 发布时间：0~90 天前，但加入近期的偏向（模拟平台增长）
        publish_days_ago = int(np.random.exponential(scale=30))
        publish_days_ago = min(publish_days_ago, 90)
        publish_time = datetime.now() - timedelta(days=publish_days_ago)

        videos.append([
            video_id,
            title,
            category,
            "|".join(tags),
            duration,
            author_id,
            publish_time.strftime("%Y-%m-%d"),
            round(video_heat_weight[video_id], 6)  # 基础热度权重
        ])

    df = pd.DataFrame(videos, columns=[
        "video_id", "title", "category", "tags",
        "duration", "author_id", "publish_time", "base_heat"
    ])

    df.to_csv(DATA_DIR / "videos.csv", index=False, encoding="utf-8-sig")
    print("videos.csv 生成完成，共", len(df), "条")
    return video_heat_weight


# =========================
# 3. 生成用户数据
# =========================

# 用户画像定义：10 个类别各出现恰好 2 次（1 次核心 + 1 次次要）
# 确保各类别流量均衡，同时画像之间有足够的兴趣差异
USER_PERSONAS = {
    "宅文化型": {
        "core_interests": ["搞笑", "游戏"],
        "secondary": ["动漫", "音乐"],
        "like_rate": (0.30, 0.05),
        "collect_rate": (0.12, 0.04),
        "share_rate": (0.04, 0.02),
    },
    "生活家型": {
        "core_interests": ["美食", "生活"],
        "secondary": ["科技", "运动"],
        "like_rate": (0.20, 0.05),
        "collect_rate": (0.10, 0.03),
        "share_rate": (0.09, 0.03),
    },
    "进步青年型": {
        "core_interests": ["学习", "科技"],
        "secondary": ["搞笑", "宠物"],
        "like_rate": (0.22, 0.05),
        "collect_rate": (0.16, 0.04),
        "share_rate": (0.05, 0.02),
    },
    "运动乐活型": {
        "core_interests": ["运动", "音乐"],
        "secondary": ["游戏", "美食"],
        "like_rate": (0.26, 0.05),
        "collect_rate": (0.08, 0.03),
        "share_rate": (0.07, 0.03),
    },
    "动漫二次元型": {
        "core_interests": ["动漫", "宠物"],
        "secondary": ["学习", "生活"],
        "like_rate": (0.28, 0.05),
        "collect_rate": (0.14, 0.04),
        "share_rate": (0.06, 0.03),
    },
}


def generate_users():
    users = []
    persona_list = list(USER_PERSONAS.keys())

    age_groups = ["18岁以下", "18-24岁", "25-30岁", "31-40岁", "40岁以上"]
    genders = ["男", "女"]

    for user_id in range(1, USER_NUM + 1):
        gender = random.choice(genders)
        age_group = random.choice(age_groups)

        # ★ 随机分配用户画像，兴趣从画像的核心+次要池中抽取
        persona = random.choice(persona_list)
        p = USER_PERSONAS[persona]
        pool = p["core_interests"] + p["secondary"]

        # 从画像池中随机选 2~4 个兴趣，核心兴趣更有偏向
        k = random.randint(2, 4)
        # 至少 1 个核心兴趣
        n_core = min(k, random.randint(1, len(p["core_interests"])))
        core_picks = random.sample(p["core_interests"], k=n_core)
        remaining = [c for c in pool if c not in core_picks]
        secondary_picks = random.sample(remaining, k=min(k - n_core, len(remaining)))
        interests = core_picks + secondary_picks

        users.append([
            user_id,
            gender,
            age_group,
            "|".join(interests),
            persona  # 保存画像名
        ])

    df = pd.DataFrame(users, columns=[
        "user_id", "gender", "age_group", "interest_categories", "persona"
    ])

    df.to_csv(DATA_DIR / "users.csv", index=False, encoding="utf-8-sig")
    print("users.csv 生成完成，共", len(df), "条")
    return dict(zip(df["user_id"], df["persona"]))


# =========================
# 4. 生成观看行为数据
# =========================

def generate_watch_logs(video_heat_weight=None, user_personas=None):
    """生成观看行为数据（优化版）。

    核心改进：
    1. 加权随机采样（二分查找累积权重） — O(log N) 替代 O(N) 逐次采样
    2. 时间衰减 — 预计算衰减值，避免逐次日期解析
    3. 构建用户-视频二分图邻接表
    4. 用户画像驱动互动率 — 不同画像用户有不同点赞/收藏/分享倾向
    """
    import json
    import bisect

    videos_df = pd.read_csv(DATA_DIR / "videos.csv")
    users_df = pd.read_csv(DATA_DIR / "users.csv")

    # 构建视频热度哈希表
    if video_heat_weight is None:
        video_heat_weight = {}
        for _, row in videos_df.iterrows():
            video_heat_weight[int(row["video_id"])] = float(row["base_heat"])

    # ★ 预计算：按类别构建 (video_ids, cumulative_weights) 用于二分查找采样
    # 同时预计算每个视频的时间衰减值
    cat_video_ids = defaultdict(list)
    cat_video_weights = defaultdict(list)
    cat_cum_weights = {}
    video_decay = {}

    for _, row in videos_df.iterrows():
        vid = int(row["video_id"])
        cat = row["category"]
        heat = video_heat_weight.get(vid, 0.01)

        cat_video_ids[cat].append(vid)
        cat_video_weights[cat].append(heat)

        # 预计算时间衰减
        publish_str = str(row["publish_time"])
        publish_date = datetime.strptime(publish_str, "%Y-%m-%d")
        days_since_publish = (datetime.now() - publish_date).days
        video_decay[vid] = max(0.15, 1.0 - 0.7 * (days_since_publish / 90))

    # 为每个类别构建累积权重数组（二分查找用）
    for cat in CATEGORIES:
        cum = []
        total = 0.0
        for w in cat_video_weights[cat]:
            total += w
            cum.append(total)
        cat_cum_weights[cat] = (cum, total)

    # ★ 用户-视频二分图（邻接表）
    user_watch_graph = defaultdict(set)
    video_watchers_graph = defaultdict(set)

    logs = []
    log_id = 1

    # 预计算所有用户的 watch_count，避免逐行 iterrows 开销
    user_records = []
    for _, user in users_df.iterrows():
        user_records.append((
            int(user["user_id"]),
            str(user["interest_categories"]).split("|"),
            random.randint(MIN_WATCH, MAX_WATCH)
        ))

    for user_id, interests, watch_count in user_records:
        # ★ 获取该用户画像的互动率参数
        persona = user_personas.get(user_id, "宅文化型") if user_personas else "宅文化型"
        pcfg = USER_PERSONAS.get(persona, USER_PERSONAS["宅文化型"])
        lr_mean, lr_std = pcfg["like_rate"]
        cr_mean, cr_std = pcfg["collect_rate"]
        sr_mean, sr_std = pcfg["share_rate"]

        for _ in range(watch_count):
            # 80% 看感兴趣类别，20% 探索其他类别
            if random.random() < 0.8:
                category = random.choice(interests)
            else:
                category = random.choice(CATEGORIES)

            # ★ 混合采样：70% Zipf 加权 + 30% 均匀随机
            #   — 保留幂律分布的头部效应
            #   — 同时保证冷门视频也有最低曝光，避免聚类出现"死簇"
            if random.random() < 0.7:
                cum, total_w = cat_cum_weights[category]
                r = random.random() * total_w
                idx = bisect.bisect_left(cum, r)
            else:
                idx = random.randint(0, len(cat_video_ids[category]) - 1)
            video_id = cat_video_ids[category][idx]

            # ★ 时间衰减：使用预计算值
            if random.random() > video_decay[video_id]:
                continue

            watch_duration = random.randint(5, 180)
            liked = 1 if random.random() < max(0.01, min(0.99, random.gauss(lr_mean, lr_std))) else 0
            collected = 1 if random.random() < max(0.01, min(0.99, random.gauss(cr_mean, cr_std))) else 0
            shared = 1 if random.random() < max(0.01, min(0.99, random.gauss(sr_mean, sr_std))) else 0

            days_ago = int(np.random.exponential(scale=7))
            days_ago = min(days_ago, 30)
            watch_time = datetime.now() - timedelta(days=days_ago)

            logs.append([
                log_id,
                user_id,
                video_id,
                category,
                watch_duration,
                liked,
                collected,
                shared,
                watch_time.strftime("%Y-%m-%d")
            ])

            user_watch_graph[user_id].add(video_id)
            video_watchers_graph[video_id].add(user_id)

            log_id += 1

    df = pd.DataFrame(logs, columns=[
        "log_id", "user_id", "video_id", "category",
        "watch_duration", "liked", "collected", "shared", "watch_time"
    ])

    df.to_csv(DATA_DIR / "watch_logs.csv", index=False, encoding="utf-8-sig")
    print("watch_logs.csv 生成完成，共", len(df), "条")

    # ★ 保存二分图的统计信息（供课程设计报告引用）
    user_degrees = [len(v) for v in user_watch_graph.values()]
    video_degrees = [len(v) for v in video_watchers_graph.values()]

    graph_stats = {
        "description": "用户-视频二分图邻接表统计",
        "data_structure": "邻接表 (Adjacency List): defaultdict(set)",
        "num_users": len(user_watch_graph),
        "num_videos": len(video_watchers_graph),
        "total_edges": len(df),
        "avg_user_degree": round(np.mean(user_degrees), 2),
        "max_user_degree": max(user_degrees),
        "avg_video_degree": round(np.mean(video_degrees), 2),
        "max_video_degree": max(video_degrees),
        "video_degree_distribution_top10": sorted(
            [(int(vid), len(users)) for vid, users in video_watchers_graph.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
    }

    with open(DATA_DIR / "graph_stats.json", "w", encoding="utf-8") as f:
        json.dump(graph_stats, f, ensure_ascii=False, indent=2)
    print("用户-视频二分图统计已保存到 graph_stats.json")

    return user_watch_graph, video_watchers_graph


# =========================
# 5. 主函数
# =========================

if __name__ == "__main__":
    ensure_data_dir()

    print("开始生成模拟数据...")
    print(f"输出目录: {DATA_DIR}")
    video_heat_weight = generate_videos()
    user_personas = generate_users()
    generate_watch_logs(video_heat_weight, user_personas)

    # 写入哨兵文件，让其他脚本自动定位到最新数据
    sentinel = DATA_DIR.parent / "LATEST_RUN.txt"
    sentinel.write_text(DATA_DIR.name, encoding="utf-8")
    print(f"已更新 LATEST_RUN.txt -> {DATA_DIR.name}")
    print("全部数据生成完成！")
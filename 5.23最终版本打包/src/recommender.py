from collections import defaultdict
import math
import pandas as pd

from config import get_data_dir


# =========================
# 1. 自实现数据结构：最小堆（用于 TopK）
# =========================

class MinHeap:
    """自实现的最小堆，用于高效维护 TopK 结果。

    不使用 heapq 库，手动实现堆的插入和弹出操作。
    堆以数组形式存储，索引 i 的左子节点为 2i+1，右子节点为 2i+2，父节点为 (i-1)//2。

    时间复杂度：
    - heappush: O(log n)
    - heappop:  O(log n)
    - topk:     O(n log k)，其中 n 为候选数，k 为保留数
    """

    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)

    def _sift_up(self, idx):
        """上浮操作：将 idx 位置的元素向上移动到合适位置。"""
        while idx > 0:
            parent = (idx - 1) // 2
            if self.heap[idx][0] < self.heap[parent][0]:
                self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
                idx = parent
            else:
                break

    def _sift_down(self, idx):
        """下沉操作：将 idx 位置的元素向下移动到合适位置。"""
        n = len(self.heap)
        while True:
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2

            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right

            if smallest != idx:
                self.heap[idx], self.heap[smallest] = self.heap[smallest], self.heap[idx]
                idx = smallest
            else:
                break

    def heappush(self, item):
        """插入元素 item = (score, data)。"""
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)

    def heappop(self):
        """弹出最小元素。"""
        if not self.heap:
            raise IndexError("堆为空")
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return root

    def topk(self, items, k, key=lambda x: x):
        """从 items 中找出 key 值最大的 k 个元素。

        使用大小为 k 的最小堆维护当前 TopK：
        - 前 k 个元素直接入堆
        - 后续元素若 key 值大于堆顶（当前第 k 大），则替换堆顶
        """
        if k <= 0:
            return []

        for item in items:
            score = key(item)
            if len(self.heap) < k:
                self.heappush((score, item))
            elif score > self.heap[0][0]:
                self.heappop()
                self.heappush((score, item))

        # 从大到小排序输出（只返回 item，不返回 score）
        result = [None] * len(self.heap)
        for i in range(len(self.heap) - 1, -1, -1):
            score, item = self.heappop()
            result[i] = item
        return result


# =========================
# 1. 路径配置
# =========================

DATA_DIR = get_data_dir()

# =========================
# 2. 推荐系统类
# =========================

class ShortVideoRecommender:
    def __init__(self, videos_df=None, users_df=None, logs_df=None):
        """
        初始化推荐系统：
        1. 读取视频、用户、观看记录数据（支持外部注入，避免重复读盘）
        2. 构建用户观看历史索引
        3. 构建视频被哪些用户观看过的倒排索引
        4. 构建视频热度索引
        """
        if videos_df is None:
            videos_df = pd.read_csv(DATA_DIR / "videos.csv", encoding="utf-8-sig")
        if users_df is None:
            users_df = pd.read_csv(DATA_DIR / "users.csv", encoding="utf-8-sig")
        if logs_df is None:
            logs_df = pd.read_csv(DATA_DIR / "watch_logs.csv", encoding="utf-8-sig")
        self.videos_df = videos_df
        self.users_df = users_df
        self.logs_df = logs_df

        self.video_info = {}
        self.user_info = {}

        self.user_watch = defaultdict(set)        # user_id -> set(video_id)
        self.video_watchers = defaultdict(set)    # video_id -> set(user_id)
        self.video_heat = defaultdict(int)        # video_id -> watch_count
        self.user_like = defaultdict(set)         # user_id -> set(liked_video_id)

        self._build_indexes()

    def _build_indexes(self):
        """构建各种索引结构"""

        # 视频信息哈希表：video_id -> 视频信息
        for row in self.videos_df.itertuples(index=False):
            self.video_info[int(row.video_id)] = {
                "title": row.title,
                "category": row.category,
                "tags": row.tags,
                "duration": row.duration,
                "author_id": row.author_id,
                "publish_time": row.publish_time
            }

        # 用户信息哈希表：user_id -> 用户信息
        for row in self.users_df.itertuples(index=False):
            interests = str(row.interest_categories).split("|")
            self.user_info[int(row.user_id)] = {
                "gender": row.gender,
                "age_group": row.age_group,
                "interest_categories": interests
            }

        # 观看行为索引
        for row in self.logs_df.itertuples(index=False):
            uid = int(row.user_id)
            vid = int(row.video_id)

            self.user_watch[uid].add(vid)
            self.video_watchers[vid].add(uid)
            self.video_heat[vid] += 1

            if int(row.liked) == 1:
                self.user_like[uid].add(vid)

        print("索引构建完成")
        print("用户数量：", len(self.user_watch))
        print("视频数量：", len(self.video_info))
        print("观看记录数量：", len(self.logs_df))

    # =========================
    # 3. F3：相似用户分析
    # =========================

    def jaccard_similarity(self, set_a, set_b):
        """
        Jaccard 相似度：
        sim(A, B) = |A ∩ B| / |A ∪ B|
        """
        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        if union == 0:
            return 0.0

        return intersection / union

    def get_similar_users(self, target_user_id, top_k=10):
        """
        找到与目标用户兴趣最相似的 TopK 用户。

        优化思路：
        不直接和所有用户比较，而是先通过“视频 -> 用户”的倒排索引，
        找到至少和目标用户看过同一个视频的候选用户。
        """

        target_user_id = int(target_user_id)

        if target_user_id not in self.user_watch:
            return []

        target_videos = self.user_watch[target_user_id]

        # 候选用户集合
        candidate_users = set()

        for video_id in target_videos:
            candidate_users.update(self.video_watchers[video_id])

        # 去掉自己
        candidate_users.discard(target_user_id)

        similar_users = []

        for user_id in candidate_users:
            sim = self.jaccard_similarity(target_videos, self.user_watch[user_id])

            if sim > 0:
                similar_users.append((user_id, sim))

        # 取相似度最高的 top_k 个用户（使用自实现堆）
        heap = MinHeap()
        top_users = heap.topk(similar_users, top_k, key=lambda x: x[1])

        result = []
        for user_id, sim in top_users:
            info = self.user_info.get(user_id, {})

            result.append({
                "user_id": user_id,
                "similarity": round(sim, 4),
                "gender": info.get("gender", "未知"),
                "age_group": info.get("age_group", "未知"),
                "interest_categories": "|".join(info.get("interest_categories", [])),
                "watch_count": len(self.user_watch[user_id])
            })

        return result

    # =========================
    # 4. F4：个性化视频推荐
    # =========================

    def recommend_videos(self, target_user_id, top_k=10, similar_user_k=20,
                         diversity_lambda=0.3):
        """根据相似用户和兴趣标签，为目标用户推荐视频。

        改进点：
        1. 评分归一化 — 候选视频分数按贡献用户数取平均
        2. 冷启动 — 新用户或无候选时推荐热门视频
        3. MMR 多样性重排 — 避免推荐结果集中在一个类别

        推荐步骤：
        ① 找相似用户
        ② 收集候选视频，综合评分（相似度+标签匹配+热度）
        ③ MMR 重排：平衡相关性与多样性
        """

        target_user_id = int(target_user_id)
        target_watched = self.user_watch.get(target_user_id, set())
        target_interests = set(
            self.user_info[target_user_id]["interest_categories"]
        ) if target_user_id in self.user_info else set()

        # ── 冷启动处理 ──
        if not target_watched or target_user_id not in self.user_info:
            return self._cold_start_recommend(top_k)

        similar_users = self.get_similar_users(
            target_user_id,
            top_k=similar_user_k
        )

        candidate_scores = defaultdict(float)
        candidate_counts = defaultdict(int)     # 记录贡献该视频的相似用户数
        candidate_reasons = defaultdict(list)

        max_heat = max(self.video_heat.values()) if self.video_heat else 1

        for user in similar_users:
            sim_user_id = user["user_id"]
            similarity = user["similarity"]

            for video_id in self.user_watch[sim_user_id]:
                if video_id in target_watched:
                    continue

                video = self.video_info.get(video_id)
                if not video:
                    continue

                category = video["category"]

                # 1. 相似用户贡献分
                similar_score = similarity

                # 2. 兴趣类别匹配分
                interest_score = 1.0 if category in target_interests else 0.3

                # 3. 视频热度分（log 压缩，消减头部效应）
                heat_score = math.log1p(self.video_heat[video_id]) / math.log1p(max_heat)

                # 综合评分
                score = (
                    0.5 * similar_score +
                    0.3 * interest_score +
                    0.2 * heat_score
                )

                candidate_scores[video_id] += score
                candidate_counts[video_id] += 1

                if category in target_interests:
                    candidate_reasons[video_id].append("兴趣标签匹配")
                candidate_reasons[video_id].append("相似用户看过")

        # ── 评分归一化（按贡献用户数平均，消除累积效应）──
        for video_id in candidate_scores:
            candidate_scores[video_id] /= candidate_counts[video_id]

        # ── 用自实现堆取 TopK × 2（为 MMR 留候选池）──
        candidates = [(vid, candidate_scores[vid]) for vid in candidate_scores]
        heap = MinHeap()
        mmr_pool_size = min(top_k * 2, len(candidates))
        initial_top = heap.topk(candidates, mmr_pool_size,
                                key=lambda x: x[1])

        if not initial_top:
            return self._cold_start_recommend(top_k)

        # ── MMR 多样性重排 ──
        selected = []
        remaining = [(vid, score, self.video_info[vid]["category"])
                     for vid, score in initial_top]

        # 第一个选分数最高的
        first = remaining.pop(0)
        selected.append(first)
        first_cat = first[2]

        while len(selected) < top_k and remaining:
            best_idx = 0
            best_mmr = -float("inf")

            for i, (vid, score, cat) in enumerate(remaining):
                # 与已选视频的最大类别重叠度
                max_overlap = 0.0
                for sel_vid, sel_score, sel_cat in selected:
                    overlap = 1.0 if cat == sel_cat else 0.0
                    if overlap > max_overlap:
                        max_overlap = overlap

                mmr = score - diversity_lambda * max_overlap
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        # ── 构建推荐结果 ──
        recommendations = []
        for rank, (video_id, score, _) in enumerate(selected):
            video = self.video_info[video_id]
            reasons = list(set(candidate_reasons[video_id]))
            recommendations.append({
                "video_id": video_id,
                "title": video["title"],
                "category": video["category"],
                "tags": video["tags"],
                "heat": self.video_heat[video_id],
                "score": round(score, 4),
                "reason": "、".join(reasons)
            })

        return recommendations

    def _cold_start_recommend(self, top_k):
        """冷启动策略：当目标用户无观看历史时，推荐全平台最热门视频。"""
        heap = MinHeap()
        hot_videos = [(vid, heat) for vid, heat in self.video_heat.items()
                      if vid in self.video_info]
        top_items = heap.topk(hot_videos, top_k, key=lambda x: x[1])

        recommendations = []
        for video_id, heat in top_items:
            video = self.video_info[video_id]
            recommendations.append({
                "video_id": video_id,
                "title": video["title"],
                "category": video["category"],
                "tags": video["tags"],
                "heat": heat,
                "score": round(heat / max(self.video_heat.values(), default=1), 4),
                "reason": "热门推荐（冷启动）"
            })
        return recommendations

    # =========================
    # 5. 用户画像
    # =========================

    def get_user_profile(self, user_id):
        """
        获取用户画像，用于界面展示。
        """

        user_id = int(user_id)

        if user_id not in self.user_info:
            return None

        watched_videos = self.user_watch[user_id]

        category_count = defaultdict(int)

        for video_id in watched_videos:
            video = self.video_info.get(video_id)
            if video:
                category_count[video["category"]] += 1

        profile = {
            "user_id": user_id,
            "gender": self.user_info[user_id]["gender"],
            "age_group": self.user_info[user_id]["age_group"],
            "interest_categories": self.user_info[user_id]["interest_categories"],
            "watch_count": len(watched_videos),
            "like_count": len(self.user_like[user_id]),
            "category_distribution": dict(category_count)
        }

        return profile


# =========================
# 6. 离线评估
# =========================

    def evaluate(self, test_user_ids=None, top_k=10, similar_user_k=20):
        """离线评估推荐系统性能。

        将每个用户的观看记录按时间划分为训练集和测试集：
        - 训练集：用户前 80% 的观看记录
        - 测试集：用户后 20% 的观看记录

        评估指标：
        - Precision@K：推荐列表中测试集视频占比
        - Recall@K：   测试集视频被推荐的比例
        - HitRate@K：   至少命中一个测试集视频的用户比例

        返回 dict 包含三项指标的平均值。
        """
        if test_user_ids is None:
            test_user_ids = list(self.user_watch.keys())[:500]

        # 为每个用户划分训练/测试集
        test_videos = {}      # user_id -> set(测试视频)
        temp_train_watch = {}  # user_id -> set(训练视频)

        # 按观看时间排序后划分
        user_logs = defaultdict(list)
        for _, row in self.logs_df.iterrows():
            user_logs[int(row["user_id"])].append(
                (int(row["video_id"]), row["watch_time"])
            )

        for user_id in test_user_ids:
            if user_id not in user_logs:
                continue
            logs = sorted(user_logs[user_id], key=lambda x: x[1])
            split = int(len(logs) * 0.8)
            temp_train_watch[user_id] = {v for v, _ in logs[:split]}
            test_videos[user_id] = {v for v, _ in logs[split:]}
            if len(test_videos[user_id]) == 0 and len(logs) > 1:
                # 至少保留 1 个测试视频
                test_videos[user_id] = {logs[-1][0]}
                temp_train_watch[user_id] = {v for v, _ in logs[:-1]}

        # 暂存原始 user_watch
        original_user_watch = self.user_watch

        precisions = []
        recalls = []
        hits = 0
        total_users = 0

        for user_id in test_user_ids:
            if user_id not in test_videos or len(test_videos[user_id]) == 0:
                continue
            if user_id not in temp_train_watch or len(temp_train_watch[user_id]) == 0:
                continue

            total_users += 1
            test_set = test_videos[user_id]

            # 临时替换为训练集
            self.user_watch[user_id] = temp_train_watch[user_id]

            try:
                recs = self.recommend_videos(
                    user_id, top_k=top_k, similar_user_k=similar_user_k
                )
            except Exception:
                recs = []

            recommended_ids = {r["video_id"] for r in recs}
            hits_in_rec = len(recommended_ids & test_set)

            precisions.append(hits_in_rec / top_k if top_k > 0 else 0)
            recalls.append(hits_in_rec / len(test_set) if len(test_set) > 0 else 0)
            if hits_in_rec > 0:
                hits += 1

        # 恢复原始数据
        self.user_watch = original_user_watch

        return {
            "Precision@K": round(sum(precisions) / len(precisions), 4) if precisions else 0,
            "Recall@K": round(sum(recalls) / len(recalls), 4) if recalls else 0,
            "HitRate@K": round(hits / total_users, 4) if total_users > 0 else 0,
            "num_test_users": total_users
        }


# =========================
# 7. 测试代码
# =========================

if __name__ == "__main__":
    recommender = ShortVideoRecommender()

    target_user_id = 100

    print("\n==============================")
    print(f"用户 {target_user_id} 的画像")
    print("==============================")
    profile = recommender.get_user_profile(target_user_id)
    print(profile)

    print("\n==============================")
    print(f"与用户 {target_user_id} 相似的用户 Top10")
    print("==============================")
    similar_users = recommender.get_similar_users(target_user_id, top_k=10)

    for user in similar_users:
        print(user)

    print("\n==============================")
    print(f"给用户 {target_user_id} 推荐的视频 Top10")
    print("==============================")
    recommendations = recommender.recommend_videos(target_user_id, top_k=10)

    for video in recommendations:
        print(video)

    print("\n==============================")
    print("离线评估（500 个测试用户）")
    print("==============================")
    metrics = recommender.evaluate(test_user_ids=list(range(1, 501)), top_k=10)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
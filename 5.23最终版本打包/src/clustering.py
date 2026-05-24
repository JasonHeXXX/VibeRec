from pathlib import Path
import pandas as pd
import numpy as np
import math
from collections import defaultdict


# ==============================================================================
# 1. 自实现 K-Means 聚类算法
# ==============================================================================

class KMeansManual:
    """手动实现的 K-Means 聚类算法（不依赖 sklearn）。

    算法步骤：
    1. K-Means++ 初始化：选择距离已有中心最远的点作为新中心
    2. 分配步：每个点分配到最近的中心
    3. 更新步：中心更新为其分配点的均值
    4. 迭代至收敛或达到最大迭代次数

    时间复杂度：O(n × k × d × iter)
    空间复杂度：O(n × d + k × d)，仅存储数据和中心，不产生中间副本
    """

    def __init__(self, n_clusters=6, max_iter=300, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def fit(self, X):
        """对数据 X 进行 K-Means 聚类。

        参数:
            X: numpy.ndarray, shape (n_samples, n_features)
        """
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        best_inertia = float("inf")
        best_labels = None
        best_centers = None
        best_n_iter = 0

        # 多次初始化，取最优结果
        for init_run in range(self.n_init):
            centers = self._kmeans_plus_plus_init(X, rng)
            labels = np.zeros(n_samples, dtype=int)
            prev_labels = np.ones(n_samples, dtype=int) * -1

            for iteration in range(self.max_iter):
                # ── 分配步：每个点找最近的中心 ──
                # 使用暴力搜索 O(n × k × d)
                for i in range(n_samples):
                    min_dist = float("inf")
                    best_cluster = 0
                    for j in range(self.n_clusters):
                        # 欧氏距离平方（避免 sqrt，不影响比较结果）
                        diff = X[i] - centers[j]
                        dist_sq = np.dot(diff, diff)
                        if dist_sq < min_dist:
                            min_dist = dist_sq
                            best_cluster = j
                    labels[i] = best_cluster

                # 检查收敛
                if np.array_equal(labels, prev_labels):
                    break
                prev_labels = labels.copy()

                # ── 更新步：重新计算每个簇的中心 ──
                new_centers = np.zeros((self.n_clusters, n_features))
                counts = np.zeros(self.n_clusters, dtype=int)

                for i in range(n_samples):
                    cluster = labels[i]
                    new_centers[cluster] += X[i]
                    counts[cluster] += 1

                for j in range(self.n_clusters):
                    if counts[j] > 0:
                        centers[j] = new_centers[j] / counts[j]
                    else:
                        # 空簇处理：随机选一个点作为新中心
                        centers[j] = X[rng.randint(0, n_samples)]

            # 计算惯性（簇内平方和）
            inertia = self._compute_inertia(X, labels, centers)

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_n_iter = iteration + 1

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self

    def _kmeans_plus_plus_init(self, X, rng):
        """K-Means++ 初始化：使初始中心尽量分散。

        算法：
        1. 随机选第一个中心
        2. 对于后续每个中心，以 D(x)² 为概率选择（D(x) 为 x 到最近中心的距离）
        """
        n_samples, n_features = X.shape
        centers = np.zeros((self.n_clusters, n_features))

        # 第一个中心随机选
        first_idx = rng.randint(0, n_samples)
        centers[0] = X[first_idx]

        # 后续中心基于距离平方加权采样
        for c in range(1, self.n_clusters):
            # 计算每个点到最近中心的最小距离平方
            min_dist_sq = np.full(n_samples, np.inf)

            for j in range(c):
                diff = X - centers[j]
                dist_sq = np.sum(diff * diff, axis=1)
                min_dist_sq = np.minimum(min_dist_sq, dist_sq)

            # 按距离平方加权随机选下一个中心
            probs = min_dist_sq / min_dist_sq.sum()
            cumsum = np.cumsum(probs)
            r = rng.rand()
            next_idx = np.searchsorted(cumsum, r)
            centers[c] = X[next_idx]

        return centers

    def _compute_inertia(self, X, labels, centers):
        """计算簇内平方和（惯性）。"""
        inertia = 0.0
        for i in range(len(X)):
            diff = X[i] - centers[labels[i]]
            inertia += np.dot(diff, diff)
        return inertia

    def predict(self, X):
        """将 X 中的点分配到最近的簇。"""
        labels = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            min_dist = float("inf")
            best = 0
            for j in range(self.n_clusters):
                diff = X[i] - self.cluster_centers_[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    best = j
            labels[i] = best
        return labels


# ==============================================================================
# 2. 自实现数据标准化
# ==============================================================================

class StandardScalerManual:
    """手动实现 StandardScaler（z-score 标准化）。

    x' = (x - mean) / std
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0   # 避免除以零
        return (X - self.mean_) / self.std_

    def transform(self, X):
        return (X - self.mean_) / self.std_


# ==============================================================================
# 3. 自实现 PCA 降维
# ==============================================================================

class PCAManual:
    """手动实现 PCA（主成分分析）降维。

    步骤：
    1. 数据中心化：X' = X - mean(X)
    2. 计算协方差矩阵：C = X'^T X' / (n - 1)
    3. 特征值分解：C·v = λ·v
    4. 按 λ 降序取前 k 个特征向量
    5. 投影：Y = X' · V_k

    特征值分解使用 numpy.linalg.eigh（对称矩阵专用），
    因为自己实现 QR 迭代算法超出了数据结构课程范围。
    """

    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit_transform(self, X):
        """对数据 X 进行 PCA 降维，返回降维后的数据。"""
        n_samples, n_features = X.shape

        # 1. 数据中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. 协方差矩阵：C = X^T X / (n - 1)
        # 这是标准的样本协方差矩阵计算方法
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # 3. 特征值分解（对称矩阵用 eigh 更稳定）
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. 按特征值降序排列
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 5. 取前 k 个主成分
        self.components_ = eigenvectors[:, :self.n_components]

        # 解释方差比
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = (
            eigenvalues[:self.n_components] / total_var
            if total_var > 0 else np.zeros(self.n_components)
        )

        # 6. 投影
        X_transformed = np.dot(X_centered, self.components_)

        return X_transformed


# ==============================================================================
# 4. 聚类评估：轮廓系数
# ==============================================================================

def silhouette_score_manual(X, labels):
    """手动实现轮廓系数（Silhouette Score）。

    对每个点 i：
      a(i) = i 到同簇其他点的平均距离
      b(i) = i 到最近异簇所有点的平均距离的最小值
      s(i) = (b(i) - a(i)) / max(a(i), b(i))

    时间复杂度 O(n²·d)，对 10 万样本太慢，因此抽样计算。
    """
    n_samples = len(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) <= 1:
        return 0.0

    # 抽样（最多 3000 个点，保证可接受时间）
    if n_samples > 3000:
        rng = np.random.RandomState(2026)
        sample_indices = rng.choice(n_samples, size=3000, replace=False)
        X = X[sample_indices]
        labels = labels[sample_indices]
        n_samples = len(X)

    scores = np.zeros(n_samples)

    for i in range(n_samples):
        label_i = labels[i]

        # a(i)：到同簇其他点的平均距离
        same_cluster = np.where(labels == label_i)[0]
        if len(same_cluster) <= 1:
            a_i = 0.0
        else:
            dists = np.sqrt(np.sum((X[i] - X[same_cluster]) ** 2, axis=1))
            a_i = np.sum(dists) / (len(same_cluster) - 1)

        # b(i)：到最近异簇的平均距离
        b_i = float("inf")
        for other_label in unique_labels:
            if other_label == label_i:
                continue
            other_cluster = np.where(labels == other_label)[0]
            dists = np.sqrt(np.sum((X[i] - X[other_cluster]) ** 2, axis=1))
            avg_dist = np.mean(dists)
            if avg_dist < b_i:
                b_i = avg_dist

        if b_i == float("inf"):
            scores[i] = 0.0
        else:
            scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    return float(np.mean(scores))


# ==============================================================================
# 5. 肘部法则确定最优 K 值
# ==============================================================================

def elbow_method(X, k_range=None, random_state=2026, sample_size=5000):
    """肘部法则：计算 k 从 2 到 max_k 的惯性，用于确定最优聚类数。

    抽样计算以减少运行时间（对大样本数据集）。
    返回每个 k 的惯性值和推荐 k 值。
    """
    if k_range is None:
        k_range = range(2, 11)

    # 抽样加速
    if len(X) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    inertias = {}
    for k in k_range:
        kmeans = KMeansManual(n_clusters=k, n_init=3,
                              max_iter=100, random_state=random_state)
        kmeans.fit(X_sample)
        inertias[k] = kmeans.inertia_

    # 简单肘部检测：惯性下降率突然变缓的点
    ks = sorted(inertias.keys())
    deltas = {}
    for i in range(1, len(ks)):
        deltas[ks[i]] = inertias[ks[i - 1]] - inertias[ks[i]]

    # 推荐 k：delta 最大变化的位置（二阶差分最大）
    if len(ks) >= 4:
        best_k = ks[0]
        max_ratio = 0
        for i in range(2, len(ks)):
            if deltas[ks[i - 1]] > 0:
                ratio = deltas[ks[i]] / deltas[ks[i - 1]]
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_k = ks[i - 1]
        recommended_k = best_k
    else:
        recommended_k = ks[len(ks) // 2]

    return inertias, recommended_k


# ==============================================================================
# 6. 路径配置
# ==============================================================================

from config import get_data_dir
DATA_DIR = get_data_dir()


# ==============================================================================
# 7. 聚类分析主类
# ==============================================================================

class ShortVideoClusterAnalyzer:
    def __init__(self, video_clusters=6, user_clusters=5, auto_k=True,
                 videos_df=None, users_df=None, logs_df=None):
        """初始化聚类分析模块。

        auto_k: 是否使用肘部法则自动选择 K 值
        videos_df / users_df / logs_df: 可选，外部传入的共享 DataFrame（避免重复读盘）
        """
        self.video_clusters = video_clusters
        self.user_clusters = user_clusters
        self.auto_k = auto_k

        if videos_df is None:
            videos_df = pd.read_csv(DATA_DIR / "videos.csv", encoding="utf-8-sig")
        if users_df is None:
            users_df = pd.read_csv(DATA_DIR / "users.csv", encoding="utf-8-sig")
        if logs_df is None:
            logs_df = pd.read_csv(DATA_DIR / "watch_logs.csv", encoding="utf-8-sig")
        self.videos_df = videos_df
        self.users_df = users_df
        self.logs_df = logs_df

        self.categories = sorted(self.videos_df["category"].unique().tolist())

        self.video_cluster_result = None
        self.user_cluster_result = None
        self.video_cluster_summary = None
        self.user_cluster_summary = None

        # 聚类评估指标（供报告引用）
        self.video_silhouette = None
        self.user_silhouette = None
        self.video_elbow = None
        self.user_elbow = None
        self.video_recommended_k = None
        self.user_recommended_k = None

        # 尝试加载已保存的结果
        self._try_load_saved_results()

    def _try_load_saved_results(self):
        """如果 data 目录下有已保存的聚类结果，直接加载。"""
        import json
        metrics_path = DATA_DIR / "clustering_metrics.json"
        if not metrics_path.exists():
            return

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            self.video_silhouette = m.get("video_silhouette")
            self.user_silhouette = m.get("user_silhouette")
            self.video_elbow = m.get("video_elbow")
            self.user_elbow = m.get("user_elbow")
            self.video_recommended_k = m.get("video_clusters")
            self.user_recommended_k = m.get("user_clusters")
            if self.video_recommended_k:
                self.video_clusters = self.video_recommended_k
            if self.user_recommended_k:
                self.user_clusters = self.user_recommended_k
        except Exception:
            return

        # 加载视频聚类结果
        vr_path = DATA_DIR / "video_cluster_result.csv"
        vs_path = DATA_DIR / "video_cluster_summary.csv"
        if vr_path.exists() and vs_path.exists():
            try:
                self.video_cluster_result = pd.read_csv(vr_path, encoding="utf-8-sig")
                self.video_cluster_summary = pd.read_csv(vs_path, encoding="utf-8-sig")
            except Exception:
                pass

        # 加载用户聚类结果
        ur_path = DATA_DIR / "user_cluster_result.csv"
        us_path = DATA_DIR / "user_cluster_summary.csv"
        if ur_path.exists() and us_path.exists():
            try:
                self.user_cluster_result = pd.read_csv(ur_path, encoding="utf-8-sig")
                self.user_cluster_summary = pd.read_csv(us_path, encoding="utf-8-sig")
            except Exception:
                pass

    # ==========================================================================
    # 8. 构建视频特征矩阵（保留原有复杂特征工程）
    # ==========================================================================

    def _build_video_features(self):
        """构建视频聚类特征矩阵。

        特征包括：
        - 视频类别 one-hot（10 维）
        - 播放量（log 压缩）
        - 平均观看时长
        - 点赞率、收藏率、分享率
        - 观看用户性别比例
        - 观看用户年龄比例
        """
        video_base = self.videos_df[[
            "video_id", "title", "category", "tags", "duration",
            "author_id", "publish_time"
        ]].copy()

        category_one_hot = pd.get_dummies(video_base["category"], prefix="category")
        video_base = pd.concat([video_base, category_one_hot], axis=1)

        video_stats = (
            self.logs_df
            .groupby("video_id")
            .agg(
                watch_count=("video_id", "count"),
                avg_watch_duration=("watch_duration", "mean"),
                like_count=("liked", "sum"),
                collect_count=("collected", "sum"),
                share_count=("shared", "sum")
            )
            .reset_index()
        )

        video_feature_df = video_base.merge(video_stats, on="video_id", how="left")
        fill_cols = ["watch_count", "avg_watch_duration",
                     "like_count", "collect_count", "share_count"]
        video_feature_df[fill_cols] = video_feature_df[fill_cols].fillna(0)

        video_feature_df["like_rate"] = (
            video_feature_df["like_count"] /
            video_feature_df["watch_count"].replace(0, np.nan)
        )
        video_feature_df["collect_rate"] = (
            video_feature_df["collect_count"] /
            video_feature_df["watch_count"].replace(0, np.nan)
        )
        video_feature_df["share_rate"] = (
            video_feature_df["share_count"] /
            video_feature_df["watch_count"].replace(0, np.nan)
        )

        video_feature_df[["like_rate", "collect_rate", "share_rate"]] = (
            video_feature_df[["like_rate", "collect_rate", "share_rate"]].fillna(0)
        )

        # 观看用户的人口统计学特征
        log_user_df = self.logs_df.merge(
            self.users_df[["user_id", "gender", "age_group"]],
            on="user_id", how="left"
        )

        gender_dist = (
            log_user_df
            .groupby(["video_id", "gender"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        age_dist = (
            log_user_df
            .groupby(["video_id", "age_group"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        video_feature_df = video_feature_df.merge(gender_dist, on="video_id", how="left")
        video_feature_df = video_feature_df.merge(age_dist, on="video_id", how="left")
        video_feature_df = video_feature_df.fillna(0)

        demographic_cols = []
        for col in ["男", "女"]:
            if col in video_feature_df.columns:
                new_col = f"gender_{col}_rate"
                video_feature_df[new_col] = (
                    video_feature_df[col] /
                    video_feature_df["watch_count"].replace(0, np.nan)
                )
                demographic_cols.append(new_col)

        age_cols = ["18岁以下", "18-24岁", "25-30岁", "31-40岁", "40岁以上"]
        for col in age_cols:
            if col in video_feature_df.columns:
                new_col = f"age_{col}_rate"
                video_feature_df[new_col] = (
                    video_feature_df[col] /
                    video_feature_df["watch_count"].replace(0, np.nan)
                )
                demographic_cols.append(new_col)

        video_feature_df[demographic_cols] = video_feature_df[demographic_cols].fillna(0)

        video_feature_df["log_watch_count"] = np.log1p(video_feature_df["watch_count"])

        category_cols = [c for c in video_feature_df.columns
                        if c.startswith("category_")]
        feature_cols = (
            category_cols +
            ["log_watch_count", "avg_watch_duration",
             "like_rate", "collect_rate", "share_rate"] +
            demographic_cols
        )

        feature_matrix = video_feature_df[feature_cols].copy()
        return video_feature_df, feature_matrix

    # ==========================================================================
    # 9. 视频聚类 F6（手动实现 K-Means + PCA）
    # ==========================================================================

    def cluster_videos(self):
        """对视频进行聚类分析（手动实现 K-Means + PCA）。

        使用数据结构：
        - numpy.ndarray: 特征矩阵的底层存储（连续内存，利于缓存）
        - dict (哈希表): 聚类名称映射
        - KMeansManual: 自实现聚类，含 K-Means++ 初始化
        - PCAManual: 自实现 PCA，手动协方差 + 特征值分解
        """
        video_feature_df, feature_matrix = self._build_video_features()
        X = feature_matrix.values.astype(np.float64)

        # ── 标准化 ──
        scaler = StandardScalerManual()
        X_scaled = scaler.fit_transform(X)

        # ── 肘部法则自动选 K ──
        if self.auto_k:
            print("正在执行肘部法则（视频聚类），寻找最优 K 值...")
            self.video_elbow, self.video_recommended_k = elbow_method(
                X_scaled,
                k_range=range(2, 11),
                random_state=2026
            )
            self.video_clusters = self.video_recommended_k
            print(f"  肘部法则推荐 K = {self.video_recommended_k}")

        # ── K-Means 聚类 ──
        print(f"开始 K-Means 聚类（K={self.video_clusters}），该过程可能需要几分钟...")
        kmeans = KMeansManual(
            n_clusters=self.video_clusters,
            n_init=5,
            max_iter=100,
            random_state=2026
        )
        kmeans.fit(X_scaled)
        labels = kmeans.labels_

        print(f"  聚类完成：{kmeans.n_iter_} 次迭代收敛，惯性 = {kmeans.inertia_:.2f}")

        # ── 轮廓系数评估 ──
        print("正在计算视频聚类轮廓系数...")
        self.video_silhouette = silhouette_score_manual(X_scaled, labels)
        print(f"  视频聚类轮廓系数 = {self.video_silhouette:.4f}")

        # ── PCA 降维到 2D 用于可视化 ──
        print("正在执行 PCA 降维...")
        pca = PCAManual(n_components=2, random_state=2026)
        pca_result = pca.fit_transform(X_scaled)
        print(f"  PCA 解释方差比：{pca.explained_variance_ratio_.round(4)}")

        video_feature_df["cluster_id"] = labels
        video_feature_df["x"] = pca_result[:, 0]
        video_feature_df["y"] = pca_result[:, 1]

        cluster_name_map = self._generate_video_cluster_names(video_feature_df)
        video_feature_df["cluster_name"] = video_feature_df["cluster_id"].map(cluster_name_map)

        self.video_cluster_result = video_feature_df[[
            "video_id", "title", "category", "tags",
            "watch_count", "avg_watch_duration",
            "like_rate", "collect_rate", "share_rate",
            "cluster_id", "cluster_name", "x", "y"
        ]].copy()

        self.video_cluster_summary = self._summarize_video_clusters(
            self.video_cluster_result
        )

        return self.video_cluster_result

    # ==========================================================================
    # 10. 构建用户特征矩阵
    # ==========================================================================

    def _build_user_features(self):
        """构建用户聚类特征。

        特征维度：
        - 类别观看比例（10维）— 兴趣偏好
        - 互动率（3维）— 点赞/收藏/分享率，区分潜水用户 vs 活跃用户
        - 内容多样性（2维）— 观看类别数 + 香农熵，区分专一 vs 泛化用户
        - 观看行为（3维）— 平均观看时长、日均观看量、完播倾向
        """
        # ── 类别观看比例 ──
        user_category_count = (
            self.logs_df
            .groupby(["user_id", "category"])
            .size()
            .unstack(fill_value=0)
        )

        for category in self.categories:
            if category not in user_category_count.columns:
                user_category_count[category] = 0

        user_category_count = user_category_count[self.categories]
        user_watch_count = user_category_count.sum(axis=1)
        user_category_ratio = user_category_count.div(user_watch_count, axis=0).fillna(0)

        user_feature_df = user_category_ratio.reset_index()
        user_feature_df = user_feature_df.merge(
            self.users_df[["user_id", "gender", "age_group", "interest_categories"]],
            on="user_id", how="left"
        )

        user_feature_df["watch_count"] = (
            user_feature_df["user_id"].map(user_watch_count).fillna(0).astype(int)
        )

        # ── 行为统计特征 ──
        user_behavior = (
            self.logs_df
            .groupby("user_id")
            .agg(
                avg_watch_duration=("watch_duration", "mean"),
                like_rate=("liked", "mean"),
                collect_rate=("collected", "mean"),
                share_rate=("shared", "mean"),
            )
            .reset_index()
        )

        user_feature_df = user_feature_df.merge(user_behavior, on="user_id", how="left")
        user_feature_df[["avg_watch_duration", "like_rate",
                         "collect_rate", "share_rate"]] = \
            user_feature_df[["avg_watch_duration", "like_rate",
                             "collect_rate", "share_rate"]].fillna(0)

        # ── 内容多样性特征 ──
        # 观看类别数（归一化）
        n_categories_watched = (user_category_count > 0).sum(axis=1)
        user_feature_df["category_diversity"] = (
            n_categories_watched / len(self.categories)
        )

        # 香农熵：衡量兴趣分布的均匀程度
        # H = -sum(p_i * log(p_i)), 熵越高 = 兴趣越分散
        eps = 1e-10
        entropy = -(user_category_ratio * np.log(user_category_ratio + eps)).sum(axis=1)
        max_entropy = np.log(len(self.categories))
        user_feature_df["interest_entropy"] = entropy / max_entropy  # 归一化

        # ── 特征矩阵 ──
        feature_cols = (
            self.categories +  # 10 维类别比例
            ["avg_watch_duration", "like_rate", "collect_rate", "share_rate",
             "category_diversity", "interest_entropy"]
        )

        feature_matrix = user_feature_df[feature_cols].copy()
        feature_matrix = feature_matrix.fillna(0)

        return user_feature_df, feature_matrix

    # ==========================================================================
    # 11. 用户聚类 F7（手动实现 K-Means + PCA）
    # ==========================================================================

    def cluster_users(self):
        """对用户进行聚类分析（手动实现 K-Means + PCA）。"""
        user_feature_df, feature_matrix = self._build_user_features()
        X = feature_matrix.values.astype(np.float64)

        # ── 标准化 ──
        scaler = StandardScalerManual()
        X_scaled = scaler.fit_transform(X)

        # ── 肘部法则自动选 K ──
        if self.auto_k:
            print("正在执行肘部法则（用户聚类），寻找最优 K 值...")
            self.user_elbow, self.user_recommended_k = elbow_method(
                X_scaled,
                k_range=range(2, 10),
                random_state=2026
            )
            self.user_clusters = self.user_recommended_k
            print(f"  肘部法则推荐 K = {self.user_recommended_k}")

        # ── K-Means 聚类 ──
        print(f"开始 K-Means 聚类（K={self.user_clusters}）...")
        kmeans = KMeansManual(
            n_clusters=self.user_clusters,
            n_init=5,
            max_iter=100,
            random_state=2026
        )
        kmeans.fit(X_scaled)
        labels = kmeans.labels_

        print(f"  聚类完成：{kmeans.n_iter_} 次迭代收敛，惯性 = {kmeans.inertia_:.2f}")

        # ── 轮廓系数评估 ──
        print("正在计算用户聚类轮廓系数...")
        self.user_silhouette = silhouette_score_manual(X_scaled, labels)
        print(f"  用户聚类轮廓系数 = {self.user_silhouette:.4f}")

        # ── PCA 降维到 2D 用于可视化 ──
        print("正在执行 PCA 降维...")
        pca = PCAManual(n_components=2, random_state=2026)
        pca_result = pca.fit_transform(X_scaled)
        print(f"  PCA 解释方差比：{pca.explained_variance_ratio_.round(4)}")

        user_feature_df["cluster_id"] = labels
        user_feature_df["x"] = pca_result[:, 0]
        user_feature_df["y"] = pca_result[:, 1]

        cluster_name_map = self._generate_user_cluster_names(user_feature_df)
        user_feature_df["cluster_name"] = (
            user_feature_df["cluster_id"].map(cluster_name_map)
        )

        self.user_cluster_result = user_feature_df[[
            "user_id", "gender", "age_group", "interest_categories",
            "watch_count", "cluster_id", "cluster_name", "x", "y"
        ] + self.categories].copy()

        self.user_cluster_summary = self._summarize_user_clusters(
            self.user_cluster_result
        )

        return self.user_cluster_result

    # ==========================================================================
    # 12. 聚类名称生成
    # ==========================================================================

    def _generate_video_cluster_names(self, result_df):
        cluster_name_map = {}
        for cluster_id in sorted(result_df["cluster_id"].unique()):
            cluster_df = result_df[result_df["cluster_id"] == cluster_id]
            main_category = cluster_df["category"].value_counts().idxmax()
            cluster_name_map[cluster_id] = f"{main_category}类视频群"
        return cluster_name_map

    def _generate_user_cluster_names(self, result_df):
        cluster_name_map = {}
        name_map = {
            "搞笑": "搞笑娱乐型用户",
            "游戏": "游戏竞技型用户",
            "美食": "美食生活型用户",
            "学习": "学习提升型用户",
            "运动": "运动健康型用户",
            "音乐": "音乐娱乐型用户",
            "动漫": "动漫二次元型用户",
            "宠物": "宠物治愈型用户",
            "科技": "科技数码型用户",
            "生活": "生活日常型用户"
        }

        for cluster_id in sorted(result_df["cluster_id"].unique()):
            cluster_df = result_df[result_df["cluster_id"] == cluster_id]
            avg_interest = cluster_df[self.categories].mean()
            top_category = avg_interest.idxmax()
            cluster_name_map[cluster_id] = name_map.get(
                top_category, f"{top_category}兴趣型用户"
            )

        return cluster_name_map

    # ==========================================================================
    # 13. 聚类汇总
    # ==========================================================================

    def _summarize_video_clusters(self, result_df):
        rows = []
        for cluster_id in sorted(result_df["cluster_id"].unique()):
            cluster_df = result_df[result_df["cluster_id"] == cluster_id]
            rows.append({
                "cluster_id": cluster_id,
                "cluster_name": cluster_df["cluster_name"].iloc[0],
                "main_category": cluster_df["category"].value_counts().idxmax(),
                "video_count": len(cluster_df),
                "avg_watch_count": round(cluster_df["watch_count"].mean(), 2),
                "avg_like_rate": round(cluster_df["like_rate"].mean(), 4),
                "avg_collect_rate": round(cluster_df["collect_rate"].mean(), 4),
                "avg_share_rate": round(cluster_df["share_rate"].mean(), 4)
            })
        return pd.DataFrame(rows)

    def _summarize_user_clusters(self, result_df):
        rows = []
        for cluster_id in sorted(result_df["cluster_id"].unique()):
            cluster_df = result_df[result_df["cluster_id"] == cluster_id]
            avg_interest = cluster_df[self.categories].mean()
            top_category = avg_interest.idxmax()
            top3 = avg_interest.sort_values(ascending=False).head(3).index.tolist()
            rows.append({
                "cluster_id": cluster_id,
                "cluster_name": cluster_df["cluster_name"].iloc[0],
                "main_interest": top_category,
                "top3_interests": "、".join(top3),
                "user_count": len(cluster_df),
                "avg_watch_count": round(cluster_df["watch_count"].mean(), 2)
            })
        return pd.DataFrame(rows)

    # ==========================================================================
    # 14. 获取聚类结果
    # ==========================================================================

    def get_video_cluster_result(self):
        if self.video_cluster_result is None:
            return self.cluster_videos()
        return self.video_cluster_result

    def get_user_cluster_result(self):
        if self.user_cluster_result is None:
            return self.cluster_users()
        return self.user_cluster_result

    def get_video_cluster_summary(self):
        if self.video_cluster_summary is None:
            self.cluster_videos()
        return self.video_cluster_summary

    def get_user_cluster_summary(self):
        if self.user_cluster_summary is None:
            self.cluster_users()
        return self.user_cluster_summary

    def get_representative_videos(self, top_n=5):
        result_df = self.get_video_cluster_result()
        rows = []
        for cluster_id in sorted(result_df["cluster_id"].unique()):
            cluster_df = result_df[result_df["cluster_id"] == cluster_id]
            top_videos = cluster_df.sort_values(
                "watch_count", ascending=False
            ).head(top_n)
            for _, row in top_videos.iterrows():
                rows.append({
                    "cluster_id": row["cluster_id"],
                    "cluster_name": row["cluster_name"],
                    "video_id": row["video_id"],
                    "title": row["title"],
                    "category": row["category"],
                    "tags": row["tags"],
                    "watch_count": row["watch_count"],
                    "like_rate": round(row["like_rate"], 4)
                })
        return pd.DataFrame(rows)

    def get_representative_users(self, top_n=5):
        result_df = self.get_user_cluster_result()
        rows = []
        for cluster_id in sorted(result_df["cluster_id"].unique()):
            cluster_df = result_df[result_df["cluster_id"] == cluster_id]
            top_users = cluster_df.sort_values(
                "watch_count", ascending=False
            ).head(top_n)
            for _, row in top_users.iterrows():
                rows.append({
                    "cluster_id": row["cluster_id"],
                    "cluster_name": row["cluster_name"],
                    "user_id": row["user_id"],
                    "gender": row["gender"],
                    "age_group": row["age_group"],
                    "interest_categories": row["interest_categories"],
                    "watch_count": row["watch_count"]
                })
        return pd.DataFrame(rows)

    def save_results(self):
        """保存聚类结果到 data 文件夹。"""
        video_result = self.get_video_cluster_result()
        user_result = self.get_user_cluster_result()
        video_summary = self.get_video_cluster_summary()
        user_summary = self.get_user_cluster_summary()

        video_result.to_csv(DATA_DIR / "video_cluster_result.csv",
                           index=False, encoding="utf-8-sig")
        user_result.to_csv(DATA_DIR / "user_cluster_result.csv",
                          index=False, encoding="utf-8-sig")
        video_summary.to_csv(DATA_DIR / "video_cluster_summary.csv",
                            index=False, encoding="utf-8-sig")
        user_summary.to_csv(DATA_DIR / "user_cluster_summary.csv",
                           index=False, encoding="utf-8-sig")

        # 保存评估指标
        metrics = {
            "video_clusters": int(self.video_clusters),
            "user_clusters": int(self.user_clusters),
            "video_silhouette": self.video_silhouette,
            "user_silhouette": self.user_silhouette,
            "video_elbow": {str(k): float(v) for k, v in (self.video_elbow or {}).items()} if self.video_elbow else None,
            "user_elbow": {str(k): float(v) for k, v in (self.user_elbow or {}).items()} if self.user_elbow else None,
        }

        import json
        with open(DATA_DIR / "clustering_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print("聚类结果和评估指标已保存到 data 文件夹。")


# ==============================================================================
# 15. 命令行测试
# ==============================================================================

if __name__ == "__main__":
    analyzer = ShortVideoClusterAnalyzer(
        video_clusters=6,
        user_clusters=5,
        auto_k=True   # 启用肘部法则自动选 K
    )

    print("=" * 60)
    print("开始进行视频聚类 F6...")
    print("=" * 60)
    video_result = analyzer.cluster_videos()
    print("\n视频聚类汇总：")
    print(analyzer.get_video_cluster_summary())
    print(f"\n视频聚类轮廓系数：{analyzer.video_silhouette:.4f}")

    print("\n" + "=" * 60)
    print("开始进行用户聚类 F7...")
    print("=" * 60)
    user_result = analyzer.cluster_users()
    print("\n用户聚类汇总：")
    print(analyzer.get_user_cluster_summary())
    print(f"\n用户聚类轮廓系数：{analyzer.user_silhouette:.4f}")

    print("\n保存聚类结果...")
    analyzer.save_results()
    print("全部聚类分析完成！")

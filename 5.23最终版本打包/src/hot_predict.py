import pandas as pd
import numpy as np
import math

from config import get_data_dir


# ==============================================================================
# 1. 自实现数据结构：双端队列 Deque（循环数组实现）
# ==============================================================================

class Deque:
    """自实现双端队列，基于循环数组。

    用于维护视频每日播放量的滑动窗口：
    - 新一天数据从右侧入队，过期数据从左侧出队
    - 所有操作 O(1)

    时间复杂度：
    - append / appendleft: O(1) 均摊
    - pop / popleft:       O(1)
    - peek / peekleft:      O(1)
    """

    def __init__(self, capacity=64):
        self.capacity = capacity
        self.arr = [None] * capacity
        self.front = 0       # 队首指针
        self.rear = 0        # 队尾指针（下一个插入位置）
        self.size = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        """从队首到队尾迭代。"""
        for i in range(self.size):
            yield self.arr[(self.front + i) % self.capacity]

    def __getitem__(self, idx):
        """按索引访问（0 = 队首）。"""
        if idx < 0 or idx >= self.size:
            raise IndexError("索引越界")
        return self.arr[(self.front + idx) % self.capacity]

    def _resize(self, new_capacity):
        """扩容。"""
        new_arr = [None] * new_capacity
        for i in range(self.size):
            new_arr[i] = self.arr[(self.front + i) % self.capacity]
        self.arr = new_arr
        self.front = 0
        self.rear = self.size
        self.capacity = new_capacity

    def append(self, item):
        """右侧入队。"""
        if self.size == self.capacity:
            self._resize(self.capacity * 2)
        self.arr[self.rear] = item
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1

    def appendleft(self, item):
        """左侧入队。"""
        if self.size == self.capacity:
            self._resize(self.capacity * 2)
        self.front = (self.front - 1) % self.capacity
        self.arr[self.front] = item
        self.size += 1

    def pop(self):
        """右侧出队。"""
        if self.size == 0:
            raise IndexError("队列为空")
        self.rear = (self.rear - 1) % self.capacity
        item = self.arr[self.rear]
        self.arr[self.rear] = None
        self.size -= 1
        return item

    def popleft(self):
        """左侧出队。"""
        if self.size == 0:
            raise IndexError("队列为空")
        item = self.arr[self.front]
        self.arr[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item

    def to_list(self):
        """返回 list 形式（从队首到队尾）。"""
        return list(self)


# ==============================================================================
# 2. 自实现数据结构：树状数组 Fenwick Tree（Binary Indexed Tree）
# ==============================================================================

class FenwickTree:
    """自实现树状数组，用于高效区间和查询。

    用于对视频过去 N 天的每日播放量进行区间求和：
    - prefix_sum(k): 前 k 个元素之和 → O(log n)
    - range_sum(l, r): 区间 [l, r] 元素之和 → O(log n)
    - update(idx, delta): 单点增量更新 → O(log n)

    原理：
    tree[i] 存储区间 (i - lowbit(i), i] 的元素之和
    lowbit(i) = i & (-i)
    """

    def __init__(self, size):
        self.n = size
        self.tree = [0] * (size + 1)   # 1-indexed

    @staticmethod
    def _lowbit(x):
        return x & (-x)

    def update(self, idx, delta):
        """在 idx 位置增加 delta（0-indexed → 内部 1-indexed）。"""
        i = idx + 1
        while i <= self.n:
            self.tree[i] += delta
            i += self._lowbit(i)

    def prefix_sum(self, idx):
        """查询前 idx 个元素之和（0-indexed，idx 为元素个数）。"""
        if idx <= 0:
            return 0
        i = min(idx, self.n)
        result = 0
        while i > 0:
            result += self.tree[i]
            i -= self._lowbit(i)
        return result

    def range_sum(self, l, r):
        """查询区间 [l, r] 之和（0-indexed，闭区间）。"""
        if l > r:
            return 0
        return self.prefix_sum(r + 1) - self.prefix_sum(l)


# ==============================================================================
# 3. 路径配置
# ==============================================================================

DATA_DIR = get_data_dir()


# ==============================================================================
# 4. 视频热度预测类
# ==============================================================================

class VideoHeatPredictor:
    def __init__(self, alpha=0.6, beta=0.3, videos_df=None, logs_df=None):
        """初始化视频热度预测模块。

        alpha: 水平平滑系数 (0 < alpha < 1)，越大越敏感于近期变化
        beta:  趋势平滑系数 (0 < beta < 1)，越大越敏感于趋势变化
        videos_df / logs_df: 可选，外部传入的共享 DataFrame（避免重复读盘）
        """
        self.alpha = alpha
        self.beta = beta

        if videos_df is None:
            videos_df = pd.read_csv(DATA_DIR / "videos.csv", encoding="utf-8-sig")
        if logs_df is None:
            logs_df = pd.read_csv(DATA_DIR / "watch_logs.csv", encoding="utf-8-sig")
        self.videos_df = videos_df
        self.logs_df = logs_df

        # watch_time 转 datetime（幂等：若已是 datetime 则跳过）
        if not pd.api.types.is_datetime64_any_dtype(self.logs_df["watch_time"]):
            self.logs_df["watch_time"] = pd.to_datetime(self.logs_df["watch_time"])

        # watch_date 用局部 Series 持有，避免污染共享 logs_df
        self._watch_dates = self.logs_df["watch_time"].dt.date

        self.video_ids = set(self.videos_df["video_id"].astype(int).tolist())
        self.current_date = self.logs_df["watch_time"].max().date()

        # 构建每日播放量哈希表：video_id -> {date -> count}
        self._build_daily_index()

    def _build_daily_index(self):
        """构建每个视频的每日播放量索引。

        使用数据结构：
        - daily_play_count:    dict[video_id, dict[date, count]]
          → 哈希表 + 哈希表，O(1) 查询某视频某天的播放量
        - video_daily_fenwick: dict[video_id, FenwickTree]
          → 为每个视频构建树状数组，O(log n) 区间求和
        """
        self.daily_play_count = {}     # video_id -> {date -> count}
        self.video_daily_fenwick = {}  # video_id -> FenwickTree

        # 按视频+日期聚合（使用局部 watch_date Series，不污染共享 df）
        grouped = (
            self.logs_df
            .groupby([self.logs_df["video_id"], self._watch_dates])
            .size()
            .reset_index(name="count")
        )
        grouped.columns = ["video_id", "watch_date", "count"]

        # 生成完整日期范围
        all_dates = pd.date_range(
            start=self._watch_dates.min(),
            end=self.current_date
        )
        self.date_list = [d.date() for d in all_dates]
        self.num_days = len(self.date_list)
        self.date_to_idx = {d: i for i, d in enumerate(self.date_list)}

        # 初始化每个视频的每日数据
        for video_id in self.video_ids:
            self.daily_play_count[video_id] = {}
            self.video_daily_fenwick[video_id] = FenwickTree(self.num_days)

        for vid, date, count in grouped.itertuples(index=False, name=None):
            vid = int(vid)
            count = int(count)
            self.daily_play_count[vid][date] = count
            if date in self.date_to_idx:
                self.video_daily_fenwick[vid].update(self.date_to_idx[date], count)

    def get_video_info(self, video_id):
        """根据 video_id 获取视频基本信息。"""
        video_id = int(video_id)
        video = self.videos_df[self.videos_df["video_id"] == video_id]
        if video.empty:
            return None

        row = video.iloc[0]
        return {
            "video_id": int(row["video_id"]),
            "title": row["title"],
            "category": row["category"],
            "tags": row["tags"],
            "duration": int(row["duration"]),
            "author_id": int(row["author_id"]),
            "publish_time": row["publish_time"]
        }

    # ==========================================================================
    # 5. 滑动窗口历史数据（使用 Deque）
    # ==========================================================================

    def get_history_heat(self, video_id, days=30):
        """获取视频过去 days 天的每日播放量。

        使用 Deque 维护滑动窗口：
        - 从最早日期到最新日期依次入队
        - 窗口大小固定为 days，超出的日期从左侧出队
        """
        video_id = int(video_id)

        if video_id not in self.video_ids:
            return pd.DataFrame()

        # 过去的 days 个日期
        end_idx = self.date_to_idx.get(self.current_date, self.num_days - 1)
        start_idx = max(0, end_idx - days + 1)

        # ★ 使用 Deque 构建滑动窗口
        dq = Deque()
        dates = []

        for i in range(start_idx, end_idx + 1):
            date = self.date_list[i]
            count = self.daily_play_count[video_id].get(date, 0)
            dq.append(count)
            dates.append(date)

        history_df = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "play_count": dq.to_list(),
            "type": "历史播放量"
        })

        return history_df

    # ==========================================================================
    # 6. 指数平滑预测（Holt 双参数法）
    # ==========================================================================

    def predict_future_heat(self, video_id, history_days=30, predict_days=7):
        """使用 Holt 双参数指数平滑法预测未来播放量。

        算法：
        - Level:  ℓ_t = α·y_t + (1-α)(ℓ_{t-1} + b_{t-1})
        - Trend:  b_t = β·(ℓ_t - ℓ_{t-1}) + (1-β)·b_{t-1}
        - 预测:   ŷ_{t+h} = ℓ_t + h·b_t

        相比旧版的线性外推：
        - 对近期数据更敏感
        - 趋势是平滑的，不会无界膨胀
        - 有理论支撑（指数加权）

        使用数据结构：
        - Deque: 维护历史序列的滑动窗口
        - FenwickTree: O(log n) 区间求和（用于计算各种移动平均）
        """
        history_df = self.get_history_heat(video_id, days=history_days)

        if history_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        history_values = history_df["play_count"].values.astype(float)
        n = len(history_values)

        # ── Holt 指数平滑参数估计 ──
        # 初始化 level 和 trend
        if n >= 2:
            l_0 = history_values[0]
            b_0 = history_values[1] - history_values[0]
        else:
            l_0 = history_values[0] if n > 0 else 0
            b_0 = 0

        l_t = l_0
        b_t = b_0
        fitted = [l_0]

        for t in range(1, n):
            y_t = history_values[t]
            l_prev, b_prev = l_t, b_t
            l_t = self.alpha * y_t + (1 - self.alpha) * (l_prev + b_prev)
            b_t = self.beta * (l_t - l_prev) + (1 - self.beta) * b_prev
            fitted.append(l_t)

        # ── 计算残差标准差（用于置信区间）──
        residuals = [history_values[i] - fitted[i] for i in range(n)]
        rmse = math.sqrt(sum(r**2 for r in residuals) / n) if n > 0 else 0

        # ── 使用树状数组计算最近 7 天与前 7 天的对比 ──
        video_id_int = int(video_id)
        bit = self.video_daily_fenwick.get(video_id_int)
        end_idx = self.date_to_idx.get(self.current_date, self.num_days - 1)

        if bit is not None and end_idx >= 6:
            recent_7_sum = bit.range_sum(end_idx - 6, end_idx)
            if end_idx >= 13:
                prev_7_sum = bit.range_sum(end_idx - 13, end_idx - 7)
            else:
                prev_7_sum = 0
        else:
            recent_7_sum = sum(history_values[-7:]) if len(history_values) >= 7 else sum(history_values)
            prev_7_sum = sum(history_values[-14:-7]) if len(history_values) >= 14 else 0

        # ── 生成预测 ──
        future_dates = pd.date_range(
            start=self.current_date + pd.Timedelta(days=1),
            periods=predict_days
        )

        predictions = []
        upper_bounds = []
        lower_bounds = []

        for h in range(1, predict_days + 1):
            forecast = l_t + h * b_t
            forecast = max(0, round(forecast))

            # 95% 置信区间（±1.96 * RMSE * sqrt(h)，预测越远越宽）
            margin = 1.96 * rmse * math.sqrt(h)
            upper = max(0, round(forecast + margin))
            lower = max(0, round(forecast - margin))

            predictions.append(forecast)
            upper_bounds.append(upper)
            lower_bounds.append(lower)

        predict_df = pd.DataFrame({
            "date": future_dates,
            "play_count": predictions,
            "upper_bound": upper_bounds,
            "lower_bound": lower_bounds,
            "type": "预测播放量"
        })

        return history_df, predict_df

    # ==========================================================================
    # 7. 合并历史与预测数据
    # ==========================================================================

    def get_heat_trend(self, video_id, history_days=30, predict_days=7):
        """返回历史 + 预测合并数据。"""
        history_df, predict_df = self.predict_future_heat(
            video_id=video_id,
            history_days=history_days,
            predict_days=predict_days
        )

        if history_df.empty:
            return pd.DataFrame()

        result_df = pd.concat([history_df, predict_df], ignore_index=True)
        return result_df

    # ==========================================================================
    # 8. 热度趋势文字分析
    # ==========================================================================

    def analyze_heat_trend(self, video_id):
        """对视频热度趋势生成文字分析报告。"""
        history_df, predict_df = self.predict_future_heat(video_id)

        if history_df.empty:
            return "该视频 ID 不存在，无法进行热度预测。"

        total_30 = int(history_df["play_count"].sum())
        recent_7 = int(history_df["play_count"].tail(7).sum())
        previous_7 = int(history_df["play_count"].tail(14).head(7).sum())
        future_7 = int(predict_df["play_count"].sum())

        if recent_7 > previous_7 * 1.1:
            trend = "上升"
            verb = "增长"
        elif recent_7 < previous_7 * 0.9:
            trend = "下降"
            verb = "回落"
        else:
            trend = "平稳"
            verb = "保持平稳"

        # 预测趋势
        pred_values = predict_df["play_count"].values
        if len(pred_values) >= 2:
            if pred_values[-1] > pred_values[0] * 1.1:
                pred_trend = "持续上升"
            elif pred_values[-1] < pred_values[0] * 0.9:
                pred_trend = "逐步下降"
            else:
                pred_trend = "趋于稳定"
        else:
            pred_trend = "单日预测"

        # 用树状数组做区间查询展示
        video_id_int = int(video_id)
        bit = self.video_daily_fenwick.get(video_id_int)
        end_idx = self.date_to_idx.get(self.current_date, self.num_days - 1)

        if bit is not None and end_idx >= 14:
            recent_14_fenwick = bit.range_sum(end_idx - 13, end_idx)
        else:
            recent_14_fenwick = None

        analysis = (
            f"该视频过去 30 天总播放量为 {total_30} 次。"
            f"最近 7 天播放量 {recent_7} 次，"
            f"前 7 天播放量 {previous_7} 次，"
            f"热度整体呈{trend}趋势。"
            f"采用 Holt 双参数指数平滑法（α={self.alpha}, β={self.beta}）"
            f"预测未来 7 天播放量约 {future_7} 次，{pred_trend}。"
        )

        if recent_14_fenwick is not None:
            analysis += (
                f"（树状数组查询：最近 14 天总播放量 = {recent_14_fenwick}）"
            )

        return analysis

    # ==========================================================================
    # 9. 离线评估（回测）
    # ==========================================================================

    def evaluate_prediction(self, video_ids=None, history_days=30,
                             predict_days=7):
        """离线评估预测性能。

        对于每个视频：
        1. 用前 (n - predict_days) 天的数据训练模型
        2. 预测后 predict_days 天
        3. 与真实值比较

        评估指标：
        - MAE (平均绝对误差)
        - RMSE (均方根误差)
        - MAPE (平均绝对百分比误差) — 仅非零真实值
        - sMAPE (对称 MAPE) — 有界 [0, 200%]，对近零值鲁棒
        - 按视频活跃度分层评估（高/中/低），揭示不同热度段的预测能力
        """
        if video_ids is None:
            video_ids = sorted(self.video_ids)[:500]

        # 收集每个视频的预测-实际对（按视频分组，用于分层）
        video_samples = {}  # video_id -> (actuals, preds, avg_daily)

        for video_id in video_ids:
            video_id = int(video_id)
            hist = self.get_history_heat(video_id, days=history_days)
            if hist.empty or len(hist) < history_days:
                continue

            values = hist["play_count"].values.astype(float)
            if len(values) < predict_days + 7:
                continue

            n = len(values)
            train = values[:n - predict_days]
            test = values[n - predict_days:]

            if len(train) < 2:
                continue

            l_t = train[0]
            b_t = train[1] - train[0] if len(train) >= 2 else 0
            for t in range(1, len(train)):
                y_t = train[t]
                l_prev, b_prev = l_t, b_t
                l_t = self.alpha * y_t + (1 - self.alpha) * (l_prev + b_prev)
                b_t = self.beta * (l_t - l_prev) + (1 - self.beta) * b_prev

            avg_daily = float(np.mean(values))
            video_samples[video_id] = {
                "actuals": [],
                "preds": [],
                "avg_daily": avg_daily
            }

            for h in range(1, predict_days + 1):
                pred = max(0, l_t + h * b_t)
                video_samples[video_id]["actuals"].append(test[h - 1])
                video_samples[video_id]["preds"].append(pred)

        if not video_samples:
            return {"MAE": None, "RMSE": None, "MAPE": None, "sMAPE": None,
                    "note": "没有足够数据进行评估"}

        def _metrics(actuals, preds):
            """计算一组样本的评估指标."""
            actuals = np.array(actuals)
            preds = np.array(preds)
            mae = float(np.mean(np.abs(actuals - preds)))
            rmse = float(math.sqrt(np.mean((actuals - preds) ** 2)))

            # MAPE: 仅非零真实值
            nonzero = actuals > 0
            mape = None
            if nonzero.sum() > 0:
                mape = float(np.mean(
                    np.abs((actuals[nonzero] - preds[nonzero]) / actuals[nonzero])
                )) * 100

            # sMAPE: 对称 MAPE，有界 [0, 200%]，对近零值鲁棒
            denom = (np.abs(actuals) + np.abs(preds))
            smape_mask = denom > 0
            if smape_mask.sum() > 0:
                smape = float(np.mean(
                    200 * np.abs(actuals[smape_mask] - preds[smape_mask])
                    / denom[smape_mask]
                ))
            else:
                smape = None

            return mae, rmse, mape, smape

        # ── 全量指标 ──
        all_actuals = []
        all_preds = []
        for v in video_samples.values():
            all_actuals.extend(v["actuals"])
            all_preds.extend(v["preds"])

        total_mae, total_rmse, total_mape, total_smape = _metrics(all_actuals, all_preds)

        # ── 按日均播放量分层（高/中/低活跃度）──
        avg_dailies = [v["avg_daily"] for v in video_samples.values()]
        avg_dailies_sorted = sorted(avg_dailies)
        n_videos = len(avg_dailies_sorted)
        low_thresh = avg_dailies_sorted[n_videos // 3]
        high_thresh = avg_dailies_sorted[2 * n_videos // 3]

        tier_actuals = {"high": [], "mid": [], "low": []}
        tier_preds = {"high": [], "mid": [], "low": []}

        for v in video_samples.values():
            if v["avg_daily"] >= high_thresh:
                tier = "high"
            elif v["avg_daily"] >= low_thresh:
                tier = "mid"
            else:
                tier = "low"
            tier_actuals[tier].extend(v["actuals"])
            tier_preds[tier].extend(v["preds"])

        tier_results = {}
        for tier in ["high", "mid", "low"]:
            mae, rmse, mape, smape = _metrics(tier_actuals[tier], tier_preds[tier])
            tier_results[tier] = {
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "MAPE": round(mape, 2) if mape is not None else None,
                "sMAPE": round(smape, 2) if smape is not None else None,
                "samples": len(tier_actuals[tier])
            }

        return {
            "MAE": round(total_mae, 4),
            "RMSE": round(total_rmse, 4),
            "MAPE": round(total_mape, 2) if total_mape is not None else None,
            "sMAPE": round(total_smape, 2) if total_smape is not None else None,
            "num_samples": len(all_actuals),
            "num_videos": n_videos,
            "by_activity": tier_results
        }


# ==============================================================================
# 10. 命令行测试
# ==============================================================================

if __name__ == "__main__":
    predictor = VideoHeatPredictor(alpha=0.6, beta=0.3)

    video_id = int(input("请输入要预测的视频 ID："))

    video_info = predictor.get_video_info(video_id)

    if video_info is None:
        print("该视频 ID 不存在。")
    else:
        print("\n==============================")
        print("视频基本信息")
        print("==============================")
        print(video_info)

        print("\n==============================")
        print("热度趋势分析")
        print("==============================")
        print(predictor.analyze_heat_trend(video_id))

        print("\n==============================")
        print("过去 30 天 + 未来 7 天播放量")
        print("==============================")

        trend_df = predictor.get_heat_trend(video_id)

        print(trend_df)

        save_path = DATA_DIR / f"video_{video_id}_heat_predict.csv"
        trend_df.to_csv(save_path, index=False, encoding="utf-8-sig")

        print(f"\n预测结果已保存到：{save_path}")

    print("\n==============================")
    print("离线评估（500 个视频回测）")
    print("==============================")
    eval_results = predictor.evaluate_prediction(video_ids=list(range(1, 501)))
    for k, v in eval_results.items():
        if k == "by_activity":
            print("  分层评估：")
            for tier, metrics in v.items():
                print(f"    [{tier}活跃度] MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, "
                      f"MAPE={metrics['MAPE']}%, sMAPE={metrics['sMAPE']}%, "
                      f"样本={metrics['samples']}")
        else:
            print(f"  {k}: {v}")

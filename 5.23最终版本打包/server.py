"""
短视频推荐系统 — Flask Web API Server
提供 5 个功能页面的数据接口，驱动前端 Dashboard。
"""
import sys
import os
import json
from pathlib import Path

# 确保 src/ 目录在 Python 路径中
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np

from config import get_data_dir
from recommender import ShortVideoRecommender
from hot_predict import VideoHeatPredictor
from clustering import ShortVideoClusterAnalyzer

app = Flask(__name__)

# ============================================================
# 全局模块实例（服务启动时初始化一次）
# ============================================================
DATA_DIR = get_data_dir()

# 集中加载一次共享数据，避免三个模块各自重复读盘
print("正在加载共享数据...")
videos_df = pd.read_csv(DATA_DIR / "videos.csv", encoding="utf-8-sig")
users_df = pd.read_csv(DATA_DIR / "users.csv", encoding="utf-8-sig")
logs_df = pd.read_csv(DATA_DIR / "watch_logs.csv", encoding="utf-8-sig")
logs_df["watch_time"] = pd.to_datetime(logs_df["watch_time"])
print(f"共享数据就绪: {len(videos_df)} videos, {len(users_df)} users, {len(logs_df)} logs")

print("正在初始化推荐系统模块...")
recommender = ShortVideoRecommender(videos_df=videos_df, users_df=users_df, logs_df=logs_df)
print("推荐系统模块就绪。")

print("正在初始化热度预测模块...")
predictor = VideoHeatPredictor(alpha=0.6, beta=0.3, videos_df=videos_df, logs_df=logs_df)
print("热度预测模块就绪。")

print("正在初始化聚类分析模块...")
analyzer = ShortVideoClusterAnalyzer(
    video_clusters=6, user_clusters=5, auto_k=False,
    videos_df=videos_df, users_df=users_df, logs_df=logs_df,
)
print("聚类分析模块就绪。")

# ============================================================
# 辅助：JSON 序列化
# ============================================================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)

app.json_encoder = NpEncoder

def json_response(data, status=200):
    """返回 JSON 响应，处理 numpy 类型。"""
    return app.response_class(
        response=json.dumps(data, cls=NpEncoder, ensure_ascii=False),
        status=status,
        mimetype="application/json"
    )

# ============================================================
# 页面路由
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")

# ============================================================
# API: Overview — 系统总览数据
# ============================================================
@app.route("/api/overview")
def api_overview():
    try:
        total_users = len(recommender.user_info)
        total_videos = len(recommender.video_info)
        total_logs = len(recommender.logs_df)

        # 热门类别统计
        category_counts = recommender.videos_df["category"].value_counts().to_dict()

        # 每日观看量趋势
        recommender.logs_df["watch_date"] = pd.to_datetime(
            recommender.logs_df["watch_time"]
        ).dt.date
        daily_views = (
            recommender.logs_df.groupby("watch_date")
            .size()
            .reset_index(name="count")
        )
        daily_views["watch_date"] = daily_views["watch_date"].astype(str)

        # 互动率
        total_likes = int(recommender.logs_df["liked"].sum())
        total_collects = int(recommender.logs_df["collected"].sum())
        total_shares = int(recommender.logs_df["shared"].sum())

        # 聚类指标
        clustering_metrics_path = DATA_DIR / "clustering_metrics.json"
        cluster_metrics = {}
        if clustering_metrics_path.exists():
            with open(clustering_metrics_path, "r", encoding="utf-8") as f:
                cluster_metrics = json.load(f)

        return json_response({
            "total_users": total_users,
            "total_videos": total_videos,
            "total_logs": total_logs,
            "total_likes": total_likes,
            "total_collects": total_collects,
            "total_shares": total_shares,
            "category_counts": category_counts,
            "daily_views": daily_views.to_dict(orient="records"),
            "video_clusters": cluster_metrics.get("video_clusters", 9),
            "user_clusters": cluster_metrics.get("user_clusters", 8),
            "video_silhouette": cluster_metrics.get("video_silhouette"),
            "user_silhouette": cluster_metrics.get("user_silhouette"),
        })
    except Exception as e:
        return json_response({"error": str(e)}, 500)

# ============================================================
# API: Recommendations (F3 + F4)
# ============================================================
@app.route("/api/recommend")
def api_recommend():
    try:
        user_id = request.args.get("user_id", type=int)
        top_k = request.args.get("top_k", default=10, type=int)

        if user_id is None:
            return json_response({"error": "缺少 user_id 参数"}, 400)

        profile = recommender.get_user_profile(user_id)
        if profile is None:
            return json_response({"error": f"用户 {user_id} 不存在"}, 404)

        similar_users = recommender.get_similar_users(user_id, top_k=top_k)
        recommendations = recommender.recommend_videos(user_id, top_k=top_k)

        return json_response({
            "user_profile": profile,
            "similar_users": similar_users,
            "recommendations": recommendations,
        })
    except Exception as e:
        return json_response({"error": str(e)}, 500)

# ============================================================
# API: Heat Prediction (F5)
# ============================================================
@app.route("/api/heat/predict")
def api_heat_predict():
    try:
        video_id = request.args.get("video_id", type=int)
        history_days = request.args.get("history_days", default=30, type=int)
        predict_days = request.args.get("predict_days", default=7, type=int)

        if video_id is None:
            return json_response({"error": "缺少 video_id 参数"}, 400)

        info = predictor.get_video_info(video_id)
        if info is None:
            return json_response({"error": f"视频 {video_id} 不存在"}, 404)

        history_df, predict_df = predictor.predict_future_heat(
            video_id, history_days=history_days, predict_days=predict_days
        )

        result = {
            "video_info": info,
            "history": [],
            "prediction": [],
        }

        if not history_df.empty:
            history_df["date"] = history_df["date"].astype(str)
            result["history"] = history_df.to_dict(orient="records")

        if not predict_df.empty:
            predict_df["date"] = predict_df["date"].astype(str)
            result["prediction"] = predict_df.to_dict(orient="records")

        return json_response(result)
    except Exception as e:
        return json_response({"error": str(e)}, 500)

@app.route("/api/heat/analyze")
def api_heat_analyze():
    try:
        video_id = request.args.get("video_id", type=int)
        if video_id is None:
            return json_response({"error": "缺少 video_id 参数"}, 400)

        analysis = predictor.analyze_heat_trend(video_id)
        return json_response({"analysis": analysis})
    except Exception as e:
        return json_response({"error": str(e)}, 500)

# ============================================================
# API: Heat Ranking — 视频热度排名
# ============================================================
@app.route("/api/heat/ranking")
def api_heat_ranking():
    try:
        video_id = request.args.get("video_id", type=int)
        top_n = request.args.get("top_n", default=50, type=int)

        sorted_videos = sorted(
            recommender.video_heat.items(), key=lambda x: x[1], reverse=True
        )

        ranking = []
        searched_rank = None

        for rank, (vid, heat) in enumerate(sorted_videos[:top_n], 1):
            info = recommender.video_info.get(vid, {})
            entry = {
                "rank": rank,
                "video_id": vid,
                "title": info.get("title", ""),
                "category": info.get("category", ""),
                "heat": heat,
            }
            ranking.append(entry)
            if vid == video_id:
                searched_rank = rank
                entry["is_searched"] = True

        # 如果搜的视频不在 top_n，补查其排名
        if searched_rank is None and video_id is not None and video_id in recommender.video_heat:
            for rank, (vid, heat) in enumerate(sorted_videos, 1):
                if vid == video_id:
                    searched_rank = rank
                    break

        return json_response({
            "ranking": ranking,
            "searched_rank": searched_rank,
            "searched_video_id": video_id,
            "total_videos": len(recommender.video_info),
        })
    except Exception as e:
        return json_response({"error": str(e)}, 500)

# ============================================================
# API: Video Clusters (F6)
# ============================================================
@app.route("/api/video-clusters")
def api_video_clusters():
    try:
        result = analyzer.get_video_cluster_result()
        summary = analyzer.get_video_cluster_summary()

        # 抽样返回（10万条太多了，前端只需要散点图数据）
        sample_size = request.args.get("sample", default=5000, type=int)
        if len(result) > sample_size:
            result_sample = result.sample(n=sample_size, random_state=42)
        else:
            result_sample = result

        return json_response({
            "points": result_sample[[
                "video_id", "title", "category", "cluster_id",
                "cluster_name", "x", "y", "watch_count"
            ]].to_dict(orient="records"),
            "summary": summary.to_dict(orient="records") if summary is not None else [],
            "silhouette": analyzer.video_silhouette,
            "total_videos": len(result),
        })
    except Exception as e:
        return json_response({"error": str(e)}, 500)

# ============================================================
# API: User Clusters (F7)
# ============================================================
@app.route("/api/user-clusters")
def api_user_clusters():
    try:
        result = analyzer.get_user_cluster_result()
        summary = analyzer.get_user_cluster_summary()

        sample_size = request.args.get("sample", default=5000, type=int)
        if len(result) > sample_size:
            result_sample = result.sample(n=sample_size, random_state=42)
        else:
            result_sample = result

        # 选取关键列
        cols = ["user_id", "gender", "age_group", "cluster_id",
                "cluster_name", "x", "y", "watch_count"]
        available = [c for c in cols if c in result_sample.columns]

        return json_response({
            "points": result_sample[available].to_dict(orient="records"),
            "summary": summary.to_dict(orient="records") if summary is not None else [],
            "silhouette": analyzer.user_silhouette,
            "total_users": len(result),
        })
    except Exception as e:
        return json_response({"error": str(e)}, 500)

# ============================================================
# API: User Search (for autocomplete)
# ============================================================
@app.route("/api/users/search")
def api_users_search():
    try:
        q = request.args.get("q", default="", type=str)
        limit = request.args.get("limit", default=20, type=int)

        if not q:
            # 返回一些示例用户
            sample_ids = sorted(recommender.user_info.keys())[:limit]
            results = []
            for uid in sample_ids:
                info = recommender.user_info[uid]
                results.append({
                    "user_id": uid,
                    "gender": info.get("gender", ""),
                    "age_group": info.get("age_group", ""),
                    "interests": "|".join(info.get("interest_categories", [])),
                    "watch_count": len(recommender.user_watch.get(uid, set())),
                })
            return json_response(results)

        # 按 ID 或兴趣搜索
        results = []
        q_lower = q.lower()
        for uid, info in recommender.user_info.items():
            if len(results) >= limit:
                break
            interests_str = "|".join(info.get("interest_categories", []))
            if (q_lower in str(uid) or
                q_lower in interests_str.lower() or
                q_lower in info.get("gender", "").lower() or
                q_lower in info.get("age_group", "").lower()):
                results.append({
                    "user_id": uid,
                    "gender": info.get("gender", ""),
                    "age_group": info.get("age_group", ""),
                    "interests": interests_str,
                    "watch_count": len(recommender.user_watch.get(uid, set())),
                })

        return json_response(results)
    except Exception as e:
        return json_response({"error": str(e)}, 500)

# ============================================================
# API: Video Search (for autocomplete)
# ============================================================
@app.route("/api/videos/search")
def api_videos_search():
    try:
        q = request.args.get("q", default="", type=str)
        limit = request.args.get("limit", default=20, type=int)

        results = []
        if not q:
            sample_ids = sorted(recommender.video_info.keys())[:limit]
            for vid in sample_ids:
                info = recommender.video_info[vid]
                results.append({
                    "video_id": vid,
                    "title": info.get("title", ""),
                    "category": info.get("category", ""),
                    "heat": recommender.video_heat.get(vid, 0),
                })
            return json_response(results)

        q_lower = q.lower()
        for vid, info in recommender.video_info.items():
            if len(results) >= limit:
                break
            if (q_lower in str(vid) or
                q_lower in info.get("title", "").lower() or
                q_lower in info.get("category", "").lower() or
                q_lower in info.get("tags", "").lower()):
                results.append({
                    "video_id": vid,
                    "title": info.get("title", ""),
                    "category": info.get("category", ""),
                    "tags": info.get("tags", ""),
                    "heat": recommender.video_heat.get(vid, 0),
                })

        return json_response(results)
    except Exception as e:
        return json_response({"error": str(e)}, 500)

# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  短视频推荐系统 — Web 服务已启动")
    print("  打开浏览器访问: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=True)

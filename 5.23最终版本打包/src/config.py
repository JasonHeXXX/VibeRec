"""
共享配置：数据目录定位。

通过 data/LATEST_RUN.txt 找到最新一次 generate_data 的输出子文件夹。
如果文件不存在（比如还在用旧结构），则回退到 data/ 根目录。
"""
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent  # 项目根目录
_DATA_ROOT = _BASE_DIR / "data"


def get_data_dir() -> Path:
    """返回当前应使用的数据目录。

    优先级：
    1. data/LATEST_RUN.txt 中记录的子文件夹
    2. data/ 根目录（向后兼容）
    """
    sentinel = _DATA_ROOT / "LATEST_RUN.txt"
    if sentinel.exists():
        subdir = sentinel.read_text(encoding="utf-8").strip()
        candidate = _DATA_ROOT / subdir
        if candidate.is_dir():
            return candidate
    return _DATA_ROOT

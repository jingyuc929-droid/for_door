"""Checkpoint loading helpers.

常见报错：
- `PytorchStreamReader failed reading zip archive: failed finding central directory`

通常意味着 `.pt` 文件被截断/损坏（例如保存中断、磁盘满、拷贝不完整）。

本模块提供：
- 更友好的错误诊断（文件大小/mtime、同目录候选 checkpoint）
- 可选的回退加载（allow_fallback=True 时尝试同目录更早的 `model_*.pt`）

设计约束：
- 默认行为不变：allow_fallback 默认 False
- 不做任何硬编码路径，仅基于传入的 checkpoint 路径所在目录扫描
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class _FileInfo:
    path: str
    exists: bool
    size_bytes: int | None
    mtime_iso: str | None

    def to_line(self) -> str:
        if not self.exists:
            return f"{self.path} (missing)"
        return f"{self.path} (size={self.size_bytes}B, mtime={self.mtime_iso})"


_MODEL_STEP_RE = re.compile(r"model_(\d+)\.pt$")


def _safe_stat(path: str) -> _FileInfo:
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return _FileInfo(path=path, exists=False, size_bytes=None, mtime_iso=None)
    except OSError:
        # 文件系统异常（权限/IO 等），尽量输出可读信息
        return _FileInfo(path=path, exists=True, size_bytes=None, mtime_iso=None)
    mtime = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    return _FileInfo(path=path, exists=True, size_bytes=int(st.st_size), mtime_iso=mtime)


def _extract_step(path: str) -> int | None:
    name = os.path.basename(path)
    m = _MODEL_STEP_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def iter_checkpoint_candidates(path: str, *, max_candidates: int = 50) -> Iterable[str]:
    """按优先级产出候选 checkpoint：

    - 第一个永远是用户指定的 `path`
    - 后续为同目录下的 `model_*.pt`，按 step 从大到小排序（新->旧）
    """
    yield path
    ckpt_dir = os.path.dirname(os.path.abspath(path))
    pattern = os.path.join(ckpt_dir, "model_*.pt")
    siblings = glob.glob(pattern)

    def _key(p: str) -> tuple[int, str]:
        step = _extract_step(p)
        return (step if step is not None else -1, p)

    siblings_sorted = sorted(siblings, key=_key, reverse=True)
    seen = {os.path.abspath(path)}
    count = 0
    for p in siblings_sorted:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        yield p
        seen.add(ap)
        count += 1
        if count >= max_candidates:
            break


def _torch_load(path: str, *, map_location: Any, weights_only: bool) -> Any:
    """兼容不同 torch 版本：旧版本可能不支持 weights_only 参数。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _is_likely_zip_corruption(err: BaseException) -> bool:
    msg = str(err)
    needles = [
        "PytorchStreamReader failed reading zip archive",
        "failed finding central directory",
        "zip archive",
        "invalid header",
        "archive is corrupted",
    ]
    return any(n in msg for n in needles)


def torch_load_checkpoint(
    path: str,
    *,
    map_location: Any,
    weights_only: bool = False,
    allow_fallback: bool = False,
    max_fallback_candidates: int = 20,
) -> Any:
    """加载 torch checkpoint，必要时可回退到同目录更早 checkpoint。"""
    if not allow_fallback:
        try:
            return _torch_load(path, map_location=map_location, weights_only=weights_only)
        except Exception as e:
            info = _safe_stat(path)
            raise RuntimeError(
                "Failed to load checkpoint.\n"
                f"- file: {info.to_line()}\n"
                f"- error: {e}"
            ) from e

    last_err: BaseException | None = None
    attempted: list[_FileInfo] = []
    candidates = list(iter_checkpoint_candidates(path, max_candidates=max_fallback_candidates))
    for cand in candidates:
        info = _safe_stat(cand)
        attempted.append(info)
        if not info.exists:
            last_err = FileNotFoundError(cand)
            continue
        try:
            return _torch_load(cand, map_location=map_location, weights_only=weights_only)
        except Exception as e:
            last_err = e
            # 非典型损坏错误：不做“静默跳过”，直接抛出，避免掩盖真实 bug（例如结构不匹配）
            if not _is_likely_zip_corruption(e):
                raise

    attempted_lines = "\n".join(f"  - {x.to_line()}" for x in attempted[:max_fallback_candidates])
    raise RuntimeError(
        "Failed to load checkpoint (all fallback candidates failed).\n"
        f"- requested: {_safe_stat(path).to_line()}\n"
        f"- attempted candidates:\n{attempted_lines}\n"
        f"- last error: {last_err}\n"
        "Tips:\n"
        "- 若是刚训练完/边训练边 play，可能 checkpoint 正在写入或写坏；尝试更早的 `model_*.pt`。\n"
        "- 若是拷贝/下载得到的文件，请核对大小并重新拷贝，避免截断。\n"
        "- 若保存时磁盘满或进程被杀，请重新训练或重新导出 checkpoint。"
    ) from last_err


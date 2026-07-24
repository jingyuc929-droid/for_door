"""外力顺应（Force compliance）相关的公共工具。

目标：
- 统一管理“外力 -> 位置偏移 delta_p”的计算与缓存，避免在多个 reward/term 中重复实现。
- 为后续引入“力指令 / 阻抗指令”等留出统一入口：只要更新这里的偏移来源即可。

约定：
- 外力来源默认读取 `env.event_apply_forces_torques_buf[:, :3]` (Fx,Fy,Fz)，其由事件侧写入。
- 输出偏移 `delta_p` 在 **Projected COM yaw frame** 下
  （与 ee_target_points 命令/奖励参考系一致）。
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply, quat_inv


def _cached_vector_tensor(
    env,
    cache_key: str,
    name: str,
    values: tuple[float, float, float],
    device: torch.device | str,
) -> torch.Tensor:
    """Return a cached (1, 3) float tensor for static compliance parameters."""
    cache = getattr(env, "_manip_force_compliance_constant_tensors", None)
    if cache is None:
        cache = {}
        setattr(env, "_manip_force_compliance_constant_tensors", cache)
    key = (cache_key, name, tuple(float(x) for x in values), torch.device(device))
    tensor = cache.get(key)
    if tensor is None:
        tensor = torch.tensor(
            values, device=device, dtype=torch.float32
        ).view(1, 3)
        cache[key] = tensor
    return tensor


def scale_delta_p_to_box(
    target_p: torch.Tensor,
    delta_p: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """将 delta_p 按比例缩放，使 target_p + delta_p 落在 [low, high] 轴对齐盒子内。

    - target_p, delta_p: shape (B, 3)
    - low, high: shape (3,) 或 (1,3)
    """
    # Broadcast to (B, 3)
    low_b = low.view(1, 3)
    high_b = high.view(1, 3)

    dp = delta_p
    tp = target_p

    # compute per-axis maximum alpha s.t. tp + alpha*dp within bounds
    dp_pos = dp > eps
    dp_neg = dp < -eps

    alpha = torch.ones((tp.shape[0],), device=tp.device, dtype=tp.dtype)

    # 不使用 ``if mask.any()``，避免每步的 GPU->CPU 同步。未沿该方向运动的轴必须
    # 对 alpha 不施加约束，因此候选比例设为 +inf 后再做逐环境归约。不能让这些轴用
    # 分母 +/-1 参与 min：那会把该轴的剩余距离（单位 m）误当成无量纲 alpha，导致
    # 合法偏移被过度缩小，甚至在任一静止轴位于边界时被直接缩成 0。
    inf = torch.full_like(dp, torch.inf)

    # dp > 0: alpha <= (high - tp)/dp
    safe_dp_pos = torch.where(dp_pos, dp, torch.ones_like(dp))
    a_pos = torch.where(dp_pos, (high_b - tp) / safe_dp_pos, inf)
    alpha = torch.minimum(alpha, torch.amin(a_pos, dim=1))

    # dp < 0: alpha <= (low - tp)/dp  (dp negative -> division flips sign appropriately)
    safe_dp_neg = torch.where(dp_neg, dp, -torch.ones_like(dp))
    a_neg = torch.where(dp_neg, (low_b - tp) / safe_dp_neg, inf)
    alpha = torch.minimum(alpha, torch.amin(a_neg, dim=1))

    alpha = torch.clamp(alpha, 0.0, 1.0).view(-1, 1)
    return delta_p * alpha


def get_force_compliance_delta_p(
    env,
    frame,
    *,
    force_to_pos_scale: float | tuple[float, float, float] = 0.0,
    force_deadzone: float | tuple[float, float, float] | None = None,
    force_clip: float | None = None,
    delta_clip: float | tuple[float, float, float] | None = None,
    target_force: torch.Tensor | None = None,
    cache_key: str = "ee_target_points",
) -> torch.Tensor:
    """从外力生成并返回位置偏移 delta_p (x,y,z)，并做 step 级缓存。

    Args:
        env: ManagerBasedEnv 实例。
        frame: projected com yaw frame（需包含 yaw_quat_w）。
        force_to_pos_scale: 外力到位置偏移的比例（m/N）。
            - 标量：对 xyz 同一比例
            - 3-tuple：对 xyz 分别缩放
        force_deadzone: 力死区（N），在 yaw frame 下按轴应用：
            - 标量：对 xyz 同一阈值
            - 3-tuple：对 xyz 分别设置阈值
            - None/0：不启用死区
        force_clip: 对输入外力逐轴 clip（N），None 表示不 clip。
        delta_clip: 对输出偏移逐轴 clip（m），None 表示不 clip。
        cache_key: 缓存命名空间，便于未来多个任务/命令并存。

    Returns:
        delta_p: shape (num_envs, 3)，Projected COM yaw frame 下的偏移。

    Notes:
        - 该函数会把结果缓存在 env 上，保证同一步内多次调用返回一致值。
        - 若同一步内用不同参数调用，会抛异常，防止“三个 reward 各用一套参数”破坏一致性。
    """
    num_envs = int(
        getattr(env, "num_envs", getattr(env, "scene").num_envs)
    )
    device = getattr(env, "device", torch.device("cpu"))

    # step cache
    step = getattr(env, "common_step_counter", None)
    cache_step_attr = f"manip_force_compliance_step__{cache_key}"
    cache_params_attr = f"manip_force_compliance_params__{cache_key}"
    cache_delta_attr = f"manip_force_compliance_delta_p__{cache_key}"

    params = (force_to_pos_scale, force_deadzone, force_clip, delta_clip)
    if (step is not None) and (getattr(env, cache_step_attr, None) == step):
        cached_params = getattr(env, cache_params_attr, None)
        if cached_params != params:
            raise ValueError(
                "Force-compliance delta_p is requested with different params "
                "within the same step. "
                f"cached={cached_params}, requested={params}. "
                "Please keep the three EE tracking rewards consistent."
            )
        return getattr(env, cache_delta_attr)

    # disabled fast-path
    if force_to_pos_scale is None:
        return torch.zeros(
            (num_envs, 3), device=device, dtype=torch.float32
        )
    if isinstance(force_to_pos_scale, (int, float)) and (
        float(force_to_pos_scale) == 0.0
    ):
        return torch.zeros(
            (num_envs, 3), device=device, dtype=torch.float32
        )
    if isinstance(force_to_pos_scale, tuple) and all(
        float(x) == 0.0 for x in force_to_pos_scale
    ):
        return torch.zeros(
            (num_envs, 3), device=device, dtype=torch.float32
        )

    wrench = getattr(env, "event_apply_forces_torques_buf", None)
    if wrench is None:
        f_w = torch.zeros(
            (num_envs, 3), device=device, dtype=torch.float32
        )
    else:
        f_w = wrench[:, :3].to(device)

    if force_clip is not None:
        fc = float(force_clip)
        f_w = torch.clamp(f_w, min=-fc, max=fc)

    # rotate world force into projected-yaw frame
    yaw_q_inv = quat_inv(frame.yaw_quat_w)
    f_p = quat_apply(yaw_q_inv, f_w)

    # deadzone in yaw frame (per-axis)
    if force_deadzone is not None:
        if isinstance(force_deadzone, tuple):
            dz = _cached_vector_tensor(
                env, cache_key, "deadzone", force_deadzone, device
            )
        else:
            dz = float(force_deadzone)
        # dz<=0 时比较恒为 False，因此可直接使用逐元素 where；无需 any()
        # 归约，也不会产生 GPU->CPU 同步。
        f_p = torch.where(
            torch.abs(f_p) < dz, torch.zeros_like(f_p), f_p
        )

    if isinstance(force_to_pos_scale, tuple):
        scale = _cached_vector_tensor(
            env, cache_key, "scale", force_to_pos_scale, device
        )
    else:
        scale = float(force_to_pos_scale)
    # 力偏移（阻抗）：delta_p = (target_force - external_force) * scale
    # target_force 在 yaw frame（命令目标力）；None 时退化为顺应外力 f_p * scale
    if target_force is not None:
        f_err = target_force.to(device=device, dtype=torch.float32) - f_p
    else:
        f_err = f_p
    delta_p = f_err * scale

    if delta_clip is not None:
        if isinstance(delta_clip, tuple):
            dc = _cached_vector_tensor(
                env, cache_key, "delta_clip", delta_clip, device
            )
            delta_p = torch.clamp(delta_p, min=-dc, max=dc)
        else:
            dc = float(delta_clip)
            delta_p = torch.clamp(delta_p, min=-dc, max=dc)

    # write cache
    if step is not None:
        setattr(env, cache_step_attr, step)
        setattr(env, cache_params_attr, params)
        setattr(env, cache_delta_attr, delta_p)

    return delta_p

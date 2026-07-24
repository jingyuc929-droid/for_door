from __future__ import annotations

from isaaclab.envs.mdp.actions import JointActionCfg, JointAction, ActionTerm
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.utils import configclass

from typing import Literal, Sequence
import torch



class JointPositionOffsetAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: JointPositionOffsetActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: JointPositionOffsetActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # 初始化 offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        else:
            if isinstance(self._joint_ids, slice):
                joint_pos_shape = self._asset.data.default_joint_pos[:, self._joint_ids].shape
                num_joints = joint_pos_shape[1]
            else:
                num_joints = len(self._joint_ids)
            self._offset = torch.zeros((env.scene.num_envs, num_joints), device=self._asset.device)

        # 初始化 random offset
        if isinstance(self._joint_ids, slice):
            joint_pos_shape = self._asset.data.default_joint_pos[:, self._joint_ids].shape
            num_joints = joint_pos_shape[1]
        else:
            num_joints = len(self._joint_ids)
        self._random_offset = torch.zeros((env.scene.num_envs, num_joints), device=self._asset.device)

    def apply_actions(self):
        # set position targets (不加 offset，offset 已在 process_actions 中用过了)
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions

        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset + self._random_offset

        # clip actions if needed
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def randomize_offset(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None = None,
        randomization_params: tuple[float | Sequence[float], float | Sequence[float]] | None = None,
        operation: Literal["add", "scale", "abs"] = "add",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        """Randomize the joint position offset.

        Should be called during environment reset through Event Manager.
        """
        if randomization_params is None:
            return

        dim_0_ids = env_ids
        selected_envs = slice(None) if env_ids is None else env_ids

        # Per-joint bounds are represented by two vectors and broadcast over
        # environments, so heterogeneous offsets still require only one sample.
        low, high = randomization_params
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            low = torch.as_tensor(low, device=self._asset.device, dtype=self._random_offset.dtype)
            high = torch.as_tensor(high, device=self._asset.device, dtype=self._random_offset.dtype)
            expected_shape = (self._random_offset.shape[1],)
            if low.shape != expected_shape or high.shape != expected_shape:
                raise ValueError(
                    "Per-joint offset bounds must match the action joint count: "
                    f"expected {expected_shape}, got low={tuple(low.shape)}, high={tuple(high.shape)}."
                )
        if torch.any(torch.as_tensor(low) > torch.as_tensor(high)):
            raise ValueError("Every joint offset lower bound must be <= its upper bound.")

        self._random_offset[selected_envs] = 0.0
        _randomize_prop_by_op(
            self._random_offset,
            (low, high),
            dim_0_ids=dim_0_ids,
            dim_1_ids=slice(None),
            operation=operation,
            distribution=distribution,
        )


@configclass
class JointPositionOffsetActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionOffsetAction` for more details.
    """

    # 提供一个默认值，让 configclass validate 能通过
    class_type: type[ActionTerm] = JointPositionOffsetAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.
    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """

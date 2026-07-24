from __future__ import annotations

from isaaclab.envs.mdp.actions import JointActionCfg, JointAction, ActionTerm
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.utils import configclass

from typing import Literal
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
        randomization_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "add",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        """Randomize the joint position offset.

        Should be called during environment reset through Event Manager.
        """
        if randomization_params is None:
            return

        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self._asset.device)

        # reset before randomizing
        self._random_offset[env_ids] = 0.0

        # apply randomization
        self._random_offset[env_ids] = _randomize_prop_by_op(
            self._random_offset[env_ids],
            randomization_params,
            dim_0_ids=None,
            dim_1_ids=slice(None),
            operation=operation,
            distribution=distribution,
        )

        print("self._random_offset:", self._random_offset)


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

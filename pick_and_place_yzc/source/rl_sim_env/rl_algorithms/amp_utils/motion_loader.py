import glob
from pathlib import Path

import numpy as np
import torch
from rl_algorithms.rsl_rl.utils.log_print import (
    print_placeholder_end,
    print_placeholder_start,
)


class AMPLoader:
    def __init__(
        self,
        device,
        amp_loader_cfg,
        time_between_frames,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_files2/*"),
    ):
        """
        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.amp_loader_cfg = amp_loader_cfg
        self.combined_indices = amp_loader_cfg['combined_indices']
        self.time_between_frames = time_between_frames

        # Values to store for each trajectory.
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_duration = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        print_placeholder_start("Loading motions")
        print(f"Loading {len(motion_files)} motions")
        print_placeholder_end()
        self.traj_length = len(motion_files)

        for i, motion_file in enumerate(motion_files):
            # 1) 名称改为直接取 npz 文件名（去掉后缀）
            name = Path(motion_file).stem
            self.trajectory_names.append(name)

            # 2) 打开 npz 而不是 yaml
            npz_data = np.load(motion_file)

            # 3) 依然按 all_keys 构造 motion_data
            motion_data = {}
            for key in self.amp_loader_cfg['all_keys']:
                if key in npz_data:
                    arr = npz_data[key]
                    # 保证一维数组变成 (N,1)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    motion_data[key] = arr
                else:
                    print(f"Warning: key '{key}' not found in {motion_file}")

            # 4) length/weight/dt 也从 npz 中读取
            data_length = int(npz_data["length"])
            weight = float(npz_data["weight"])
            frame_duration = float(npz_data["dt"])

            # 5) 和之前完全一样的位移／旋转处理
            first_root_pos = motion_data["root_position_world"][0].copy()
            for f_i in range(data_length):
                root_pos = motion_data["root_position_world"][f_i]
                root_pos[:2] -= first_root_pos[:2]
                root_pos[2] += 0.026
                motion_data["root_position_world"][f_i] = root_pos

                root_rot = motion_data["root_quaternion_wxyz"][f_i]
                root_rot = self.QuaternionNormalize(root_rot)
                root_rot = self.standardize_quaternion(root_rot)
                motion_data["root_quaternion_wxyz"][f_i] = root_rot

            # 6) 拼接、存入 trajectories_full、idx、weights、durations、num_frames
            concatenated_data_all = np.concatenate(
                [motion_data[key] for key in self.amp_loader_cfg['all_keys'] if key in motion_data], axis=1
            )
            self.trajectories_full.append(torch.tensor(concatenated_data_all, dtype=torch.float32, device=device))
            self.trajectory_idxs.append(i)
            self.trajectory_weights.append(weight)
            self.trajectory_frame_durations.append(frame_duration)
            traj_len = (data_length - 1) * frame_duration
            self.trajectory_duration.append(traj_len)
            self.trajectory_num_frames.append(float(data_length))

            # print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_duration = np.array(self.trajectory_duration)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        self.all_trajectories_full = torch.vstack(self.trajectories_full)
        self.traj_lengths = torch.tensor(
            [t.shape[0] for t in self.trajectories_full],
            device=self.device, dtype=torch.long
        )  # 形如 [L0, L1, L2, ...]
        self.traj_row_offsets = torch.cat([
            torch.zeros(1, device=self.device, dtype=torch.long),
            torch.cumsum(self.traj_lengths, dim=0)[:-1]
        ])  # 形如 [0, L0, L0+L1, ...]

        # Preload transitions.
        print(f"Preloading {num_preload_transitions} transitions")
        traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
        times = self.traj_time_sample_batch(traj_idxs)
        self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
        self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)

        print("Finished preloading")

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_duration[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_duration[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        device = self.device
        # 输入统一成 torch 同设备
        traj_idxs_t = torch.as_tensor(traj_idxs, device=device, dtype=torch.long) \
            if not torch.is_tensor(traj_idxs) else traj_idxs.to(device=device, dtype=torch.long)
        times_t = torch.as_tensor(times, device=device, dtype=torch.float32) \
            if not torch.is_tensor(times) else times.to(device=device, dtype=torch.float32)

        # 取 n 与 duration（注意：duration = (n-1)*dt，已按你构建 self.trajectory_duration）
        n_frames = torch.as_tensor(self.trajectory_num_frames, device=device, dtype=torch.long)[traj_idxs_t]  # [B]
        duration = torch.as_tensor(self.trajectory_duration, device=device, dtype=torch.float32)[traj_idxs_t]  # [B]

        # 代码二的映射：p = t/duration; x = p*n; lo=floor(x); hi=ceil(x)
        # ——不做任何 clamp（严格复现代码二）
        p = times_t / duration                               # [B]
        x = p * n_frames.to(torch.float32)                   # [B]
        idx_lo = torch.floor(x).to(torch.long)               # [B]
        idx_hi = torch.ceil(x).to(torch.long)                # [B]
        alpha = (x - idx_lo.to(torch.float32)).unsqueeze(1)  # [B,1]

        # 下面与原代码一保持一致：用全局 row 偏移做矢量化两帧索引
        offsets = self.traj_row_offsets.index_select(0, traj_idxs_t)   # [B]
        g_lo = offsets + idx_lo
        g_hi = offsets + idx_hi     # ⚠️ 若 idx_hi==n，会越界（与代码二一致）

        big = self.all_trajectories_full
        rows_lo = big.index_select(0, g_lo)   # [B, TOTAL_SIZE]
        rows_hi = big.index_select(0, g_hi)   # [B, TOTAL_SIZE]

        POS_S, POS_E = self.amp_loader_cfg['ROOT_POS_START_IDX'], self.amp_loader_cfg['ROOT_POS_END_IDX']
        ROT_S, ROT_E = self.amp_loader_cfg['ROOT_ROT_START_IDX'], self.amp_loader_cfg['ROOT_ROT_END_IDX']
        AMP_S, AMP_E = ROT_E, self.amp_loader_cfg['TOTAL_SIZE']

        pos_lo, pos_hi = rows_lo[:, POS_S:POS_E], rows_hi[:, POS_S:POS_E]
        rot_lo, rot_hi = rows_lo[:, ROT_S:ROT_E], rows_hi[:, ROT_S:ROT_E]
        amp_lo, amp_hi = rows_lo[:, AMP_S:AMP_E], rows_hi[:, AMP_S:AMP_E]

        pos_blend = (1.0 - alpha) * pos_lo + alpha * pos_hi
        amp_blend = (1.0 - alpha) * amp_lo + alpha * amp_hi
        rot_blend = self.quaternion_slerp(rot_lo, rot_hi, alpha)

        return torch.cat([pos_blend, rot_blend, amp_blend], dim=1)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        # print("combined_indices:",combined_indices)
        for _ in range(num_mini_batch):
            idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
            s = self.preloaded_s[idxs][:, self.amp_loader_cfg['combined_indices']]
            s_next = self.preloaded_s_next[idxs][:, self.amp_loader_cfg['combined_indices']]

            yield s, s_next

    @property
    def observation_dim(self):
        return len(self.amp_loader_cfg['combined_indices'])

    def get_root_pos(self, pose):
        return pose[self.amp_loader_cfg['ROOT_POS_START_IDX'] : self.amp_loader_cfg['ROOT_POS_END_IDX']]

    def get_root_pos_batch(self, poses):
        return poses[:, self.amp_loader_cfg['ROOT_POS_START_IDX'] : self.amp_loader_cfg['ROOT_POS_END_IDX']]

    def get_root_rot(self, pose):
        return pose[self.amp_loader_cfg['ROOT_ROT_START_IDX'] : self.amp_loader_cfg['ROOT_ROT_END_IDX']]

    def get_linear_vel_batch(self, poses):
        return poses[:, self.amp_loader_cfg['ROOT_LINEAR_VEL_START_IDX'] : self.amp_loader_cfg['ROOT_LINEAR_VEL_END_IDX']]

    def get_angular_vel_batch(self, poses):
        return poses[:, self.amp_loader_cfg['ROOT_ANGULAR_VEL_START_IDX'] : self.amp_loader_cfg['ROOT_ANGULAR_VEL_END_IDX']]

    def get_root_rot_batch(self, poses):
        return poses[:, self.amp_loader_cfg['ROOT_ROT_START_IDX'] : self.amp_loader_cfg['ROOT_ROT_END_IDX']]

    def get_joint_pos_batch(self, poses):
        return poses[:, self.amp_loader_cfg['JOINT_POS_START_IDX'] : self.amp_loader_cfg['JOINT_POS_END_IDX']]

    def get_joint_vel_batch(self, poses):
        return poses[:, self.amp_loader_cfg['JOINT_VEL_START_IDX'] : self.amp_loader_cfg['JOINT_VEL_END_IDX']]

    def get_frame_pos_batch(self, poses):
        return poses[:, self.amp_loader_cfg['FRAME_POS_START_IDX'] : self.amp_loader_cfg['FRAME_POS_END_IDX']]

    def QuaternionNormalize(self, q):
        """Normalizes the quaternion to length 1.

        Divides the quaternion by its magnitude.  If the magnitude is too
        small, returns the quaternion identity value (1.0).

        Args:
        q: A quaternion to be normalized.

        Raises:
        ValueError: If input quaternion has length near zero.

        Returns:
        A quaternion with magnitude 1 in a numpy array [x, y, z, w].

        """
        q_norm = np.linalg.norm(q)
        if np.isclose(q_norm, 0.0):
            raise ValueError(f"Quaternion may not be zero in QuaternionNormalize: |q| = {q_norm:f}, q = {q}")
        return q / q_norm

    def standardize_quaternion(self, q):
        """Returns a quaternion where q.w >= 0 to remove redundancy due to q = -q.

        Args:
        q: A quaternion to be standardized.

        Returns:
        A quaternion with q.w >= 0.

        """
        if q[-1] < 0:
            q = -q
        return q

    def quaternion_slerp(self, q0, q1, fraction, spin=0, shortestpath=True):
        """Batch quaternion spherical linear interpolation."""
        _EPS = np.finfo(float).eps * 4.0
        out = torch.zeros_like(q0)

        zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
        ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
        out[zero_mask] = q0[zero_mask]
        out[ones_mask] = q1[ones_mask]

        d = torch.sum(q0 * q1, dim=-1, keepdim=True)
        dist_mask = (torch.abs(torch.abs(d) - 1.0) < _EPS).squeeze()
        out[dist_mask] = q0[dist_mask]

        if shortestpath:
            d_old = torch.clone(d)
            d = torch.where(d_old < 0, -d, d)
            q1 = torch.where(d_old < 0, -q1, q1)

        angle = torch.acos(d) + spin * torch.pi
        angle_mask = (torch.abs(angle) < _EPS).squeeze() | torch.isnan(angle.squeeze())
        out[angle_mask] = q0[angle_mask]

        final_mask = torch.logical_or(zero_mask, ones_mask)
        final_mask = torch.logical_or(final_mask, dist_mask)
        final_mask = torch.logical_or(final_mask, angle_mask)
        final_mask = torch.logical_not(final_mask)

        isin = 1.0 / angle
        q0 *= torch.sin((1.0 - fraction) * angle) * isin
        q1 *= torch.sin(fraction * angle) * isin
        q0 += q1
        out[final_mask] = q0[final_mask]
        return out

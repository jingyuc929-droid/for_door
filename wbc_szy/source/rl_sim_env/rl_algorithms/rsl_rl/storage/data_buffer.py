import torch


class DataBuffer:
    def __init__(self, num_envs, num_data, data_history_length, device):

        self.num_envs = num_envs
        self.num_data = num_data
        self.data_history_length = data_history_length
        self.device = device

        self.num_data_total = num_data * data_history_length

        self.data_buf = torch.zeros(
            self.num_envs,
            self.num_data_total,
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )

    def reset(self, data_ids, new_data=None):
        for data_id in data_ids:
            if new_data is None:
                self.data_buf[data_id] = torch.zeros(self.num_data_total, device=self.device, dtype=torch.float)
            else:
                self.data_buf[data_id] = new_data.repeat(1, self.data_history_length)

    def insert(self, new_data):
        # Shift observations back.
        self.data_buf[:, : self.num_data * (self.data_history_length - 1)] = self.data_buf[
            :, self.num_data : self.num_data * self.data_history_length
        ].clone()

        # Add new observation.
        self.data_buf[:, -self.num_data :] = new_data.clone()

    def get_all_data(self) -> torch.Tensor:
        """
        获取全量历史数据，按时间顺序从最旧到最新平铺（flatten）。

        返回:
            Tensor，shape=(num_envs, num_data * data_history_length)
            每一行最左边是最旧的数据段，最右边是最新的数据段。
        """
        # 如果不希望后续修改影响到内部 buffer，可 clone() 一份
        return self.data_buf.clone()

    def get_data_vec(self, data_ids=None):
        """Gets history of data indexed by data_ids.

        Arguments:
            obs_ids: An array of integers with which to index the desired
                observations, where 0 is the latest observation and
                obs_history_length - 1 is the oldest observation.
        """

        if data_ids is None:
            return self.data_buf[:, -self.num_data :]
        data = []
        for data_id in reversed(sorted(data_ids)):
            slice_idx = self.data_history_length - data_id - 1
            data.append(self.data_buf[:, slice_idx * self.num_data : (slice_idx + 1) * self.num_data])
        return torch.cat(data, dim=-1)

    def get_data_his(self, his_list):
        """
        获取每个环境中指定历史时刻的数据。

        参数：
            his_list (Tensor): 形状为 (num_envs,) 的整数张量，
                            每个元素代表该环境取历史数据的索引，
                            其中 0 表示最新，data_history_length-1 表示最旧。

        返回：
            Tensor: 形状为 (num_envs, num_data)，每行对应一个环境指定历史时刻的数据。
        """
        if his_list.shape[0] != self.num_envs:
            raise ValueError("his_list 的长度必须等于 num_envs")

        if ((his_list < 0) | (his_list >= self.data_history_length)).any():
            raise ValueError("每个历史索引必须在 [0, data_history_length-1] 范围内")

        # 先将 data_buf 重塑为 (num_envs, data_history_length, num_data)
        data_buf_reshaped = self.data_buf.view(self.num_envs, self.data_history_length, self.num_data)
        # 计算每个环境对应的 slice 索引：
        # 0 表示最新数据，对应的 slice index 为 data_history_length - 1，
        # data_history_length - 1 表示最旧，对应的 slice index 为 0。
        slice_indices = self.data_history_length - his_list - 1  # 形状为 (num_envs,)
        env_indices = torch.arange(self.num_envs, device=self.device)
        # 利用批量索引选取每个环境对应的历史数据
        result = data_buf_reshaped[env_indices, slice_indices, :]
        return result

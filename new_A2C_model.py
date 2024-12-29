from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from stable_baselines3 import A2C
from gymnasium import spaces
import torch as th
from stable_baselines3.common.utils import explained_variance

class MixedPrecisionA2C(A2C):
    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        self.policy.set_training_mode(True)  # 設置為訓練模式
        self._update_learning_rate(self.policy.optimizer)

        scaler = GradScaler()  # 創建 GradScaler

        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            # 自動混合精度支持
            with autocast():
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # 計算損失
                policy_loss = -(advantages * log_prob).mean()
                value_loss = F.mse_loss(rollout_data.returns, values)
                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # 梯度縮放和反向傳播
            scaler.scale(loss).backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            # 梯度步驟和更新縮放因子
            scaler.step(self.policy.optimizer)
            scaler.update()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)

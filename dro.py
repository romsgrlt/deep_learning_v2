import torch

class GroupDROLoss:
    def __init__(self, n_groups, step_size=0.01, gamma=0.1):
        self.n_groups = n_groups
        self.step_size = step_size
        self.gamma = gamma
        self.adv_probs = torch.ones(n_groups) / n_groups
        self.exp_avg_loss = torch.zeros(n_groups)
        self.exp_avg_init = torch.zeros(n_groups, dtype=torch.bool)

    def to(self, device):
        self.adv_probs = self.adv_probs.to(device)
        self.exp_avg_loss = self.exp_avg_loss.to(device)
        self.exp_avg_init = self.exp_avg_init.to(device)
        return self

    def compute_group_avg(self, per_sample_losses, group):
        group_map = (group.unsqueeze(0) == torch.arange(self.n_groups).unsqueeze(1).to(group.device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()
        group_loss = (group_map @ per_sample_losses) / group_denom
        return group_loss, group_count

    def update_exp_avg(self, group_loss, group_count):
        previous_weights = (1 - self.gamma * (group_count > 0).float()) * self.exp_avg_init.float()
        current_weights = 1 - previous_weights
        self.exp_avg_loss = self.exp_avg_loss * previous_weights + group_loss * current_weights
        self.exp_avg_init = self.exp_avg_init | (group_count > 0)

    def loss(self, per_sample_losses, group):
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group)
        self.update_exp_avg(group_loss, group_count)

        self.adv_probs = self.adv_probs * torch.exp(self.step_size * group_loss.detach())
        self.adv_probs = self.adv_probs / self.adv_probs.sum()

        return (group_loss * self.adv_probs).sum()
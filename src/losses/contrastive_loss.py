import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.total_neg_per_epoch = []
        self.neg_per_epoch = []
        self.pos_per_epoch = []


    def forward(self, inputs_col, targets_col, inputs_row, target_row):

        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        epsilon = 1e-5
        loss = list()

        total_neg_count = list()
        neg_count = list()
        pos_count = list()
        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets_col[i] == target_row)
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
            neg_pair_ = torch.masked_select(sim_mat[i], targets_col[i] != target_row)
            total_neg_count.append(len(neg_pair_))

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)

            pos_loss = torch.sum(-pos_pair_ + 1)
            pos_count.append(len(pos_pair_))
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
                neg_count.append(len(neg_pair))
            else:
                neg_loss = torch.tensor(0.)
                neg_count.append(0)

            loss.append(pos_loss + neg_loss)
        self.total_neg_per_epoch.append(sum(total_neg_count) / len(total_neg_count) if len(total_neg_count) > 0 else 0.0)
        self.neg_per_epoch.append(sum(neg_count) / len(neg_count) if len(neg_count) > 0 else 0.0)
        self.pos_per_epoch.append(sum(pos_count) / len(pos_count) if len(pos_count) > 0 else 0.0)
        total_loss = sum(loss) / n
        return total_loss

    def log_info(self, epoch, wandb):
        tneg = self.total_neg_per_epoch
        neg = self.neg_per_epoch
        pos = self.pos_per_epoch
        wandb.log({'total_neg_count': sum(tneg) / len(tneg) if len(tneg) > 0 else 0.0}, step=epoch)
        wandb.log({'neg_count': sum(neg) / len(neg) if len(neg) > 0 else 0.0}, step=epoch)
        wandb.log({'pos_count': sum(pos) / len(pos) if len(pos) > 0 else 0.0}, step=epoch)
        self.total_neg_per_epoch.clear()
        self.neg_per_epoch.clear()
        self.pos_per_epoch.clear()

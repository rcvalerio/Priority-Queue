import torch
from torch import nn

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.total_neg_per_epoch = []
        self.neg_per_epoch = []
        self.pos_per_epoch = []

    def forward(self, inputs_col, targets_col, inputs_row, targets_row):
        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        # split the positive and negative pairs
        eyes_ = torch.eye(n, dtype=torch.uint8).cuda()
        pos_mask = (targets_col.expand(
            targets_row.shape[0], n
        ).t() == targets_row.expand(n, targets_row.shape[0])).int()
        neg_mask = 1 - pos_mask
        pos_mask[:, :n] = pos_mask[:, :n] - eyes_

        loss = list()
        total_neg_count = list()
        neg_count = list()
        pos_count = list()
        for i in range(n):
            pos_pair_idx = torch.nonzero(pos_mask[i, :]).view(-1)
            if pos_pair_idx.shape[0] > 0:
                pos_pair_ = sim_mat[i, pos_pair_idx]
                pos_pair_ = torch.sort(pos_pair_)[0]

                neg_pair_idx = torch.nonzero(neg_mask[i, :]).view(-1)
                neg_pair_ = sim_mat[i, neg_pair_idx]
                neg_pair_ = torch.sort(neg_pair_)[0]
                total_neg_count.append(len(neg_pair_))

                select_pos_pair_idx = torch.nonzero(
                    pos_pair_ < neg_pair_[-1] + self.margin
                ).view(-1)
                pos_pair = pos_pair_[select_pos_pair_idx]

                if pos_pair_.shape[0]>1:
                    select_neg_pair_idx = torch.nonzero(
                        neg_pair_ > max(0.9, pos_pair_[-2]) - self.margin
                    ).view(-1)
                else:
                    select_neg_pair_idx = torch.nonzero(
                        neg_pair_ > max(0.9, pos_pair_[-1]) - self.margin
                    ).view(-1)
                neg_pair = neg_pair_[select_neg_pair_idx]

                pos_loss = torch.sum(1 - pos_pair).cuda()
                pos_count.append(len(pos_pair))
                if len(neg_pair) >= 1:
                    neg_loss = torch.sum(neg_pair).cuda()
                    neg_count.append(len(neg_pair))
                else:
                    neg_loss = 0.
                    neg_count.append(0)
                loss.append(pos_loss + neg_loss)
            else:
                loss.append(torch.tensor(0.).cuda())
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

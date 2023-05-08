import torch

class PriorityQueueBatch:
    def __init__(self, queue_size, batch_size, embedding_size, rm_old, rm_loss):
        self.K = queue_size
        self.batch_size = batch_size
        self.feats = torch.zeros(self.K, embedding_size).cuda()
        self.targets = torch.zeros(self.K, dtype=torch.int64).cuda()
        self.loss = torch.zeros(int(self.K/self.batch_size)).cuda()
        self.updated = torch.zeros(int(self.K / self.batch_size), dtype=torch.bool).cuda()
        self.indices = torch.zeros(self.K, dtype=torch.int64).cuda()
        self.oldest = torch.ones(int(self.K/self.batch_size), dtype=torch.long).cuda()
        self.size = 0
        self.rm_old = rm_old
        self.rm_loss = rm_loss
        self.mode = 'rm_old' if rm_old > 0 else 'rm_loss'
        self.rm_count = getattr(self, self.mode)

    def is_full(self):
        return self.size >= self.K

    def get(self):
        if self.is_full():
            return self.feats, self.targets
        else:
            return self.feats[:self.size], self.targets[:self.size]

    def enqueue_dequeue(self, feats, targets, loss, indices, iteration):
        q_size = len(targets)
        if self.is_full():
            if self.rm_count <= 0 and self.rm_old > 0 and self.rm_loss > 0:
                self.mode = 'rm_old' if self.mode == 'rm_loss' else 'rm_loss'
                self.rm_count = getattr(self, self.mode)
            self.rm_count = self.rm_count - 1

            _, i = torch.topk(self.loss if self.mode == 'rm_loss' else self.oldest, k=1, largest=False)
            #print("enqueue_dequeue_index:", i.item(), "old_loss:", self.loss[i].item(), "new_loss:", loss.item())
            index = int(i*self.batch_size)
            self.loss[i] = loss.cuda()
            self.feats[index: index+q_size] = feats.cuda()
            self.targets[index: index+q_size] = targets.cuda()
            self.indices[index: index+q_size] = indices.cuda()
            self.oldest[i] = iteration
            self.updated[i] = False
        else:
            self.feats[self.size: self.size + q_size] = feats.cuda()
            self.targets[self.size: self.size + q_size] = targets.cuda()
            self.loss[int(self.size/self.batch_size)] = loss.cuda()
            self.indices[self.size: self.size + q_size] = indices.cuda()
            self.oldest[int(self.size/self.batch_size)] = iteration
            self.updated[int(self.size/self.batch_size)] = False
            self.size += q_size

    def update(self, feats, loss, indices, i , iteration):
        a = torch.zeros(self.K, dtype=torch.bool)
        index = int(i * self.batch_size)
        a[index:index+self.batch_size] = True
        self.feats[a] = feats
        self.loss[i] = loss
        self.oldest[i] = iteration
        self.updated[i] = True
        assert torch.sum(self.indices[a]) == (torch.sum(indices))

    def get_oldest(self, num_batches=1):
        self.oldest[self.updated] = self.oldest[self.updated] + 1000
        _, i = torch.topk(self.oldest, k=num_batches, largest=False)
        self.oldest[self.updated] = self.oldest[self.updated] - 1000
        a = torch.zeros(self.K,dtype=torch.bool)
        for i_ in i:
            index = int(i_ * self.batch_size)
            a[index:index+self.batch_size] = True
        return self.feats[a],\
               self.targets[a],\
               self.loss[i],\
               self.indices[a],\
               i

    def get_worst(self, num_batches=1):
        self.loss[self.updated] = self.loss[self.updated] - 1000.
        _, i = torch.topk(self.loss, k=num_batches, largest=True)
        self.loss[self.updated] = self.loss[self.updated] + 1000.
        a = torch.zeros(self.K,dtype=torch.bool)
        for i_ in i:
            index = int(i_ * self.batch_size)
            a[index:index+self.batch_size] = True
        return self.feats[a],\
               self.targets[a],\
               self.loss[i],\
               self.indices[a],\
               i

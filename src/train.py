import torch, argparse, os, glob
from torch.utils.data import DataLoader, Subset
import dataset, utils

from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from losses.contrastive_loss import ContrastiveLoss
from losses.triplet_loss import TripletLoss
from losses.multi_similarity_loss import MultiSimilarityLoss
import subprocess

from tqdm import *
import wandb

from pq_batch import PriorityQueueBatch

print("Using", torch.cuda.device_count(), "GPUs")

parser = argparse.ArgumentParser(description='Priority Queue')
parser.add_argument('--LOG_DIR',
                    default='../logs',
                    help='Path to log folder'
                    )
parser.add_argument('--dataset',
                    choices=['Inshop', 'cub', 'cars', 'SOP'],
                    help='Training dataset, e.g. cub, cars, SOP, Inshop'
                    )
parser.add_argument('--embedding-size', default=128, type=int,
                    dest='sz_embedding',
                    help='Size of embedding that is appended to backbone model.'
                    )
parser.add_argument('--batch-size', default=64, type=int,
                    dest='sz_batch',
                    help='Number of samples per batch.'
                    )
parser.add_argument('--epochs', default=60, type=int,
                    dest='nb_epochs',
                    help='Number of training epochs.'
                    )
parser.add_argument('--gpu-id', default=0, type=int,
                    help='ID of GPU that is used for training.'
                    )
parser.add_argument('--workers', default=4, type=int,
                    dest='nb_workers',
                    help='Number of workers for dataloader.'
                    )
parser.add_argument('--model', default='bn_inception',
                    choices=['googlenet', 'bn_inception', 'resnet18', 'resnet50', 'resnet101'],
                    help='Model for training'
                    )
parser.add_argument('--loss', default='Contrastive',
                    choices=['MS', 'Contrastive', 'Triplet'],
                    help='Criterion for training'
                    )
parser.add_argument('--optimizer', default='adam',
                    choices=['sgd', 'adam', 'rmsprop', 'adamw'],
                    help='Optimizer setting'
                    )
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Learning rate setting'
                    )
parser.add_argument('--weight-decay', default=5e-4, type=float,
                    help='Weight decay setting'
                    )
parser.add_argument('--lr-decay-step', default=10, type=int,
                    help='Learning decay step setting'
                    )
parser.add_argument('--lr-decay-gamma', default=0.5, type=float,
                    help='Learning decay gamma setting'
                    )
parser.add_argument('--alpha', default=32, type=float,
                    help='Scaling Parameter setting'
                    )
parser.add_argument('--IPC', type=int,
                    help='Balanced sampling, images per class'
                    )
parser.add_argument('--warm', default=1, type=int,
                    help='Warmup training epochs'
                    )
parser.add_argument('--bn-freeze', default=1, type=int,
                    help='Batch normalization parameter freeze'
                    )
parser.add_argument('--l2-norm', default=1, type=int,
                    help='L2 normlization'
                    )
parser.add_argument('--pq-size', default=8192, type=int,
                    help='Size of pq'
                    )
parser.add_argument('--pq-warmup', default=3, type=int,
                    help='Pq warmup epochs'
                    )
parser.add_argument('--pq-scheduler', default='reduce',
                    choices=['reduce', 'step'],
                    help='Scheduler'
                    )
parser.add_argument('--grad-acc', default=1, type=int,
                    help='Gradient accumulation steps'
                    )
parser.add_argument('--contrastive-margin', default=0.43, type=float,
                    help='Contrastive loss margin'
                    )
parser.add_argument('--triplet-margin', default=0.15, type=float,
                    help='Triplet loss margin'
                    )
parser.add_argument('--ms-pos-scale', default=2.0, type=float,
                    help='Multi similarity loss positive scale'
                    )
parser.add_argument('--ms-neg-scale', default=40.0, type=float,
                    help='Multi similarity loss negative scale'
                    )
parser.add_argument('--ms-margin', default=0.1, type=float,
                    help='Multi similarity loss margin'
                    )
parser.add_argument('--pq-type', default='normal',
                    choices=['normal', 'update'],
                    help='Priority queue type'
                    )
parser.add_argument('--pq-update-type', default='normal',
                    choices=['batch'],
                    help='Priority queue update type'
                    )
parser.add_argument('--pq-update-criterion',
                    choices=['oldest', 'worst', 'none'],
                    help='Priority queue removal criterion'
                    )
parser.add_argument('--pq-rm-old', default=1, type=int,
                    help='Priority queue iterations to remove oldest'
                    )
parser.add_argument('--pq-rm-loss', default=0, type=int,
                    help='Priority queue iterations to remove worst loss'
                    )
parser.add_argument('--pq-upd-int', default=0, type=int,
                    help='Priority queue update interval'
                    )
parser.add_argument('--job-id', required=True, type=str,
                    help='Priority queue iterations to remove worst loss'
                    )
parser.add_argument('--remark', default='',
                    help='Any remark'
                    )

args = parser.parse_args()

#if args.gpu_id != -1:
    #torch.cuda.set_device(args.gpu_id)
    #torch.cuda.set_device('cuda')

# Directory for Log
LOG_DIR = args.LOG_DIR + '/logs_{}/{}_{}_{}_embedding{}_{}_lr{}_batch{}_type{}_size{}_acc{}_old{}_loss{}{}'.format(
    args.dataset, args.job_id, args.model, args.loss, args.sz_embedding,
    args.optimizer, args.lr, args.sz_batch, args.pq_type, args.pq_size,
    args.grad_acc, args.pq_rm_old, args.pq_rm_loss, args.remark)
# Wandb Initialization
wandb.init(project=args.dataset + '_git', notes=LOG_DIR)
wandb.config.update(args)


command = 'git rev-parse --abbrev-ref HEAD'.split()
branch = subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()
curr_branch = branch.strip().decode('utf-8')
wandb.run.name = curr_branch + "_" + args.job_id

os.chdir('../data/')
data_root = os.getcwd()
# Dataset Loader and Sampler
if args.dataset != 'Inshop':
    trn_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        mode='train',
        transform=dataset.utils.make_transform(
            is_train=True,
            is_inception=(args.model == 'bn_inception')
        ))
else:
    trn_dataset = Inshop_Dataset(
        root=data_root,
        mode='train',
        transform=dataset.utils.make_transform(
            is_train=True,
            is_inception=(args.model == 'bn_inception')
        ))

if args.IPC:
    balanced_sampler = sampler.BalancedSampler(trn_dataset, batch_size=args.sz_batch, images_per_class=args.IPC)
    batch_sampler = BatchSampler(balanced_sampler, batch_size=args.sz_batch, drop_last=True)
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers=args.nb_workers,
        pin_memory=True,
        batch_sampler=batch_sampler
    )
    print('Balanced Sampling')

else:
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size=args.sz_batch,
        shuffle=True,
        num_workers=args.nb_workers,
        drop_last=True,
        pin_memory=True
    )
    print('Random Sampling')

if args.dataset != 'Inshop':
    ev_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        mode='eval',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

else:
    query_dataset = Inshop_Dataset(
        root=data_root,
        mode='query',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

    gallery_dataset = Inshop_Dataset(
        root=data_root,
        mode='gallery',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_gallery = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

nb_classes = trn_dataset.nb_classes()

# Backbone Model
if args.model.find('googlenet') + 1:
    model = googlenet(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
elif args.model.find('bn_inception') + 1:
    model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm,
                         bn_freeze=args.bn_freeze)
elif args.model.find('resnet18') + 1:
    model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
elif args.model.find('resnet50') + 1:
    model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
elif args.model.find('resnet101') + 1:
    model = Resnet101(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)

model = nn.DataParallel(model)
model = model.cuda()


# DML Losses
if args.loss == 'MS':
    criterion = MultiSimilarityLoss(args.ms_pos_scale, args.ms_neg_scale, args.ms_margin)
elif args.loss == 'Contrastive':
    criterion = ContrastiveLoss(args.contrastive_margin)
elif args.loss == 'Triplet':
    criterion = TripletLoss(args.triplet_margin)

# Train Parameters
param_groups = [
    {'params': list(
        set(model.parameters()).difference(set(model.model.embedding.parameters()))) if args.gpu_id != -1 else
    list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
    {'params': model.model.embedding.parameters() if args.gpu_id != -1 else model.module.model.embedding.parameters(),
     'lr': float(args.lr) * 1},
]

# Optimizer Setting
if args.optimizer == 'sgd':
    opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
elif args.optimizer == 'adam':
    opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
elif args.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay=args.weight_decay, momentum=0.9)
elif args.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=4, threshold=0.001)

print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args.nb_epochs))
losses_list = []
pqlosses_list = []
emb_drif_list = []
pq_update_loss_list = []
best_recall = [0]
best_epoch = 0
iteration = 0
pq = PriorityQueueBatch(args.pq_size, int(args.sz_batch / args.grad_acc), args.sz_embedding, args.pq_rm_old,
                        args.pq_rm_loss)

for epoch in range(0, args.nb_epochs):
    model.train()
    bn_freeze = args.bn_freeze
    if bn_freeze:
        modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
        for m in modules:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    losses_per_epoch = []
    pqlosses_per_epoch = []
    emb_drift_per_epoch = []
    pq_update_loss_per_epoch = []

    # Warmup: Train only new params, helps stabilize learning.
    if args.warm > 0:
        if args.gpu_id != -1:
            unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion.parameters())
        else:
            unfreeze_model_param = list(model.module.model.embedding.parameters()) + list(criterion.parameters())

        if epoch == 0:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = False
        if epoch == args.warm:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = True

    pbar = tqdm(enumerate(dl_tr))

    for batch_idx, (x, y, dl_index) in pbar:
        x = x.squeeze().cuda()
        y = y.squeeze().cuda()

        if args.grad_acc > 1:
            with torch.no_grad():
                pos_neg = model(x)

        loss = 0
        #update pq
        if args.pq_type == 'update' and iteration % args.pq_upd_int == 0 and epoch > args.pq_warmup:
            with torch.no_grad():
                model.eval()
                if args.pq_update_criterion == "oldest":
                    pq_feats, pq_targets, pq_loss, pq_indices, pq_index = pq.get_oldest(1)
                elif args.pq_update_criterion == "worst":
                    pq_feats, pq_targets, pq_loss, pq_indices, pq_index = pq.get_worst(1)
                else:
                    assert False

                pq_feats_old = torch.clone(pq_feats)
                pq_loss_old = torch.clone(pq_loss)
                pq_loader = DataLoader(dataset=Subset(trn_dataset, pq_indices),
                                       batch_size=args.sz_batch,
                                       num_workers=0)
                for ii, (pq_images_, pq_targets_, pq_indices_) in enumerate(pq_loader):
                    index = int(ii * args.sz_batch)
                    pq_feats_ = model(pq_images_.cuda())
                    pq_feats[index:index+args.sz_batch] = pq_feats_
                    pq_targets_ = pq_targets_.cuda()

                    assert torch.equal(pq_targets[index:index+args.sz_batch],pq_targets_)
                    assert torch.equal(pq_indices[index:index + args.sz_batch], pq_indices_)

                    pq_feats_all, pq_targets_all = pq.get()
                    a = torch.ones(args.pq_size,dtype=torch.bool)
                    a[index:index+args.sz_batch] = False
                    pq_feats_all = pq_feats_all[a]
                    pq_targets_all = pq_targets_all[a]

                    pq_feats_all = torch.cat((pq_feats_, pq_feats_all), dim=0)
                    pq_targets_all = torch.cat((pq_targets_, pq_targets_all), dim=0)
                    pq_loss_ = criterion(pq_feats_all, pq_targets_all, pq_feats_all, pq_targets_all)
                    pq_loss[ii] = pq_loss_
                    pq.update(pq_feats_.detach(), pq_loss_.detach(), pq_indices_.detach(), pq_index[ii], iteration)

                emb_drift = torch.sum(torch.abs(pq_feats_old - pq_feats)) / args.pq_size
                emb_drift_per_epoch.append(emb_drift.data.cpu().numpy())
                pq_update_loss = (torch.sum(pq_loss) - torch.sum(pq_loss_old))
                pq_update_loss_per_epoch.append(pq_update_loss.data.cpu().numpy())

        split_images = torch.split(x, int(args.sz_batch / args.grad_acc), dim=0)
        split_targets = torch.split(y, int(args.sz_batch / args.grad_acc), dim=0)
        split_indices = torch.split(dl_index, int(args.sz_batch / args.grad_acc), dim=0)
        opt.zero_grad()
        model.train()
        for m_, y_, i_ in zip(split_images, split_targets, split_indices):
            anchors = model(m_)
            loss = criterion(anchors, y_, anchors, y_) / args.grad_acc
            if epoch >= args.pq_warmup:
                pq_feats, pq_targets = pq.get()
                pq_feats = torch.cat((anchors, pq_feats), dim=0)
                pq_targets = torch.cat((y_, pq_targets), dim=0)
                pq_loss = criterion(anchors, y_, pq_feats, pq_targets) / args.grad_acc
                pq.enqueue_dequeue(anchors.detach(), y_.detach(), pq_loss.detach(), i_, iteration)
                loss = loss + pq_loss
                pqlosses_per_epoch.append(pq_loss.data.cpu().numpy())
            if iteration % 20 == 0:
                print("Iter:", iteration, "loss:", loss.item())
            loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        opt.step()
        opt.zero_grad()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Iteration: {}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                       100. * batch_idx / len(dl_tr),
                loss.item(), iteration))
        iteration += 1

    if iteration > 1:
        emb_drif_list.append(np.mean(emb_drift_per_epoch))
        pq_update_loss_list.append(np.mean(pq_update_loss_per_epoch))
        wandb.log({'emb_drift': emb_drif_list[-1]}, step=epoch)
        wandb.log({'pq_update_loss': pq_update_loss_list[-1]}, step=epoch)

    pqlosses_list.append(np.mean(pqlosses_per_epoch))
    wandb.log({'pqloss': pqlosses_list[-1]}, step=epoch)
    wandb.log({'iteration': iteration}, step=epoch)
    wandb.log({'max mem (GB)': torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0}, step=epoch)
    torch.cuda.reset_peak_memory_stats()

    if epoch >= 0:
        with torch.no_grad():
            print("**Evaluating...**")
            if args.dataset == 'Inshop':
                Recalls = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)
            elif args.dataset != 'SOP':
                Recalls = utils.evaluate_cos(model, dl_ev)
            else:
                Recalls = utils.evaluate_cos_SOP(model, dl_ev)

        # Logging Evaluation Score
        if args.dataset == 'Inshop':
            for i, K in enumerate([1, 10, 20, 30, 40, 50]):
                wandb.log({"R@{}".format(K): Recalls[i]}, step=epoch)
        elif args.dataset != 'SOP':
            for i in range(6):
                wandb.log({"R@{}".format(2 ** i): Recalls[i]}, step=epoch)
        else:
            for i in range(4):
                wandb.log({"R@{}".format(10 ** i): Recalls[i]}, step=epoch)

        scheduler.step(Recalls[0])
        wandb.log({"learning rate": opt.param_groups[0]['lr']}, step=epoch)
        criterion.log_info(epoch, wandb)

        # Best model save
        if best_recall[0] < Recalls[0]:
            best_recall = Recalls
            best_epoch = epoch
            if not os.path.exists('{}'.format(LOG_DIR)):
                os.makedirs('{}'.format(LOG_DIR))
            torch.save({'model_state_dict': model.state_dict()},
                       '{}/{}_{}_best.pth'.format(LOG_DIR, args.dataset, args.model))
            with open('{}/{}_{}_{}_best_results.txt'.format(LOG_DIR, args.job_id, args.dataset, args.model), 'w') as f:
                f.write('Best Epoch: {}\n'.format(best_epoch))
                if args.dataset == 'Inshop':
                    for i, K in enumerate([1, 10, 20, 30, 40, 50]):
                        f.write("Best Recall@{}: {:.4f}\n".format(K, best_recall[i] * 100))
                elif args.dataset != 'SOP':
                    for i in range(6):
                        f.write("Best Recall@{}: {:.4f}\n".format(2 ** i, best_recall[i] * 100))
                else:
                    for i in range(4):
                        f.write("Best Recall@{}: {:.4f}\n".format(10 ** i, best_recall[i] * 100))
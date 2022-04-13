import os
import argparse
import time
import torch
import torch.nn.parallel
import torch.optim
import numpy as np
from config import cfg
from utils.logger import Logger, make_print_to_file
from utils.evaluation import AverageMeter
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from networks import network
from dataloader.head_dataload import Head_dataloader
import random


parser = argparse.ArgumentParser(description='PyTorch CPN Training')
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--gpu', type=list, default=[0, 1, 2, 3])
parser.add_argument('--log-interval', type=int, default=10, metavar='N')
parser.add_argument('--epochs', default=70, type=int, metavar='N')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')

args = parser.parse_args()
args_dict = vars(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.save_folder)

make_print_to_file(path=args.checkpoint)

# create model
model = network.FDGRNet(cfg.backbone_name, cfg.output_shape, cfg.num_class, pretrained = cfg.pretranined)
model.to(args.gpu[0])
model = torch.nn.DataParallel(model, args.gpu)

criterion1 = torch.nn.MSELoss().cuda()
criterion2 = torch.nn.MSELoss(reduce=False).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

if args.resume:
    if isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_dict)
        args.start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
        logger = Logger(join(args.checkpoint, 'log.txt'), resume=True)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    logger = Logger(join(args.checkpoint, 'log.txt'))
    logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss'])
    print('Total params: %.4fMB' % (sum(p.numel() for p in model.parameters()) / (1000000)))


train_loader = torch.utils.data.DataLoader(
    Head_dataloader(cfg, train=True),
    batch_size=cfg.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)


val_loader = torch.utils.data.DataLoader(
    Head_dataloader(cfg, train=False),
    batch_size=cfg.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

def ohkm(loss, top_k):
    ohkm_loss = 0.
    for i in range(loss.size()[0]):
        sub_loss = loss[i]
        topk_val, topk_idx = torch.topk(sub_loss, k=top_k, dim=0, sorted=False)
        tmp_loss = torch.gather(sub_loss, 0, topk_idx)
        ohkm_loss += torch.sum(tmp_loss) / top_k
    ohkm_loss /= loss.size()[0]
    return ohkm_loss

def main():
    print('Train begins...')
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        lr = adjust_learning_rate(optimizer, epoch, cfg.lr_dec_epoch, cfg.lr_gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # train for one epoch
        train_loss = train()
        val_loss = evaluate()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s '
              '| epoch loss {:.6f} | val loss {:.6f} |'.format(
            epoch, time.time() - epoch_start_time, train_loss, val_loss))
        print('-' * 89)

        # append logger file
        logger.append([epoch + 1, lr, train_loss, val_loss])
        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint=args.checkpoint)
    logger.close()

def train():
    model.train()
    losses = AverageMeter()
    for i, (inputs, targets, valid, meta) in enumerate(train_loader):
        input_var = torch.autograd.Variable(inputs.cuda())
        target15, target11, target9, target7 = targets
        refine_target_var = torch.autograd.Variable(target7.cuda(async=True))
        valid_var = torch.autograd.Variable(valid.cuda(async=True))

        global_outputs, refine_output = model(input_var)
        loss = 0.
        global_loss_record = 0.
        refine_loss_record = 0.

        for global_output, label in zip(global_outputs, targets):
            num_points = global_output.size()[1]
            global_label = label * (valid > 1.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
            global_loss = criterion1(global_output, torch.autograd.Variable(global_label.cuda(async=True))) / 2.0
            loss += global_loss
            global_loss_record += global_loss.data.item()
        refine_loss = criterion2(refine_output, refine_target_var)
        refine_loss = refine_loss.mean(dim=3).mean(dim=2)
        refine_loss *= (valid_var > 0.1).type(torch.cuda.FloatTensor)
        refine_loss = ohkm(refine_loss, 8)
        loss += refine_loss
        refine_loss_record += refine_loss.data.item()
        losses.update(loss.data.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i % args.log_interval == 0 and i != 0):
            print('iteration {} | loss: {}, global loss: {}, refine loss: {}, avg loss: {}'
                  .format(i, loss.data.item(), global_loss_record,
                          refine_loss_record, losses.avg))
    return losses.avg

def evaluate():
    #model.eval()
    val_losses = AverageMeter()
    with torch.no_grad():
        for i, (inputs, meta, adj, target7) in enumerate(val_loader):
            input_var = torch.autograd.Variable(inputs.cuda())
            adj_var = torch.autograd.Variable(adj.cuda())
            refine_target_var = torch.autograd.Variable(target7.cuda(async=True))
            global_outputs, refine_output = model(input_var)
            val_loss = criterion1(refine_output, refine_target_var)
            val_losses.update(val_loss.data.item(), inputs.size(0))
    return val_losses.avg


if __name__ == '__main__':
    main()
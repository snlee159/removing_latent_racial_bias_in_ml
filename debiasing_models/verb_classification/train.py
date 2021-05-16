import math, os, random, json, pickle, sys, pdb
import string, shutil, time, argparse
import numpy as np
import itertools

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from tqdm import tqdm as tqdm
from PIL import Image

import torch.nn.functional as F
import torch, torchvision
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
from torch.utils.data import DataLoader

from data_loader import ImSituVerbGender
from model import VerbClassification
from logger import Logger

verb_id_map = pickle.load(open('./data/verb_id.map', 'rb'))

verb2id = verb_id_map['verb2id']
id2verb = verb_id_map['id2verb']
# gender_ratios = pickle.load(open('./data/gender_ratios.p', 'rb'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str,
            help='path for saving checkpoints')
    parser.add_argument('--log_dir', type=str,
            help='path for saving log files')

    parser.add_argument('--ratio', type=str,
            default = '1')
    parser.add_argument('--num_verb', type=int,
            default = 507)

    parser.add_argument('--annotation_dir', type=str,
            default='./data',
            help='annotation files path')
    parser.add_argument('--image_dir',
            default = './data/of500_images_resized',
            help='image directory')

    parser.add_argument('--balanced', action='store_true',
            help='use balanced subset for training')
    parser.add_argument('--gender_balanced', action='store_true',
            help='use balanced subset for training, ratio will be 1/2/3')
    parser.add_argument('--batch_balanced', action='store_true',
            help='in every batch, gender balanced')

    parser.add_argument('--no_image', action='store_true',
            help='do not load image in dataloaders')

    parser.add_argument('--blackout', action='store_true')
    parser.add_argument('--blackout_box', action='store_true')
    parser.add_argument('--blackout_face', action='store_true')
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--edges', action='store_true')

    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # create model save directory
    args.save_dir = os.path.join('./models', args.save_dir)
    if os.path.exists(args.save_dir) and not args.resume:
        print('Path {} exists! and not resuming'.format(args.save_dir))
        return
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # create log save directory for train and val
    args.log_dir = os.path.join('./logs', args.log_dir)
    train_log_dir = os.path.join(args.log_dir, 'train')
    val_log_dir = os.path.join(args.log_dir, 'val')
    if not os.path.exists(train_log_dir): os.makedirs(train_log_dir)
    if not os.path.exists(val_log_dir): os.makedirs(val_log_dir)
    # train_logger = Logger(train_log_dir)
    # val_logger = Logger(val_log_dir)
    train_logger = None
    val_logger = None

    #save all hyper-parameters for training
    with open(os.path.join(args.log_dir, "arguments.txt"), "a") as f:
        f.write(str(args)+'\n')

    # image preprocessing
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    val_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])

    # Data samplers.
    train_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir, split = 'train', transform = train_transform)

    val_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir, split = 'val', transform = val_transform)

    # Data loaders / batch assemblers.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size,
            shuffle = True, num_workers = 6, pin_memory = True)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size,
            shuffle = False, num_workers = 4, pin_memory = True)

    # build the models
    model = VerbClassification(args, args.num_verb).cuda()

    # build loss
    verb_weights = torch.FloatTensor(train_data.getVerbWeights())
    criterion = nn.CrossEntropyLoss(weight=verb_weights, reduction='elementwise_mean').cuda()

    # build optimizer for trainable loss
    def trainable_params():
        for param in model.parameters():
            if param.requires_grad:
                yield param

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num_trainable_params:', num_trainable_params)
    optimizer = torch.optim.Adam(trainable_params(), args.learning_rate, weight_decay = 1e-5)

    best_performance = 0
    if args.resume:
        if os.path.isfile(os.path.join(args.save_dir, 'checkpoint.pth.tar')):
            print("=> loading checkpoint '{}'".format(args.save_dir))
            checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pth.tar'))
            args.start_epoch = checkpoint['epoch']
            best_performance = checkpoint['best_performance']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.save_dir))

    print('before training, evaluate the model')
    test(args, 0, model, criterion, val_loader, val_logger, logging=False)

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        train(args, epoch, model, criterion, train_loader, optimizer, \
                train_logger, logging = True)
        current_performance = test(args, epoch, model, criterion, val_loader, \
                val_logger, logging = True)
        is_best = current_performance > best_performance
        best_performance = max(current_performance, best_performance)
        model_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_performance': best_performance}
        save_checkpoint(args, model_state, is_best, os.path.join(args.save_dir, \
                'checkpoint.pth.tar'))

        # at the end of every run, save the model
        if epoch == args.num_epochs:
            torch.save(model_state, os.path.join(args.save_dir, \
                'checkpoint_%s.pth.tar' % str(args.num_epochs)))

def save_checkpoint(args, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.save_dir, 'model_best.pth.tar'))


def train(args, epoch, model, criterion, train_loader, optimizer, \
    train_logger, logging=True):
    model.train()
    nProcessed = 0
    nTrain = len(train_loader.dataset) # number of images
    loss_logger = AverageMeter()

    res = list()
    t = tqdm(train_loader, desc = 'Train %d' % epoch)
    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        #if batch_idx == 100: break # constrain epoch size

        if args.batch_balanced:
            man_idx = genders[:, 0].nonzero().squeeze()
            if len(man_idx.size()) == 0: man_idx = man_idx.unsqueeze(0)
            woman_idx = genders[:, 1].nonzero().squeeze()
            if len(woman_idx.size()) == 0: woman_idx = woman_idx.unsqueeze(0)
            selected_num = min(len(man_idx), len(woman_idx))

            if selected_num < args.batch_size / 2:
                continue # skip the batch if the selected num is too small
            else:
                selected_num = args.batch_size / 2
                selected_idx = torch.cat((man_idx[:selected_num], woman_idx[:selected_num]), 0)

            images = torch.index_select(images, 0, selected_idx)
            targets = torch.index_select(targets, 0, selected_idx)
            genders = torch.index_select(genders, 0, selected_idx)

        # set mini-batch dataset
        images = images.cuda()
        targets = targets.cuda()

        # forward, Backward and Optimize
        preds = model(images)

        # compute loss and add softmax to preds (crossentropy loss integrates softmax)
        loss = criterion(preds, targets.max(1, keepdim=False)[1])
        loss_logger.update(loss.item())

        preds = F.softmax(preds, dim=1)
        preds_max = preds.max(1, keepdim=True)[1]

        # save the exact preds (binary)
        tensor = torch.tensor((), dtype=torch.float64)
        preds_exact = tensor.new_zeros(preds.size())
        for idx, item in enumerate(preds_max):
        	preds_exact[idx, item] = 1

        res.append((image_ids, preds.detach().cpu(), targets.detach().cpu(), genders, preds_exact))

        # backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        nProcessed += len(images)
        t.set_postfix(loss = loss_logger.avg, completed = nProcessed)


    # compute mean average precision score for verb classifier
    total_preds   = torch.cat([entry[1] for entry in res], 0)
    total_targets = torch.cat([entry[2] for entry in res], 0)
    total_genders = torch.cat([entry[3] for entry in res], 0)
    total_preds_exact = torch.cat([entry[4] for entry in res], 0)

    # compute f1 score (no threshold as we simple take the max for multi-classification)
    task_f1_score = f1_score(total_targets.numpy(), total_preds_exact.numpy(), average = 'macro')

    man_idx = total_genders[:, 0].nonzero().squeeze()
    woman_idx = total_genders[:, 1].nonzero().squeeze()

    preds_man = torch.index_select(total_preds, 0, man_idx)
    preds_woman = torch.index_select(total_preds, 0, woman_idx)
    targets_man = torch.index_select(total_targets, 0, man_idx)
    targets_woman = torch.index_select(total_targets, 0, woman_idx)

    meanAP = average_precision_score(total_targets.numpy(), total_preds.numpy(), average='macro')
    meanAP_man = average_precision_score(targets_man.numpy(), preds_man.numpy(), average='macro')
    meanAP_woman = average_precision_score(targets_woman.numpy(), preds_woman.numpy(), average='macro')

    if logging:
        # train_logger.scalar_summary('loss', loss_logger.avg, epoch)
        # train_logger.scalar_summary('task_f1_score', task_f1_score, epoch)
        # train_logger.scalar_summary('meanAP', meanAP, epoch)
        # train_logger.scalar_summary('meanAP_man', meanAP_man, epoch)
        # train_logger.scalar_summary('meanAP_woman', meanAP_woman, epoch)
        pass

    print('man size: {} woman size: {}'.format(len(man_idx), len(woman_idx)))
    print('Train epoch  : {}, meanAP: {:.2f}, meanAP_man: {:.2f}, meanAP_woman: {:.2f}'.format( \
        epoch, meanAP*100, meanAP_man*100, meanAP_woman*100))

def test(args, epoch, model, criterion, val_loader, val_logger, logging=True):

    # set the eval mode
    model.eval()
    nProcessed = 0
    nVal = len(val_loader.dataset) # number of images
    loss_logger = AverageMeter()

    res = list()
    t = tqdm(val_loader, desc = 'Val %d' % epoch)
    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        #if batch_idx == 100: break # constrain epoch size

        # Set mini-batch dataset
        images = images.cuda()
        targets = targets.cuda()

        # Forward, Backward and Optimize
        preds = model(images)

        loss = criterion(preds, targets.max(1, keepdim=False)[1])
        loss_logger.update(loss.item())

        preds = F.softmax(preds, dim=1)
        preds_max = preds.max(1, keepdim=True)[1]

        tensor = torch.tensor((), dtype=torch.float64)
        preds_exact = tensor.new_zeros(preds.size())
        for idx, item in enumerate(preds_max):
            preds_exact[idx, item] = 1

        res.append((image_ids, preds.detach().cpu(), targets.detach().cpu(), genders, preds_exact))


        # Print log info
        nProcessed += len(images)
        t.set_postfix(loss = loss_logger.avg, completed = nProcessed)

    # compute mean average precision score for verb classifier
    total_preds   = torch.cat([entry[1] for entry in res], 0)
    total_targets = torch.cat([entry[2] for entry in res], 0)
    total_genders = torch.cat([entry[3] for entry in res], 0)
    total_preds_exact = torch.cat([entry[4] for entry in res], 0)

    task_f1_score = f1_score(total_targets.numpy(), total_preds_exact.numpy(), average = 'macro')

    man_idx = total_genders[:, 0].nonzero().squeeze()
    woman_idx = total_genders[:, 1].nonzero().squeeze()

    preds_man = torch.index_select(total_preds, 0, man_idx)
    preds_woman = torch.index_select(total_preds, 0, woman_idx)
    targets_man = torch.index_select(total_targets, 0, man_idx)
    targets_woman = torch.index_select(total_targets, 0, woman_idx)

    meanAP = average_precision_score(total_targets.numpy(), total_preds.numpy(), average='macro')
    meanAP_man = average_precision_score(targets_man.numpy(), preds_man.numpy(), average='macro')
    meanAP_woman = average_precision_score(targets_woman.numpy(), preds_woman.numpy(), average='macro')

    if logging:
        # val_logger.scalar_summary('loss', loss_logger.avg, epoch)
        # val_logger.scalar_summary('task_f1_score', task_f1_score, epoch)
        # val_logger.scalar_summary('meanAP', meanAP, epoch)
        # val_logger.scalar_summary('meanAP_man', meanAP_man, epoch)
        # val_logger.scalar_summary('meanAP_woman', meanAP_woman, epoch)
        pass

    print('man size: {} woman size: {}'.format(len(man_idx), len(woman_idx)))
    print('Val epoch  : {}, meanAP: {:.2f}, meanAP_man: {:.2f}, meanAP_woman: {:.2f}'.format( \
        epoch, meanAP*100, meanAP_man*100, meanAP_woman*100))

    return task_f1_score

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()

import math, os, random, json, pickle, sys, pdb
import string, shutil, time, argparse
import numpy as np
import itertools

from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
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
from torch.autograd import Function

from torchvision.utils import save_image
from torch.utils.data import DataLoader

from data_loader import ImSituVerbGender
from ae_adv_model import VerbClassificationAdv
from logger import Logger

## AA change (comment out)
# verb_id_map = pickle.load(open('../data/verb_id.map'))

## AA change '../data/verb_id.map' --> '../data/verb_id_fulldata.map'
verb_id_map = pickle.load(open('../data/verb_id_fulldata.map', 'rb'))

verb2id = verb_id_map['verb2id']
id2verb = verb_id_map['id2verb']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str,
            help='experiment id')
    parser.add_argument('--log_dir', type=str,
            help='path for saving log files')

    parser.add_argument('--annotation_dir', type=str,
            default='../data',
            help='annotation files path')
    parser.add_argument('--image_dir',
            default = '../data/of500_images_resized',
            help='image directory')

    parser.add_argument('--ratio', type=str,
            default = '0')
    
    ## AA change (comment out)
    # parser.add_argument('--num_verb', type=int,
    #         default = 211)

    ## AA change (211 -> 507)
    parser.add_argument('--num_verb', type=int,
            default = 507)

    parser.add_argument('--no_image', action='store_true',
            help='do not load image in dataloaders')

    parser.add_argument('--balanced', action='store_true',
            help='use balanced subset for training')
    parser.add_argument('--gender_balanced', action='store_true',
            help='use gender balanced subset for training')
    parser.add_argument('--batch_balanced', action='store_true',
            help='in every batch, gender balanced')

    parser.add_argument('--beta', type=float, default=1.0,
            help='autoencoder l1 loss weight')

    parser.add_argument('--adv_on', action='store_true',
            help='start adv training')
    parser.add_argument('--layer', type=str,
            help='extract image feature for adv at this layer')
    parser.add_argument('--adv_conv', action='store_true',
            help='add conv layers to adv component')
    parser.add_argument('--no_avgpool', action='store_true',
            help='remove avgpool layer for adv component')
    parser.add_argument('--adv_capacity', type=int,
            help='linear layer dimension for adv component')
    parser.add_argument('--adv_lambda', type=float,
            help='weight assigned to adv loss')
    parser.add_argument('--dropout', type=float,
            help='parameter for dropout layter in adv component')

    parser.add_argument('--blackout', action='store_true')
    parser.add_argument('--blackout_box', action='store_true')
    parser.add_argument('--blackout_face', action='store_true')
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--edges', action='store_true')

    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--autoencoder_finetune', action = 'store_true')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.00001)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.save_dir = os.path.join('./models', args.layer + '_' + str(args.adv_lambda) + '_' + \
            str(args.beta) + '_' + args.exp_id)

    if os.path.exists(args.save_dir) and not args.resume:
        print('Path {} exists! and not resuming'.format(args.save_dir))
        return
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    args.log_dir = os.path.join('./logs', args.layer + '_' + str(args.adv_lambda) + '_' + \
            str(args.beta) + '_' + args.exp_id)

    train_log_dir = os.path.join(args.log_dir, 'train')
    val_log_dir = os.path.join(args.log_dir, 'val')
    if not os.path.exists(train_log_dir): os.makedirs(train_log_dir)
    if not os.path.exists(val_log_dir): os.makedirs(val_log_dir)

    ## AA question: need?
    # train_logger = Logger(train_log_dir)
    train_logger = None
    # val_logger = Logger(val_log_dir)
    val_logger = None

    #save all parameters for training
    with open(os.path.join(args.log_dir, "arguments.txt"), "a") as f:
        f.write(str(args)+'\n')

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    # Image preprocessing
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

    args.gender_balanced = True
    val_data_gender_balanced = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir, split = 'val', transform = val_transform)
    args.gender_balanced = False

    # Data loaders / batch assemblers.
    if args.batch_balanced:
        train_batch_size = int(2.5 * args.batch_size)
    else:
        train_batch_size = int(args.batch_size)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = train_batch_size,
            shuffle = True, num_workers = 6, pin_memory = True)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size,
            shuffle = False, num_workers = 4, pin_memory = True)

    val_loader_gender_balanced = torch.utils.data.DataLoader(val_data_gender_balanced, \
        batch_size = args.batch_size, shuffle = False, num_workers = 4, pin_memory = True)

    # Build the models
    model = VerbClassificationAdv(args, args.num_verb, args.adv_capacity, args.dropout, args.adv_lambda).cuda()

    checkpoint = torch.load('./origin/model_best.pth.tar')
    # load partial weights
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    verb_weights = torch.FloatTensor(train_data.getVerbWeights())
    criterion = nn.CrossEntropyLoss(weight=verb_weights, reduction='mean').cuda()
    criterionL1 = torch.nn.L1Loss(reduction='mean')
    # print model
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
            # load partial weights
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.save_dir))

    print('before training, evaluate the model')
    test_balanced(args, 0, model, criterion, criterionL1, val_loader_gender_balanced,
        val_logger, logging=False)
    test(args, 0, model, criterion, criterionL1, val_loader,  val_logger, logging=False)

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        train(args, epoch, model, criterion, criterionL1, train_loader, optimizer, \
                train_logger, logging=False)
        test_balanced(args, epoch, model, criterion, criterionL1, val_loader_gender_balanced,
            val_logger, logging=False)

        current_performance = test(args, epoch, model, criterion, criterionL1,  val_loader, \
                val_logger, logging = False)
        is_best = current_performance > best_performance
        best_performance = max(current_performance, best_performance)
        model_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_performance': best_performance}
        save_checkpoint(args, model_state, is_best, os.path.join(args.save_dir, \
                'checkpoint.pth.tar'))

def save_checkpoint(args, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.save_dir, 'model_best.pth.tar'))

def get_acc(preds, y):
    pred_val = np.argmax(preds, axis=1)
    y_val = np.argmax(y, axis=1)

    return sum(pred_val == y_val)/len(preds)

def get_precision(preds, y, val):
    pred_val = np.argmax(preds, axis=1)
    y_val = np.argmax(y, axis=1)
    
    return sum(pred_val == y_val)/sum(pred_val = val)

def get_recall(preds, y, val):
    pred_val = np.argmax(preds, axis=1)
    y_val = np.argmax(y, axis=1)
    pred_val_true = pred_val == val
    y_val_false = y_val != val
    false_neg = sum(pred_val_true == y_val_false)

    return sum(pred_val == y_val)/(sum(pred_val == y_val) + false_neg)

def train(args, epoch, model, criterion, criterionL1,  train_loader, optimizer, \
        train_logger, logging=True):
    model.train()
    nProcessed = 0
    task_preds, adv_preds = [], []
    task_truth, adv_truth = [], []
    nTrain = len(train_loader.dataset) # number of images

    ## AA question: need?
    task_loss_logger = AverageMeter()
    adv_loss_logger = AverageMeter()


    t = tqdm(train_loader, desc = 'Train %d' % epoch)
    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        #if batch_idx == 100: break # constrain epoch size

        # Set mini-batch dataset
        if args.batch_balanced:
            ## AA change: comment out
            # man_idx = genders[:, 0].nonzero().squeeze()
            # if len(man_idx.size()) == 0: man_idx = man_idx.unsqueeze(0)
            # woman_idx = genders[:, 1].nonzero().squeeze()
            # if len(woman_idx.size()) == 0: woman_idx = woman_idx.unsqueeze(0)
            # selected_num = min(len(man_idx), len(woman_idx))

            ## AA changes: gender -> race
            black_idx = genders[:, 0].nonzero().squeeze()
            if len(black_idx.size()) == 0: black_idx = black_idx.unsqueeze(0)
            eastAsian_idx = genders[:, 1].nonzero().squeeze()
            if len(eastAsian_idx.size()) == 0: eastAsian_idx = eastAsian_idx.unsqueeze(0)
            indian_idx = genders[:, 2].nonzero().squeeze()
            if len(indian_idx.size()) == 0: indian_idx = indian_idx.unsqueeze(0)
            latino_idx = genders[:, 3].nonzero().squeeze()
            if len(latino_idx.size()) == 0: latino_idx = latino_idx.unsqueeze(0)
            middleEastern_idx = genders[:, 4].nonzero().squeeze()
            if len(middleEastern_idx.size()) == 0: middleEastern_idx = middleEastern_idx.unsqueeze(0)
            southeastAsian_idx = genders[:, 5].nonzero().squeeze()
            if len(southeastAsian_idx.size()) == 0: southeastAsian_idx = southeastAsian_idx.unsqueeze(0)
            white_idx = genders[:, 6].nonzero().squeeze()
            if len(white_idx.size()) == 0: white_idx = white_idx.unsqueeze(0)
            selected_num = min(len(black_idx), len(eastAsian_idx), len(indian_idx), len(latino_idx), len(middleEastern_idx), len(southeastAsian_idx), len(white_idx))

            ## AA change: batch_size / 2 -> batch_size / 7
            if selected_num < args.batch_size / 7:
                continue
            else:
                ## AA change: batch_size / 2 -> batch_size / 7
                selected_num = args.batch_size / 7
                ## AA change: gender -> race
                selected_idx = torch.cat((black_idx[:selected_num], eastAsian_idx[:selected_num], indian_idx[:selected_num], latino_idx[:selected_num], middleEastern_idx[:selected_num], southeastAsian_idx[:selected_num], white_idx[:selected_num], ), 0)

            images = torch.index_select(images, 0, selected_idx)
            targets = torch.index_select(targets, 0, selected_idx)
            genders = torch.index_select(genders, 0, selected_idx)

        images = images.cuda()
        targets = targets.cuda()
        genders = genders.cuda()

        # Forward, Backward and Optimizer
        task_pred, adv_pred, autoencoded_images = model(images)
        l1_loss = criterionL1(autoencoded_images, images)

        task_loss = criterion(task_pred, targets.max(1, keepdim=False)[1])

        adv_loss = F.cross_entropy(adv_pred, genders.max(1, keepdim=False)[1], reduction='mean')

        ## AA question: need?
        task_loss_logger.update(task_loss.item())
        adv_loss_logger.update(adv_loss.item())

        adv_pred = np.argmax(F.softmax(adv_pred, dim=1).cpu().detach().numpy(), axis=1)
        adv_preds += adv_pred.tolist()
        adv_truth += genders.cpu().max(1, keepdim=False)[1].numpy().tolist()


        task_pred = F.softmax(task_pred, dim=1)
        if batch_idx > 0 and len(task_preds) > 0:
            task_preds = torch.cat((task_preds, task_pred.detach().cpu()), 0)
            task_truth = torch.cat((task_truth, targets.cpu()), 0)
            total_genders = torch.cat((total_genders, genders.cpu()), 0)
        else:
            task_preds = task_pred.detach().cpu()
            task_truth = targets.cpu()
            total_genders = genders.cpu()

        loss = task_loss + adv_loss + args.beta * l1_loss

        # backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    task_f1_score = f1_score(task_truth.max(1)[1].numpy(), task_preds.max(1)[1].numpy(), average = 'macro')

    ## AA change: comment out
    # man_idx = total_genders[:, 0].nonzero().squeeze()
    # woman_idx = total_genders[:, 1].nonzero().squeeze()
    # preds_man = torch.index_select(task_preds, 0, man_idx)
    # preds_woman = torch.index_select(task_preds, 0, woman_idx)
    # targets_man = torch.index_select(task_truth, 0, man_idx)
    # targets_woman = torch.index_select(task_truth, 0, woman_idx)
    # meanAP = average_precision_score(task_truth.numpy(), task_preds.numpy(), average='macro')
    # meanAP_man = average_precision_score(targets_man.numpy(), preds_man.numpy(), average='macro')
    # meanAP_woman = average_precision_score(targets_woman.numpy(), preds_woman.numpy(), average='macro')
    # adv_acc = accuracy_score(adv_truth, adv_preds)

    ## AA change: gender -> race
    black_idx = total_genders[:, 0].nonzero().squeeze()
    eastAsian_idx = total_genders[:, 1].nonzero().squeeze()
    indian_idx = total_genders[:, 2].nonzero().squeeze()
    latino_idx = total_genders[:, 3].nonzero().squeeze()
    middleEastern_idx = total_genders[:, 4].nonzero().squeeze()
    southeastAsian_idx = total_genders[:, 5].nonzero().squeeze()
    white_idx = total_genders[:, 6].nonzero().squeeze()

    preds_black = torch.index_select(task_preds, 0, black_idx)
    preds_eastAsian = torch.index_select(task_preds, 0, eastAsian_idx)
    preds_indian = torch.index_select(task_preds, 0, indian_idx)
    preds_latino = torch.index_select(task_preds, 0, latino_idx)
    preds_middleEastern = torch.index_select(task_preds, 0, middleEastern_idx)
    preds_southeastAsian = torch.index_select(task_preds, 0, southeastAsian_idx)
    preds_white = torch.index_select(task_preds, 0, white_idx)

    targets_black = torch.index_select(task_truth, 0, black_idx)
    targets_eastAsian = torch.index_select(task_truth, 0, eastAsian_idx)
    targets_indian = torch.index_select(task_truth, 0, indian_idx)
    targets_latino = torch.index_select(task_truth, 0, latino_idx)
    targets_middleEastern = torch.index_select(task_truth, 0, middleEastern_idx)
    targets_southeastAsian = torch.index_select(task_truth, 0, southeastAsian_idx)
    targets_white = torch.index_select(task_truth, 0, white_idx)

    meanAP = accuracy_score(task_truth.numpy(), task_preds.numpy())
    meanAP_black = accuracy_score(targets_black.cpu().numpy().argmax(axis=1), preds_black.cpu().numpy().argmax(axis=1))
    meanAP_eastAsian = accuracy_score(targets_eastAsian.cpu().numpy().argmax(axis=1), preds_eastAsian.cpu().numpy().argmax(axis=1))
    meanAP_indian = accuracy_score(targets_indian.cpu().numpy().argmax(axis=1), preds_indian.cpu().numpy().argmax(axis=1))
    meanAP_latino = accuracy_score(targets_latino.cpu().numpy().argmax(axis=1), preds_latino.cpu().numpy().argmax(axis=1))
    meanAP_middleEastern = accuracy_score(targets_middleEastern.cpu().numpy().argmax(axis=1), preds_middleEastern.cpu().numpy().argmax(axis=1))
    meanAP_southeastAsian = accuracy_score(targets_southeastAsian.cpu().numpy().argmax(axis=1), preds_southeastAsian.cpu().numpy().argmax(axis=1))
    meanAP_white = accuracy_score(targets_white.cpu().numpy().argmax(axis=1), preds_white.cpu().numpy().argmax(axis=1))
    
    adv_acc = accuracy_score(adv_truth, adv_preds)

    ## AA question: need?
    if logging:
        # train_logger.scalar_summary('task loss', task_loss_logger.avg, epoch)
        # train_logger.scalar_summary('adv loss', adv_loss_logger.avg, epoch)
        # train_logger.scalar_summary('task_f1_score', task_f1_score, epoch)
        # train_logger.scalar_summary('meanAP', meanAP, epoch)
        # train_logger.scalar_summary('meanAP_man', meanAP_man, epoch)
        # train_logger.scalar_summary('meanAP_woman', meanAP_woman, epoch)
        # train_logger.scalar_summary('adv acc', adv_acc, epoch)
        None

    ## AA change gender -> race
    print('black size: {} eastAsian size: {} indian size: {} latino size: {} middleEastern size: {} southeastAsian size: {} white size: {}'.format(len(black_idx), len(eastAsian_idx), len(indian_idx), len(latino_idx), len(middleEastern_idx), len(southeastAsian_idx), len(white_idx)))
    print('Train epoch  : {}, meanAP: {:.2f}, meanAP_black: {:.2f}, meanAP_eastAsian: {:.2f}, meanAP_indian: {:.2f}, meanAP_latino: {:.2f}, meanAP_middleEastern: {:.2f}, meanAP_southeastAsian: {:.2f}, meanAP_white: {:.2f}, adv acc: {:.2f}, '.format( \
        epoch, meanAP*100, meanAP_black*100, meanAP_eastAsian*100,  meanAP_indian*100,  meanAP_latino*100,  meanAP_middleEastern*100,  meanAP_southeastAsian*100,  meanAP_white*100, adv_acc*100))

def test_balanced(args, epoch, model, criterion, criterionL1,  val_loader, val_logger, print_every=10000, logging=True):
    model.eval()
    nProcessed = 0
    task_preds, adv_preds = [], []
    task_truth, adv_truth = [], []
    nTest = len(val_loader.dataset) # number of images

    ## AA question: need?
    task_loss_logger = AverageMeter()
    adv_loss_logger = AverageMeter()

    t = tqdm(val_loader, desc = 'Val balanced %d' % epoch)
    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        #if batch_idx == 100: break # constrain epoch size

        # Set mini-batch dataset
        images = images.cuda()
        targets = targets.cuda()
        genders = genders.cuda()

        # Forward, Backward and Optimizer
        task_pred, adv_pred, autoencoded_images = model(images)
        l1_loss = criterionL1(autoencoded_images, images)

        task_loss = criterion(task_pred, targets.max(1, keepdim=False)[1])

        adv_loss = F.cross_entropy(adv_pred, genders.max(1, keepdim=False)[1], reduction='mean')

        ## AA question: need?
        task_loss_logger.update(task_loss.item())
        adv_loss_logger.update(adv_loss.item())

        adv_pred = np.argmax(F.softmax(adv_pred, dim=1).cpu().detach().numpy(), axis=1)
        adv_preds += adv_pred.tolist()
        adv_truth += genders.cpu().max(1, keepdim=False)[1].numpy().tolist()

        task_pred = F.softmax(task_pred, dim=1)
        if batch_idx > 0 and len(task_preds) > 0:
            task_preds = torch.cat((task_preds, task_pred.detach().cpu()), 0)
            task_truth = torch.cat((task_truth, targets.cpu()), 0)
            total_genders = torch.cat((total_genders, genders.cpu()), 0)
        else:
            task_preds = task_pred.detach().cpu()
            task_truth = targets.cpu()
            total_genders = genders.cpu()

        loss = task_loss + adv_loss + args.beta * l1_loss

    task_f1_score = f1_score(task_truth.max(1)[1].numpy(), task_preds.max(1)[1].numpy(), average = 'macro')

    ## AA change: comment out
    # man_idx = total_genders[:, 0].nonzero().squeeze()
    # woman_idx = total_genders[:, 1].nonzero().squeeze()
    # preds_man = torch.index_select(task_preds, 0, man_idx)
    # preds_woman = torch.index_select(task_preds, 0, woman_idx)
    # targets_man = torch.index_select(task_truth, 0, man_idx)
    # targets_woman = torch.index_select(task_truth, 0, woman_idx)
    # meanAP = average_precision_score(task_truth.numpy(), task_preds.numpy(), average='macro')
    # meanAP_man = average_precision_score(targets_man.numpy(), preds_man.numpy(), average='macro')
    # meanAP_woman = average_precision_score(targets_woman.numpy(), preds_woman.numpy(), average='macro')
    # adv_acc = accuracy_score(adv_truth, adv_preds)

    ## AA change: gender -> race
    black_idx = total_genders[:, 0].nonzero().squeeze()
    eastAsian_idx = total_genders[:, 1].nonzero().squeeze()
    indian_idx = total_genders[:, 2].nonzero().squeeze()
    latino_idx = total_genders[:, 3].nonzero().squeeze()
    middleEastern_idx = total_genders[:, 4].nonzero().squeeze()
    southeastAsian_idx = total_genders[:, 5].nonzero().squeeze()
    white_idx = total_genders[:, 6].nonzero().squeeze()

    preds_black = torch.index_select(task_preds, 0, black_idx)
    preds_eastAsian = torch.index_select(task_preds, 0, eastAsian_idx)
    preds_indian = torch.index_select(task_preds, 0, indian_idx)
    preds_latino = torch.index_select(task_preds, 0, latino_idx)
    preds_middleEastern = torch.index_select(task_preds, 0, middleEastern_idx)
    preds_southeastAsian = torch.index_select(task_preds, 0, southeastAsian_idx)
    preds_white = torch.index_select(task_preds, 0, white_idx)

    targets_black = torch.index_select(task_truth, 0, black_idx)
    targets_eastAsian = torch.index_select(task_truth, 0, eastAsian_idx)
    targets_indian = torch.index_select(task_truth, 0, indian_idx)
    targets_latino = torch.index_select(task_truth, 0, latino_idx)
    targets_middleEastern = torch.index_select(task_truth, 0, middleEastern_idx)
    targets_southeastAsian = torch.index_select(task_truth, 0, southeastAsian_idx)
    targets_white = torch.index_select(task_truth, 0, white_idx)

    meanAP = accuracy_score(task_truth.numpy(), task_preds.numpy())
    meanAP_black = accuracy_score(targets_black.cpu().numpy().argmax(axis=1), preds_black.cpu().numpy().argmax(axis=1))
    meanAP_eastAsian = accuracy_score(targets_eastAsian.cpu().numpy().argmax(axis=1), preds_eastAsian.cpu().numpy().argmax(axis=1))
    meanAP_indian = accuracy_score(targets_indian.cpu().numpy().argmax(axis=1), preds_indian.cpu().numpy().argmax(axis=1))
    meanAP_latino = accuracy_score(targets_latino.cpu().numpy().argmax(axis=1), preds_latino.cpu().numpy().argmax(axis=1))
    meanAP_middleEastern = accuracy_score(targets_middleEastern.cpu().numpy().argmax(axis=1), preds_middleEastern.cpu().numpy().argmax(axis=1))
    meanAP_southeastAsian = accuracy_score(targets_southeastAsian.cpu().numpy().argmax(axis=1), preds_southeastAsian.cpu().numpy().argmax(axis=1))
    meanAP_white = accuracy_score(targets_white.cpu().numpy().argmax(axis=1), preds_white.cpu().numpy().argmax(axis=1))

    adv_acc = accuracy_score(adv_truth, adv_preds)   

    ## AA question: need
    if logging:
        val_logger.scalar_summary('adv loss balanced', adv_loss_logger.avg, epoch)
        val_logger.scalar_summary('adv acc balanced', adv_acc, epoch)

    ## AA change gender -> race
    print('black size: {} eastAsian size: {} indian size: {} latino size: {} middleEastern size: {} southeastAsian size: {} white size: {}'.format(len(black_idx), len(eastAsian_idx), len(indian_idx), len(latino_idx), len(middleEastern_idx), len(southeastAsian_idx), len(white_idx)))
    print('Train epoch  : {}, meanAP: {:.2f}, meanAP_black: {:.2f}, meanAP_eastAsian: {:.2f}, meanAP_indian: {:.2f}, meanAP_latino: {:.2f}, meanAP_middleEastern: {:.2f}, meanAP_southeastAsian: {:.2f}, meanAP_white: {:.2f}, adv acc: {:.2f}, '.format( \
        epoch, meanAP*100, meanAP_black*100, meanAP_eastAsian*100,  meanAP_indian*100,  meanAP_latino*100,  meanAP_middleEastern*100,  meanAP_southeastAsian*100,  meanAP_white*100, adv_acc*100))

    return task_f1_score

def test(args, epoch, model, criterion, criterionL1, val_loader, val_logger, print_every=10000, logging=True):
    model.eval()
    nProcessed = 0
    task_preds, adv_preds = [], []
    task_truth, adv_truth = [], []
    nTest = len(val_loader.dataset) # number of images

    ## AA question: need?
    task_loss_logger = AverageMeter()
    adv_loss_logger = AverageMeter()

    t = tqdm(val_loader, desc = 'Val %d' % epoch)
    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        #if batch_idx == 100: break # constrain epoch size

        # Set mini-batch dataset
        images = images.cuda()
        targets = targets.cuda()
        genders = genders.cuda()

        # Forward, Backward and Optimizer
        task_pred, adv_pred, autoencoded_images = model(images)
        l1_loss = criterionL1(autoencoded_images, images)

        task_loss = criterion(task_pred, targets.max(1, keepdim=False)[1])

        adv_loss = F.cross_entropy(adv_pred, genders.max(1, keepdim=False)[1], reduction='mean')

        task_loss_logger.update(task_loss.item())
        adv_loss_logger.update(adv_loss.item())

        adv_pred = np.argmax(F.softmax(adv_pred, dim=1).cpu().detach().numpy(), axis=1)
        adv_preds += adv_pred.tolist()
        adv_truth += genders.cpu().max(1, keepdim=False)[1].numpy().tolist()

        task_pred = F.softmax(task_pred, dim=1)
        if batch_idx > 0 and len(task_preds) > 0:
            task_preds = torch.cat((task_preds, task_pred.detach().cpu()), 0)
            task_truth = torch.cat((task_truth, targets.cpu()), 0)
            total_genders = torch.cat((total_genders, genders.cpu()), 0)
        else:
            task_preds = task_pred.detach().cpu()
            task_truth = targets.cpu()
            total_genders = genders.cpu()

        loss = task_loss + adv_loss + args.beta * l1_loss

    task_f1_score = f1_score(task_truth.max(1)[1].numpy(), task_preds.max(1)[1].numpy(), average = 'macro')

    ## AA change: comment out
    # man_idx = total_genders[:, 0].nonzero().squeeze()
    # woman_idx = total_genders[:, 1].nonzero().squeeze()
    # preds_man = torch.index_select(task_preds, 0, man_idx)
    # preds_woman = torch.index_select(task_preds, 0, woman_idx)
    # targets_man = torch.index_select(task_truth, 0, man_idx)
    # targets_woman = torch.index_select(task_truth, 0, woman_idx)
    # meanAP = average_precision_score(task_truth.numpy(), task_preds.numpy(), average='macro')
    # meanAP_man = average_precision_score(targets_man.numpy(), preds_man.numpy(), average='macro')
    # meanAP_woman = average_precision_score(targets_woman.numpy(), preds_woman.numpy(), average='macro')
    # adv_acc = accuracy_score(adv_truth, adv_preds)

    ## AA change: gender -> race
    black_idx = total_genders[:, 0].nonzero().squeeze()
    eastAsian_idx = total_genders[:, 1].nonzero().squeeze()
    indian_idx = total_genders[:, 2].nonzero().squeeze()
    latino_idx = total_genders[:, 3].nonzero().squeeze()
    middleEastern_idx = total_genders[:, 4].nonzero().squeeze()
    southeastAsian_idx = total_genders[:, 5].nonzero().squeeze()
    white_idx = total_genders[:, 6].nonzero().squeeze()

    preds_black = torch.index_select(task_preds, 0, black_idx)
    preds_eastAsian = torch.index_select(task_preds, 0, eastAsian_idx)
    preds_indian = torch.index_select(task_preds, 0, indian_idx)
    preds_latino = torch.index_select(task_preds, 0, latino_idx)
    preds_middleEastern = torch.index_select(task_preds, 0, middleEastern_idx)
    preds_southeastAsian = torch.index_select(task_preds, 0, southeastAsian_idx)
    preds_white = torch.index_select(task_preds, 0, white_idx)

    targets_black = torch.index_select(task_truth, 0, black_idx)
    targets_eastAsian = torch.index_select(task_truth, 0, eastAsian_idx)
    targets_indian = torch.index_select(task_truth, 0, indian_idx)
    targets_latino = torch.index_select(task_truth, 0, latino_idx)
    targets_middleEastern = torch.index_select(task_truth, 0, middleEastern_idx)
    targets_southeastAsian = torch.index_select(task_truth, 0, southeastAsian_idx)
    targets_white = torch.index_select(task_truth, 0, white_idx)

    meanAP = average_precision_score(task_truth.numpy(), task_preds.numpy(), average='macro')
    meanAP_black = average_precision_score(targets_black.numpy(), preds_black.numpy(), average='macro')
    meanAP_eastAsian = average_precision_score(targets_eastAsian.numpy(), preds_eastAsian.numpy(), average='macro')
    meanAP_indian = average_precision_score(targets_indian.numpy(), preds_indian.numpy(), average='macro')
    meanAP_latino = average_precision_score(targets_latino.numpy(), preds_latino.numpy(), average='macro')
    meanAP_middleEastern = average_precision_score(targets_middleEastern.numpy(), preds_middleEastern.numpy(), average='macro')
    meanAP_southeastAsian = average_precision_score(targets_southeastAsian.numpy(), preds_southeastAsian.numpy(), average='macro')
    meanAP_white = average_precision_score(targets_white.numpy(), preds_white.numpy(), average='macro')

    adv_acc = accuracy_score(adv_truth, adv_preds)    

    ## AA question: need?
    if logging:
        val_logger.scalar_summary('task loss', task_loss_logger.avg, epoch)
        val_logger.scalar_summary('adv loss', adv_loss_logger.avg, epoch)
        val_logger.scalar_summary('task_f1_score', task_f1_score, epoch)
        val_logger.scalar_summary('meanAP', meanAP, epoch)
        val_logger.scalar_summary('meanAP_man', meanAP_man, epoch)
        val_logger.scalar_summary('meanAP_woman', meanAP_woman, epoch)
        val_logger.scalar_summary('adv acc', adv_acc, epoch)

    ## AA change gender -> race
    print('black size: {} eastAsian size: {} indian size: {} latino size: {} middleEastern size: {} southeastAsian size: {} white size: {}'.format(len(black_idx), len(eastAsian_idx), len(indian_idx), len(latino_idx), len(middleEastern_idx), len(southeastAsian_idx), len(white_idx)))
    print('Train epoch  : {}, meanAP: {:.2f}, meanAP_black: {:.2f}, meanAP_eastAsian: {:.2f}, meanAP_indian: {:.2f}, meanAP_latino: {:.2f}, meanAP_middleEastern: {:.2f}, meanAP_southeastAsian: {:.2f}, meanAP_white: {:.2f}, adv acc: {:.2f}, '.format( \
        epoch, meanAP*100, meanAP_black*100, meanAP_eastAsian*100,  meanAP_indian*100,  meanAP_latino*100,  meanAP_middleEastern*100,  meanAP_southeastAsian*100,  meanAP_white*100, adv_acc*100))

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

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import math, os, random, json, pickle, sys, pdb
import string, shutil, time, argparse, operator, collections
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from tqdm import tqdm as tqdm

from data_loader import ImSituVerbGender, ImSituVerbGenderFeature
from model import VerbClassification, GenderClassifier

####### data preparation #########################
verb_id_map = pickle.load(open('./data/verb_id.map'))
verb2id = verb_id_map['verb2id']
id2verb = verb_id_map['id2verb']
gender_ratios = pickle.load(open('./data/gender_ratios.p'))

def test(args, model, data_loader):

    # set the eval mode
    model.eval()
    nProcessed = 0

    res = list()
    t = tqdm(data_loader, desc = 'Val')
    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        #if batch_idx == 100: break # constrain epoch size

        # Set mini-batch dataset
        images = images.cuda()
        targets = targets.cuda()

        # Forward, Backward and Optimize
        preds = model(images)

        preds = F.softmax(preds, dim=1)
        preds_max = preds.max(1, keepdim=True)[1]

        tensor = torch.tensor((), dtype=torch.float64)
        preds_exact = tensor.new_zeros(preds.size())
        for idx, item in enumerate(preds_max):
            preds_exact[idx, item] = 1

        res.append((image_ids, preds.detach().cpu(), targets.detach().cpu(), genders, preds_exact))

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


    print('man size: {} woman size: {}'.format(len(man_idx), len(woman_idx)))
    print('Val epoch, meanAP: {:.2f}, meanAP_man: {:.2f}, meanAP_woman: {:.2f}'.format( \
        meanAP*100, meanAP_man*100, meanAP_woman*100))
    print('f1 score: ', task_f1_score)

    return task_f1_score


def generate_image_feature(split, image_features_path, data_loader, encoder):

    targets = list()
    genders = list()
    image_ids = list()
    potentials = list()

    for ind, (images_, targets_, genders_, image_ids_) in enumerate(data_loader):
        images_ = images_.cuda()
        preds = encoder(images_)
        potentials.append(preds.detach().cpu())
        targets.append(targets_.cpu())
        genders.append(genders_.cpu())
        image_ids.append(image_ids_.cpu())

    targets = torch.cat(targets, 0)
    genders = torch.cat(genders, 0)
    image_ids = torch.cat(image_ids, 0)
    potentials = torch.cat(potentials, 0)

    torch.save(targets, os.path.join(image_features_path, '{}_targets.pth'.format(split)))
    torch.save(genders, os.path.join(image_features_path, '{}_genders.pth'.format(split)))
    torch.save(image_ids, os.path.join(image_features_path, '{}_image_ids.pth'.format(split)))
    torch.save(potentials, os.path.join(image_features_path, '{}_potentials.pth'.format(split)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str,
            help='experiment id, e.g. conv4_300_1.0_0.2_1')

    parser.add_argument('--num_rounds', type=int,
            default = 5)

    parser.add_argument('--annotation_dir', type=str,
            default='./data',
            help='annotation files path')
    parser.add_argument('--image_dir',
            default = './data/of500_images_resized',
            help='image directory')

    parser.add_argument('--gender_balanced', action='store_true',
            help='use gender balanced subset for training')

    parser.add_argument('--balanced', action='store_true',
            help='use balanced subset for training')
    parser.add_argument('--ratio', type=str,
            default = '0')

    parser.add_argument('--num_verb', type=int,
            default = 211)

    parser.add_argument('--blackout', action='store_true')
    parser.add_argument('--blackout_box', action='store_true')
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--edges', action='store_true')
    parser.add_argument('--blackout_face', action='store_true')

    parser.add_argument('--noise', action='store_true',
            help='add noise to image features')
    parser.add_argument('--noise_scale', type=float, default=0.2,
            help='std in gaussian noise')
    parser.add_argument('--no_image', action='store_true')

    parser.add_argument('--hid_size', type=int, default=300,
            help='linear layer dimension for attacker')

    parser.add_argument('--attacker_capacity', type=int, default=300,
            help='linear layer dimension for attacker')
    parser.add_argument('--attacker_dropout', type=float, default=0.2,
            help='parameter for dropout layter in attacker')

    ## training setting for attacker
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=0.00005,
            help='attacker learning rate')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    test_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])

    #Build the encoder from adv model
    encoder = VerbClassification(args, args.num_verb).cuda()
    model_path = os.path.join('./models', args.exp_id)
    if os.path.isfile(os.path.join(model_path, 'model_best.pth.tar')):
        print("=> loading encoder from '{}'".format(model_path))
        checkpoint = torch.load(os.path.join(model_path, 'model_best.pth.tar'))
        best_performance = checkpoint['best_performance']
        encoder.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    encoder.eval()

    # Data samplers.
    val_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir,split = 'val', transform = test_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size, \
            shuffle = False, num_workers = 4,pin_memory = True)

    test_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir,split = 'test', transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, \
            shuffle = False, num_workers = 4,pin_memory = True)

    print('val set performance:')
    test(args, encoder, val_loader)

    print('test set performance:')
    test(args, encoder, test_loader)

    acc_list = {}
    acc_list['potential'] = []

    args.gender_balanced = True
    for i in range(args.num_rounds):

        train_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir,split = 'train', transform = train_transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size,
                    shuffle = True, num_workers = 6, pin_memory = True)

        # Data samplers for val set.
        val_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
                image_dir = args.image_dir,split = 'val', transform = test_transform)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size, \
                shuffle = False, num_workers = 4,pin_memory = True)

        # Data samplers for test set.
        test_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
                image_dir = args.image_dir,split = 'test', transform = test_transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, \
                shuffle = False, num_workers = 4,pin_memory = True)

        image_features_path = os.path.join(model_path, 'image_features')
        if not os.path.exists(image_features_path):
            os.makedirs(image_features_path)

        # get image features from encoder
        generate_image_feature('train', image_features_path, train_loader, encoder)
        generate_image_feature('val', image_features_path, val_loader, encoder)
        generate_image_feature('test', image_features_path, test_loader, encoder)

        train_data = ImSituVerbGenderFeature(args, image_features_path, split = 'train')
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size,
                    shuffle = True, num_workers = 6, pin_memory = True)

        val_data = ImSituVerbGenderFeature(args, image_features_path, split = 'val')
        val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size,
                    shuffle = True, num_workers = 6, pin_memory = True)

        test_data = ImSituVerbGenderFeature(args, image_features_path, split = 'test')
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size,
                    shuffle = True, num_workers = 6, pin_memory = True)



        model_save_dir = './attacker'
        if args.noise:
            args.exp_id += '_noise' + str(args.noise_scale)
        model_save_dir = os.path.join(model_save_dir, str(args.exp_id))

        if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)

        for feature_type in acc_list.keys():

            #import pdb
            #pdb.set_trace()

            attacker = GenderClassifier(args, args.num_verb)

            attacker = attacker.cuda()

            optimizer = optim.Adam(attacker.parameters(), lr=args.learning_rate, weight_decay = 1e-5)

            train_attacker(args.num_epochs, optimizer, attacker, encoder, train_loader, val_loader, \
               model_save_dir, feature_type)

            # evaluate best attacker on balanced test split
            best_attacker = torch.load(model_save_dir + '/best_attacker.pth.tar')
            attacker.load_state_dict(best_attacker['state_dict'])
            _, val_acc = epoch_pass(0, val_loader, attacker, encoder, None, False, feature_type)
            val_acc = 0.5 + abs(val_acc - 0.5)
            _, test_acc = epoch_pass(0, test_loader, attacker, encoder, None, False, feature_type)
            test_acc = 0.5 + abs(test_acc - 0.5)
            acc_list[feature_type].append(test_acc)
            print('round {} feature type: {}, test acc: {}, val acc: {}'.format(i, feature_type, test_acc, val_acc))

    for feature_type in acc_list.keys():
        print(acc_list[feature_type], np.std(np.array(acc_list[feature_type])))
        print('{} average leakage: {}'.format(feature_type, np.mean(np.array(acc_list[feature_type]))))

def train_attacker(num_epochs, optimizer, attacker, encoder, train_loader, test_loader, model_save_dir, feature_type, print_every=500):
    # training setting
    encoder.eval()
    attacker.train()

    train_acc_arr, train_loss_arr = [], []
    dev_acc_arr, dev_loss_arr = [], []
    best_model_epoch = 1
    best_score = 0.0

    for epoch in xrange(1, num_epochs + 1):

        # train
        loss, train_task_acc = epoch_pass(epoch, train_loader, attacker, encoder, optimizer, True, feature_type, print_every)
        train_acc_arr.append(train_task_acc)
        train_loss_arr.append(loss)
        if epoch % 10 == 0:
          print('train, {0}, adv acc: {1:.2f}'.format(epoch, train_task_acc*100))

        # dev
        loss, dev_task_acc = epoch_pass(epoch, test_loader, attacker, encoder, optimizer, False, feature_type, print_every)
        dev_acc_arr.append(dev_task_acc)
        dev_loss_arr.append(loss)
        if epoch % 10 == 0:
          print('dev, {0}, adv acc: {1:.2f}'.format(epoch, dev_task_acc*100))

        if dev_task_acc > best_score:
            best_score = dev_task_acc
            best_model_epoch = epoch
            torch.save({'epoch': epoch, 'state_dict': attacker.state_dict()}, model_save_dir + '/best_attacker.pth.tar')

        if epoch % 10 == 0:
          print('current best dev score: {:.2f}'.format(best_score*100))


def epoch_pass(epoch, data_loader, attacker, encoder, optimizer, training, feature_type, print_every=500):

    t_loss = 0.0
    preds, truth = [], []
    n_processed = 0

    if training:
        attacker.train()
    else:
        attacker.eval()

    for ind, (targets, genders, image_ids, potentials) in enumerate(data_loader):

        features = potentials.float().cuda()
        adv_pred = attacker(features)
        loss = F.cross_entropy(adv_pred, genders.cuda().max(1, keepdim=False)[1], reduction='elementwise_mean')

        adv_pred = np.argmax(F.softmax(adv_pred, dim=1).cpu().detach().numpy(), axis=1)
        preds += adv_pred.tolist()
        truth += genders.cpu().max(1, keepdim=False)[1].numpy().tolist()

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t_loss += loss.item()
        n_processed += len(targets)

        acc_score = accuracy_score(truth, preds)

    return t_loss / n_processed, acc_score

if __name__ == '__main__':
    main()

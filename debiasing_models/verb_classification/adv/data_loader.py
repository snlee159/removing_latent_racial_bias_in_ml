import json, os, string, random, time, pickle, gc, pdb
from PIL import Image
from PIL import ImageFilter
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import random

# number of races in the dataset
numRaces = 7

class ImSituVerbGender(data.Dataset):
    def __init__(self, args, annotation_dir, image_dir, split = 'train', transform = None, \
        balanced_val=False, balanced_test=False):
        print("ImSituVerbGender dataloader")

        self.split = split
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.args = args

        # verb_id_map = pickle.load(open('../data/verb_id.map', 'rb'))
        verb_id_map = pickle.load(open('../data/verb_id_fulldata.map', 'rb'))
        self.verb2id = verb_id_map['verb2id']
        self.id2verb = verb_id_map['id2verb']

        print("loading %s annotations.........." % self.split)
        self.ann_data = pickle.load(open(os.path.join(annotation_dir, split+"_race.ids"), "rb"))

        if args.balanced and split == 'train':
            # balanced_subset = pickle.load(open("../data/{}_ratio_{}.ids".format(split, \
                # args.ratio)))
            balanced_subset = pickle.load(open("balanced_race.ids".format(split, \
                args.ratio)))                           
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        if balanced_val and split == 'val':
            balanced_subset = pickle.load(open("balanced_race.ids".format(split, \
                args.ratio)))
            # balanced_subset = pickle.load(open("../data/{}_ratio_{}_race.ids".format(split, \
                # args.ratio)))                
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        if balanced_test and split == 'test':
            balanced_subset = pickle.load(open("../data/{}_ratio_{}.ids".format(split, \
                args.ratio)))
            # balanced_subset = pickle.load(open("../data/{}_ratio_{}_race.ids".format(split, \
            #     args.ratio)))                
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        print("dataset size: %d" % len(self.ann_data))
        self.verb_ann = np.zeros((len(self.ann_data), len(self.verb2id)))
        # self.gender_ann = np.zeros((len(self.ann_data), 2), dtype=int)
        self.gender_ann = np.zeros((len(self.ann_data), numRaces), dtype=int)

        for index, ann in enumerate(self.ann_data):
            self.verb_ann[index][ann['verb']] = 1
            self.gender_ann[index][ann['gender']] = 1

        if args.gender_balanced:
            black_idx = np.nonzero(self.gender_ann[:, 0])[0]
            eastAsian_idx = np.nonzero(self.gender_ann[:, 1])[0]
            indian_idx = np.nonzero(self.gender_ann[:, 2])[0]
            latino_idx = np.nonzero(self.gender_ann[:, 3])[0]
            middleEastern_idx = np.nonzero(self.gender_ann[:, 4])[0]
            southeastAsian_idx = np.nonzero(self.gender_ann[:, 5])[0]
            white_idx = np.nonzero(self.gender_ann[:, 6])[0]
            random.shuffle(black_idx)
            random.shuffle(eastAsian_idx)
            random.shuffle(indian_idx)
            random.shuffle(latino_idx)
            random.shuffle(middleEastern_idx)
            random.shuffle(southeastAsian_idx)
            random.shuffle(white_idx)
            min_len = 200 if self.split == 'train' else 100
            selected_idxs = list(black_idx[:min_len]) + list(eastAsian_idx[:min_len]) + list(indian_idx[:min_len]) + list(latino_idx[:min_len]) + list(middleEastern_idx[:min_len]) + list(southeastAsian_idx[:min_len]) + list(white_idx[:min_len])
            self.ann_data = [self.ann_data[idx] for idx in selected_idxs]
            self.verb_ann = np.take(self.verb_ann, selected_idxs, axis=0)
            self.gender_ann = np.take(self.gender_ann, selected_idxs, axis=0)

            # selected_idxs = list(man_idxs[:min_len]) + list(woman_idxs[:min_len])
            # man_idxs = np.nonzero(self.gender_ann[:, 0])[0]
            # woman_idxs = np.nonzero(self.gender_ann[:, 1])[0]

        self.image_ids = range(len(self.ann_data))
        
        for i in range(self.gender_ann.shape[1]):
            print("Size of label {}: {}".format(i, len(np.nonzero(self.gender_ann[:, i])[0])))

    def __getitem__(self, index):
        if self.args.no_image:
            return torch.Tensor([1]), torch.Tensor(self.verb_ann[index]), \
                torch.LongTensor(self.gender_ann[index]), torch.Tensor([1])

        img = self.ann_data[index]
        image_name = img['image_name']
        image_path_ = os.path.join(self.image_dir, image_name)

        img_ = Image.open(image_path_).convert('RGB')

        if self.transform is not None:
            img_ = self.transform(img_)

        return img_, torch.Tensor(self.verb_ann[index]), \
                torch.LongTensor(self.gender_ann[index]), torch.LongTensor([self.image_ids[index]])

    def getGenderWeights(self):
        return (self.gender_ann == 0).sum(axis = 0) / (1e-15 + \
                (self.gender_ann.sum(axis = 0) + (self.gender_ann == 0).sum(axis = 0) ))

    def getVerbWeights(self):
        return (self.verb_ann == 0).sum(axis = 0) / (1e-15 + self.verb_ann.sum(axis = 0))

    def __len__(self):
        return len(self.ann_data)


class ImSituVerbGenderFeature(data.Dataset):
    def __init__(self, args, feature_dir, split = 'train'):
        print("ImSituVerbGenderFeature dataloader")

        self.split = split
        self.args = args

        print("loading %s annotations.........." % self.split)

        self.targets = torch.load(os.path.join(feature_dir, '{}_targets.pth'.format(split)))
        self.genders = torch.load(os.path.join(feature_dir, '{}_genders.pth'.format(split)))
        self.image_ids = torch.load(os.path.join(feature_dir, '{}_image_ids.pth'.format(split)))
        self.potentials = torch.load(os.path.join(feature_dir, '{}_potentials.pth'.format(split)))

        # print("man size : {} and woman size: {}".format(len(self.genders[:, 0].nonzero().squeeze()), \
        #     len(self.genders[:, 1].nonzero().squeeze())))

        for i in range(self.genders.shape[1]):
            print("Size of race {}: {}".format(i, len(self.genders[:, i].nonzero().squeeze())))

    def __getitem__(self, index):
        return self.targets[index], self.genders[index], self.image_ids[index], self.potentials[index]

    def __len__(self):
        return len(self.targets)


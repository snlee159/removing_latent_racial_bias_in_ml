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

        verb_id_map = pickle.load(open('./data/verb_id_fulldata.map', 'rb'))
        # verb_id_map = pickle.load(open('./data/verb_id.map', 'rb'))
        self.verb2id = verb_id_map['verb2id']
        self.id2verb = verb_id_map['id2verb']

        print("loading %s annotations.........." % self.split)
        # self.ann_data = pickle.load(open(os.path.join(annotation_dir, split+ ".data"), 'rb'))
        self.ann_data = pickle.load(open("./data/train_ratio_1_race.ids", "rb"))

        if args.balanced and split == 'train':
            # balanced_subset = pickle.load(open("./data/{}_ratio_{}.ids".format(split, \
                # args.ratio)))
            file_train = "./data/{}_ratio_{}_race.ids".format(split, args.ratio)
            print("Training file is:", file_train)
            balanced_subset = pickle.load(open(file_train))                
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        if balanced_val and split == 'val':
            # balanced_subset = pickle.load(open("./data/{}_ratio_{}.ids".format(split, \
                # args.ratio)))
            # balanced_subset = pickle.load(open("./data/{}_ratio_{}_race.ids".format(split, \
                # args.ratio)))                
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        if balanced_test and split == 'test':
            balanced_subset = pickle.load(open("./data/{}_ratio_{}.ids".format(split, \
                args.ratio)))
            # balanced_subset = pickle.load(open("./data/{}_ratio_{}_race.ids".format(split, \
                # args.ratio)))   
        print("dataset size: %d" % len(self.ann_data))
        self.verb_ann = np.zeros((len(self.ann_data), len(self.verb2id)))
        # self.gender_ann = np.zeros((len(self.ann_data), 2), dtype=int)
        self.gender_ann = np.zeros((len(self.ann_data), numRaces), dtype=int)


        for index, ann in enumerate(self.ann_data):
            self.verb_ann[index][ann['verb']] = 1
            self.gender_ann[index][ann['gender']] = 1

        if args.gender_balanced:
            man_idxs = np.nonzero(self.gender_ann[:, 0])[0]
            woman_idxs = np.nonzero(self.gender_ann[:, 1])[0]
            random.shuffle(man_idxs)# only blackout box is available for imSitu
            random.shuffle(woman_idxs)
            min_len = 7300 if self.split == 'train' else 3000
            selected_idxs = list(man_idxs[:min_len]) + list(woman_idxs[:min_len])

            self.ann_data = [self.ann_data[idx] for idx in selected_idxs]
            self.verb_ann = np.take(self.verb_ann, selected_idxs, axis=0)
            self.gender_ann = np.take(self.gender_ann, selected_idxs, axis=0)

        self.image_ids = range(len(self.ann_data))

        # print("man size : {} and woman size: {}".format(len(np.nonzero( \
        #         self.gender_ann[:, 0])[0]), len(np.nonzero(self.gender_ann[:, 1])[0])))
        print(self.gender_ann.shape)
        for i in range(self.gender_ann.shape[1]):
            print("Size of label {}: {}".format(i, len(np.nonzero(self.gender_ann[:, i])[0])))

        if args.blackout_box:
            self.masks_ann = json.load(open(os.path.join(annotation_dir, \
                'masks/masks/'+split+'.json')))

        if args.blackout_face:
            self.faces = pickle.load(open(split+'_faces.p'), 'rb')

    def __getitem__(self, index):
        if self.args.no_image:
            return torch.Tensor([1]), torch.Tensor(self.verb_ann[index]), \
                torch.LongTensor(self.gender_ann[index]), torch.Tensor([1])

        img = self.ann_data[index]
        image_name = img['image_name']
        image_path_ = os.path.join(self.image_dir, image_name)

        img_ = Image.open(image_path_).convert('RGB')
        if self.args.blackout_box: # only blackout box is available for imSitu

            img_ = self.blackout_img(image_name, img_)

        elif self.args.blackout_face:
            img_ = self.blackout_face(image_name, img_)

        if self.transform is not None:
            img_ = self.transform(img_)

        return img_, torch.Tensor(self.verb_ann[index]), \
                torch.LongTensor(self.gender_ann[index]), torch.LongTensor([self.image_ids[index]])

    def getGenderWeights(self):
        return (self.gender_ann == 0).sum(axis = 0) / (1e-15 + \
                (self.gender_ann.sum(axis = 0) + (self.gender_ann == 0).sum(axis = 0) ))

    def getVerbWeights(self):
        return (self.verb_ann == 0).sum(axis = 0) / (1e-15 + self.verb_ann.sum(axis = 0))

    def blackout_img(self, img_name, img):
        if 'agent' not in self.masks_ann[img_name]['bb']:
            return img # if mask is not available, return the original img

        bb = self.masks_ann[img_name]['bb']['agent']
        if -1 in bb:
            return img # if mask if not available, return the original img
        else:
            xmin, ymin, xmax, ymax = self.masks_ann[img_name]['bb']['agent']
            width = self.masks_ann[img_name]['width']
            height = self.masks_ann[img_name]['height']
            black_img = Image.fromarray(np.zeros((img.size[1], img.size[0])))
            mask = np.zeros((width, height))
            for i in range(xmin, xmax):
                for j in range(ymin, ymax):
                    mask[j][i] = 1
            img_mask = Image.fromarray(255 * (mask > 0).astype('uint8')).resize((img.size[0], img.size[1]), Image.ANTIALIAS)
            return Image.composite(black_img, img, img_mask)

    def blackout_face(self, img_name, img):

        try:
            vertices = self.faces[img_name]
        except:
            return img

        # vertices = self.faces[img_name]
        width = img.size[1]
        height = img.size[0]

        black_img = Image.fromarray(np.zeros((img.size[1], img.size[0])))
        mask = np.zeros((width, height))
        for poly in vertices:
            xmin, ymin = poly[0].strip('()').split(',')
            xmax, ymax = poly[2].strip('()').split(',')
            for i in range(int(xmin), int(xmax)):
                for j in range(int(ymin), int(ymax)):
                    mask[j][i] = 1
        img_mask = Image.fromarray(255 * (mask > 0).astype('uint8')).resize((img.size[0], \
                img.size[1]), Image.ANTIALIAS)

        return Image.composite(black_img, img, img_mask)

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

        print("man size : {} and woman size: {}".format(len(self.genders[:, 0].nonzero().squeeze()), \
            len(self.genders[:, 1].nonzero().squeeze())))
        # for i in range(self.genders.size(1)):
            # print("Size of race {}: {}".format(i, len(self.genders[:, i].nonzero().squeeze())))


    def __getitem__(self, index):
        return self.targets[index], self.genders[index], self.image_ids[index], self.potentials[index]

    def __len__(self):
        return len(self.targets)


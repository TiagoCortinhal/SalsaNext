#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp
import os
import zipfile
import h5py

import __init__ as booger
import numpy as np
from common.avgmeter import *
from common.logger import Logger
from common.warmupLR import *
from tasks.semantic.modules.ioueval import *
from tasks.semantic.modules.segmentator import *


def open_npz(filename):
    data = np.load(filename,mmap_mode='r')
    in_vol = data['in_vol']
    proj_mask = data['proj_mask']
    proj_labels = data['proj_labels']
    path_seq = [data['path_seq'][0]]
    path_name = [data['path_name'][0]]

    return in_vol, proj_mask, proj_labels, path_seq, path_name


class SaveDataSet:
    def __init__(self, ARCH, DATA, datadir, logdir, path=None):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.path = path

        # put logger where it belongs
        self.tb_logger = Logger(self.log + "/tb")
        self.info = {"train_update": 0,
                     "train_loss": 0,
                     "train_acc": 0,
                     "train_iou": 0,
                     "valid_loss": 0,
                     "valid_acc": 0,
                     "valid_iou": 0,
                     "backbone_lr": 0,
                     "decoder_lr": 0,
                     "head_lr": 0,
                     "post_lr": 0}

        # get the data

        parserModule = imp.load_source("parserModule",
                                       booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                       self.DATA["name"] + '/parser.py')
        self.parser = parserModule.Parser(root=self.datadir,
                                          train_sequences=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                                          valid_sequences=self.DATA["split"]["valid"],
                                          # test_sequences=self.DATA["split"]["test"],
                                          test_sequences=None,
                                          labels=self.DATA["labels"],
                                          color_map=self.DATA["color_map"],
                                          learning_map=self.DATA["learning_map"],
                                          learning_map_inv=self.DATA["learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"],
                                          max_points=self.ARCH["dataset"]["max_points"],
                                          batch_size=self.ARCH["train"]["batch_size"],
                                          workers=self.ARCH["train"]["workers"],
                                          gt=True,
                                          shuffle_train=False)

        train_loader = self.parser.get_train_set()
        prev_i = 0
        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(
                train_loader):

            if not os.path.exists('/home/tiago/Downloads/dataset/projected_saved_h5py/{}'.format(path_seq[0])):
                os.makedirs('/home/tiago/Downloads/dataset/projected_saved_h5py/{}'.format(path_seq[0]))
                prev_i = 0
            with h5py.File("/home/tiago/Downloads/dataset/projected_saved_h5py/{}/{}.h5".format(path_seq[0], prev_i), "w") as file:
                dset = file.create_dataset("in_vol", data=np.squeeze(in_vol.numpy()))
                dset = file.create_dataset("proj_mask", data=np.squeeze(proj_mask.numpy()))
                dset = file.create_dataset("proj_labels", data=np.squeeze(proj_labels.numpy()))

            # np.savez_compressed("/home/tiago/Downloads/dataset/projected_saved_h5py/{}/{}".format(path_seq[0], prev_i),
            #                     in_vol=np.squeeze(in_vol),
            #                     proj_mask=np.squeeze(proj_mask),
            #                     proj_labels=np.squeeze(proj_labels),
            #                     path_seq=path_seq,
            #                     path_name=path_name, fix_imports=False)
            # saved_correctly = False
            # while not saved_correctly:
            #     try:
            #         in_vol, proj_mask, proj_labels, path_seq, path_name = open_npz(
            #             "/home/tiago/Downloads/dataset/projected_saved/{}/{}".format(path_seq[0],
            #                                                                                prev_i) + '.npz')
            #         file = zipfile.ZipFile(
            #             "/home/tiago/Downloads/dataset/projected_saved/{}/{}".format(path_seq[0],
            #                                                                                prev_i) + '.npz')
            #         file = file.testzip()
            #         if file:
            #             print("Testzip failed for {}-{}".format(path_seq[0], prev_i))
            #             saved_correctly = False
            #             raise zipfile.BadZipFile
            #         else:
            #             saved_correctly = True
            #     except Exception as e:
            #         print(e)
            #         print("Error saving {}-{}".format(path_seq[0], prev_i))
            #         np.savez_compressed(
            #             "/home/tiago/Downloads/dataset/projected_saved/{}/{}".format(path_seq[0], prev_i),
            #             in_vol=np.squeeze(in_vol),
            #             proj_mask=np.squeeze(proj_mask),
            #             proj_labels=np.squeeze(proj_labels),
            #             path_seq=path_seq,
            #             path_name=path_name, fix_imports=False)

            prev_i += 1
            print("{}/{}".format(i, len(train_loader)))

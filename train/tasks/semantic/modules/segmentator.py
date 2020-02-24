#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import imp

import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F
from tasks.semantic.postproc.CRF import CRF



class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3), stride=1):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size,padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA = self.bn1(resA)

        resA = self.conv3(resA)
        resA = self.act3(resA)
        resA = self.bn2(resA)
        return resA + shortcut


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, use_concrete_dropout, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True,drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size,padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=kernel_size,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size, stride=(1,2),padding=1)
            self.act4 = nn.LeakyReLU()
            self.bn3 = nn.BatchNorm2d(out_filters)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA = self.bn1(resA)

        resA = self.conv3(resA)
        resA = self.act3(resA)
        resA = self.bn2(resA)
        resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB  = resA
            resB = self.conv4(resB)
            resB = self.act4(resB)
            resB = self.bn3(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, use_concrete_dropout, dropout_rate, kernel_size=(3, 3),
                 layer_name="dec", training=True,drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.trans = nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride=(1,2), padding=1)
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm2d(out_filters)

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(out_filters, out_filters, kernel_size,padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size,padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x,skip):
        upA = self.trans(x)
        upA = F.pad(upA, (0,1,0,0), mode='replicate')

        upA = self.trans_act(upA)
        upA = self.trans_bn(upA)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = upA + skip
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE = self.bn1(upE)

        upE = self.conv2(upE)
        upE = self.act2(upE)
        upE = self.bn2(upE)

        upE = self.conv3(upE)
        upE = self.act3(upE)
        upE = self.bn3(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


class SalsaNet(nn.Module):
    def __init__(self, ARCH, nclasses, path=None, path_append="", strict=False):
        super(SalsaNet, self).__init__()
        self.ARCH = ARCH
        self.nclasses = nclasses
        self.path = path
        self.path_append = path_append
        self.strict = False

        self.downCntx = ResContextBlock(5, 32)
        self.resBlock1 = ResBlock(32, 32, False, 0.5, pooling=True,drop_out=False)
        self.resBlock2 = ResBlock(32, 2 * 32, False, 0.5, pooling=True)
        self.resBlock3 = ResBlock(2 * 32, 4 * 32, False, 0.5, pooling=True)
        self.resBlock4 = ResBlock(4 * 32, 8 * 32, False, 0.5, pooling=True)
        self.resBlock5 = ResBlock(8 * 32, 16 * 32, False, 0.5, pooling=True)
        self.resBlock6 = ResBlock(16 * 32, 16 * 32, False, 0.5, pooling=False,drop_out=True)


        self.upBlock1 = UpBlock(16 * 32,16 * 32,False,0.5,drop_out=True)
        self.upBlock2 = UpBlock(16 * 32, 8 * 32,False,0.5)
        self.upBlock3 = UpBlock(8 * 32, 4 * 32, False, 0.5)
        self.upBlock4 = UpBlock(4 * 32,2 * 32, False, 0.5)
        self.upBlock5 = UpBlock(2 * 32, 32, False, 0.5, drop_out=False)

        self.logits = nn.Conv2d(32, nclasses,kernel_size=(1,1))


    def forward(self, x):

        downCntx = self.downCntx(x)
        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c,down4b = self.resBlock5(down3c)
        down5b = self.resBlock6(down4c)

        up4e = self.upBlock1(down5b,down4b)
        up3e = self.upBlock2(up4e,down3b)
        up2e = self.upBlock3(up3e,down2b)
        up1e = self.upBlock4(up2e,down1b)
        up0e = self.upBlock5(up1e, down0b)

        logits = self.logits(up0e)
        logits = F.softmax(logits, dim=1)
        return logits





class Segmentator(nn.Module):
    def __init__(self, ARCH, nclasses, path=None, path_append="", strict=False):
        super().__init__()
        self.ARCH = ARCH
        self.nclasses = nclasses
        self.path = path
        self.path_append = path_append
        self.strict = False

        # get the model
        bboneModule = imp.load_source("bboneModule",
                                      '/data/tiacor/PycharmProjects/RangeNet++ Adapted SalsaNet/train' + '/backbones/' +
                                      self.ARCH["backbone"]["name"] + '.py')
        self.backbone = bboneModule.Backbone(params=self.ARCH["backbone"])

        # do a pass of the backbone to initialize the skip connections
        stub = torch.zeros((1,
                            self.backbone.get_input_depth(),
                            self.ARCH["dataset"]["sensor"]["img_prop"]["height"],
                            self.ARCH["dataset"]["sensor"]["img_prop"]["width"]))

        if torch.cuda.is_available():
            stub = stub
            self.backbone
        _, stub_skips = self.backbone(stub)

        decoderModule = imp.load_source("decoderModule",
                                        '/data/tiacor/PycharmProjects/RangeNet++ Adapted SalsaNet/train' + '/tasks/semantic/decoders/' +
                                        self.ARCH["decoder"]["name"] + '.py')
        self.decoder = decoderModule.Decoder(params=self.ARCH["decoder"],
                                             stub_skips=stub_skips,
                                             OS=self.ARCH["backbone"]["OS"],
                                             feature_depth=self.backbone.get_last_depth())

        self.head = nn.Sequential(nn.Dropout2d(p=ARCH["head"]["dropout"]),
                                  nn.Conv2d(self.decoder.get_last_depth(),
                                            self.nclasses, kernel_size=3,
                                            stride=1, padding=1))

        if self.ARCH["post"]["CRF"]["use"]:
            self.CRF = CRF(self.ARCH["post"]["CRF"]["params"], self.nclasses)
        else:
            self.CRF = None

        # train backbone?
        if not self.ARCH["backbone"]["train"]:
            for w in self.backbone.parameters():
                w.requires_grad = False

        # train decoder?
        if not self.ARCH["decoder"]["train"]:
            for w in self.decoder.parameters():
                w.requires_grad = False

        # train head?
        if not self.ARCH["head"]["train"]:
            for w in self.head.parameters():
                w.requires_grad = False

        # train CRF?
        if self.CRF and not self.ARCH["post"]["CRF"]["train"]:
            for w in self.CRF.parameters():
                w.requires_grad = False

        # print number of parameters and the ones requiring gradients
        # print number of parameters and the ones requiring gradients
        weights_total = sum(p.numel() for p in self.parameters())
        weights_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total number of parameters: ", weights_total)
        print("Total number of parameters requires_grad: ", weights_grad)

        # breakdown by layer
        weights_enc = sum(p.numel() for p in self.backbone.parameters())
        weights_dec = sum(p.numel() for p in self.decoder.parameters())
        weights_head = sum(p.numel() for p in self.head.parameters())
        print("Param encoder ", weights_enc)
        print("Param decoder ", weights_dec)
        print("Param head ", weights_head)
        if self.CRF:
            weights_crf = sum(p.numel() for p in self.CRF.parameters())
            print("Param CRF ", weights_crf)

        # get weights
        if path is not None:
            # try backbone
            try:
                w_dict = torch.load(path + "/backbone" + path_append,
                                    map_location=lambda storage, loc: storage)
                self.backbone.load_state_dict(w_dict, strict=True)
                print("Successfully loaded model backbone weights")
            except Exception as e:
                print()
                print("Couldn't load backbone, using random weights. Error: ", e)
                if strict:
                    print("I'm in strict mode and failure to load weights blows me up :)")
                    raise e

            # try decoder
            try:
                w_dict = torch.load(path + "/segmentation_decoder" + path_append,
                                    map_location=lambda storage, loc: storage)
                self.decoder.load_state_dict(w_dict, strict=True)
                print("Successfully loaded model decoder weights")
            except Exception as e:
                print("Couldn't load decoder, using random weights. Error: ", e)
                if strict:
                    print("I'm in strict mode and failure to load weights blows me up :)")
                    raise e

            # try head
            try:
                w_dict = torch.load(path + "/segmentation_head" + path_append,
                                    map_location=lambda storage, loc: storage)
                self.head.load_state_dict(w_dict, strict=True)
                print("Successfully loaded model head weights")
            except Exception as e:
                print("Couldn't load head, using random weights. Error: ", e)
                if strict:
                    print("I'm in strict mode and failure to load weights blows me up :)")
                    raise e

            # try CRF
            if self.CRF:
                try:
                    w_dict = torch.load(path + "/segmentation_CRF" + path_append,
                                        map_location=lambda storage, loc: storage)
                    self.CRF.load_state_dict(w_dict, strict=True)
                    print("Successfully loaded model CRF weights")
                except Exception as e:
                    print("Couldn't load CRF, using random weights. Error: ", e)
                    if strict:
                        print("I'm in strict mode and failure to load weights blows me up :)")
                        raise e
        else:
            print("No path to pretrained, using random init.")

    def forward(self, x, mask=None):
        y, skips = self.backbone(x)
        y = self.decoder(y, skips)
        y = self.head(y)
        y = F.softmax(y, dim=1)
        if self.CRF:
            assert (mask is not None)
            y = self.CRF(x, y, mask)
        return y

    def save_checkpoint(self, logdir, suffix=""):
        # Save the weights
        torch.save(self.backbone.state_dict(), logdir +
                   "/backbone" + suffix)
        torch.save(self.decoder.state_dict(), logdir +
                   "/segmentation_decoder" + suffix)
        torch.save(self.head.state_dict(), logdir +
                   "/segmentation_head" + suffix)
        if self.CRF:
            torch.save(self.CRF.state_dict(), logdir +
                       "/segmentation_CRF" + suffix)

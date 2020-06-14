# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp

import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F
from tasks.semantic.modules.ConcreteDropout import ConcreteDropoutConvolutional

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)


    def forward(self, x):

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = ConcreteDropoutConvolutional(temp=2./3.)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = ConcreteDropoutConvolutional(temp=2./3.)

    def forward(self, x):
        reg_total = 0
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA


        if self.pooling:
            if self.drop_out:
                resB,reg = self.dropout(resA,nn.Identity())
                reg_total += reg
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA,reg_total
        else:
            if self.drop_out:
                resB,reg = self.dropout(resA,nn.Identity())
                reg_total += reg
            else:
                resB = resA
            return resB,reg_total


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters,drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = ConcreteDropoutConvolutional(temp=2. / 3.)
        self.dropout2 = ConcreteDropoutConvolutional(temp=2./3.)

        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)
        self.dropout3 = ConcreteDropoutConvolutional(temp=2./3.)

    def forward(self, x, skip):
        reg_total = 0
        upA = nn.PixelShuffle(2)(x)

        if self.drop_out:
            upA,reg = self.dropout1(upA,nn.Identity())
            reg_total += reg

        upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB, reg = self.dropout2(upB,nn.Identity())
            reg_total += reg

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        if self.drop_out:
            concat,reg = self.dropout3(concat,nn.Identity())
            reg_total += reg
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE,reg = self.dropout3(upE,nn.Identity())
            reg_total += reg

        return upE,reg_total


class SalsaNextUncertainty(nn.Module):
    def __init__(self, nclasses):
        super(SalsaNextUncertainty, self).__init__()
        self.nclasses = nclasses
        self.strict = False

        self.downCntx = ResContextBlock(5, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)
        self.downCntx4 = ResContextBlock(32, 32)
        self.downCntx5 = ResContextBlock(32, 32)


        self.resBlock1 = ResBlock(32, 2 * 32, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 4 * 32, pooling=True)
        self.resBlock3 = ResBlock(4 * 32, 8 * 32,  pooling=True)
        self.resBlock4 = ResBlock(8 * 32, 8 * 32, pooling=True)
        self.resBlock5 = ResBlock(8 * 32, 8 * 32, pooling=False)

        self.upBlock1 = UpBlock(8 * 32, 4 * 32)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32)
        self.upBlock4_logvar = UpBlock(2 * 32, 32, drop_out=False)
        self.upBlock4_mu = UpBlock(2 * 32, 32, drop_out=False)

        self.logits_mu = nn.Conv2d(32, nclasses, kernel_size=(1, 1))
        self.logits_logvar = nn.Conv2d(32, nclasses, kernel_size=(1, 1))
        self.concrete_logvar = ConcreteDropoutConvolutional(temp=2. / 3.)
        self.concrete_mu = ConcreteDropoutConvolutional(temp=2. / 3.)

    def forward(self, x):
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)
        downCntx = self.downCntx4(downCntx)
        downCntx = self.downCntx5(downCntx)


        down0c, down0b, reg1 = self.resBlock1(downCntx)
        down1c, down1b, reg2 = self.resBlock2(down0c)
        down2c, down2b, reg3 = self.resBlock3(down1c)
        down3c, down3b, reg4 = self.resBlock4(down2c)
        down5c, reg5 = self.resBlock5(down3c)

        up4e, reg6 = self.upBlock1(down5c,down3b)
        up3e, reg7 = self.upBlock2(up4e, down2b)
        up2e, reg8 = self.upBlock3(up3e, down1b)
        up1e_mu, reg9 = self.upBlock4_mu(up2e, down0b)
        up1e_logvar, reg10 = self.upBlock4_logvar(up2e, down0b)

        logits_mu,reg12 = self.concrete_mu(up1e_mu,self.logits_mu)
        logits_logvar, reg11 = self.concrete_logvar(up1e_logvar,self.logits_logvar)

        logits_mu = F.softmax(logits_mu, dim=1)
        logits_logvar = F.softmax(logits_logvar, dim=1)

        return logits_logvar,logits_mu,reg1+reg2+reg3+reg4+reg5+reg6+reg7+reg8+reg9+reg10+reg11+reg12
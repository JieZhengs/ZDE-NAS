"""
from __future__ import print_function

import copy
import time

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchprofile import profile_macs
from utils import Utils, Log
import load_dataset.data_loader_LC25000 as data_loader
import os
from datetime import datetime
import multiprocessing

from compute_zen_score import compute_nas_score
from utils import Utils
from sklearn.metrics import cohen_kappa_score, roc_auc_score, f1_score, balanced_accuracy_score
from tqdm import tqdm
import sys
from typing import Optional

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hswish()'


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def sigmoid(x, inplace=False):
    return x.sigmoid_() if inplace else x.sigmoid()


class SELayer(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25):
        super(SELayer, self).__init__()
        reduce_chs = max(1, int(in_chs * se_ratio))
        self.act_fn = F.relu
        self.gate_fn = sigmoid
        reduced_chs = reduce_chs or in_chs
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        # NOTE adaptiveavgpool can be used here, but seems to cause issues with NVIDIA AMP performance
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se, inplace=True)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU6())
    return conv


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 middle_dw_downsample,
                 stride,
                 expand_ratio,
                 attention
                 ):
        super().__init__()
        # Starting depthwise conv.
        self.attention = attention
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                      groups=expand_filters)
        # Projection with 1x1 convs.
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)

        # Ending depthwise conv.
        # this not used
        # _end_dw_kernel_size = 0
        # self._end_dw = conv_2d(oup, oup, kernel_size=_end_dw_kernel_size, stride=stride, groups=inp, act=False)
        # attention
        if attention:
            self.se = SELayer(expand_filters)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        if self.attention:
            x = self.se(x)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        return x


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        self.Hswish = Hswish(inplace=True)
        #generated_init


    def forward(self, x):
        #generate_forward

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.Hswish(self.conv_end1(out))
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.linear(out)
        return out


class TrainModel(object):
    def __init__(self, is_test):
        if is_test:
            full_trainloader = data_loader.get_train_loader('../datasets/LC25000_data/train', batch_size=8, augment=True,shuffle=True, num_workers=4, pin_memory=True)
            testloader = data_loader.get_test_loader('../datasets/LC25000_data/val', batch_size=8, shuffle=False,num_workers=4, pin_memory=True)
            self.full_trainloader = full_trainloader
            self.testloader = testloader
        net = EvoCNNModel()
        cudnn.benchmark = True
        net = net.cuda()
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        self.net = net
        self.params = 0
        self.flops = 0
        self.zen_score = 0
        self.criterion = criterion.cuda()
        self.best_acc = best_acc
        self.best_epoch = 0
        # self.trainloader = trainloader
        # self.validate_loader = validate_loader
        # self.testloader = testloader
        self.file_id = os.path.basename(__file__).split('.')[0]
        #self.testloader = testloader
        #self.log_record(net, first_time=True)
        #self.log_record('+'*50, first_time=False)

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def zero_proxy(self):
        start_timer = time.time()
        info = compute_nas_score(model=self.net, resolution=128, batch_size=16)
        time_cost = (time.time() - start_timer)
        self.zen_score = info['avg_nas_score']
        Log.info(f'zen-score={self.zen_score:.4g}, time cost={time_cost:.4g} second(s)')

        inputs = torch.randn(8, 3, 128, 128)
        inputs = Variable(inputs.cuda())
        self.params = sum(p.numel() for p in self.net.parameters())
        self.flops = profile_macs(copy.deepcopy(self.net), inputs)
        self.log_record('#parameters:%d, #FLOPs:%d' % (self.params, self.flops))
        self.log_record('Evaluate Total time: %4f second(s), Avg Time cost: %4f second(s), zen-score: %4f' % (
            time.time() - start_timer, float(time_cost / 32), self.zen_score))

    def final_train(self, epoch, optimizer):
        self.net.train()
        # lr = 0.01
        # optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum = 0.9, weight_decay=5e-4)
        running_loss = 0.0
        total = 0
        correct = 0
        full_trainloader = tqdm(self.full_trainloader, file=sys.stdout, leave=True)
        for ii, data in enumerate(full_trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
            full_trainloader.desc = "[Final Train epoch {}]  Loss {} Acc {} ".format(epoch + 1, running_loss / total,
                                                                                    correct / total)
            if epoch == 0 and ii == 0:
                inputs = torch.randn(1, 3, 128, 128)
                inputs = Variable(inputs.cuda())
                self.params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
                # flops1, params1 = profile(self.net, inputs=(inputs,))
                self.flops = profile_macs(copy.deepcopy(self.net), inputs)
                self.log_record('#parameters:%d, #FLOPs:%d' % (self.params, self.flops))
        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f' % (epoch + 1, running_loss / total, (correct / total)))


    def process(self):
        self.zero_proxy()
        return self.zen_score, self.params, self.flops

    def process_test(self):
        params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        inputs = torch.randn(1, 3, 128, 128).cuda()
        flops = profile_macs(copy.deepcopy(self.net), inputs)
        self.log_record('#parameters:%d' % (params))
        total_epoch = Utils.get_params('network', 'epoch_test')
        lr_rate = 0.001
        optimizer = optim.SGD(self.net.parameters(), lr=lr_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
        for p in range(total_epoch):
            self.final_train(p, optimizer)
            self.test(p)
            scheduler.step()
        return self.best_acc, params, flops

    def test(self, epoch):
        with torch.no_grad():
            self.net.eval()
            test_loss = 0.0
            total = 0
            correct = 0

            all_predictions = []
            all_labels = []
            testloader = tqdm(self.testloader, file=sys.stdout, leave=True)
            for _, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                outputs = self.net(inputs)

                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.data).sum()

                outputs = F.softmax(outputs, dim=1)
                all_predictions.extend(outputs.cpu().detach().numpy())
                all_labels.extend(labels.cpu().numpy())

                testloader.desc = "[Final Test epoch {}]  Loss {} Acc {} ".format(epoch + 1, test_loss / total,
                                                                            correct / total)
        roc_auc = roc_auc_score(all_labels, all_predictions, multi_class='ovo', average='weighted')
        cohen_kappa = cohen_kappa_score(all_labels, np.argmax(all_predictions, axis=1))
        f1 = f1_score(all_labels, np.argmax(all_predictions, axis=1), average='weighted')
        IBA = balanced_accuracy_score(all_labels, np.argmax(all_predictions, axis=1))

        if correct / total > self.best_acc:
            torch.save(self.net.state_dict(), './trained_models/best_CNN.pth')
            self.best_acc = correct / total

        self.log_record(
            'Test-Loss:%.4f, Acc:%.4f, cohen_kappa:%.4f, ROC-AUC:%.4f, F1-Score:%.4f, IBA:%.4f' % (
                test_loss / total, correct / total, cohen_kappa, roc_auc, f1, IBA))

class RunModel(object):
    def do_work(self, gpu_id, curr_gen, file_id, is_test, return_dict):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0
        params = 1e9
        flops = 1e9
        try:
            m = TrainModel(is_test)
            m.log_record('Used GPU#%s, worker name:%s[%d]'%(gpu_id, multiprocessing.current_process().name, os.getpid()), first_time=True)
            if is_test:
                best_acc, params, flops = m.process_test()
            else:
                zen_score, params, flops = m.process()
            # return_dict[file_id] = best_acc
        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))
        finally:
            if is_test:
                m.log_record('Finished-Acc:%.4f'%best_acc)
                f = open('./populations/after_%02d.txt'%(curr_gen), 'a+')
                f.write('%s=%.5f\n'%(file_id, best_acc))
                f.flush()
                f.close()
            else:
                m.log_record('Finished-ZenScore:%.4f' % zen_score)
                f = open('./populations/after_%02d.txt'%(curr_gen), 'a+')
                f.write('%s=%.5f\n'%(file_id, zen_score))
                f.flush()
                f.close()

            f = open('./populations/params_%02d.txt' % (curr_gen), 'a+')
            f.write('%s=%d\n' % (file_id, params))
            f.flush()
            f.close()

            f = open('./populations/flops_%02d.txt' % (curr_gen), 'a+')
            f.write('%s=%d\n' % (file_id, flops))
            f.flush()
            f.close()
"""

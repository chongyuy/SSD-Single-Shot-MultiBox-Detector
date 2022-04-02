import argparse
from cProfile import label
import os
import numpy as np
import time
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

num_epochs = 100
batch_size = 16


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)
network.cuda()
network.load_state_dict(torch.load('network.pth'))
network.eval()
cudnn.benchmark = True


if args.test:
    dataset = COCO("CMPT733-Lab3-Workspace/data/data/data/train/images/", "CMPT733-Lab3-Workspace/data/data/data/train/annotations/", class_num, boxs_default, validate=True, train = True, image_size=320)
    dataset_test = COCO("CMPT733-Lab3-Workspace/data/data/data/train/images/", "CMPT733-Lab3-Workspace/data/data/data/train/annotations/", class_num, boxs_default, validate=False, train = True, image_size=320)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    
    optimizer = optim.Adam(network.parameters(), lr = 1e-4)
    #feel free to try other optimizers and parameters.
    train_loss = []
    val_loss = []
    start_time = time.time()
    for epoch in range(num_epochs):
        #TRAINING
        network.train()
        print("starting epoch:" + str(epoch))
        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_,width,height  = data
            # image_res = images_[0].numpy()
            # image_res = np.transpose(image_res, (1,2,0)).astype(np.uint8)
            # plt.imshow(image_res)
            # plt.savefig("img.png")
            # print(ann_box_[0].shape)
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()
            # print(images_[0].numpy())
            
            pred_confidence, pred_box = network(images)
            optimizer.zero_grad()
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            # for name, parms in network.named_parameters():	
            #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
            #         ' -->grad_value:',parms.grad)
            # print(loss_net.data)
            optimizer.step()
            avg_loss += loss_net.data
            avg_count += 1
            m = nn.Softmax(dim=2)
            pred_confidence = m(pred_confidence)
            pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            pred_box_ = pred_box[0].detach().cpu().numpy()
            # visualize_pred("train_before_nms", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)

            pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
            res = visualize_pred("train_after_nms", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
            filename = '{}.txt'.format(str(i).zfill(5))
            file = open('CMPT733-Lab3-Workspace/data/data/data/train/annotations_res/'+ filename,'w')
            for k in range(len(res)):
                res[k][1] = int((res[k][1] / 320) * width)
                res[k][2] = int((res[k][2] / 320) * height)
                res[k][3] = int(res[k][3] * width)
                res[k][4] = int(res[k][4] * height)
                for m in res[k]:
                    file.writelines(str(m)+" ")
                file.writelines("\n")
            # time.sleep(0.5)
        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        # train_loss.append(avg_loss.cpu()/avg_count)
        
        # #visualize
        # pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        # pred_box_ = pred_box[0].detach().cpu().numpy()
        # pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        # visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        
        
        #VALIDATION
        network.eval()
        
        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        avg_val_loss = 0
        avg_val_count = 0
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_,width,height  = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            avg_val_loss += loss_net.data
            avg_val_count += 1
            m = nn.Softmax(dim=2)
            pred_confidence = m(pred_confidence)
            pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            pred_box_ = pred_box[0].detach().cpu().numpy()
            # visualize_pred("val_before_nms", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)

            pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
            res = visualize_pred("val_after_nms", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
            filename = '{}.txt'.format(str(i+5752).zfill(5))
            file = open('CMPT733-Lab3-Workspace/data/data/data/train/annotations_res/'+ filename,'w')
            for k in range(len(res)):
                res[k][1] = int((res[k][1] / 320) * width)
                res[k][2] = int((res[k][2] / 320) * height)
                res[k][3] = int(res[k][3] * width)
                res[k][4] = int(res[k][4] * height)
                for m in res[k]:
                    file.writelines(str(m)+" ")
                file.writelines("\n")
        
            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
        val_loss.append(avg_val_loss.cpu()/avg_val_count)
        #visualize
        m = nn.Softmax(dim=2)
        pred_confidence = m(pred_confidence)
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        # visualize_pred("val", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        
        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)
        
        #save weights
        if epoch%10==9:
            #save last network
            print('saving net...')
            torch.save(network.state_dict(), 'network_new.pth')
    x_axix = range(num_epochs)
    plt.title('Training loss and validation loss')
    plt.plot(x_axix, train_loss, color='red',label='Trainning loss')
    plt.plot(x_axix, val_loss, color='yellow',label='Validation loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.savefig('./result.png')


else:
   #TEST
    dataset_test = COCO("CMPT733-Lab3-Workspace/data/data/data/test/images/", "CMPT733-Lab3-Workspace/data/data/data/test/annotations/", class_num, boxs_default,validate=True, train = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network.pth'))
    network.eval()

    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_,width,height = data
        # print(width, height)
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)
        m = nn.Softmax(dim=2)
        pred_confidence = m(pred_confidence)
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred("test_before_nms", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        
        res = visualize_pred("test_after_nms", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        filename = '{}.txt'.format(str(i).zfill(5))
        file = open('CMPT733-Lab3-Workspace/data/data/data/test/annotations/'+filename,'w')
        for k in range(len(res)):
            res[k][1] = int((res[k][1] / 320) * width)
            res[k][2] = int((res[k][2] / 320) * height)
            res[k][3] = int(res[k][3] * width)
            res[k][4] = int(res[k][4] * height)
            for m in res[k]:
                file.writelines(str(m)+" ")
            file.writelines("\n")
        # print(res)
        # ann_confidence = ann_confidence_.cuda()
        # time.sleep(0.5)
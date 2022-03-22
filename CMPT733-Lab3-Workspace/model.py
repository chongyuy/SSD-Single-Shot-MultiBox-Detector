import os
import random
import numpy as np

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

def conv_basic(in_planes, out_planes, kernelsize, stride):
    padding = (kernelsize-1) // 2
    x = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
        )
    return x
def conv_reshape(x,in_planes, out_planes, kernelsize, stride):
    padding = (kernelsize-1) // 2
    x = nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding)(x)
    x = x.reshape((x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))
    return x


# def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
#     #input:
#     #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
#     #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
#     #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
#     #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
#     #
#     #output:
#     #loss -- a single number for the value of the loss function, [1]
    
#     #TODO: write a loss function for SSD
#     #
#     #For confidence (class labels), use cross entropy (F.cross_entropy)
#     #You can try F.binary_cross_entropy and see which loss is better
#     #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
#     #
#     #Note that you need to consider cells carrying objects and empty cells separately.
#     #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
#     #and reshape box to [batch_size*num_of_boxes, 4].
#     #Then you need to figure out how you can get the indices of all cells carrying objects,
#     #and use confidence[indices], box[indices] to select those cells.





class SSD(nn.Module):
    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        self.conv1 = conv_basic(3, 64, 3, 2)
        self.conv2 = conv_basic(64, 64, 3, 1)
        self.conv3 = conv_basic(64, 64, 3, 1)
        self.conv4 = conv_basic(64, 128, 3, 2)
        self.conv5 = conv_basic(128, 128, 3, 1)
        self.conv6 = conv_basic(128, 128, 3, 1)
        self.conv7 = conv_basic(128, 256, 3, 2)
        self.conv8 = conv_basic(256, 256, 3, 1)
        self.conv9 = conv_basic(256, 256, 3, 1)
        self.conv10 = conv_basic(256, 512, 3, 2)
        self.conv11 = conv_basic(512, 512, 3, 1)
        self.conv12 = conv_basic(512, 512, 3, 1)
        self.conv13 = conv_basic(512, 256, 3, 2)
        self.conv14 = conv_basic(256, 256, 1, 1)
        self.conv15 = conv_basic(256, 256, 3, 2)
        self.conv16 = conv_basic(256, 256, 1, 1)
        self.conv17 = conv_basic(256, 256, 3, 2)
        self.conv18 = conv_basic(256, 256, 1, 1)
        self.conv19 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3,stride=3,padding=1),
            nn.ReLU(True)
        )
        #TODO: define layers
        
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out)
        out = self.conv12(out)
        out = self.conv13(out)
        right_res1 = conv_reshape(out,256,16,3,1)
        left_res1 = conv_reshape(out,256,16,3,1)
        out = self.conv14(out)
        out = self.conv15(out)
        right_res2 = conv_reshape(out,256,16,3,1)
        left_res2 = conv_reshape(out,256,16,3,1)
        out = self.conv16(out)
        out = self.conv17(out)
        right_res3 = conv_reshape(out,256,16,3,1)
        left_res3 = conv_reshape(out,256,16,3,1)
        out = self.conv18(out)
        print(out.shape)
        out = self.conv19(out)
        right_res4 = conv_reshape(out,256,16,1,1)
        left_res4 = conv_reshape(out,256,16,1,1)

        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        # print(left_res1.shape)
        # print(left_res2.shape)
        # print(left_res3.shape)
        # print(left_res4.shape)
        confidence = torch.cat((left_res1,left_res2,left_res3,left_res4),2)
        bboxes = torch.cat((right_res1,right_res2,right_res3,right_res4),2)
        confidence =  torch.permute(confidence, (0, 2, 1))
        bboxes =  torch.permute(bboxes, (0, 2, 1))
        confidence = confidence.reshape((confidence.shape[0],540,4))
        m = nn.Softmax()
        confidence = m(confidence)
        bboxes = bboxes.reshape((bboxes.shape[0],540,4))
        return confidence,bboxes





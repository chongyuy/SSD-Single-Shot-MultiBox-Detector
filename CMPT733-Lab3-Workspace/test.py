import numpy as np

def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    boxes = []
    grid_num = 10 * 10 + 5 * 5 + 3 * 3 + 1 * 1
    box_num = 4 * (10 * 10 + 5 * 5 + 3 * 3 + 1 * 1)
    for k in range(len(layers)):
        start_x = (1/layers[k])/2
        start_y = (1/layers[k])/2
        grid_side = 1/layers[k]
        ssize = small_scale[k]
        lsize = large_scale[k]
        for i in range(layers[k]*layers[k]):
            box_for_4 = []
            row = i//layers[k]
            col = i % layers[k]
            x_center = start_x+row*grid_side
            y_center = start_y+col*grid_side
            box_for_4.append([x_center,y_center,ssize,ssize,x_center-ssize/2,y_center-ssize/2,x_center+ssize/2,y_center+ssize/2])
            box_for_4.append([x_center,y_center,lsize,lsize,x_center-lsize/2,y_center-lsize/2,x_center+lsize/2,y_center+lsize/2])
            box_for_4.append([x_center,y_center,lsize*np.sqrt(2),lsize/np.sqrt(2),x_center-(lsize*np.sqrt(2))/2,y_center-(lsize/np.sqrt(2))/2,x_center+(lsize*np.sqrt(2))/2,y_center+(lsize/np.sqrt(2))/2])
            box_for_4.append([x_center,y_center,lsize/np.sqrt(2),lsize*np.sqrt(2),x_center-(lsize/np.sqrt(2))/2,y_center-(lsize*np.sqrt(2))/2,x_center+(lsize/np.sqrt(2))/2,y_center+(lsize*np.sqrt(2))/2])
            boxes.append(box_for_4)
    boxes = np.array(boxes)     
    boxes = np.reshape(boxes,(box_num,8))         
  
    return boxes

larers = [10,5,3,1]
large_scale = [0.2,0.4,0.6,0.8]
small_scale =  [0.1,0.3,0.5,0.7]

# boxes = default_box_generator(larers, large_scale, small_scale)
# print(boxes[0])

def conv_basic(in_planes, out_planes, kernelsize, stride):
    padding = (kernelsize-1) // 2
    x = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
        )
    return x



import torch
import torch.nn as nn
from dataset import *
# x = np.zeros([1,3,320,320])
# x = torch.Tensor(x)
# x = conv_basic(3, 64, 3, 2)(x)
# x = conv_basic(64, 64, 3, 1)(x)
# x = conv_basic(64, 64, 3, 1)(x)
# x = conv_basic(64, 128, 3, 2)(x)
# x = conv_basic(128, 128, 3, 1)(x)
# x = conv_basic(128, 128, 3, 1)(x)
# x = conv_basic(128, 256, 3, 2)(x)
# x = conv_basic(256, 256, 3, 1)(x)
# x = conv_basic(256, 256, 3, 1)(x)
# x = conv_basic(256, 512, 3, 2)(x)
# x = conv_basic(512, 512, 3, 1)(x)
# x = conv_basic(512, 512, 3, 1)(x)
# x = conv_basic(512, 256, 3, 2)(x)
# def conv_reshape(x,in_planes, out_planes, kernelsize, stride):
#     padding = (kernelsize-1) // 2
#     x = nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding)(x)
#     x = x.reshape((x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))
#     return x
# y = conv_reshape(x,256,16,3,1)
# z = np.zeros([1,16,1])
# z = torch.Tensor(z)
# a = np.zeros([1,16,9])
# a = torch.Tensor(a)
# b = torch.cat((y,z,a),2)
# print(b.shape)


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
# conv1 = conv3x3(3, 64, 2)
# out = conv1(x)
# print(out.shape)
from model import *

# network = SSD(4)
# z,y = network.forward(x)
# print(z.shape)
# print(y.shape)
# x = torch.rand((1, 256, 3, 3))
# x = nn.Conv2d(256,256,kernel_size=3,stride=3,padding=1)(x)
# print(x.shape)
import torch.nn.functional as F


# x = torch.randn((2,3))

# print(x)
# print(x[0:1])
# print(x[1:2])
# input = torch.randn(1,8)
# target = torch.randn(1, 8).softmax(dim=1)
# entropy_loss = nn.CrossEntropyLoss()

# print(F.smooth_l1_loss(input,target))

# pred_confidence, pred_box, ann_confidence = torch.randn(32,540,4), torch.randn(32,540,4), torch.randn(32,540,4)
# ann_box = torch.zeros(32,540,4)
# x = torch.randint(0,100,8)
# ann_box[:,:,-1] = 1
# print()
# print(SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box))
class_num = 4
boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])
# print(boxs_default)
# # res = iou(boxs_default,0, 0, 0.1, 0.1)
image, ann_box, ann_confidence = COCO("CMPT733-Lab3-Workspace/data/data/data/train/images/", "CMPT733-Lab3-Workspace/data/data/data/train/annotations/", class_num, boxs_default, train = True, image_size=320).__getitem__(9)
x_obj = []
x_noob = []
# for i in range(ann_confidence.shape[0]):
#     if ann_confidence[i][3] == 1:
#         x_obj.append(0)
#         x_noob.append(1)
#     else:
#         x_obj.append(1)
#         x_noob.append(0)
x = ann_confidence[:,3]
y = list(filter(lambda z:z==0,x))
print(y)
# x,y=torch.randn(540,4), torch.randn(540,4)
# loss_en = F.cross_entropy(x,y)
# print(loss_en)
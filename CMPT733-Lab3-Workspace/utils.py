import numpy as np
import cv2
from dataset import iou
import torch

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                gt_xcenter = (ann_box[i][0] * boxs_default[i][2]) + boxs_default[i][0]
                gt_ycenter = (ann_box[i][1] * boxs_default[i][3]) + boxs_default[i][1]
                gt_box_width = boxs_default[i][2] * np.exp(ann_box[i][2])
                gt_box_height = boxs_default[i][3] * np.exp(ann_box[i][3])
                gt_xmin = gt_xcenter - gt_box_width/2
                gt_ymin = gt_ycenter - gt_box_height/2
                gt_xmax = gt_xcenter + gt_box_width/2
                gt_ymax = gt_ycenter + gt_box_height/2
                gt_xmin = gt_xmin * image.shape[0]
                gt_ymin = gt_ymin * image.shape[1]
                gt_xmax = gt_xmax * image.shape[0]
                gt_ymax = gt_ymax * image.shape[1]
                start_point = (int(gt_xmin), int(gt_ymin))
                end_point = (int(gt_xmax), int(gt_ymax)) 
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                df_xmin =int(boxs_default[i][4] *  image.shape[0])
                df_ymin =int(boxs_default[i][5] *  image.shape[1])
                df_xmax =int(boxs_default[i][6] *  image.shape[0])
                df_ymax =int(boxs_default[i][7] *  image.shape[1])
                start_point_df = (df_xmin, df_ymin)
                end_point_df = (df_xmax, df_ymax)
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                color1 = (255, 0, 0)
                thickness = 1
                image1 = cv2.rectangle(image1, start_point, end_point, colors[j], thickness)
                image2 = cv2.rectangle(image2, start_point_df, end_point_df, colors[j], thickness)

    # cv2.imwrite(filename, image1)
    #pred
    res = []
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.3:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                gt_xcenter = (pred_box[i][0] * boxs_default[i][2]) + boxs_default[i][0]
                gt_ycenter = (pred_box[i][1] * boxs_default[i][3]) + boxs_default[i][1]
                gt_box_width = boxs_default[i][2] * np.exp(pred_box[i][2])
                gt_box_height = boxs_default[i][3] * np.exp(pred_box[i][3])
                gt_xmin = gt_xcenter - gt_box_width/2
                gt_ymin = gt_ycenter - gt_box_height/2
                gt_xmax = gt_xcenter + gt_box_width/2
                gt_ymax = gt_ycenter + gt_box_height/2
                gt_xmin = gt_xmin * image.shape[0]
                gt_ymin = gt_ymin * image.shape[1]
                gt_xmax = gt_xmax * image.shape[0]
                gt_ymax = gt_ymax * image.shape[1]
                start_point = (int(gt_xmin), int(gt_ymin))
                end_point = (int(gt_xmax), int(gt_ymax)) 
                df_xmin =int(boxs_default[i][4] *  image.shape[0])
                df_ymin =int(boxs_default[i][5] *  image.shape[1])
                df_xmax =int(boxs_default[i][6] *  image.shape[0])
                df_ymax =int(boxs_default[i][7] *  image.shape[1])
                start_point_df = (df_xmin, df_ymin)
                end_point_df = (df_xmax, df_ymax)
                # print(start_point)
                if gt_xmin < 0 or gt_ymin < 0 or gt_xmax > 320 or gt_ymax > 320:
                    continue
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                thickness = 1
                image3 = cv2.rectangle(image3, start_point, end_point, colors[j], thickness)
                image4 = cv2.rectangle(image4, start_point_df, end_point_df, colors[j], thickness)
                res.append([j, gt_xmin, gt_ymin, gt_box_width, gt_box_height])
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    filename = windowname+".jpg"
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.
    # print(res)
    return res
def overlap_res(Axmin,Aymin,Axmax,Aymax, boxB, boxs_default):

    Bxcenter = (boxB[0] * boxs_default[2]) + boxs_default[0]
    Bycenter = (boxB[1] * boxs_default[3]) + boxs_default[1]
    Bbox_width = boxs_default[2] * np.exp(boxB[2])
    Bbox_height = boxs_default[3] * np.exp(boxB[3])
    Bxmin = Bxcenter - Bbox_width/2
    Bymin = Bycenter - Bbox_height/2
    Bxmax = Bxcenter + Bbox_width/2
    Bymax = Bycenter + Bbox_height/2
    inter = np.maximum(np.minimum(Bxmax,Axmax)-np.maximum(Bxmin,Axmin),0)*np.maximum(np.minimum(Bymax,Aymax)-np.maximum(Aymin,Bymin),0)
    area_a = (Bxmax-Bxmin)*(Bymax-Bymin)
    area_b = (Axmax-Axmin)*(Aymax-Aymin)
    union = area_a + area_b - inter
    res = inter/np.maximum(union,1e-8)
    # print(res)
    return res

def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.1, threshold=0.3):
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    x = confidence_[:,0:3]
    new_confidence = np.zeros((confidence_.shape[0],confidence_.shape[1]))
    x_noob = np.unravel_index(np.argmax(x), x.shape)
    # print(x[x_noob])
    while x[x_noob] > threshold:
        new_confidence[x_noob[0]] = confidence_[x_noob[0]]
        box = box_[x_noob[0]]
        Axcenter = (box[0] * boxs_default[x_noob[0]][2]) + boxs_default[x_noob[0]][0]
        Aycenter = (box[1] * boxs_default[x_noob[0]][3]) + boxs_default[x_noob[0]][1]
        Abox_width = boxs_default[x_noob[0]][2] * np.exp(box[2])
        Abox_height = boxs_default[x_noob[0]][3] * np.exp(box[3])
        Axmin = Axcenter - Abox_width/2
        Aymin = Aycenter - Abox_height/2
        Axmax = Axcenter + Abox_width/2
        Aymax = Aycenter + Abox_height/2
        for i in range(len(box_)):
            if overlap_res(Axmin,Aymin,Axmax,Aymax, box_[i], boxs_default[i]) > overlap:
                x[i] = np.zeros(3)
        x_noob = np.unravel_index(np.argmax(x), x.shape)
        # print(x[x_noob])
    return new_confidence, box_
    #TODO: non maximum suppression
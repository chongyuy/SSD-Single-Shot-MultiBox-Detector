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
    return boxes

larers = [10,5,3,1]
large_scale = [0.2,0.4,0.6,0.8]
small_scale =  [0.1,0.3,0.5,0.7]

boxes = default_box_generator(larers, large_scale, small_scale)
print(boxes[0][3])
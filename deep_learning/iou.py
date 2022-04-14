import numpy as np
#多个物体与多个物体的 IOU 计算：
def iou(gtboxes, dtboxes):
   '''numpy version of calculating IoU between two set of 2D bboxes.
    Args:
        gtboxes (np.ndarray): Shape (B,4) of ..,  4 present [x1,y1,x2,y2]
        dtboxes,np.ndarray,shape:(N,4), 4 present [x1,y1,x2,y2].
    Returns:
        np.ndarray: Shape (B,N).
    '''
    gtboxes = gtboxes[:, np.newaxis, :]
    ixmin = np.maximum(gtboxes[:, :, 0], dtboxes[:, 0])
    iymin = np.maximum(gtboxes[:, :, 1], dtboxes[:, 1])
    ixmax = np.minimum(gtboxes[:, :, 2], dtboxes[:, 2])
    iymax = np.minimum(gtboxes[:, :, 3], dtboxes[:, 3])
    intersection = (ixmax - ixmin + 1) * (iymax - iymin + 1)
    union = (gtboxes[:,:,2]-gtboxes[:,:,0]+1)*(gtboxes[:,:,3]-gtboxes[:,:,1]+1)\
            +(dtboxes[:,2]-dtboxes[:,0]+1)*(dtboxes[:,3]-dtboxes[:,1]+1)-intersection
    return intersection / union


#单个物体和单个物体的 IOU 计算：
def compute_IOU(rec1,rec2):
    """
    rec1:(y0,x0,y1,x1)  x1>x0,y1>y0
    rec2:(y0,x0,y1,x1)  x1>x1,y1>y1
    """
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    sum_area = S_rec1 + S_rec2
    
    left_line = max(rec1[1],rec2[1])
    right_line = min(rec1[3],rec2[3])
    up_line = max(rec1[0],rec2[0])
    bottom_line = min(rec1[2],rec2[2])
    
    if left_line>=right_line or up_line>=bottom_line:
        return 0
    else:
        intersect = (right_line-left_line)*(bottom_line-up_line)
        return intersect/(sum_area-intersect) * 1.0

rec1, rec2 = (0,0,2,2),(1,1,3,3)
compute_IOU(rec1, rec2)
import numpy as np
import cv2
#多个物体与多个物体的 IOU 计算：
def iou(gtboxes, dtboxes):
    '''
    numpy version of calculating IoU between two set of 2D bboxes.
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


# 中心点 矩形的w h, 旋转的theta（角度，不是弧度）
def iou_rotate_calculate(boxes1, boxes2):
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    ious = []
    for i, box1 in enumerate(boxes1):
        temp_ious = []
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        for j, box2 in enumerate(boxes2):
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
        ious.append(temp_ious)
    return np.array(ious, dtype=np.float32)

if __name__ == '__main__':
    gtboxes = np.random.random((10, 4))
    dtboxes = np.random.random((10, 4))
    res = iou(gtboxes, dtboxes)
    
    print(res)
    max_idx = np.argmax(res, axis=1)
    print(dtboxes[max_idx])
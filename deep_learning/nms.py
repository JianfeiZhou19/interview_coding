import numpy as np
def non_max_suppress(predict_dict, threshold=0.2):
    predicts_dict={}
    for object_name, bbox in predict_dict.items():
        bbox_array = np.array(bbox, dtype = np.float)
        x0,y0,x1,y1,scores= bbox_array[:,0],bbox_array[:,1],bbox_array[:,2],\
                      bbox_array[:,3],bbox_array[:,4]
        areas = (x1-x0+1)*(y1-y0+1)
        orders = scores.argsort()[::-1]
        keep = []
        while orders.size > 0:
            i = orders[0]
            keep.append(i)
            
            xx1 = np.maximum(x0[i],x0[orders[1:]])
            yy1 = np.maximum(y0[i],y0[orders[1:]])
            xx2 = np.minimum(x1[i],x1[orders[1:]])
            yy2 = np.minimum(y1[i],y1[orders[1:]])
            
            inter = np.maximum(0,xx2-xx1+1)*np.maximum(0,yy2-yy1+1)
            iou = inter / (areas[i]+areas[orders[1:]] - inter) 
            
            indexs = np.where(iou<=threshold)[0]+1
            orders = orders[indexs]
        bbox = bbox_array[keep]
        predicts_dict[object_name]=bbox.tolist()
    return predicts_dict

predict_dict =  {'cup': [[59, 120, 137, 368, 0.124648176], 
                         [221, 89, 369, 367, 0.35818103], 
                         [54, 154, 148, 382, 0.13638769]]}
result = non_max_suppress(predict_dict)
print(result)
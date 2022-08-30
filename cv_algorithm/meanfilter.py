import numpy as np
import cv2

def integral(img):
    integral_res = np.zeros((img.shape[0]+1, img.shape[1]+1), dtype=np.int32)
    for x in range(img.shape[0]):
        sum_col = 0
        for y in range(img.shape[1]):
            sum_col = sum_col+img[x][y]
            integral_res[x+1][y+1] = integral_res[x][y+1]+sum_col
    return integral_res

def meanfilter_fast(img, kh=3, kw=3, ):
    dst = np.zeros_like(img)
    hh, ww = kh//2, kw//2
    img = np.pad(img, (hh, ww))
    integral_res = integral(img)
    h, w = dst.shape
    mean = 0
    for i in range(hh+1, h+hh+1):
        for j in range(ww+1, w+ww+1):
            top_left = integral_res[i-hh-1, j-ww-1]
            top_right = integral_res[i-hh-1, j+ww]
            bottom_left = integral_res[i+hh, j-ww-1]
            bottom_right = integral_res[i+hh, j+ww]
            mean = (bottom_right-top_right-bottom_left+top_left)/(kh*kw)
            if mean<0:
                mean = 0
            if mean>255:
                mean = 255

            dst[i-hh-1, j-ww-1] = int(mean+0.5)
    return dst

def meanfilter(img, padding):
    padding_img = np.pad(img, pad_width=padding)
    dst = np.zeros_like(img)
    h, w = img.shape
    for i in range(1, h+1):
        for j in range(1, w+1):
            dst[i-1, j-1] = np.mean(padding_img[i-1:i+2, j-1:j+2])
    return dst

if __name__ == "__main__":
    img = cv2.imread("luna.png", 0)
    stand = cv2.blur(img, (3, 3))
    ours = meanfilter(img)
    img = np.random.random((5, 5))
    print(meanfilter(img, 1))
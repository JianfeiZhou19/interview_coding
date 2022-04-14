import numpy as np

def meanfilter(img, padding):
    padding_img = np.pad(img, pad_width=padding)
    dst = np.zeros_like(img)
    h, w = img.shape
    for i in range(1, h+1):
        for j in range(1, w+1):
            dst[i-1, j-1] = np.mean(padding_img[i-1:i+2, j-1:j+2])
    return dst

img = np.random.random((5, 5))
print(meanfilter(img, 1))
import numpy as np
#卷积
def conv(data, filter, p, s):
    h,w = data.shape
    m,n = filter.shape
    input=np.zeros((h+2*p,w+2*p))
    input[p:p+h,p:p+w] = data
    h,w = h+2*p,w+2*p
    output = []
    for i in range(0,h,s):
        line = []
        for j in range(0,w,s):
            x = input[i:i+m,j:j+n]
            if x.shape == filter.shape: #判断不能少
                line.append(np.sum(x*filter))
            if line!=[]: #这句不能忘！！！！！
                output.append(line)
    return output

#最大池化
def max_pooling(data, m, n):
    a,b = data.shape
    img_new = []
    for i in range(0,a,m):
        line = []
        for j in range(0,b,n):
            x = data[i:i+m,j:j+n]
            line.append(np.max(x))
        img_new.append(line)
    return np.array(img_new)

#均值池化
def average_pooling(data, m, n):
    a,b = data.shape
    img_new = []
    for i in range(0,a,m):
        line = []#记录每一行
        for j in range(0,b,n):
            x = data[i:i+m,j:j+n]#选取池化区域
            line.append(np.sum(x)/(n*m))
        img_new.append(line)
    return np.array(img_new)

def img2col_conv(img, kernel, padding, stride):
    img_padding = np.pad(img, pad_width=padding)
    kernel_flatten = kernel.flatten()
    h, w = img.shape
    k_h, k_w = kernel.shape

    dst = []
    for i in range(int(k_w/2), w+1, stride):
        for j in range(int(k_h/2), h+1, stride):
            region = img_padding[i-int(k_w/2):i+int(k_w/2)+1, j-int(k_h/2):j+int(k_h/2)+1].flatten()
            tmp = np.sum(region * kernel_flatten)
            dst.append(tmp)
    dst = np.array(dst)

    return np.reshape(dst, (int(w/stride), int(h/stride)))

def img2col_conv(img, kernel, padding, stride):
    img_padding = np.pad(img, pad_width=padding)
    kernel_flatten = kernel.flatten()
    h, w = img.shape
    k_h, k_w = kernel.shape

    dst = []
    for i in range(int(k_w/2), w+1, stride):
        for j in range(int(k_h/2), h+1, stride):
            region = img_padding[i-int(k_w/2):i+int(k_w/2)+1, j-int(k_h/2):j+int(k_h/2)+1].flatten()
            tmp = np.sum(region * kernel_flatten)
            dst.append(tmp)
    dst = np.array(dst)

    return np.reshape(dst, (int(w/stride), int(h/stride)))


if __name__ == "__main__":
    image = np.ones((3,5,5))
    filter = np.ones((3,3,3))
    channel = image.shape[0]
    result = []
    for i in range(channel):
        result.append(conv(image[i],filter[i],0,1))
        # result.append(average_pooling(image[i], 2, 2))
    print(np.sum(np.array(result),axis=0))
    img = np.random.random((10,10))
    kernel = np.ones((3, 3))
    print(img2col_conv(img, kernel, 1, 1))
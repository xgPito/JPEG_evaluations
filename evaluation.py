import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# 差分攻击和秘钥敏感性评价指标：NPCR和UACI
# 计算像素数变化率NPCR----99.6094%
def NPCR(img1, img2, isColor=1):
    # opencv颜色通道顺序为BGR
    img1 = cv2.imread(img1, isColor)
    img2 = cv2.imread(img2, isColor)
    w, h = img1.shape[:2]

    # 如果是灰度图，直接得出结果
    if isColor == 0:
        ar, num = np.unique((img1 != img2), return_counts=True)
        R_npcr = (num[0] if ar[0] == True else num[1])/(w*h)
        return R_npcr

    # 彩色图图像通道拆分
    B1, G1, R1 = cv2.split(img1)
    B2, G2, R2 = cv2.split(img2)

    # 返回数组的排序后的唯一元素和每个元素重复的次数
    ar, num = np.unique((R1 != R2), return_counts=True)
    R_npcr = (num[0] if ar[0] == True else num[1])/(w*h)
    ar, num = np.unique((G1 != G2), return_counts=True)
    G_npcr = (num[0] if ar[0] == True else num[1])/(w*h)
    ar, num = np.unique((B1 != B2), return_counts=True)
    B_npcr = (num[0] if ar[0] == True else num[1])/(w*h)

    return R_npcr, G_npcr, B_npcr


# 两张图像之间的平均变化强度UACI----33.4635%
def UACI(img1, img2, isColor=1):
    img1 = cv2.imread(img1, isColor)
    img2 = cv2.imread(img2, isColor)
    w, h = img1.shape[:2]

    # 如果是灰度图，直接得出结果
    if not isColor:
        img1 = img1.astype(np.int16)
        img2 = img2.astype(np.int16)
        print(np.sum(abs(img1-img2)))
        return np.sum(abs(img1-img2))/255/(w*h)

    # 彩色图图像通道拆分
    B1, G1, R1 = cv2.split(img1)
    B2, G2, R2 = cv2.split(img2)
    # 元素为uint8类型取值范围：0到255
    # print(R1.dtype)

    # 强制转换元素类型，为了运算
    R1 = R1.astype(np.int16)
    R2 = R2.astype(np.int16)
    G1 = G1.astype(np.int16)
    G2 = G2.astype(np.int16)
    B1 = B1.astype(np.int16)
    B2 = B2.astype(np.int16)

    sumR = np.sum(abs(R1-R2))
    sumG = np.sum(abs(G1-G2))
    sumB = np.sum(abs(B1-B2))
    R_uaci = sumR/255/(w*h)
    G_uaci = sumG/255/(w*h)
    B_uaci = sumB/255/(w*h)

    return R_uaci, G_uaci, B_uaci


# 计算两幅图像的峰值信噪比PSNR
def PSNR(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    w, h, _ = img1.shape
    B1, G1, R1 = cv2.split(img1)
    B2, G2, R2 = cv2.split(img2)

  # 强制转换元素类型，为了运算
    R1 = R1.astype(np.int32)
    R2 = R2.astype(np.int32)
    G1 = G1.astype(np.int32)
    G2 = G2.astype(np.int32)
    B1 = B1.astype(np.int32)
    B2 = B2.astype(np.int32)

  # 计算均方误差,初始化64位无符号整型，防止累加中溢出
    R_mse = np.uint64(0)
    G_mse = np.uint64(0)
    B_mse = np.uint64(0)
    for i in range(w):
        for j in range(h):
            R_mse += (R1[i][j]-R2[i][j])**2
            G_mse += (G1[i][j]-G2[i][j])**2
            B_mse += (B1[i][j]-B2[i][j])**2
    R_mse /= (w*h)
    G_mse /= (w*h)
    B_mse /= (w*h)
    R_psnr = 10*math.log((255**2)/R_mse, 10)
    G_psnr = 10*math.log((255**2)/G_mse, 10)
    B_psnr = 10*math.log((255**2)/B_mse, 10)

    return R_psnr, G_psnr, B_psnr


# 2.1结构相似性预处理函数
def ssim(img1, img2):
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  
  C1 = (0.01 * 255)**2
  C2 = (0.03 * 255)**2
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())
  mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
  mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
  sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
  
  return ssim_map.mean()

# 2.2开始计算两幅图像的SSIM
def calculate_ssim(img1, img2):
  img1=cv2.imread(img1)
  img2=cv2.imread(img2)
  if not img1.shape == img2.shape:
    raise ValueError('Input images must have the same dimensions.')
  if img1.ndim == 2:
    return ssim(img1, img2)
  elif img1.ndim == 3:
    if img1.shape[2] == 3:
      ssims = []
      for i in range(3):
        ssims.append(ssim(img1, img2))
      return np.array(ssims).mean()
    elif img1.shape[2] == 1:
      return ssim(np.squeeze(img1), np.squeeze(img2))
  else:
    raise ValueError('Wrong input image dimensions.')


# 绘制像素直方图
def HIST(img):
    img = cv2.imread(img)
    B, G, R = cv2.split(img)
    # 转成一维
    R = R.flatten(order='C')
    G = G.flatten(order='C')
    B = B.flatten(order='C')

    # 全局设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体，中文乱码
    plt.rcParams['figure.figsize'] = (12, 6)

    plt.subplot(232)
    # plt.imshow(img[:,:,(2,1,0)])
    plt.hist(img.flatten(order='C'), bins=range(257), color='gray')
    plt.title('原图像')
    # 子图2，通道R
    plt.subplot(234)
    # imshow()对图像进行处理，画出图像，show()进行图像显示
    plt.hist(R, bins=range(257), color='red')
    plt.title('通道R')
    # plt.show()
    # plt.axis('off')    # 不显示坐标轴

    # 子图3，通道G
    plt.subplot(235)
    plt.hist(G, bins=range(257), color='green')
    plt.title('通道G')
    # plt.show()
    # plt.axis('off')

    # 子图4，通道B
    plt.subplot(236)
    plt.hist(B, bins=range(257), color='blue')
    plt.title('通道B')
    # plt.axis('off')
    # #设置子图默认的间距
    plt.tight_layout()
    plt.show()


# 1.1分别计算图像通道相邻像素的水平、垂直和对角线的相关系数并返回
def RGB_correlation(channel, N):
    # 计算channel通道
    h, w = channel.shape
    # 随机产生pixels个[0,w-1)范围内的整数序列
    row = np.random.randint(0, h-1, N)
    col = np.random.randint(0, w-1, N)
    # 绘制相邻像素相关性图,统计x,y坐标
    x = []  # 存放原始像素
    x_h = []  # 存放水平像素
    x_v = []
    x_d = []
    for i in range(N):
        # 选择当前一个像素
        x.append(channel[row[i]][col[i]])
        # 水平相邻像素是它的右侧也就是同行下一列的像素
        x_h.append(channel[row[i]][col[i]+1])
        # 垂直相邻像素是它的下方也就是同列下一行的像素
        x_v.append(channel[row[i]+1][col[i]])
        # 对角线相邻像素是它的右下即下一行下一列的那个像素
        x_d.append(channel[row[i]+1][col[i]+1])
    # 三个方向的合到一起
    x = x*3
    y = x_h+x_v+x_d

    # 计算E(x)，计算三个方向相关性时，x没有重新选择也可以更改
    ex = 0
    ex_h = 0
    ex_v = 0
    ex_d = 0
    for i in range(N):
        ex += x[i]
        ex_h += x_h[i]
        ex_v += x_v[i]
        ex_d += x_d[i]
    ex /= N
    ex_h /= N
    ex_v /= N
    ex_d /= N

    # 计算D(x)
    dx = 0
    dx_h = 0
    dx_v = 0
    dx_d = 0
    for i in range(N):
        dx += (x[i]-ex)**2
        dx_h += (x_h[i]-ex_h)**2
        dx_v += (x_v[i]-ex_v)**2
        dx_d += (x_d[i]-ex_d)**2
    dx /= N
    dx_h /= N
    dx_v /= N
    dx_d /= N

    # 计算协方差cov(x,y)
    covx_h = 0
    covx_v = 0
    covx_d = 0
    for i in range(N):
        covx_h += (x[i]-ex)*(x_h[i]-ex_h)
        covx_v += (x[i]-ex)*(x_v[i]-ex_v)
        covx_d += (x[i]-ex)*(x_d[i]-ex_d)
    covx_h /= N
    covx_v /= N
    covx_d /= N

    # 计算相关性R
    Rx_h = covx_h/(np.sqrt(dx)*np.sqrt(dx_h))
    Rx_v = covx_v/(np.sqrt(dx)*np.sqrt(dx_v))
    Rx_d = covx_d/(np.sqrt(dx)*np.sqrt(dx_d))

    return Rx_h, Rx_v, Rx_d, x, y

# 1.2分别计算图像img的各通道相邻像素的相关系数，默认随机选取3000对相邻像素
def correlation(img, N=3000):
    img = cv2.imread(img)
    h, w, _ = img.shape
    B, G, R = cv2.split(img)
    R_Rxy = RGB_correlation(R, N)
    G_Rxy = RGB_correlation(G, N)
    B_Rxy = RGB_correlation(B, N)

    # 结果展示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    # plt.rcParams['figure.figsize'] = (12, 6)

    plt.subplot(221)
    plt.imshow(img[:, :, (2, 1, 0)])
    plt.title('原图像')
    # 子图2
    plt.subplot(222)
    plt.scatter(R_Rxy[3], R_Rxy[4], s=1, c='red')
    plt.title('通道R')

    # 子图3
    plt.subplot(223)
    plt.scatter(G_Rxy[3], G_Rxy[4], s=1, c='green')
    plt.title('通道G')
    # 子图4
    plt.subplot(224)
    plt.scatter(B_Rxy[3], B_Rxy[4], s=1, c='blue')
    plt.title('通道B')

    plt.tight_layout()
    plt.show()

    return R_Rxy[0:3], G_Rxy[0:3], B_Rxy[0:3]


# 计算信息熵----8
def entropy(img):
    img = cv2.imread(img)
    w, h, _ = img.shape
    B, G, R = cv2.split(img)
    gray, num1 = np.unique(R, return_counts=True)
    gray, num2 = np.unique(G, return_counts=True)
    gray, num3 = np.unique(B, return_counts=True)
    R_entropy = 0
    G_entropy = 0
    B_entropy = 0

    for i in range(len(gray)):
        p1 = num1[i]/(w*h)
        p2 = num2[i]/(w*h)
        p3 = num3[i]/(w*h)
        R_entropy -= p1*(math.log(p1, 2))
        G_entropy -= p2*(math.log(p2, 2))
        B_entropy -= p3*(math.log(p3, 2))
    return R_entropy, G_entropy, B_entropy


if __name__ == '__main__':
    # print(NPCR("lena_gray.bmp","lena_85.jpeg",0))
    # print(UACI("lena_gray.bmp","lena_85.jpeg"))
    # print(PSNR("JPEG_color.jpeg", "lena_color.bmp"))
    print(calculate_ssim("lena_gray.bmp","lena_85.jpeg"))
    # HIST("lena_color.bmp")
    # print(correlation("lena_color.bmp"))
    # print(entropy("lena_color.bmp"))

    # img=cv2.imread("lena_color.bmp")
    # cv2.imwrite("JPEG_color.jpeg",img,[int(cv2.IMWRITE_JPEG_QUALITY),85])

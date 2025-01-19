import os
import cv2
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math

# 飽和度
def SaturationMap(im):
    img_hsv = Image.fromarray((im * 255).astype(np.uint8)).convert("HSV")
    h, s, v = img_hsv.split()
    return np.array(s) / 255.0

# 暗通道
def DarkChannel(im, win=15):
    if isinstance(im, PIL.Image.Image):
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    dark = cv2.erode(dc, kernel)
    return np.array(dark, dtype=float) / 255.0  # 轉換為 float 並正規化

#傅立葉轉換，然後高/低頻濾波，然後逆傅立葉轉換
def apply_gaussian_filter(image, sigma, high_pass=True):
    # 轉換為灰階圖像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 轉換到頻率域
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 創建高斯濾波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    gaussian_filter = np.exp(-(distance**2) / (2*(sigma**2)))

    if high_pass:
        # 高通濾波器
        gaussian_filter = 1 - gaussian_filter
    else:
        # 低通濾波器
        gaussian_filter = gaussian_filter

    # 應用濾波器
    fshift = dft_shift * gaussian_filter[:, :, np.newaxis]

    # 傅立葉逆轉換
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back

# def calculate_entropy(image_path):
#     image = Image.open(image_path).convert('L')
#     image_np = np.array(image)

#     # 计算灰度直方图
#     histogram = Counter(image_np.flatten())
#     total_pixels = image_np.size

#     # 计算归一化直方图（灰度级别的概率）
#     probabilities = [count / total_pixels for count in histogram.values()]

#     # 计算熵
#     entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

#     return entropy
    

def main(image_path):
    # 載入圖片
    original_img = Image.open(image_path)
    im_array = np.array(original_img) / 255.0

    # 計算飽和度圖像
    saturation_img = SaturationMap(im_array)

    # 計算暗通道圖像
    dark_channel_img = DarkChannel(original_img)

    # sigma = 10
    # # 高斯HIGH
    # high_img = np.array(original_img.convert('L'))  # 轉換為灰階圖像
    # HIGH = apply_gaussian_filter(high_img, sigma, high_pass=True)
    # # 高斯LOW
    # LOW = apply_gaussian_filter(high_img, sigma, high_pass=False)

    # #圖像熵
    # entropy = calculate_entropy(image_path)

    # 顯示結果
    plt.figure(figsize=(12, 4))

    # 顯示原始圖像
    plt.subplot(1, 5, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')

    # 顯示飽和度圖像
    plt.subplot(1, 5, 2)
    plt.imshow(saturation_img, cmap='gray')
    plt.title('Saturation Map')
    plt.axis('off')

    # 顯示暗通道圖像
    plt.subplot(1, 5, 3)
    plt.imshow(dark_channel_img, cmap='gray')
    plt.title('Dark Channel')
    plt.axis('off')

    # # 顯示HIGH
    # plt.subplot(1, 5, 4)
    # plt.imshow(HIGH, cmap='gray')
    # plt.title('HIGH')
    # plt.axis('off')
    # # 顯示LOW
    # plt.subplot(1, 5, 5)
    # plt.imshow(LOW, cmap='gray')
    # plt.title('LOW')
    # plt.axis('off')

    # plt.subplot(1, 6, 6)
    # plt.text(0.5, 0.5, f'Entropy: {entropy:.2f}', fontsize=12, ha='center')
    # plt.title('Entropy')
    # plt.axis('off')

    plt.show()

if __name__ == "__main__":
    image_path = '077-202303011720.jpg'  # 請更改為你的圖像路徑
    main(image_path)

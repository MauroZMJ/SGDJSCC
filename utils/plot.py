import matplotlib.pyplot as plt
import numpy as np
import os
def imshow(img):
   plt.imshow(np.transpose(img, (1, 2, 0)))
def insert_newlines(text, interval):
    # 将字符串按照指定间隔分割成子串
    substrings = [text[i:i+interval] for i in range(0, len(text), interval)]
    # 插入换行符号并拼接成新的字符串
    new_text = '\n'.join(substrings)
    return new_text
def showOrigNoiseOut(origin_img, model_img,save_path,caption_list):
    n = len(origin_img)
    plt.figure(figsize=(30, 10))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        imshow(origin_img[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy image
        ax = plt.subplot(2, n, i + 1 + n)
        imshow(model_img[i])
        ax.text(-50, 400, insert_newlines(caption_list[i],25), fontsize=12, color='r',wrap=True)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.figtext(0.5, 0.95, "ORIGINAL IMAGES", ha="center", va="top", fontsize=14, color="r")
    plt.figtext(0.5, 0.45, "NOISY IMAGES", ha="center", va="top", fontsize=14, color="r")
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(save_path)
def save_image(image_set,save_path,number_offset=0):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    image_num = len(image_set)
    origin_num = image_num //2
    method_dict = {0:'origin',1:'predict'}
    from torchvision.utils import save_image as save_image_fn
    for image_idx in range(image_num):
        save_image_fn(image_set[image_idx,:],save_path + '/%s_%d.png'%(method_dict[image_idx//origin_num],image_idx%origin_num+number_offset))
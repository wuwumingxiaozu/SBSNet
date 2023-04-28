import cv2
from PIL import Image


def keep_image_size_open(path, size=(1024, 1024)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('P', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    # print(f'keep_image_size_open:{len(mask.split())}')
    return mask


def keep_image_size_open_rgb(path, size=(1024, 1024)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask
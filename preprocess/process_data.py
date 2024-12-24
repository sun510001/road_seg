import os
import cv2
import shutil
import numpy as np

from glob import glob
from tqdm import tqdm


def generate_contour(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if np.max(image).item() == 1:
        return None
    lower_gray = 100
    upper_gray = 254
    mask = cv2.inRange(image, lower_gray, upper_gray)
    mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour_image = np.zeros_like(image)
    polygons = []
    for contour in contours:
        polygons.append(contour)
        # cv2.drawContours(contour_image, [contour], -1, 255, 2)

    return polygons, image.shape


def process_data(source_folder, type, file_name_list):
    ori_images = glob(os.path.join(source_folder, type, "*.jpg"))
    mask_images = glob(os.path.join(source_folder, type, "*.png"))
    pair_dict = {}
    for mask in tqdm(mask_images):
        if file_name_list[0] in os.path.split(mask)[1][:-4]:
            number = os.path.split(mask)[1][:-4].replace(file_name_list[0], '')
            polygons, shape = generate_contour(mask)
            pair_dict[number] = [mask, polygons, shape]

    for image in tqdm(ori_images):
        number = os.path.split(image)[1][:-4].replace(file_name_list[1], '')
        pair_dict[number].append(image)
    return pair_dict


def generate_yolo_dataset(target_folder, type, pair_dict):
    images_folder = os.path.join(target_folder, type, 'images')
    labels_folder = os.path.join(target_folder, type, 'labels')
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)
    for _, values in pair_dict.items():
        _, polygons, (h, w), image_path = values
        output_file = os.path.join(
            labels_folder, f"{os.path.split(image_path)[1][:-4]}.txt")
        with open(output_file, 'w') as f:
            for contour in polygons:
                contour_normalized = contour.flatten().tolist()
                contour_normalized = [(point[0] / w, point[1] / h)
                                      for point in zip(contour_normalized[::2], contour_normalized[1::2])]
                contour_flattened = [f"0"] + \
                    [f"{x} {y}" for x, y in contour_normalized]
                f.write(" ".join(contour_flattened) + "\n")
        shutil.copy(image_path, images_folder)


def main():
    source_folder = "data/UAS_Dataset/UAS_UESTC_All-day_Scenery/sun_sight"
    type = "train"
    target_folder = "data/yolo/sun_sight"
    # [mask name, origin image name]
    file_name_list = ['SunLabelGraph', 'SunSight']

    pair_dict = process_data(source_folder, type, file_name_list)
    generate_yolo_dataset(target_folder, type, pair_dict)


if __name__ == '__main__':
    main()

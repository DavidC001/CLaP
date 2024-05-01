import cv2
import os
import numpy as np
from tqdm import tqdm
from pychromakey import ChromaKey
import matplotlib.pyplot as plt

def is_completely_green(image_path):

    if not os.path.isfile(image_path):
        print(f"Image file not found: {image_path}")
        return False

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    red, green, blue = cv2.split(image)
    mean_blue = np.array(blue).mean() 
    mean_red= np.array(red).mean()
    mean_green = np.array(green).mean()
    
    # cv2.imshow(image_path, image)
    # cv2.waitKey(0)

    if mean_green > 10*mean_blue and mean_green > 10*mean_red:
        if mean_blue > 1300 or mean_red > 1300:
        # print(mean_blue, mean_red, mean_green)
            cv2.imshow(image_path, image)
            cv2.waitKey(0)
        return True
    else:
        return False

def is_green_pychromakey(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(image, lower_green, upper_green)
    plt.imshow(mask, cmap="gray")
    plt.show()
    if mask.all(mask == 255) or np.any(mask == 0):
        return True
    else:
        return False

def main():

    base_folder_path = "datasets/Panoptic/"
    green_images = []

    for folder_name in os.listdir(base_folder_path):
        print(f"Processing folder: {folder_name}")
        folder_path = os.path.join(base_folder_path, folder_name)
        for root, _, files in os.walk(folder_path):
            for file in tqdm(files):
                if file.endswith(".jpg"):
                    image_path = os.path.join(root, file)
                    if is_completely_green(image_path):
                        green_images.append(file)

    #Check only in the folder "170228_haggling_a2"
    # folder_path = os.path.join(base_folder_path, "170228_haggling_a2")
    # for root, _, files in os.walk(folder_path):
    #     for file in tqdm(files):
    #         if file.endswith(".jpg"):
    #             image_path = os.path.join(root, file)
    #             if is_completely_green(image_path):
    #                 green_images.append(file)
                        
    print(f"Total green images: {len(green_images)}")

    with open("green_images.txt", "w") as f:
        for image_name in green_images:
            f.write(image_name + "\n")

if __name__ == "__main__":
    main()



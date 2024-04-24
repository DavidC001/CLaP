import cv2
import os
import numpy as np

def is_completely_green(image_path):

    if not os.path.isfile(image_path):
        print(f"Image file not found: {image_path}")
        return False

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    red, green, blue = cv2.split(image)
    mean_blue = np.array(blue).sum() 
    mean_red= np.array(red).sum()
    


    # cv2.imshow(image_path, image)
    # cv2.waitKey(0)

    if mean_blue == 0 and mean_red == 0:
        print(mean_blue, mean_red)
        # cv2.imshow(image_path, image)
        # cv2.waitKey(0)

        return True
    else:
        return False




# def is_black_and_green(image_path):

#     if not os.path.isfile(image_path):
#         print(f"Image file not found: {image_path}")
#         return False
#     image = cv2.imread(image_path)

#     b, g, r = cv2.split(image)

#     if (b == 0).all() and (r == 0).all() and (g != 0).any():
#         return True
#     else:
#         return False
from tqdm import tqdm

def main():
    folder = [
        '160226_haggling1', '160422_haggling1', '170221_haggling_b1', '170221_haggling_b2', '170221_haggling_b3', '170221_haggling_m1', '170221_haggling_m2', '170221_haggling_m3', '170224_haggling_a1', '170224_haggling_a2', '170224_haggling_a3', '170224_haggling_b1', '170224_haggling_b2', '170224_haggling_b3', '170228_haggling_a1', '170228_haggling_a2', '170228_haggling_a3', '170228_haggling_b1', '170228_haggling_b2', '170228_haggling_b3', '170404_haggling_a1', '170404_haggling_a2', '170404_haggling_a3', '170404_haggling_b1', '170404_haggling_b2', '170404_haggling_b3', '170407_haggling_a1', '170407_haggling_a2', '170407_haggling_a3', '170407_haggling_b1', '170407_haggling_b2', '170407_haggling_b3',
        '160422_ultimatum1', '160906_band1', '160906_band2', '160906_band3', '160906_ian1', '160906_ian2', '160906_ian3', '160906_ian5', 
        '161029_flute1', '161029_piano1', '161029_piano2', '161029_piano3', '161029_piano4', '170307_dance5', '171026_cell03', '171026_pose1', '171026_pose2', '171026_pose3', '171024_pose1', '171024_pose2', '171024_pose3', '171024_pose4', '171024_pose5', '171024_pose6'
            ]
    base_folder_path = "datasets/Panoptic/"
    green_images = []


    for folder_name in folder:
        print(f"Processing folder: {folder_name}")
        folder_path = os.path.join(base_folder_path, folder_name)
        for root, _, files in os.walk(folder_path):
            for file in tqdm(files):
                if file.endswith(".jpg"):
                    image_path = os.path.join(root, file)
                    if is_completely_green(image_path):
                        green_images.append(image_path)

    with open("green_images.txt", "w") as f:
        for image_name in green_images:
            f.write(image_name + "\n")

if __name__ == "__main__":
    main()



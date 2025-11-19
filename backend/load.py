
import os
import cv2
from PIL import Image
import numpy as np

def process_images_in_folders(root_directory):
    data = []
    labels = []

    subdirectories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

    for label, subdirectory in enumerate(subdirectories):
        subdirectory_path = os.path.join(root_directory, subdirectory)

        image_files = os.listdir(subdirectory_path)

        for image_file in image_files:
            image_path = os.path.join(subdirectory_path, image_file)

            imag = cv2.imread(image_path)

            img_from_ar = Image.fromarray(imag, 'RGB')
            resized_image = img_from_ar.resize((224, 224))

            data.append(np.array(resized_image))

            labels.append(label)

    return data, labels

# Specify the root directory containing subfolders for each category
root_directory = "///path/to/your/image/folders"

data, labels = process_images_in_folders(root_directory)

# Convert data and labels to NumPy arrays
herbs = np.array(data)
labels = np.array(labels)

# Save the processed data and labels
np.save("crops", herbs)
np.save("labels", labels)
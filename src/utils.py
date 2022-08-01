from cgi import test
from email.mime import image
import os, json, pdb, cv2
from venv import create
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import groupby
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
import torch
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sn

def tsne_viz_feature_vecs(feature_vectors, labels, output_dir):
    # pdb.set_trace()
    standardized_data = StandardScaler().fit_transform(feature_vectors)
    # Picking the top 1000 points as TSNE takes a lot of time for 15K points

    model = TSNE(n_components=2, random_state=0)
    # configuring the parameteres
    # the number of components = 2
    # default perplexity = 30
    # default learning rate = 200
    # default Maximum number of iterations for the optimization = 1000

    tsne_data = model.fit_transform(standardized_data)
    # creating a new data frame which help us in ploting the result data
    tsne_data = np.vstack((tsne_data.T, labels)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

    # Ploting the result of tsne
    sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()  
    plt.savefig(output_dir)


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    last_elem = 0
    running_length = 0
    for i, elem in enumerate(binary_mask.ravel(order='F')):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1
    counts.append(running_length)
    return rle
def get_bbox_from_mask_img(binary_mask):
    x = int(min(np.where(binary_mask==255)[1]))
    y = int(min(np.where(binary_mask==255)[0]))
    width = int(max(np.where(binary_mask==255)[1])-x)
    height = int(max(np.where(binary_mask==255)[0])-y)

    return [x,y,width,height]
def create_test_coco_json_annotation(test_annotation_folder):
    print("CAUTION: THIS SCRIPT ONLY WORKS IF THERE IS ONLY ONE OBJECT PER IMAGE")
    full_dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 0,
                "name": "Cheezit_box"
            }
        ]
    }
    # Iterate directory
    image_id = 0
    for file in tqdm(os.listdir(test_annotation_folder)):
        # check only text files
        if file.endswith('.txt'):
            if file == 'model_corners.txt':
                continue
            full_dict['images'].append({
                "file_name": file.split("_corners")[0]+".jpg",
                "height": 1920,
                "width": 1080,
                "id": image_id
            })
            
            full_dict['annotations'].append({
                "id" : image_id,
                "image_id" : image_id,
                "category_id" : 0,
                "segmentation": [[]],
                "area" : 0,
                "bbox": [],
                "iscrowd" : 0,
            }) 
            img = cv2.imread(os.path.join(test_annotation_folder,file.split("_corners")[0]+"_mask.png"), 0)
            rle_mask = binary_mask_to_rle(img)
            bbox = get_bbox_from_mask_img(img)
            full_dict['annotations'][image_id]["segmentation"] = rle_mask
            full_dict['annotations'][image_id]["bbox"] = bbox
            full_dict['annotations'][image_id]["area"] = full_dict['annotations'][image_id]["bbox"][2]*full_dict['annotations'][image_id]["bbox"][3]
            image_id += 1  
    return full_dict

# test_annotation_folder = "C:/Users/t-ssudhakar/Synthetic Dataset Sample/datasets/YCB/Captures/4-20-22-studio-x-GT/cheeze-zoom-1-lights-2-gt/results/cheeze-zoom-1-lights-2"
# new_result = create_test_coco_json_annotation(test_annotation_folder)

# with open("C:/Users/t-ssudhakar/Synthetic Dataset Sample/datasets/YCB/Captures/4-20-22-studio-x-GT/cheeze-zoom-1-lights-2-gt/results/cheeze-zoom-1-lights-2/labels.json", "w") as outfile:
#     json.dump(new_result, outfile)


# test_annotation_folder = "C:/Users/t-ssudhakar/Synthetic Dataset Sample/datasets/YCB/Captures/4-20-22-studio-x-GT/cheeze-zoom05-lights1-gt/results/cheeze-zoom05-lights1/gt_labels"
# new_result = create_test_coco_json_annotation(test_annotation_folder)

# with open("C:/Users/t-ssudhakar/Synthetic Dataset Sample/datasets/YCB/Captures/4-20-22-studio-x-GT/cheeze-zoom05-lights1-gt/results/cheeze-zoom05-lights1/labels.json", "w") as outfile:
#     json.dump(new_result, outfile)

# test_annotation_folder = "C:/Users/t-ssudhakar/Synthetic Dataset Sample/datasets/YCB/Captures/4-20-22-studio-x-GT/cheeze-zoom-05-lights-2-gt/results/cheeze-zoom-05-lights-2/gt_labels"
# new_result = create_test_coco_json_annotation(test_annotation_folder)

# with open("C:/Users/t-ssudhakar/Synthetic Dataset Sample/datasets/YCB/Captures/4-20-22-studio-x-GT/cheeze-zoom-05-lights-2-gt/results/cheeze-zoom-05-lights-2/labels.json", "w") as outfile:
#     json.dump(new_result, outfile)


# train_annotation_folder = "C:/Users/t-ssudhakar/Synthetic Dataset Sample/MSR_table_cheezit_coco"
# train_labels_viz(train_annotation_folder)

# tsne_viz("C:/Users/t-ssudhakar/MINST Dataset")
import torch
from torch.optim import Adam, SGD
import time
import asyncio

import os
import numpy as np

# from data.data import *
from data.data_al import get_dataset_al
from data.data_remaker_al import remake_data
from unet_model import *
from loss import *
from metrics import *
import config

from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2

import numpy as np
import torch
import itertools
import time
from collections import defaultdict
from PIL import Image
from torch.utils.data import DataLoader

from copy import copy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

def get_meta_data(DATASET_PATH):
    
    DATASET = os.listdir(DATASET_PATH)
    DATASET = [file for file in DATASET if  file.endswith(".npy") and re.search("Features", file)]

    META_DATA = dict()

    for file_name in DATASET:
        file = np.load(os.path.join(DATASET_PATH, file_name))
        #print(file.shape)
        file_height, file_width, _ = file.shape
        #print(file_height)
        #print(file_width)

        # elev_data = file[:, :, 3]
        # file_elev_max = np.max(elev_data)
        # file_elev_min = np.min(elev_data)
        # print(file_elev_max)
        # print(file_elev_min)

        # if file_elev_max>config.GLOBAL_MAX:
        #     config.GLOBAL_MAX = file_elev_max
        # if file_elev_min<config.GLOBAL_MIN:
        #     config.GLOBAL_MIN = file_elev_min


        META_DATA[file_name] = {"height": file_height,
                                "width": file_width}
        
    return META_DATA

def run_pred(model, data_loader):

    ## Model gets set to evaluation mode
    model.eval()
    pred_patches_dict = dict()

    for data_dict in tqdm(data_loader):

        ## RGB data
        rgb_data = data_dict['rgb_data'].float().to(DEVICE)

        ## Data labels
        labels = data_dict['labels'].float().to(DEVICE)

        ## Get filename
        filename = data_dict['filename']
        # print("filename: ", filename)

        ## Get model prediction
        pred = model(rgb_data)


        ## Remove pred and GT from GPU and convert to np array
        pred_labels_np = pred.detach().cpu().numpy()
        gt_labels_np = labels.detach().cpu().numpy()

        ## Save Image and RGB patch
        for idx in range(rgb_data.shape[0]):
            pred_patches_dict[filename[idx]] = pred_labels_np[idx, :, :, :]

    return pred_patches_dict


def find_patch_meta(pred_patches_dict):
    y_max = 0
    x_max = 0

    for item in pred_patches_dict:

        temp = int(item.split("_")[3])
        if temp>y_max:
            y_max = temp

        temp = int(item.split("_")[5])
        if temp>x_max:
            x_max = temp


    y_max+=1
    x_max+=1
    
    return y_max, x_max


def stitch_patches_GT_labels(pred_patches_dict, TEST_REGION):
    cropped_data_path = f"./data/Region_{TEST_REGION}_TEST/cropped_data_val_test_al"
    y_max, x_max = find_patch_meta(pred_patches_dict)
    
    for i in range(y_max):
        for j in range(x_max):
            dict_key = f"Region_{TEST_REGION}_y_{i}_x_{j}_features.npy"
            dict_key_label = f"Region_{TEST_REGION}_y_{i}_x_{j}_label.npy"
            #print(dict_key)
        
            pred_patch = pred_patches_dict[dict_key]
            pred_patch = np.transpose(pred_patch, (1, 2, 0))

            label_patch = np.load(os.path.join(cropped_data_path, dict_key_label))

            if j == 0:
                label_x_patches = label_patch
                pred_x_patches = pred_patch
            else:
                label_x_patches = np.concatenate((label_x_patches, label_patch), axis = 1)
                pred_x_patches = np.concatenate((pred_x_patches, pred_patch), axis = 1)

            ## rgb_patches.append(rgb_patch)
            ## pred_patches.append(pred_patch)
    
        if i == 0:
            label_y_patches = label_x_patches
            pred_y_patches = pred_x_patches
        else:
            label_y_patches = np.vstack((label_y_patches, label_x_patches))
            pred_y_patches = np.vstack((pred_y_patches, pred_x_patches))
        

    label_stitched = label_y_patches
#     pred_stitched = np.argmax(pred_y_patches, axis = -1)
    pred_stitched = pred_y_patches.copy()
    
    return label_stitched, pred_stitched

def stitch_patches_augmented(pred_patches_dict, TEST_REGION):
    cropped_data_path = f"./data/Region_{TEST_REGION}_TEST/cropped_data_val_test_al"
    y_max, x_max = find_patch_meta(pred_patches_dict)

    for i in range(y_max):
        for j in range(x_max):
            dict_key = f"Region_{TEST_REGION}_y_{i}_x_{j}_features.npy"

            pred_patch = pred_patches_dict[dict_key]
            pred_patch = np.transpose(pred_patch, (1, 2, 0))

            rgb_patch = np.load(os.path.join(cropped_data_path, dict_key))[:, :, :3]

            if j == 0:
                rgb_x_patches = rgb_patch[config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS, config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS, :3]
                pred_x_patches = pred_patch[config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS, config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS] # 4:124, 4:124
            else:
                rgb_x_patches = np.concatenate((rgb_x_patches, rgb_patch[config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS, config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS,:3]), axis = 1)
                pred_x_patches = np.concatenate((pred_x_patches, pred_patch[config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS, config.EXTRA_PIXELS:config.SPATIAL_SIZE-config.EXTRA_PIXELS,:3]), axis = 1)

        if i == 0:
            rgb_y_patches = rgb_x_patches
            pred_y_patches = pred_x_patches
        else:
            rgb_y_patches = np.vstack((rgb_y_patches, rgb_x_patches))
            pred_y_patches = np.vstack((pred_y_patches, pred_x_patches))


    rgb_stitched = rgb_y_patches.astype('uint8')
    # pred_stitched = np.argmax(pred_y_patches, axis = -1)
    pred_stitched = pred_y_patches.copy()

    return rgb_stitched, pred_stitched

def stitch_patches(pred_patches_dict, TEST_REGION):
    cropped_data_path = f"./data/Region_{TEST_REGION}_TEST/cropped_data_val_test_al"
    y_max, x_max = find_patch_meta(pred_patches_dict)
    
    for i in range(y_max):
        for j in range(x_max):
            dict_key = f"Region_{TEST_REGION}_y_{i}_x_{j}_features.npy"
            #print(dict_key)
        
            pred_patch = pred_patches_dict[dict_key]
            pred_patch = np.transpose(pred_patch, (1, 2, 0))

            rgb_patch = np.load(os.path.join(cropped_data_path, dict_key))[:, :, :3]


            if j == 0:
                rgb_x_patches = rgb_patch
                pred_x_patches = pred_patch
            else:
                rgb_x_patches = np.concatenate((rgb_x_patches, rgb_patch), axis = 1)
                pred_x_patches = np.concatenate((pred_x_patches, pred_patch), axis = 1)

            ## rgb_patches.append(rgb_patch)
            ## pred_patches.append(pred_patch)
    
        if i == 0:
            rgb_y_patches = rgb_x_patches
            pred_y_patches = pred_x_patches
        else:
            rgb_y_patches = np.vstack((rgb_y_patches, rgb_x_patches))
            pred_y_patches = np.vstack((pred_y_patches, pred_x_patches))
        

    rgb_stitched = rgb_y_patches.astype('uint8')
#     pred_stitched = np.argmax(pred_y_patches, axis = -1)
    pred_stitched = pred_y_patches.copy()
    
    return rgb_stitched, pred_stitched

def center_crop(stictched_data, original_height, original_width, image = False):

    current_height, current_width = stictched_data.shape[0], stictched_data.shape[1]

    height_diff = current_height-original_height
    width_diff = current_width-original_width

    cropped = stictched_data[height_diff//2:current_height-height_diff//2, width_diff//2: current_width-width_diff//2]

    return cropped

def center_crop_augmented(stictched_data, original_height, original_width, image = False):

    current_height, current_width = stictched_data.shape[0], stictched_data.shape[1]

    height_diff = current_height-original_height
    width_diff = current_width-original_width

    cropped = stictched_data[0:current_height-height_diff, 0:current_width-width_diff]

    return cropped


class EarlyStopping:
	"""
	Early stopping to stop the training when the loss does not improve after
	certain epochs.
	"""

	def __init__(self, patience=5, min_delta=0):
		"""
		:param patience: how many epochs to wait before stopping when loss is
			   not improving
		:param min_delta: minimum difference between new loss and old loss for
			   new loss to be considered as an improvement
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = None
		self.early_stop = False

	def __call__(self, val_loss):
		if self.best_loss == None:
			self.best_loss = val_loss
		elif self.best_loss - val_loss > self.min_delta:
			self.best_loss = val_loss
			# reset counter if validation loss improves
			self.counter = 0
		elif self.best_loss - val_loss < self.min_delta:
			self.counter += 1
			# logging.info(f"Early stopping counter {self.counter} of {self.patience}")
			if self.counter >= self.patience:
				# logging.info('Early stop!')
				self.early_stop = True

def convert_to_rgb(input_array):
    # Ensure the input array is in the range [0, 1]
    input_array = np.clip(input_array, 0, 1)

    # Create a colormap from blue to red
    cmap = plt.get_cmap('RdYlGn')

    # Apply the colormap to the input array
    rgb_image = cmap(input_array)

    # Remove the alpha channel if it exists
    if rgb_image.shape[-1] == 4:
        rgb_image = rgb_image[:, :, :3]

    return rgb_image


# TODO: check for index..................
def ann_to_labels(png_image):
    ann = cv2.imread(png_image)
    ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)

    forest = ann[:, :, 1] > 0
    not_forest = ann[:, :, 2] > 0

    forest_arr = np.where(forest, 1, 0)
    not_forest_arr = np.where(not_forest, -1, 0)

    final_arr = forest_arr + not_forest_arr

    final_arr[final_arr == 0] = -99
    final_arr[final_arr == -1] = 0
    final_arr[final_arr == -99] = -1

    return final_arr


def train(TEST_REGION, student_id, al_cycle):
    print("Retraining the Model with new labels")

    if not os.path.exists(f"./users/{student_id}"):
        os.mkdir(f"./users/{student_id}")

    if not os.path.exists(f"./users/{student_id}/output"):
        os.mkdir(f"./users/{student_id}/output")

    if not os.path.exists(f"./users/{student_id}/saved_models"):
        os.mkdir(f"./users/{student_id}/saved_models")
    
    if not os.path.exists(f"./users/{student_id}/saved_models/Region_{TEST_REGION}_TEST"):
        os.mkdir(f"./users/{student_id}/saved_models/Region_{TEST_REGION}_TEST")

    if not os.path.exists(f"./users/{student_id}/resume_epoch"):
        os.mkdir(f"./users/{student_id}/resume_epoch")
    
    if not os.path.exists(f"./users/{student_id}/al_cycles"):
        os.mkdir(f"./users/{student_id}/al_cycles")

    # time.sleep(5)
    # return # TODO: remove after test


    model = UNet(config.IN_CHANNEL, config.N_CLASSES, ultrasmall = True).to(DEVICE)
    optimizer = SGD(model.parameters(), lr = 1e-7)
    criterion = torch.nn.CrossEntropyLoss(reduction = 'sum', ignore_index=2)
    elev_eval = Evaluator()

    # read resume epoch from text file if exists
    try:
        with open(f"./users/{student_id}/resume_epoch/R{TEST_REGION}.txt", 'r') as file:
            content = file.read()
            resume_epoch = int(content) 
    except FileNotFoundError:
        resume_epoch = 0

    model_path = f"./users/{student_id}/saved_models/Region_{TEST_REGION}_TEST/saved_model_{resume_epoch}.ckpt"
    if not os.path.exists(model_path):
        print("Model path doesn't exist; using pretrained model!!!")
        model_path = f"./saved_models/initial_model/saved_model_{resume_epoch}.ckpt"

    # if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model'])
    print(f"Resuming from epoch {resume_epoch}")
    

    updated_labels_orig = ann_to_labels(f'./users/{student_id}/output/R{TEST_REGION}_labels.png')
    new_labels = ann_to_labels(f'./users/{student_id}/output/R{TEST_REGION}_pred_label.png')

    merged = np.where((updated_labels_orig == -1), new_labels, updated_labels_orig)
    # print(merged)
    updated_labels = merged

    merged_copy = copy(merged)
    merged_copy = merged_copy.astype("float32")
    merged_copy_new = np.where(merged_copy == -1, 0, merged_copy)
    new_arr_2 = np.zeros((merged_copy_new.shape[0], merged_copy_new.shape[1], 3), dtype=np.uint8)
    new_arr_2[:,:,1] = merged_copy_new*128

    plt.imsave(f'./users/{student_id}/output/R{TEST_REGION}_NEW_LABELS.png', new_arr_2)

    np.save(f"R_{TEST_REGION}_updated_labels.npy", updated_labels)

    # need to remake labels after getting updated labels
    remake_data(updated_labels, TEST_REGION)

    # return
    
    cropped_data_path_al = f"./data/Region_{TEST_REGION}_TEST/cropped_data_val_test_al"
    elev_val_test_dataset_al = get_dataset_al(cropped_data_path_al)
    elev_val_test_seq = np.arange(0, len(elev_val_test_dataset_al), dtype=int)
    elev_val_test_dataset_al = torch.utils.data.Subset(elev_val_test_dataset_al, elev_val_test_seq)
    al_loader = DataLoader(elev_val_test_dataset_al, batch_size = config.BATCH_SIZE)
    
    # Retrain
    ################################## Training Loop#####################################    
    al_loss_dict = dict()
    val_loss_dict = dict()
    min_val_loss = 1e10   

    total_epochs = resume_epoch + config.EPOCHS

    last_epoch = resume_epoch
    for epoch in range(resume_epoch, total_epochs):
        print(f"EPOCH: {epoch}/{total_epochs} \r")
        ## Model gets set to training mode
        model.train()
        al_loss = 0 

        for data_dict in tqdm(al_loader):

            ## Retrieve data from data dict and send to deivce

            ## RGB data
            rgb_data = data_dict['rgb_data'].float().to(DEVICE)
            rgb_data.requires_grad = True

            """
            ## Data labels
            Elev Loss function label format: Flood = 1, Unknown = 0, Dry = -1 
            """
            labels = data_dict['labels'].long().to(DEVICE)
            labels.requires_grad = False  

            ## Get model prediction
            pred = model(rgb_data)

            ## Backprop Loss
            optimizer.zero_grad()

#             pred = F.log_softmax(pred, dim=1)
#             labels = F.softmax(labels, dim = 1)
            # loss = criterion.forward(pred, torch.unsqueeze(labels, dim=1))

            loss = criterion.forward(pred, labels)

            loss.backward()
            optimizer.step()

            ## Record loss for batch
            al_loss += loss.item()

        al_loss /= len(al_loader)
        al_loss_dict[epoch+1] = al_loss
        last_epoch = epoch + 1
        print(f"Epoch: {epoch+1} AL Loss: {al_loss}" )
    

    print("Saving Model")
    torch.save({'epoch': epoch + 1,  # when resuming, we will start at the next epoch
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                f"./users/{student_id}/saved_models/Region_{TEST_REGION}_TEST/saved_model_{last_epoch}.ckpt")
    
    with open(f"./users/{student_id}/resume_epoch/R{TEST_REGION}.txt", 'w') as file:
        file.write(str(last_epoch))
    
    with open(f"./users/{student_id}/al_cycles/R{TEST_REGION}.txt", 'w') as file:
        file.write(str(al_cycle + 1))
    
    # call AL pipeline once the model is retrained
    run_prediction(TEST_REGION, student_id, updated_labels = updated_labels, updated_labels_orig=updated_labels_orig,  al_cycle=al_cycle+1)
    
    return

def run_prediction(TEST_REGION, student_id, updated_labels = None, updated_labels_orig=None, al_cycle=0):

    if not os.path.exists(f"./users/{student_id}"):
        os.mkdir(f"./users/{student_id}")

    if not os.path.exists(f"./users/{student_id}/output"):
        os.mkdir(f"./users/{student_id}/output")

    if not os.path.exists(f"./users/{student_id}/saved_models"):
        os.mkdir(f"./users/{student_id}/saved_models")
    
    if not os.path.exists(f"./users/{student_id}/saved_models/Region_{TEST_REGION}_TEST"):
        os.mkdir(f"./users/{student_id}/saved_models/Region_{TEST_REGION}_TEST")
    
    if not os.path.exists(f"./users/{student_id}/resume_epoch"):
        os.mkdir(f"./users/{student_id}/resume_epoch")


    # return # TODO: remove after test


    start = time.time()
    DATASET_PATH = "./data/repo/images"

    # read resume epoch from text file if exists
    try:
        with open(f"./users/{student_id}/resume_epoch/R{TEST_REGION}.txt", 'r') as file:
            content = file.read()
            resume_epoch = int(content) 
    except FileNotFoundError:
        resume_epoch = 0

    print("Starting from epoch: ", resume_epoch)

    test_filename = f"Region_{TEST_REGION}_image.npy"

    SAVE_FREQUENCY = 1

    features = np.load(f"./data/repo/images/Region_{TEST_REGION}_image.npy")
    
    if features.shape[0] == 3:
        height, width = features.shape[1], features.shape[2]
    else:
        height, width = features.shape[0], features.shape[1]

    try:
        gt_labels = np.load(f"./data/repo/groundTruths/Region_{TEST_REGION}_GT_Labels.npy")
    except FileNotFoundError:
        gt_labels = np.zeros((height, width))

    height, width = gt_labels.shape[0], gt_labels.shape[1]
    print("features.shape: ", features.shape)


    model = UNet(config.IN_CHANNEL, config.N_CLASSES, ultrasmall = True).to(DEVICE)


    model_path = f"./users/{student_id}/saved_models/Region_{TEST_REGION}_TEST/saved_model_{resume_epoch}.ckpt"
    if not os.path.exists(model_path):
        print("Model path doesn't exist; using pretrained model!!!")
        model_path = f"./saved_models/initial_model/saved_model_0.ckpt"
    
    checkpoint = torch.load(model_path, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model'])
    pretrained_epoch = checkpoint['epoch']
    print(f"Resuming from epoch {pretrained_epoch}")

    
    ## Model gets set to evaluation mode
    model.eval()
    pred_patches_dict = dict()

    # META_DATA = get_meta_data(DATASET_PATH)
    
    ## Run prediciton
    cropped_data_path_al = f"./data/Region_{TEST_REGION}_TEST/cropped_data_val_test_al"
    # test_dataset = get_dataset_al(cropped_data_path_al, config.IN_CHANNEL)
    test_dataset = get_dataset_al(cropped_data_path_al)

    test_loader = DataLoader(test_dataset, batch_size = config.BATCH_SIZE)

    pred_patches_dict = run_pred(model, test_loader)

    rgb_stitched, pred_stitched = stitch_patches_augmented(pred_patches_dict, TEST_REGION)
    pred_unpadded = center_crop_augmented(pred_stitched, height, width, image = False)

    print("pred_unpadded.shape: ", pred_unpadded.shape)
    print(np.max(pred_unpadded))
    pred_forest = pred_unpadded[:,:,1]

    if updated_labels is not None:
        forest_pixels = np.where(updated_labels == 1)
        not_forest_pixels = np.where(updated_labels == 0)
        pred_forest[forest_pixels] = 1
        pred_forest[not_forest_pixels] = 0
    
    pred_labels = np.where(pred_forest > 0.5, 1, 0)

    np.save(f"./users/{student_id}/output/R{TEST_REGION}_pred_np.npy", pred_forest)

    pred_forest = pred_forest.astype("float32")
    new_arr = np.zeros((pred_forest.shape[0], pred_forest.shape[1], 3), dtype=np.uint8)
    new_arr[:,:,1] = pred_forest*128

    plt.imsave(f'./users/{student_id}/output/R{TEST_REGION}_pred_test.png', new_arr)

    elev_eval = Evaluator()

    gt_labels = np.load(f"./data/repo/groundTruths/Region_{TEST_REGION}_GT_Labels.npy")
    metrices = elev_eval.run_eval(pred_labels, gt_labels)

    if updated_labels is None:
        updated_labels = np.full((height, width), -1)

    if updated_labels_orig is None:
        updated_labels_orig = np.full((height, width), -1)

    annotated_pixels = np.where(updated_labels_orig != -1, 1, 0)
    annotated_pixels_percent = (np.sum(annotated_pixels) / (updated_labels_orig.shape[0] * updated_labels_orig.shape[1])) * 100
    print("Ann pixels percent: ", annotated_pixels_percent)

    annotated_pixels_percent = float("{:.2f}".format(annotated_pixels_percent))

    metrices += "\n"
    metrices += f"Annotated Pixels: {annotated_pixels_percent} %"

    file_path = f"./users/{student_id}/output/Region_{TEST_REGION}_Metrics_C{al_cycle}.txt"
    with open(file_path, "w") as fp:
        fp.write(metrices)


    # forest = np.where(pred_unpadded < 0.5, 1, 0)
    # not_forest = np.where(pred_unpadded >= 0.5, 1, 0)

    # forest = np.expand_dims(forest, axis=-1)
    # not_forest = np.expand_dims(not_forest, axis=-1)

    # forest = forest*np.array([ [ [0, 255, 0] ] ])
    # not_forest = not_forest*np.array([ [ [0, 0, 255] ] ])
    # pred_labels = (forest + not_forest).astype('uint8')

    # pim = Image.fromarray(pred_labels)
    # pim.convert('RGB').save("./R1_pred_test.png")

    return

if __name__ == "__main__":
    TEST_REGION = "1"

    








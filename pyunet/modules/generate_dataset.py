import os
import glob
import shutil
from sklearn.model_selection import train_test_split

class GenerateDataset:
    def __init__(self, params={}):
        self.dataset_name   = params.get('dataset_name')
        self.input_img_dir  = params.get('input_img_dir')
        self.input_mask_dir = params.get('input_mask_dir')
    
    def execute(self):
        train_img_dir   = "{}/train/images".format(self.dataset_name)
        train_mask_dir  = "{}/train/masks".format(self.dataset_name)
        test_img_dir    = "{}/test/images".format(self.dataset_name)
        test_mask_dir   = "{}/test/masks".format(self.dataset_name)

        # Create train img and mask dir if it does not exist
        print("Creating train_img_dir: {}".format(train_img_dir))
        os.makedirs(train_img_dir)

        print("Creating train_mask_dir: {}".format(train_mask_dir))
        os.makedirs(train_mask_dir)

        print("Creating test_img_dir: {}".format(test_img_dir))
        os.makedirs(test_img_dir)
        print("Creating test_mask_dir: {}".format(test_mask_dir))
        os.makedirs(test_mask_dir)

        images  = sorted(glob.glob("{}/*".format(self.input_img_dir)))
        masks   = sorted(glob.glob("{}/*".format(self.input_mask_dir)))

        train_images, test_images, train_masks, test_masks = train_test_split(
            images,
            masks,
            test_size=0.3
        )

        for img_file in train_images:
            print("[TRAIN] Copying {} to {}".format(img_file, train_img_dir))
            shutil.copy(img_file, train_img_dir)
            
        for img_file in train_masks:
            print("[TRAIN] Copying {} to {}".format(img_file, train_mask_dir))
            shutil.copy(img_file, train_mask_dir)
            
        for img_file in test_images:
            print("[TEST] Copying {} to {}".format(img_file, train_img_dir))
            shutil.copy(img_file, test_img_dir)
            
        for img_file in test_masks:
            print("[TEST] Copying {} to {}".format(img_file, train_mask_dir))
            shutil.copy(img_file, test_mask_dir)

        print("Done")

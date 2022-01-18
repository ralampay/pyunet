import sys
import argparse
import os
import json
import cv2
import numpy as np

def main():
    parser  = argparse.ArgumentParser(description="Convert JSON COCO config to image masks")
    
    parser.add_argument("--img-dir", help="Image directory", required=True, type=str)
    parser.add_argument("--config", help="JSON configuration file", required=True, type=str, default="config.json")

    args = parser.parse_args()

    img_dir = args.img_dir
    config  = args.config

    config = json.load(open(config))

    objects = []

    for idx in range(len(config["images"])):
        obj = {
            "file": os.path.join(img_dir, config["images"][idx]["file_name"]),
            "segs": []
        }

        for ann in config["annotations"]:
            if ann["image_id"] == config["images"][idx]["id"]:
                obj["segs"].append(ann["segmentation"])

        objects.append(obj)

    for obj in objects:
        original_image = cv2.imread(obj["file"])

        segs = [np.array(seg, np.int32).reshape((1, -1, 2))
                    for seg in obj["segs"]]

        result = np.zeros(original_image.shape, dtype=np.uint8)

        cv2.fillPoly(result, segs, color=(255,255,255))

        filename, extension = obj["file"].split(".")

        cv2.imwrite(os.path.join(img_dir, "{}{}.{}".format(filename, "_mask", extension)), result)

if __name__ == '__main__':
    main()

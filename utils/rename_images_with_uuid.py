import sys
import argparse
import os
import cv2
import json
import glob
import uuid

def main():
    parser = argparse.ArgumentParser(description="Rename images in a folder to {uuid}.png")
    parser.add_argument("--img-dir", help="Image Directory", required=True, type=str)

    args    = parser.parse_args()
    img_dir = args.img_dir
    
    images = []

    ext = ['png']

    files = []
    [files.extend(glob.glob(img_dir + "/*." + e)) for e in ext]

    for file in files:
        new_id = str(uuid.uuid4())
        new_filename = "{}{}.png".format(img_dir, new_id)
        print("Renaming {} to {}".format(file, new_filename))

        os.rename(file, new_filename)

if __name__ == '__main__':
    main()


from PIL import Image
import os, sys
sys.path.append('/Users/yahanyang/Desktop/CIS620/gradual_domain_adaptation')
import shutil
from shutil import copyfile
import numpy as np
import datasets
import scipy
import pickle
# Resize images.
def resize(path, size=64):
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((size,size), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG')

#for folder in ['./dataset_64x64/M/', './dataset_64x64/F/']:
#    resize(folder, size=64)

#datasets.save_data(data_dir='dataset_64x64', save_file='dataset_64x64.mat', target_size=(64,64))

##add 64 size portrait data 
##for folder in ['./dataset_64x64/M/', './dataset_64x64/F/']:
#    resize(folder, size=64)

#datasets.save_data(data_dir='dataset_64x64', save_file='dataset_64x64.mat', target_size=(64,64))

image_options = {
    'batch_size': 100,
    'class_mode': 'binary',
    'color_mode': 'grayscale',
}


d = {}
def save_data(target_size=(32, 32)):
    for subdir, dirs, files in os.walk("./CURE-TSR/Real_Test"):
        if subdir != dirs:
            var = subdir.split("/")[-1].split("-")[0]
            if var not in d:
                d[var] = []
            for file in files:
                y = file.split("_")[1]
                directory = subdir+"/"+y
                if not os.path.exists(directory):
                    os.mkdir(directory)
                shutil.move(subdir+"/"+file, directory+"/"+file)
                d[var].append(file)

save_data()

We strongly suggest installing the requirements on a python virtual environment, to limit any dependency issues. Below are instructions to do so (for more info, see https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
1. Install virtualenv  
python3 -m pip install --user virtualenv

2. Create a virtual environment called env.  
python3 -m venv env

1. Activate virtual environment. From the level above the env/ folder, run:  
source env/bin/activate

2. Perform installations (no need for sudo) and run code as usual. Installations located in env/lib/ directory. The packages and appropriate versions necessary for this project are located in the requirements.txt file. You can install them by running:  
python3 -m pip install -r requirements.txt

3. To deactivate virtual environment, run:  
deactivate

# How to get CURE-TSR dataset and setup in folder
Download from https://ieee-dataport.org/open-access/cure-tsd-challenging-unreal-and-real-environment-traffic-sign-detection#files  

cd gradual_domain_adaptation

Extract to gradual_domain_adaptation/CURE_TSR

Then run: ln -s ./CURE_TSR ../CAML-Pytorch/CURE_TSR_OG

Then run the following: python create_dataset.py

Then run: ln -s ./CURE_TSR ../CAML-Pytorch/CURE_TSR_Yahan_Shortcut

# GDA on CURE-TSR (Table 1)
cd gradual_domain_adaptation

specify the environment by chnaging argument 'natural_type' on ine 647 in datasets.py (possible options: "Darkening", "Decolorization", "Snow", "Rain", "LensBlur", "Haze", "Exposure", "GaussianBlur", "Noise", "Shadow")

python gradual_shift_better.py

# MAML based CAML (Table 2 and 3)
cd CAML-Pytorch
## Choose environment, e.g. snow
In CURE_TSR_tasksets.py, change line 28 to " lvl5_test_dir = './CURE_TSR_OG/Real_Train/Snow-5/' " to perform experiments on snow environment.

## Source Model
### 5 way, 1 shot
python cure_tsr_train.py --model resnet18 --gpu 0 --lr 0.5 --ways 5 --shots 1  
python cure_tsr_train.py --model resnet50 --gpu 0 --lr 0.1 --ways 5 --shots 1  
python cure_tsr_train.py --model densenet --gpu 0 --lr 0.5 --ways 5 --shots 1  
### 3 way, 2 shot
python cure_tsr_train.py --model resnet18 --gpu 0 --lr 0.5 --ways 3 --shots 2  
python cure_tsr_train.py --model resnet50 --gpu 0 --lr 0.1 --ways 3 --shots 2  
python cure_tsr_train.py --model densenet --gpu 0 --lr 0.5 --ways 3 --shots 2  

## Target ST
### 5 way, 1 shot
python cure_tsr_target.py --model resnet18 --gpu 0 --lr 0.5 --ways 5 --shots 1  
python cure_tsr_target.py --model resnet50 --gpu 0 --lr 0.1 --ways 5 --shots 1  
python cure_tsr_target.py --model densenet --gpu 0 --lr 0.5 --ways 5 --shots 1  
### 3 way, 2 shot
python cure_tsr_target.py --model resnet18 --gpu 0 --lr 0.5 --ways 3 --shots 2  
python cure_tsr_target.py --model resnet50 --gpu 0 --lr 0.1 --ways 3 --shots 2  
python cure_tsr_target.py --model densenet --gpu 0 --lr 0.5 --ways 3 --shots 2  

## All ST
### 5 way, 1 shot
python cure_tsr_all.py --model resnet18 --gpu 0 --lr 0.5 --ways 5 --shots 1  
python cure_tsr_all.py --model resnet50 --gpu 0 --lr 0.1 --ways 5 --shots 1  
python cure_tsr_all.py --model densenet --gpu 0 --lr 0.5 --ways 5 --shots 1  
### 3 way, 2 shot
python cure_tsr_all.py --model resnet18 --gpu 0 --lr 0.5 --ways 3 --shots 2  
python cure_tsr_all.py --model resnet50 --gpu 0 --lr 0.1 --ways 3 --shots 2  
python cure_tsr_all.py --model densenet --gpu 0 --lr 0.5 --ways 3 --shots 2  

## Gradual ST (a.k.a CAML)
### 5 way, 1 shot
python cure_tsr_gradual.py --model resnet18 --gpu 0 --lr 0.5 --ways 5 --shots 1  
python cure_tsr_gradual.py --model resnet50 --gpu 0 --lr 0.1 --ways 5 --shots 1  
python cure_tsr_gradual.py --model densenet --gpu 0 --lr 0.5 --ways 5 --shots 1  
### 3 way, 2 shot
python cure_tsr_gradual.py --model resnet18 --gpu 0 --lr 0.5 --ways 3 --shots 2  
python cure_tsr_gradual.py --model resnet50 --gpu 0 --lr 0.1 --ways 3 --shots 2  
python cure_tsr_gradual.py --model densenet --gpu 0 --lr 0.5 --ways 3 --shots 2  


# Protonets based CAML (Table 4 and 5)
## Source Model
### 5 way, 1 shot
python cure_tsr_proto.py --gpu-id 1 --model resnet18   
python cure_tsr_proto.py --gpu-id 1 --model resnet50  
python cure_tsr_proto.py --gpu-id 1 --model densenet   
### 3 way, 2 shot
python cure_tsr_proto.py --gpu-id 1 --model resnet18 --train-way 3 --shot 2 --train-query 2 --test-way 3 --test-shot 2 --test-query 2  
python cure_tsr_proto.py --gpu-id 1 --model resnet50 --train-way 3 --shot 2 --train-query 2 --test-way 3 --test-shot 2 --test-query 2  
python -cure_tsr_proto.py --gpu-id 1 --model densenet --train-way 3 --shot 2 --train-query 2 --test-way 3 --test-shot 2 --test-query 2  

## Target ST
### 5 way, 1 shot
python cure_tsr_target_protonet.py --gpu-id 1 --model resnet18   
python cure_tsr_target_protonet.py --gpu-id 1 --model resnet50  
python cure_tsr_target_protonet.py --gpu-id 1 --model densenet   
### 3 way, 2 shot
python cure_tsr_target_protonet.py --gpu-id 1 --model resnet18 --train-way 3 --shot 2 --train-query 2 --test-way 3 --test-shot 2 --test-query 2  
python cure_tsr_target_protonet.py --gpu-id 1 --model resnet50 --train-way 3 --shot 2 --train-query 2 --test-way 3 --test-shot 2 --test-query 2  
python -cure_tsr_target_protonet.py --gpu-id 1 --model densenet --train-way 3 --shot 2 --train-query 2 --test-way 3 --test-shot 2 --test-query 2  

## All ST
### 5 way, 1 shot
python cure_tsr_all_protonet.py --gpu-id 1 --model resnet18   
python cure_tsr_all_protonet.py --gpu-id 1 --model resnet50  
python cure_tsr_all_protonet.py --gpu-id 1 --model densenet   
### 3 way, 2 shot
python cure_tsr_all_protonet.py --gpu-id 1 --model resnet18 --train-way 3 --shot 2 --train-query 2 --test-way 3 --test-shot 2 --test-query 2  
python cure_tsr_all_protonet.py --gpu-id 1 --model resnet50 --train-way 3 --shot 2 --train-query 2 --test-way 3 --test-shot 2 --test-query 2  
python -cure_tsr_all_protonet.py --gpu-id 1 --model densenet --train-way 3 --shot 2 --train-query 2 --test-way 3 --test-shot 2 --test-query 2  

## Gradual ST (a.k.a CAML)
### 5 way, 1 shot
python cure_tsr_gradual_protonet.py --gpu-id 1 --model resnet18 --train-way 5 --shot 1 --train-query 1 --test-way 5 --test-shot 1 --test-query 1  
python cure_tsr_gradual_protonet.py --gpu-id 1 --model resnet50 --train-way 5 --shot 1 --train-query 1 --test-way 5 --test-shot 1 --test-query 1  
python -cure_tsr_gradual_protonet.py --gpu-id 1 --model densenet --train-way 5 --shot 1 --train-query 1 --test-way 5 --test-shot 1 --test-query 1  
### 3 way, 2 shot
python cure_tsr_gradual_protonet.py --gpu-id 1 --model resnet18 --train-way 3 --shot 2 --train-query 2 --test-way 3 --test-shot 2 --test-query 2  
python cure_tsr_gradual_protonet.py --gpu-id 1 --model resnet50 --train-way 3 --shot 2 --train-query 2 --test-way 3 --test-shot 2 --test-query 2  
python cure_tsr_gradual_protonet.py --gpu-id 1 --model densenet --train-way 3 --shot 2 --train-query 2 --test-way 3 --test-shot 2 --test-query 2  

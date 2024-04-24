# <ins>The Problem<ins>
## Segmentation of the hippocampus on magnetic resonance images (MRI) of human brains.


# <ins>Method<ins>
## Data
#### Target:  Hippocampus head and body
####  Modality: Mono - modal MRI
#### Size: 394 3D volumes(263 Training + 131 Testing)
#### Source: Vanderbilt University Medical Center



## Model

-| Name      | Type       | Params |
--- |-----------|------------|-------- 
0 | net       | UNet       | 122 K    
1 | criterion | DiceCELoss | 0    

### UNet 
#### Architecture
#### Design
#### Advantages

### Loss Function



# <ins>Setup<ins>

Using conda for virtual environment and installation.

From project root directory:
```console
git clone 
conda create –n 766-ppml
conda activate 766-ppml
conda install python=3.8 torchcsprng cudatoolkit=10.2 -c pytorch -c conda-forge
pip install -r requirements.txt
```


# Sources
Credit goes to:

GradAttack: attack.py, gradientinversion.py, TrainingPipline.py, utils.py
: https://github.com/Princeton-SysML/GradAttack/tree/master

Deiping: inverting gradients reconstructing algorithms
: https://github.com/JonasGeiping/invertinggradients

Monai: unet.py, model.py
: https://github.com/Project-MONAI/tutorials/blob/main/modules/TorchIO_MONAI_PyTorch_Lightning.ipynb

https://arxiv.org/pdf/2001.02610.pdf

#TODO
https://github.com/Project-MONAI/MONAI/discussions/4007
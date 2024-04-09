##  Attacks


## repos
Contains git submodules for each model implementation
Visit each repositories README for installation and running instructions.
iDash-2020, DPBFL, and GradAttack were all successfully setup and run.

## data
Contains BC-TCGA dataset as well as 3-D image datasets for testing.
Currently testing 3-D Brain Tumor Images for classification.


## src
### Not fully implemented, code has errors.
### BCTCGADataModule.py
Implementation of LightningDataModule for BC-TCGA. Will be _used regardless of model.
Implementation of a 3-D dataset DataModule ..._

### GradAttack.py
Example Gradient Inversion attack on CIFAR10 dataset. Taken from GradAttack/examples. Trying to convert this
to work with BCTCGADataModule.

### DPFLAttack.py
Example running on MNIST dataset.Trying to integrate GradAttack into this pipeline. Need to wrap model in Lightning.
Then conver this to work with BCTCGADataModule.


## requirements.txt
### Will include more installation instructions later.
List of project dependencies. Use pip install -r requirements.txt.
Preferably install dependencies in virtual environment, either venv or conda.
Personally used conda for initial setup. Requires manual installation of torchcsprng.





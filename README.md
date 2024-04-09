# 766-ppml

## Privacy Preservation Machine Learning with Attacks and Defenses

### Introduction
This repository contains the code and resources for implementing Privacy Preservation Machine Learning (PPML) techniques along with various attacks and defense mechanisms. PPML aims to protect sensitive information in machine learning models and datasets from unauthorized access or inference attacks. In its current state, there are two branches off of main. The "attack" branch is authored by Michael Xu and showcases his progress thus far on the attacks.
The "defend" branch is authored by Anna Theodore and showcases her progress thus far on the defenses.

### Problem
This research aims to tackle security concerns related to safeguarding the privacy of personal data. As the field of Bioinformatics continues to expand, it is vital to protect the personal health data of millions of individuals. A challenge arises when attempting to distribute sensitive data. There is a need to conceal enough information to protect the data owner’s privacy while also sharing enough information to optimize model performance. To protect this private data, this project will implement and test gradient inversion attacks and defenses against these attacks.

### Data
Michael fill this out

## Related Repositories
- [Py-FHE Repository](https://github.com/sarojaerabelli/py-fhe.git): Python Implementation of Fully Homomorphic Encryption.
Michael add yours

### Attack Algorithms
Michael fill this out w/ yours

### Defend Algorithms
Currently, the algorithms for defense consist of Homomorphic Encryption (HE). The basic methodology used is from the Py-FHE Repository referenced above. These algorithms currently only exist in the "defend" branch.

The following file paths that have been changed from the Py-FHE implementation are as follows:

py-fhe/tests/bfv/test_he.py -> New file with Microsoft SEAL attemps
py-fhe/tests/bfv/BC-TCGA-Normal.txt -> New file with dataset example
py-fhe/tests/bfv/test_bfv_encrypt_decrypt.py -> File present in Py-FHE with new parsing algorithms for the specific use case

### Installation
1. Future instructions will be written for installation.

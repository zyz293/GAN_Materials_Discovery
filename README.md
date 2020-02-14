# Microstructural materials design via deep adversarial learning methodology
This software is an deep learning application for generating materials microstructure images using generative adversarial networks. The proposed approach is trained on synthetic 2D microstructure images.

To use this software, what the algorithm requires as input are a numpy array. The shape of data point is (x, 51, 51, 51) where x is the number of microscale volume elements (MVEs) and the dimension of microstructure should be three-dimensional (i.e. 51x51x51). For crystal plasticity dataset, the shape of data point is (x, 224, 224) where x is the number of strain profile images/crops and the size of strain profile image/crop is 224x224. The software will take the row data and corresponding two-point correlation funcion (i.e. integerated domain knowledge with same size as row data) as input, and train the predictive models. (The detail about data preprocessing and model is in related sections of published paper). (Note that two-point correlation funcion can be computed using PyMKS software at http://pymks.org/en/latest/rst/README.html).

## Requirements ##
Python 2.7
Numpy 1.12.1
Sklearn 0.19.1
Keras 2.0.0
Scipy
Pandas
Pickle
TensorFlow
H5PY

## Files ##
1. gan_training.py: The script to train the GAN that can generate two-phase microstructure image.
2. ScalableG.py: Trained generator of the proposed GAN.
3. weights.pickle: Weights of the trained generator of the proposed GAN.

## How to run it
1. To run gan_training.py: use commend 'python sgan_ST_v4.py'. The script will train the GAN and save your GAN.
2. To use trained generator: 
	1. Download "weights.pickle" and "ScalableG.py" in the same folder.
	2. Change the "zval" variables in "ScalableG.py" to generate different microstructure images.
	3. Use command line "python ScalableG.py" to load the weights and run the model. 

## Acknowledgement
The Rigorous Couple Wave Analysis simulation is supported by Prof. Cheng Sun's lab at Northwestern University. This work is primarily supported by the Center of Hierarchical Materials Design (NIST CHiMaD 70NANB14H012) and Predictive Science and Engineering Design Cluster (PS&ED, Northwestern University). Partial support from NSF awards DMREF-1818574, DMREF-1729743, DIBBS-1640840, CCF-1409601; DOE awards DE-SC0007456, DE-SC0014330; AFOSR award FA9550-12-1-0458; and Northwestern Data Science Initiative is also acknowledged. 

## Related Publications ##
Microstructural Materials Design via Deep Adversarial Learning Methodology

## Contact
Ankit Agrawal <ankitag@eecs.northwestern.edu>




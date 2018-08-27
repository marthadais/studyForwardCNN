Copyright 2017 Martha Dais Ferreira, Rodrigo Fernandes de Mello

This file is part of studyForwardCNN.

studyForwardCNN is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

studyForwardCNN is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ImageFalseNearest.  If not, see <http://www.gnu.org/licenses/>.

See the file "COPYING" for the text of the license.

Contact: 
	Martha D. Ferreira: daismf@icmc.usp.br
	Rodrigo Fernandes de Mello: mello@icmc.usp.br

----------------------------------------------
# Study of CNN process

This source codes are used to execute the forward process of Convolutional Neural Networks (CNN) for images input. Such source code was written in the C++ Language, in which can be running in sequential or in parallel. In addition, a source code of Multilayer Perceptron (MLP) was developed in R language to study the impact of dropout and regularization terms during the learning process.

##Information about directories

./include	- This directory contains all CNN classes. It is ready for 3-layer CNNs but it can be easily adapted to work with more layers

./database 	- This directory contains datasets

./src		- This directory contains file ThreeLayerCNN.cpp which has function main

./R		- Optimization functions:

			- Architecture optimization - defines number of neurons per layer, kernel sizes, strides, pooling, etc.

			- Kernel optimization - optimizes kernel masks (used for convolution)

			- Compute Accuracy - it is used to compute CNN accuracy for a particular configuration

./test		- Tests used for learning OpenCV and check codes developing

./xml		- Xml files with parameters

## Getting Started

These instructions of how to use studyForwardCNN come with a copy of the source code, which contains the forward process of a CNN and MLP code to evaluated techniques used to avoid overfitting.

## Requirements

Before starting, it is necessary to have those libraries:

1. opencv

2. xerces

3. armadillo:

	3.1 LAPACK
	3.2 BLAS
	3.3 ATLAS

## Compiling

In the terminal, use 'cd' command to enter the folder studyForwardCNN and type the command:

```
> make
```

## Example

The example requires the dataset also provided in this source code. To execute an example in terminal, run the command:

```
> make run
```

This command run in sequential mode as the command:

```
> make run-sequencial
```

For run in parallel mode, execute:

```
> make run-parallel
```

## CNN architecture:

### Summary:
 - For each image:
     - For each layer:
         - Normalize image
        - Convolve image in each depth
        - Sum the images by depth
        - Apply maximum pooling (in one image)
    - Return the new Image
- Select the minimum rows and columns of the results images
- Save the results images using the class information:
	- Put the image in a row
	- Put the class information

### Kernel Constructor:
- Create a random Kernel without depth, passing number of rows and colunms
- Create a random Kernel, passing number of rows, colunms, and depth
- Create a Kernel without depth, passing the weights, the number of rows and colunms
- Create a Kernel, passing the weights, the number of rows, columns, and depth
- Create a random Kernel without depth, passing number of rows and columns, and the minimum and maximum values
- Create a random Kernel, passing number of rows, columns, depth, and the minimum and maximum values
	
### Neuron Constructor:
Set the values
- Build Network:
    - For each Layer:
	    - Create Kernels (Kernel Constructor)
		- Create Neurons (Neuron Constructor)
			
- Maximum:
    - Return the maximum value
	
- Pooling:
    - Do subsample of the image using maximum value by a stride (maximum)
	
- Norm:
    - Return the sum of the squares of the members in a neighbor
	- Return the sum of the squares of the members in a neighbor in depth
	
- Normalize Image/ Normalize Image Depth
    - Divides all elements of the image by a value of the neighbors (norm)
	- Divides all elements of the image by a value of the neighbors in depth (norm)
		
- Convolve:
    - Do a fourier convolve
	
- Neuron Process:
    - Do the convolution of image/images depth with the kernel/kernel depth
	- Sum the images by depth
	- Apply threshold, removing negative numbers
	- Apply maximum pooling (applyPooling)
	
- Forward Step/Forward Step Thread:
    - Read the image passed
	- Convert the uchar image to float image
	- For each layer:
	    - Normalize image (normalizeImage/normalizeImageDepth)
		- Process image (process)
	- Normalize Image
	- Return results of this image
		
- Forward / Parallel Forward:
	- Build network (buildNetwork)
	- Processing files (forwardStep/forwardStepThread)
		
- Sum Image Depth:
	- Not used


## Citing this code 

Please cite this thesis in your publications if this code helps your research:

    TODO

Or:

    TODO

### TODO

1. Implement include/CNN.hpp to create generic CNNs and not only 3-layer neural networks.
2. Implemente src/CNN.cpp to execute the generic CNN and:
    - reading all input parameters from an XML file
	- giving option to read a file with all kernel weights (this is usefull after another program optimizes kernels)
	- without any approach for feature selection as it is in src/ThreeLayerCNN.cpp
	- producing only an output file including all features (assuming all input images have the same size)
	- also running in parallel as it is currently
3. Implement include/CNNEntropyMaximizer.hpp and src/CNNEntropyMaximizer.cpp to maximize entropy as already implemented in file test/cnn-training/training-multilayer-cnn-architecture.r

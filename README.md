# Implementation of PseudoEdgeNet: Nuclei Segemtnation only with Point Annotations

## dataset:

* https://nucleisegmentationbenchmark.weebly.com/

## Paper:

* [PseudoEdgeNet: Nuclei Segmentation only with Point Annotations](https://arxiv.org/abs/1906.02924)

## Implementation and Result:

* training time: around 30 sec on a single NVIDIA Tesla V100 with 16 GB memory
* batch_size = 2
* epochs = 60
* could produce same result showing the paper if use same evalutaion method (10-fold CV)

## structure
     
* Note.ipynb: how I implement it, thoughts, methods
* main.ipynb: main function file
* libs: source code

## dependency:

* tensorflow >= 2.2.0
* imgaug == 0.4.0

## Usage:

* Download the dataset using the link.
* create a folder named 'dataset' (no quotes), under the dataset folder, create two folder named 'Annotations' and 'Tissue images', separately (no quotas). Then put *.xml under 'Annotations' folder, put *.png under 'Tissue images' (from the dataset you download). 
* put main.ipynb, libs folder, dataset folder in same directory.
* run main.ipynb cell by cell...
# [AI Programming with Python Project](https://github.com/udacity/aipnd-project)

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

> **Note: This is an edited version after project submission, contents may not follow project rubric. And the main objective of this edited version is to explore the use of Git, GitHub and Google Colab, hence the training epoch use is 1 just to ensure the model is working.**  
</br>

## Installation

* Python version 3.7 or later
* PyTorch (refer [Pytorch website](https://pytorch.org/get-started/locally/))
* Jupyter Notebook
* Matplotlib
* Matplotlib-inline
* Numpy
* Pillow  
</br>

## Dataset

Dataset use for training -
 [Download split dataset](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz)  
Steps:  

1. Download and unzip the split dataset  
2. Rename the folder name from `flower_data` to **`flowers`**  
3. Move the renamed **`flowers`** folder to the same directory where these programmes are stored.  
</br>

## Usage

**Build and train a model, using command line:**  
Default arguments

```bash
$ python3 train.py
# or
$ python3 train.py --arch --hidden_units --learning_rate --epochs
```  

Specified arguments

```bash
$ python3 train.py --arch vgg16 --hidden_units 4096 --learning_rate 0.01 --epochs 1
```

</br>

**Rebuild trained model and predict, using command line:**  
Default arguments

```bash
$ python3 predict.py
# or
$ python3 predict.py --topk
```  

Specified arguments

```bash
$ python3 predict.py --topk 5
```  

Alternatively, use Image Classifier Project-test20.ipynb to train and predict in one file.  
</br>

## License  

![license](https://img.shields.io/github/license/perrigoh/image_classifier_pytorch_udacity)  
MIT License

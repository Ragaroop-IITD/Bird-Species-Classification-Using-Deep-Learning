# Bird Species Classification Using Deep Learning

## Introduction

This assignment involves solving a practical machine learning problem. We will gain experience with the iterative process of designing and validating a neural network architecture for a prediction problem using a popular deep learning library (PyTorch). The focus will be on understanding basic deep learning practices and techniques.

## Problem Statement

We are given a dataset consisting of images of different bird species with a total of K = 10 bird species. Each species has between 500 and 1,200 images in the dataset, and each image contains a single bird. The task is to design a neural network that takes a bird image as input and predicts the class label (one of the K possible labels) corresponding to each image.

## What We Do in This Assignment

In this project, we aim to build a deep-learning model to classify bird species based on image data. The main tasks we will undertake include:

1. **Data Preprocessing and Augmentation:** Prepare the dataset for training by organizing images, applying transformations, and performing an 80-20 split to create training and test sets. We will use image augmentations to improve the model's robustness to variations.
2. **Model Selection and Design:** Develop a convolutional neural network (CNN) suitable for image classification.
3. **Model Training:** Train the neural network on the bird images, adjusting hyperparameters to optimize performance. We will use the GPU to speed up training.
4. **Evaluation and Testing:** Assess the model's accuracy on the test dataset, examining metrics like accuracy, precision, and recall.
5. **Fine-tuning and Optimization:** Based on the results, we will fine-tune the model to improve accuracy and reduce overfitting. Techniques such as regularization, learning rate adjustments, or more data augmentation will be considered.
6. **Results and Analysis:** Present the model's performance and discuss its strengths, limitations, and potential areas for future improvement.

## Specifications of the Model

- Our model has about 32 million parameters and processes 224x224 images into ten classes.
- We used various techniques to improve performance, which will be discussed in the sections below.
- The maximum accuracy attained by our model is about 94%.

![image](https://github.com/user-attachments/assets/c3084059-b2be-4362-b9e7-055d3091f14f)


### Model Architecture

The table and image below show the architecture of the neural network we trained for this assignment. We have six convolution layers and used Batch Normalization and Max pooling.

## Observations

### Train and Validation Loss vs. Epochs

![image](https://github.com/user-attachments/assets/634e1cca-747e-4323-9c0e-0dde603bb54d)


The observations are as expected. Since the model is never trained on the test set, train loss is always less than validation loss. And both the losses decrease over time.

### Train and Validation Accuracy vs. Epochs

![image](https://github.com/user-attachments/assets/738ab24c-3375-4b01-a6b3-fc9f5792cd75)


The observations are as expected. Since the model is never trained on the test set, train accuracy is always greater than validation accuracy. And both the losses decrease over time.

### Effect of Model Optimization

| Implementation                                      | Validation Accuracy |
|-----------------------------------------------------|---------------------|
| Without any data augmentation, batch normalization, and dropout | 79.6%               |
| With data augmentation but without dropout and normalization | 81.3%               |
| With data augmentation, dropout, and batch normalization       | 90.6%               |
| With initialized class weights                                  | 91.2%               |
| With adaptive learning rate step                                | 93.6%               |

1. When dropout, data augmentation, and batch normalization were not used, the accuracy was the lowest compared to the final model in which all techniques are used, which indicates these techniques are necessary.
   - Dropout prevents the model from overfitting problems.
   - Data augmentation improves our model by exposing it to varied examples.
   - Batch normalization helps by stabilizing the learning process and making training faster and more reliable. It adds regularization to the model.
2. Initializing class weights helps by addressing imbalanced class distributions in training data, which can lead to biased models that favor the majority class. It accelerates convergence in the imbalanced data.
3. A learning rate step (also known as learning rate scheduling or learning rate decay) helps by systematically reducing the learning rate as training progresses. It improves convergence and reduces the risk of overfitting and balances exploration and exploitation.

### Analysis of Class Activation Maps (CAM)

The area of focus is mainly on the bird's face when it is predicting correctly. This suggests that the model relies on these specific characteristics for classification. The background is less highlighted, indicating that the model ignores irrelevant details and focuses on critical areas for identifying the bird's class.

![image](https://github.com/user-attachments/assets/c91c329f-8070-4699-b34d-ead69c568d73)


## References

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Training an Image Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Udacity Deep Learning Book](https://udlbook.github.io/udlbook/)

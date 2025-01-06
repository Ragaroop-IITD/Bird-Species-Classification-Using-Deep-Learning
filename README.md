# Bird Species Classification Using Deep Learning

This assignment involves solving a practical machine learning problem. We will gain experience with the iterative process of designing and validating a neural network architecture for a prediction problem using a popular deep learning library (PyTorch). The focus will be on understanding basic deep learning practices and techniques.

## Problem Statement

We are given a dataset consisting of images of different bird species with a total of K = 10 bird species. Each species has between 500 and 1,200 images in the dataset, and each image contains a single bird. The task is to design a neural network that takes a bird image as input and predicts the class label (one of the K possible labels) corresponding to each image.

## Model Design Guidelines

### Data Preparation
- Split the dataset into training and validation sets using an 80-20 split.
- Create a data loader to handle loading the images for training and validation, applying any necessary preprocessing steps such as resizing and normalization.

### Network Layers
- Use a basic CNN architecture with convolutional layers, ReLU activations, pooling layers, and fully connected layers.
- Implement the final layer using an activation such as softmax (PyTorch's `nn.CrossEntropyLoss` handles this internally).

### Loss Function
- Use `nn.CrossEntropyLoss` for multi-class classification, and handle class imbalance by assigning class weights inversely proportional to the number of samples per class.

### Optimization
- Start with the Adam optimizer, and experiment with different learning rates and data augmentation techniques.

### Regularization
- Use techniques like early stopping, dropout, and batch normalization to prevent overfitting.

### Visualizing Class Activation Maps
- Apply Grad-CAM to visualize the important regions in an image that contribute to the model's decision.

## Training and Evaluation Process

### Training Code
- The main code for training and testing the model should be in `bird.py`.
- Save the trained model in the same directory with the name `bird.pth`.

### Model Saving and Loading
- Save the model using `torch.save(model.state_dict(), PATH)`.
- Load the model using `model.load_state_dict(torch.load(PATH))`.

### Testing
- During testing, produce a `bird.csv` file with the predicted label for each image.

### Evaluation
- Evaluate the model using the average of Macro and Micro F1 scores.

## Implementation Guidelines

- Use CNNs for this task, and do not use pre-trained features or existing implementations of common models like ResNet or AlexNet.
- Ensure that the training time does not exceed 2 hours, and the model size does not exceed 80 million parameters.
- Set the seed to ensure reproducibility: `torch.manual_seed(0)`.


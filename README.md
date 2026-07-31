# Java Visual Pavement

### Pavement Distress Detection and Classification

To maintain road infrastructure in good condition, periodic evaluations are needed to determine its status and plan appropriate intervention actions. The objective of the [VisualPavement repository](https://github.com/ximenarios/VisualPavement) was to develop a software tool for the automatic classification of surface distresses in flexible pavements, carried out in accordance with the standards established in Colombia by the National Road Institute (INVIAS).

This repository presents an adaptation of the deep learning models developed in the Visual Pavement project, originally created in Python and now built and executed entirely in Java.



## Dataset

![Dataset and Label Distribution](images/DatosE_2.png)

* The original images were captured on two roads in Colombia (La Tebaida, Puerto Tejada) and labeled manually by civil engineers from our work team.
* For training and validation, categories with more than 145 labels were selected, resulting in a total of 15 categories.
* The classes include distresses such as Alligator Cracking (FPC), Fatigue Cracking (FLF), and Patching (B), among others.
* The dataset loader reads the images from the local dataset directory and automatically resizes them to a uniform scale of 224x224 pixels.
* Classification labels are extracted automatically based on the folder names.

## Convolutional Neural Network Architecture

![CNN Architecture](images/ConvNet_2.png)

This project replicates the structure of the [original Python model](https://github.com/ximenarios/VisualPavement/blob/master/VisualPavConvnets.ipynb) using Deeplearning4j ([View code](https://github.com/ximenarios/JavaVisualPavement/blob/main/src/main/java/com/ximena/VisualPavementModel.java)). The model is built using Java.

The architecture consists of:
* **Four Convolutional Blocks:** Each block contains a `ConvolutionLayer` with a ReLU activation function, immediately followed by a `SubsamplingLayer` (Max-Pooling) to extract features.
* **Dense Layer:** A flattening phase connects to a densely connected layer of 512 outputs with a ReLU activation.
* **Output Layer (Classifier):** A final softmax output layer classifies the image into one of the 15 output categories using the MCXENT loss function.

## Project Structure

The Java application is divided into specific components to handle the machine learning pipeline:

* **`ImageDatasetLoader.java`:** Responsible for loading and preprocessing the pavement images. It randomizes the order of the images using `FileSplit` for better training results.
* **`VisualPavementModel.java`:** Defines the multi-layer neural network architecture and its hyperparameters.
* **`Main.java`:** The main entry point that coordinates data loading, model building, and training. It normalizes the images, trains the network over the configured epochs, evaluates performance on testing data, and saves the trained model to disk as a `.zip` file.
* **`Inferencia.java`:** A class dedicated exclusively to prediction. It restores the pre-trained model and analyzes a single pavement image, returning the detected distress type and the AI confidence level.

## How to Run (Docker)

This project is fully containerized. You only need Docker installed on your system to compile and execute the application.

### 1. Build the Image
To compile the Java source code and download all Maven dependencies, run the following command in your terminal:
```bash
docker build --no-cache -t java-visual-pavement .
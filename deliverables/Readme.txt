Monkeypox Skin Image Classification

Overview

This project aims to classify images of Monkeypox skin conditions using a deep learning model. The dataset is preprocessed by organizing it into training, validation, and testing sets. A pre-trained model (ResNet50V2) is used for feature extraction and classification.

Dataset

The dataset consists of Monkeypox skin images categorized into different classes. The dataset is split into:

Training Set (60%): Used to train the model.

Validation Set (20%): Used to tune hyperparameters.

Testing Set (20%): Used to evaluate model performance.

Directory Structure

Monkeypox Skin Image Dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── test/
│   ├── class_1/
│   ├── class_2/
│   └── ...

Prerequisites

Ensure you have the following dependencies installed:

pip install numpy pandas tensorflow tensorflow-hub keras plotly matplotlib scikit-learn

Code Breakdown

Data Preparation

The dataset is split into training (60%), validation (20%), and testing (20%).

Images are copied into respective directories.

Class distribution is visualized using a pie chart.

Model Setup

Uses ResNet50V2 for feature extraction.

A fully connected neural network is added on top for classification.

Implements EarlyStopping and ModelCheckpoint callbacks.

Running the Project

Place the dataset in the Monkeypox Skin Image Dataset/ directory.

Run the Python script to preprocess and split the data.

Train the model using the dataset.

Evaluate model performance using test data.

Visualization

A pie chart is plotted to visualize the distribution of different classes in the dataset.

License

This project is for educational and research purposes only.

Contributors

Saroja 

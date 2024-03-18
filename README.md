# Fake News Detector AI

This repository contains code for a Fake News Detector AI, designed to classify news articles as either real or fake using artificial intelligence techniques.

## Introduction
The main idea behind this project is exercising, and since Fake news has become a significant concern in recent times due to its potential to manipulate public opinion and spread misinformation. This project aims to address this issue by leveraging machine learning models to automatically identify fake news articles.

## Features

- Utilizes state-of-the-art natural language processing techniques.
- Get embeddings from BERT-based model for text classification.
- Provides functionalities for preprocessing textual data, including tokenization and embedding generation.
- Includes a Convolutional Neural Network (CNN) model for feature extraction.
- Offers scripts for model training, testing, and inference.
## Neural network description 

This function is responsible for constructing and compiling a neural network model architecture for fake news detection. It utilizes convolutional and dense layers to process both textual and title inputs, aiming to learn features that distinguish between real and fake news articles.

### Input Parameters:

- `titles`: Input tensor representing the titles of news articles. It has a shape of (768,) and data type of `tf.float32`.
- `text`: Input tensor representing the textual content of news articles. It has a shape of [16, 768] (indicating a maximum of 16 tokens per input) and data type of `tf.float32`. The `ragged` parameter is set to `False`.
  
### Model Architecture:

1. **Text Processing Layers:**
   - `Conv1D`: A one-dimensional convolutional layer with 256 filters, kernel size of 4, and ReLU activation function.
   - `MaxPooling1D`: Max pooling layer with pool size of 5.
   - `Flatten`: Flatten layer to convert the output tensor from the convolutional layer into a 1D tensor.

2. **Title Processing Layers:**
   - `Dense`: Fully connected layer with 128 units and ReLU activation function.

3. **Concatenation:**
   - The outputs from the text and title processing layers are concatenated along the feature axis (-1).

4. **Additional Dense Layers:**
   - Two dense layers with 256 and 128 units, respectively, followed by ReLU activation functions and dropout regularization (50% dropout rate).

5. **Output Layer:**
   - Final dense layer with 1 unit and sigmoid activation function, producing the probability of the input being fake news.

### Model Compilation:

- **Optimizer:** Adam optimizer is used for gradient descent optimization.
- **Loss Function:** Binary cross-entropy is used as the loss function, suitable for binary classification tasks.
- **Metrics:** Accuracy and crossentropy metrics are monitored during training to evaluate model performance.

### Model Summary:

- The summary of the constructed model, including the layer types, output shapes, and number of parameters, is displayed.

### Output:

- The function returns the compiled Keras model ready for training and inference.


## Getting Started

To use this project, follow these steps:

1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/your_username/fake-news-detector-ai.git
   
Download the dataset containing real and fake news articles.

Run the provided scripts to preprocess the data and train the models.

## Usage

To utilize this project effectively you need to have poetry installed first, then you can follow the guidelines below:

- `dataset_preprocessing.py`: This script contains functions responsible for reading and preprocessing the dataset. Ensure you run this script before training your model to prepare the data appropriately.

- `fake_news_detection.py`: This script is the core of the project and implements the main functionality for fake news detection. It encompasses model creation, training, testing, and inference. Execute this script to train your model, evaluate its performance, and make predictions on new data.

- `README.md`: This documentation provides an overview of the project and instructions for usage. You're currently viewing it!
  
  ```bash
  poetry update

## Contributors
Milad Kiwan

## Acknowledgments
Special thanks to Hugging Face for providing pre-trained BERT models.
Inspired by the need to combat misinformation in the digital age.


Feel free to use this README file for your project. Let me know if you need any further assistance!


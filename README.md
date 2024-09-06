# Project Title - Real-Time Facial Emotion Detection
This project is a Python-based application that uses OpenCV and Keras to detect emotions from facial expressions in real-time using a webcam.

## Overview
The project detects faces in real-time using a Haar Cascade Classifier and predicts emotions such as anger, happiness, sadness, and more through a pre-trained convolutional neural network (CNN). It utilizes deep learning models saved in .json format and their corresponding weights in .h5 format.

## Features
Real-time face detection using OpenCV.
Emotion recognition with a pre-trained deep learning model.
The system identifies seven emotions:
Angry
Disgust
Fear
Happy
Neutral
Sad
Surprise

## Project Files
realtimedetection.py: This file contains the main logic for capturing video input from the webcam, detecting faces, and recognizing the emotion of the detected faces.
trainmodel.ipynb: Jupyter notebook containing the training code for the emotion detection model. It includes data preprocessing, model architecture definition, and training the model on a facial emotion dataset.

## Dependencies
Make sure to install the following dependencies before running the project:
pip install opencv-python keras numpy

## Usage
Clone the repository:
Navigate to the project directory:
Ensure the model JSON and weights file (emotiondetector.json and emotiondetector.h5) are in the correct path as defined in the script. You may need to adjust the path to your local environment.
Run the script: python realtimedetection.py
The webcam will open, and the system will begin detecting faces and predicting their emotions.

## Model Training
To train the model from scratch, you can use the trainmodel.ipynb file. Ensure you have the necessary dataset and adjust the paths accordingly. The model used here is a simple CNN architecture trained on a dataset of labeled facial emotions.

## Future Improvements
Add support for more emotions.
Improve the model accuracy with more data or by fine-tuning the architecture.
Create a graphical user interface (GUI) for easier interaction.

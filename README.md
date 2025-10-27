Distracted Driving Detection using GAT–BiLSTM (Graph Attention Network + BiLSTM)
Overview

This project presents a spatio-temporal deep learning model for detecting distracted driving behavior using Graph Attention Networks (GAT) integrated with Bidirectional LSTMs (BiLSTM).
The model effectively captures spatial relationships among frame-level features and temporal dependencies across video sequences, achieving a classification accuracy of 98.45% on six distraction classes.

The system is designed for applications in driver safety monitoring, real-time behavior analysis, and intelligent transportation systems.

Objectives

Detect and classify driver distraction behaviors from in-vehicle video data.

Utilize Graph Attention Networks (GAT) for spatial feature correlation learning.

Incorporate BiLSTM for modeling temporal sequences of driver activities.

Develop a framework suitable for real-time deployment on embedded systems.

Model Architecture
Components:

Feature Extractor: Pretrained CNN backbone (VGG16, InceptionV3, or DenseNet121)

Spatial Modeling: Multi-head Graph Attention Network (GAT)

Temporal Modeling: Bidirectional LSTM layers

Classifier: Fully Connected + Softmax output layer

Distraction Classes

The model classifies input driving video into six behavior categories:

Normal driving

Billboard viewing

Wildlife or side-road viewing

Electronics item usage

Rubbernecking

Passenger interaction

Dataset and Preprocessing

Video data is converted into frames and processed using a pretrained CNN to obtain high-level feature embeddings.

Each video sequence is represented as a spatio-temporal graph where:

Nodes represent frame-level features

Edges represent temporal adjacency and correlation between frames

Data augmentation applied:

Rainy (with slight contrast), yellow tint, blue tint, orange tint, and rotation variants.

Dataset split: 80% training, 10% validation, 10% testing.

Training Configuration
Parameter	Value
Optimizer	Adam
Learning Rate	0.0001
Batch Size	16
Epochs	50
Loss Function	Cross Entropy
Dropout	0.3
Results
Metric	Value
Accuracy	98.45%
Precision	97.9%
Recall	98.2%
F1-Score	98.05%

Visualization

Training and validation accuracy/loss plots

Confusion matrix across all six distraction classes

Attention heatmaps highlighting the driver’s focus regions

Future Work

Extend to multi-driver and multi-camera datasets.

Deploy on embedded hardware such as Raspberry Pi or Jetson Nano.

Integrate temporal self-attention for finer behavior segmentation.


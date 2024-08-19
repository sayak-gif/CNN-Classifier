# CNN-Classifier

A Convolutional Neural Network (CNN) classifier designed for brain tumor classification using the ResNet-50 model is an advanced deep learning approach tailored to accurately identify and categorize brain tumors from medical imaging data, such as MRI scans. ResNet-50, a 50-layer deep residual network, is particularly well-suited for this task due to its ability to mitigate the vanishing gradient problem that often hampers the performance of very deep networks.

# Best Trained Model:

*Link of the Model* - 
[*MODEL*](https://drive.google.com/file/d/1ggAWl4ab4pJQYaHJ6elvtT_ZR0dQzD20/view?usp=drive_link)
*Our best model returned* - 
Training accuracy of 99.69% and loss of 0.0049.
Validation accuracy of 95.70% and loss of 0.1665.
Testing accuracy of 94.62% and loss of 0.3212.


# Key Features:

1. *ResNet-50 Architecture*: 
   - ResNet-50 is composed of multiple convolutional layers, batch normalization, ReLU activations, and identity or residual connections, which allow the model to learn complex features and patterns from the input images while maintaining high efficiency in training and inference.
   - The residual connections are crucial as they enable the network to learn incremental modifications, improving its ability to generalize across different tumor types and sizes.

2. *Preprocessing and Data Augmentation*:
   - The classifier typically starts with preprocessing steps, including resizing images to match the input dimensions required by ResNet-50 (224x224 pixels), normalization to standardize pixel intensity values, and data augmentation techniques such as rotations, flips, and zooming. These augmentations help the model become more robust to variations in the input data.

3. *Feature Extraction and Classification*:
   - Once the MRI images are fed into the ResNet-50 model, it extracts hierarchical features through its convolutional layers. These features range from simple edge detectors in the initial layers to complex tumor-specific patterns in the deeper layers.
   - The final output of the ResNet-50 model is typically a feature map that is passed through fully connected layers or a global average pooling layer, followed by a softmax activation function to produce probabilities for each tumor class (e.g., benign, malignant, or normal).

4. *Transfer Learning*:
   - Often, the ResNet-50 model is pre-trained on large datasets like ImageNet and then fine-tuned on the specific brain tumor dataset. This approach leverages the pre-learned general features from a vast dataset, allowing the model to converge faster and achieve higher accuracy with the relatively smaller medical dataset.

5. *Performance Metrics*:
   - The classifier’s performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and the area under the ROC curve (AUC-ROC). These metrics provide a comprehensive view of the model’s ability to correctly classify brain tumors, taking into account both false positives and false negatives.

6. *Applications*:
   - This CNN classifier is invaluable in medical diagnostics, providing radiologists and medical professionals with a powerful tool to assist in the early detection and classification of brain tumors, potentially leading to more effective treatment planning and improved patient outcomes.

# Conclusion:
Using ResNet-50 for brain tumor classification combines the strengths of deep residual networks with domain-specific adaptations to achieve high accuracy and reliability in medical image analysis. This approach not only enhances the accuracy of tumor classification but also offers a scalable and efficient solution for handling large volumes of medical imaging data in clinical settings.

# Dataset:
[*DATASET*](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

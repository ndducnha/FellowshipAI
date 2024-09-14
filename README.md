
# Flower Classification Using ResNet-50

This project tackles the **Computer Vision (CV) Challenge** by leveraging transfer learning with a pre-trained ResNet-50 model to classify images of flowers from the Oxford 102 Flower Category Dataset. The dataset consists of images belonging to 102 different flower categories, providing a comprehensive classification challenge.

## Author Information

- **Full Name**: Dinh Duc Nha Nguyen (Tony)
- **Date**: 14/09/2024
- **Email**: ndducnha@gmail.com

## Project Overview

- **Objective**: Use a pre-trained ResNet-50 model to classify flowers into 102 categories.
- **Dataset**: The Oxford 102 Flower Category Dataset, which includes 8,189 images labeled into 102 flower classes.

## Solution Approach

### 1. Dataset Preparation
- **Download and Extract**: The dataset includes flower images (`102flowers.tgz`), labels (`imagelabels.mat`), and data splits (`setid.mat`).
- **Organize Data**: Images are organized into training, validation, and test sets based on predefined splits.

### 2. Data Augmentation
- **Techniques Used**: 
  - Rescaling images to normalize pixel values.
  - Random horizontal flipping, zooming, and shearing to increase data variability and improve model generalization.

### 3. Model Architecture
- **Base Model**: ResNet-50 pre-trained on ImageNet, excluding the top classification layers to use it as a feature extractor.
- **Custom Layers**: Added global average pooling and dense layers for classification into 102 flower categories.

### 4. Training Process
- **Optimizer**: Adam optimizer for adaptive learning rate.
- **Loss Function**: Categorical crossentropy for multi-class classification.
- **Metrics**: Accuracy to measure model performance.
- **Training**: Trained for 10 epochs with data augmentation to enhance model robustness.

## Implementation

The main code can be found in the `flower_classification.py` script. The code follows these steps:

1. **Download and Extract the Dataset**: Automatically downloads and prepares the dataset.
2. **Load and Organize Data**: Images are arranged into labeled folders for training, validation, and testing.
3. **Define and Compile Model**: The ResNet-50 model is loaded with custom layers added and compiled with appropriate settings.
4. **Train the Model**: The model is trained using augmented data to maximize generalization performance.
5. **Save the Model**: After training, the model is saved as `resnet50_flowers_model.h5`.

### Prerequisites

Ensure the following libraries are installed:

- Python 3.x
- TensorFlow
- Keras
- SciPy
- Matplotlib (for plotting results, if needed)

Install the required packages using:

```bash
pip install tensorflow keras scipy matplotlib
```

### Running the Code

To run the code, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/flower-classification-resnet50.git
    cd flower-classification-resnet50
    ```

2. Run the script:

    ```bash
    python flower_classification.py
    ```

This will download the dataset, train the model, and save the trained model.


## Acknowledgements

- The dataset used in this project is from the Oxford 102 Flower Category Dataset: [Oxford 102 Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/).

## Contact

For any questions or suggestions, please contact:

- **Dinh Duc Nha Nguyen (Tony)**
- **Email**: ndducnha@gmail.com

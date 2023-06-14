# Forest_Fire_Detection
  This project aims to detect forest fires. The code provided demonstrates the implementation of a Convolutional Neural Network (CNN) model to classify images as either "Fire" or "No Fire". The project utilizes the TensorFlow and Keras libraries for building and training the model.
  
## Dependencies

The project requires the following dependencies:

- TensorFlow
- NumPy
- OpenCV (cv2)
- Matplotlib

Make sure to install these dependencies before running the code.

## Dataset

The dataset for this project should be organized in the following structure:

```
forest_fire_dataset/
    ├── Training and Validation/
    │   ├── fire/
    │   ├── no_fire/
    ├── Testing/
    │   ├── fire/
    │   ├── no_fire/
```

The dataset should contain separate folders for training and validation images, as well as testing images. Within each folder, the images should be organized into two subfolders: "fire" and "no_fire", representing the respective classes.

## Data Preprocessing

The script utilizes the `ImageDataGenerator` class from Keras to preprocess the image data. The images are rescaled by dividing the pixel values by 255 to normalize them between 0 and 1. The `flow_from_directory` method is used to load and preprocess the training and testing datasets. The images are resized to a target size of (150, 150) and grouped into batches.

## Model Architecture

The CNN model architecture is defined using the Keras Sequential API. The model consists of several convolutional layers followed by max-pooling layers to extract features from the images. The flattened feature maps are then passed through dense layers for classification. The final output layer uses a sigmoid activation function to predict the probability of fire.

## Training the Model

The model is trained using the `fit` method, which takes the training dataset as input. The training progress is recorded, and the loss and accuracy curves are plotted using Matplotlib.

## Prediction

The `predimg` function is defined to predict the class of an input image. The function loads the image, preprocesses it, and passes it through the trained model for prediction. The predicted value is then rounded to the nearest integer. Finally, the function displays the image and labels it as "Fire" or "No Fire" based on the prediction.

## Usage

To use this project, follow these steps:

1. Ensure that the dataset is organized as described above.
2. Install the required dependencies.
3. Run the script.

## Results

After running the script, the model will make predictions on the testing dataset. The predicted values will be rounded to either 0 or 1, representing "No Fire" and "Fire" classes, respectively. Additionally, loss and accuracy plots will be displayed to visualize the training progress.

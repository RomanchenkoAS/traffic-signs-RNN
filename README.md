# Pattern Recognition RNN for Traffic Signs (GTSRB)
## Description
This repository contains a Python program for recognizing traffic signs using a Recurrent Neural Network (RNN). The dataset used is the German Traffic Sign Recognition Benchmark (GTSRB). This was developed during me taking CS50AI course as one of the assignments.

## Getting Started
### Prerequisites
- Python 3.x
- TensorFlow
- OpenCV (cv2)
- scikit-learn
- NumPy

You can install the required packages using pip:
```bash
pip install tensorflow opencv-python scikit-learn numpy 
```

or 

```bash
pip -r requirements.txt
```


### Dataset
Download the GTSRB dataset from here and unzip it.

## Usage
### Training the Model
- Navigate to the directory containing the main.py file.
- Run the following command:
```bash
python main.py path/to/GTSRB/dataset 
```
Replace path/to/GTSRB/dataset with the directory path where you unzipped the GTSRB dataset.

If you want to save the model after training, run the following command:

```bash
python main.py path/to/GTSRB/dataset model.h5
```
This will save the trained model as model.h5 in the current directory.

### Code Overview
- load_data(): Reads the image data and labels from the specified directory.
- get_model(): Defines and compiles the RNN model using TensorFlow.
- main(): The main function that controls the flow of the program.
### Constants
- EPOCHS = 10: Number of training epochs.
- IMG_WIDTH = 30, IMG_HEIGHT = 30: Dimensions to which all images will be resized.
- NUM_CATEGORIES = 43: Number of different signs/categories in the dataset.
- TEST_SIZE = 0.4: Fraction of data to be used for testing.

## Contributing
Feel free to open issues or pull requests with improvements or corrections.

## Acknowledgments
Thanks to the creators of the GTSRB dataset and tensorflow developers.
Inspired by the CS50 AI course.




# Digit Recognition Using Perceptron
  A simple implementaion of a autograd based perceptron library and a simple working model which will predict a speicfic digit from the MNIST dataset
 
# How it works
 * Data Loading: MNIST dataset is loaded and preprocessed. Images are normalized, and labels are converted to binary (target digit vs. others).
 
 * Model Training: The Perceptron is trained on the processed MNIST data using mini-batch gradient descent.
 Loss and accuracy are logged and displayed for each epoch.
 
 * Model Saving/Loading: After training, the model parameters (weights and bias) are saved to a file.
 The model can be reloaded for future use without retraining.
 
 * Model Testing: The model can be tested interactively by selecting an image index.
 The predicted label is displayed, and the corresponding image is shown.
 The learned weights are visualized as a 28x28 image grid.

https://github.com/user-attachments/assets/c971e0eb-151c-4103-a026-1dcad10e2241

 In the video shown above the selected target digit i have set is 7, thus the perceptron predicts the index at which digit 7 is present.
# Note
 The code works on binary classification tasks and can be adapted to other datasets or target digits.

# Dependencies
Python libraries: numpy, sklearn, matplotlib, tqdm

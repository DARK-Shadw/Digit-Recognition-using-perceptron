import json
import os
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random, math


class Node:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.label = label
        self.grad = 0.0
        self._backward = self._def_back
        self._op = _op
        self._prev = set(_children)

    def _def_back(self):
        pass

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        output = Node(self.data + other.data, (self, other), '+')
        def _backward():
          self.grad += output.grad
          other.grad += output.grad
        output._backward = _backward
        return output

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        output = Node(self.data * other.data, (self, other), '*', label="")
        def _backward():
          self.grad += (output.grad * other.data)
          other.grad += (output.grad * self.data)

        output._backward = _backward
        return output

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        output = Node(self.data**other, (self,), f'**{other}', label="")

        def _backward():
            self.grad += (other * self.data**(other-1)) * output.grad
        output._backward = _backward

        return output

    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __rsub__(self, other): # other - self
        return other + (-self)

    def __radd__(self, other): # other + self
        return self + other

    def __rmul__(self, other):
       return self * other
    
    def __neg__(self): # -self
        return self * -1
    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def exp(self):
        # Calculate e^(self.data)
        output = Node(math.exp(self.data), (self,), "exp")
        
        def _backward():
            # Derivative of e^x is e^x
            self.grad += output.data * output.grad
        
        output._backward = _backward
        return output

    def sigmoid(self):
        # Calculate sigmoid = 1 / (1 + exp(-self.data))
        exp_neg_self = (-self).exp()
        output = Node(1 / (1 + exp_neg_self.data), (self,), "sigmoid")
        
        def _backward():
            # Derivative of sigmoid is σ(x) * (1 - σ(x))
            sigmoid_grad = output.data * (1 - output.data)
            self.grad += sigmoid_grad * output.grad
        
        output._backward = _backward
        return output

    def relu(self):
        output = Node(0 if self.data < 0 else self.data, (self, ), "relu")
        def _backward():
            self.grad += (output.data > 0) * output.grad
        output._backward = _backward
        return output

    def backward(self):
      topo = []
      visited = set()
      def build_topo(v):
        if v not in visited:
          visited.add(v)
          for child in v._prev:
            build_topo(child)
          topo.append(v)

      build_topo(self)
      self.grad = 1.0

      for v in reversed(topo):
        v._backward()

    def __repr__(self):
        return f"{self.data}"


def test_model_on_image(model, image, choice, X_test, target_digit=7):
    # Flatten the image to match the input size
    image_flat = image.flatten() / 255.0  # Normalize it to 0-1
    
    # Get the prediction
    prediction = model.predict_single(X_test[choice])
    print(f"Prediction for choice {choice} = {prediction}")
    # Display the image
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Prediction: {'Target Digit' if prediction == 1 else 'Not Target Digit'}")
    plt.show()
    
    if prediction == 1:
        print(f"The image is classified as the target digit {target_digit}.")
    else:
        print(f"The image is not classified as the target digit {target_digit}.")


def choose_image(X_test):
    while True:
        # Ask the user to pick an image index
        try:
            image_idx = int(input(f"Enter an image index (0 to {len(X_test) - 1}): "))
            if image_idx == -1:
                return None, image_idx
            if image_idx < 0 or image_idx >= len(X_test):
                print(f"Please choose a number between 0 and {len(X_test) - 1}.")
                continue
            return X_test[image_idx], image_idx
        except ValueError:
            print("Please enter a valid number.")



def load_mnist_binary(target_digit=7):
    print("Loading MNIST dataset...")
    # Load MNIST dataset
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Normalize pixel values and take a subset for faster training
    X = X[:10000] / 255.0
    y = y[:10000].astype(int)
    
    # Convert to binary classification
    y = (y == target_digit).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

class MNISTPerceptron:
    def __init__(self, input_size=784):
        self.input_size = input_size
        # Initialize with smaller weights for better training
        self.weights = [Node(random.uniform(-0.1, 0.1)) for _ in range(input_size)]
        self.bias = Node(random.uniform(-0.1, 0.1))
        self.history = {'loss': [], 'accuracy': []}

    def forward(self, x):
        # Convert input to list of Nodes if necessary
        x_nodes = [Node(xi) for xi in x]
        weighted_sum = sum(xi * wi for xi, wi in zip(x_nodes, self.weights)) + self.bias
        return weighted_sum.sigmoid()

    def parameters(self):
        return self.weights + [self.bias]

    def train_batch(self, X_batch, y_batch, learning_rate):
        batch_size = len(X_batch)
        batch_loss = Node(0.0)

        # Forward pass and loss computation for batch
        for x, y in zip(X_batch, y_batch):
            y_pred = self.forward(x)
            # Using MSE loss since it's already implemented in your library
            batch_loss = batch_loss + (y_pred - y)**2

        # Average loss for the batch
        batch_loss = batch_loss * (1.0 / batch_size)

        # Zero gradients
        for p in self.parameters():
            p.grad = 0

        # Backward pass
        batch_loss.backward()

        # Update parameters
        for p in self.parameters():
            p.data += -learning_rate * p.grad

        return batch_loss.data
    
    def predict_single(self, x):
        return 1 if self.forward(x).data > 0.5 else 0

    def predict(self, X, verbose=False):
        predictions = []
        i = 0
        for x in X:
            pred = self.forward(x).data
            pred = 1 if pred > 0.5 else 0
            if pred == 1 and verbose:
                print(f"The {i}'th Sample was Successfully Found to be 7!!")
            predictions.append(pred)
            i += 1
        return predictions

    def train(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32, learning_rate=0.1):
        n_samples = len(X_train)
        n_batches = n_samples // batch_size
        
        print(f"Training on {n_samples} samples with batch size {batch_size}")
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            epoch_loss = 0
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train in batches
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                batch_loss = self.train_batch(X_batch, y_batch, learning_rate)
                epoch_loss += batch_loss
                
                if i % 10 == 0:
                    print(f"Batch {i}/{n_batches}, Loss: {batch_loss:.4f}")
            
            # Compute validation accuracy
            val_pred = self.predict(X_val)
            val_acc = sum(p == y for p, y in zip(val_pred, y_val)) / len(y_val)
            
            epoch_loss /= n_batches
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(val_acc)
            
            print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history['loss'])
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        
        ax2.plot(self.history['accuracy'])
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        
        plt.tight_layout()
        plt.show()

    def visualize_weights(self):
        # Convert weights to numpy array
        weight_array = np.array([w.data for w in self.weights])
        # Reshape weights to 28x28 image
        w_img = weight_array.reshape(28, 28)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(w_img, cmap='viridis')
        plt.colorbar()
        plt.title('Learned weights')
        plt.show()

def run_experiment(target_digit=7):
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_mnist_binary(target_digit)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create and train model
    model = MNISTPerceptron()
    model.train(X_train, y_train, X_test, y_test, 
                epochs=5, batch_size=32, learning_rate=0.01)
    
    # Visualize results
    model.plot_training_history()
    model.visualize_weights()
    
    # Final test accuracy
    test_pred = model.predict(X_test, verbose=True)
    test_acc = sum(p == y for p, y in zip(test_pred, y_test)) / len(y_test)
    print(f"\nFinal test accuracy: {test_acc:.4f}")
    
    return model, X_test


def save_model(model, filename):
    params = {
        'weights': [w.data for w in model.weights],
        'bias': model.bias.data
    }
    with open(filename, 'w') as f:
        json.dump(params, f)

def load_model(filename):
    model = MNISTPerceptron()
    with open(filename, 'r') as f:
        params = json.load(f)

    for w, saved_w in zip(model.weights, params['weights']):
        w.data = saved_w
    model.bias.data = params['bias']

    return model




def main():
    model_filename = "Models/perceptron_model.saved"
    target_digit = 7
    model = None
    X_test = None

    if not os.path.exists(model_filename):
        # Train the model
        model, X_test = run_experiment(target_digit)
        save_model(model, model_filename)
    else:
        X_train, X_test, y_train, y_test = load_mnist_binary(target_digit)
        model = load_model(model_filename)

    while True:
        # print(X_test[0])
        # Choose an image from the test set
        new_image, choice = choose_image(X_test)

        if choice == -1:
            break
        
        # # Test the model on the selected image
        test_model_on_image(model, new_image, choice, X_test, target_digit)
        

if __name__ == "__main__":
    main()


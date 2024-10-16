import matplotlib.pyplot as plt
import time
from MLPwDim_Recduction_RBM import MLP
from preprocess_RBM import *
#? so sánh 3 mô hình : 1.MLP normal  2.MLP with dim_rec RBM 3.MLP with dim_rec RBMsklearn
import matplotlib.pyplot as plt
import time

# >Function to train and evaluate a model
def train_and_evaluate(model, train_data, train_labels, test_data, test_labels, batch_size=64, epochs=10):
    start_time = time.time()

    # Train the model
    history = model.model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=0)

    # Measure time elapsed
    elapsed_time = time.time() - start_time

    # Evaluate the model
    loss, accuracy = model.model.evaluate(test_data, test_labels, verbose=0)

    return accuracy, elapsed_time, history

#> Function to plot results for multiple models
def plot_results(results):
    plt.figure(figsize=(18, 6))

    # Plot Loss for each model
    plt.subplot(1, 2, 1)
    for label, result in results.items():
        plt.plot(result['history'].history['loss'], label=f'{label} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()

    # Plot Accuracy for each model
    plt.subplot(1, 2, 2)
    for label, result in results.items():
        plt.plot(result['history'].history['accuracy'], label=f'{label} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print final accuracy and time for each model
    for label, result in results.items():
        print(f"{label} Accuracy: {result['accuracy']:.4f}, Time: {result['time']:.2f} seconds")

#% Main comparison function for different MLP models with dimensionality reduction
if __name__ == "__main__":
    #>Load data
    x_train, y_train, x_test, y_test = load_mnist_data()

    #> Define the models and their reduction techniques
    models = {
        'MLP Normal': {
            'model': MLP(input_dim=784, hidden_dims=[128, 64], output_dim=10),
            'preprocess': lambda data: data  # No dimensionality reduction
        },
        'MLP + RBM': {
            'model': MLP(input_dim=128, hidden_dims=[128, 64], output_dim=10),
            'preprocess': preprocess_data_with_rbm  # Dimensionality reduction using custom RBM
        },
        'MLP + RBM (sklearn)': {
            'model': MLP(input_dim=128, hidden_dims=[128, 64], output_dim=10),
            'preprocess': preprocess_data_with_rbmsklearn  # Dimensionality reduction using sklearn's RBM
        }
        # Additional models can be easily added here
    }

    results = {}

    #> Train, evaluate, and collect results for each model
    for label, info in models.items():
        reduced_train_data = info['preprocess'](x_train)
        reduced_test_data = info['preprocess'](x_test)
        accuracy, elapsed_time, history = train_and_evaluate(info['model'], reduced_train_data, y_train, reduced_test_data, y_test)
        
        # Store the results
        results[label] = {
            'accuracy': accuracy,
            'time': elapsed_time,
            'history': history
        }

    # Plot and compare the results
    plot_results(results)

import os
import pandas as pd
import tensorflow as tf
import numpy as np


def high_outliers_indices(df: pd.DataFrame):
    upper_limit_dict = {'fixed acidity': 14.5,
                        'volatile acidity': 1.15,
                        'citric acid': 0.8,
                        'residual sugar': 10,
                        'chlorides': 0.5,
                        'free sulfur dioxide': 60,
                        'total sulfur dioxide': 200,
                        'density': None,
                        'pH': 3.8,
                        'sulphates': 1.4,
                        'alcohol': 14.5}
    ans = []
    for key, value in upper_limit_dict.items():
        indices = df[df[key] > value].index
        for i in indices:
            if i not in ans:
                ans.append(i)
    return ans


def main():
    # import the data
    os.chdir(r"../data")
    wine_df = pd.read_csv("winequality-red.csv", delimiter=';')

    # remove high outliers
    wine_df.drop(high_outliers_indices(wine_df), inplace=True)

    # Prepare the data
    my_input_data = wine_df.iloc[:, :11]  # Example input data (100 samples, 11 features)
    my_output_data = wine_df.iloc[:, 11]  # Example output data (binary labels)

    # Normalize the input data (optional)
    # input_data = (input_data - np.mean(input_data)) / np.std(input_data)


def machine_learn(input_data, output_data):
    # Split the data into training and testing sets
    train_ratio = 0.8  # 80% for training, 20% for testing
    train_samples = int(train_ratio * input_data.shape[0])
    train_input = input_data[:train_samples]
    train_output = output_data[:train_samples]
    test_input = input_data[train_samples:]
    test_output = output_data[train_samples:]

    # # Define the network architecture
    input_dim = 9
    # output_dim = 1
    #
    # inputs = tf.keras.Input(shape=(input_dim,), name='inputs')

    # will this work?
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(9, activation='relu', input_shape=(input_dim,)),  # Hidden layer with 9 neurons
    #     tf.keras.layers.Dense(output_dim, activation='softmax')])  # Output

    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(input_dim,), name='inputs'),
        tf.keras.layers.Dense(9, activation='relu', input_shape=(9,)),  # Hidden layer with 9 neurons
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Define the output operation
    # output = tf.keras.layers.Dense(output_dim)(inputs)
    #
    # # Create the model
    # model = tf.keras.Model(inputs=inputs, outputs=output)
    #
    # # Compile the model
    # model.compile(optimizer='adam', loss='mse')

    # Print the model summary
    model.summary()

    # Train the model
    epochs = 1000
    model.fit(train_input, train_output, epochs=epochs, verbose=1)

    # Evaluate the model
    test_loss = model.evaluate(test_input, test_output, verbose=0)
    print("Testing Loss:", test_loss)

    # Get the final weights and biases
    # weights = model.layers[1].get_weights()[0]  # Assuming there is only one layer
    # biases = model.layers[1].get_weights()[1]  # Assuming there is only one layer
    #
    # print("Final Weights:")
    # print(weights)
    # print("Final Biases:")
    # print(biases)

    for layer in model.layers:
        weights, biases = layer.get_weights()
        print("Layer Weights:")
        print(weights)
        print("Layer Biases:")
        print(biases)


if __name__ == "__main__":
    machine_learn(my_input_data, my_output_data)


"""
First 100: 
Testing Loss: 0.4394053518772125
Final Weights:
[[ 0.03312918]
 [-0.9286719 ]
 [-0.11080444]
 [ 0.00932213]
 [-1.1462734 ]
 [ 0.00248567]
 [-0.00274192]
 [ 0.6426    ]
 [ 0.16413184]
 [ 1.1680039 ]
 [ 0.3099082 ]]
Final Biases:
[0.92430377]

All: Unnormalized 
Testing Loss: 0.4509418308734894
Final Weights:
[[ 0.05973548]
 [-1.027153  ]
 [-0.2547749 ]
 [ 0.0045553 ]
 [-0.93602085]
 [ 0.00216343]
 [-0.00248034]
 [ 0.32539114]
 [ 0.35694644]
 [ 1.0199566 ]
 [ 0.3231147 ]]
Final Biases:
[0.4254112]

normalized and dimension reduced
Testing Loss: 0.43485674262046814
Final Weights:
[[-0.05233626]
 [ 0.2821826 ]
 [-0.19712853]
 [ 0.02563123]
 [-0.0879453 ]
 [-0.01019083]
 [-0.09990207]
 [ 0.068753  ]
 [ 0.12720628]]
Final Biases:
[5.6673555]

Earlier you wrote a tensorflow script for me
how do i add a hidden layor of 9 neurons?

"""

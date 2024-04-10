import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv(
    "C:/Users/Usuario/Desktop/TFG/DatosGenes/DatosTratadosTodos.csv",  delimiter=';',
    names=["code_column", "Probability"])

# Convert the code column to string
data['code_column'] = data['code_column'].astype(str)

# Split the code column into separate digits
data['code_digits'] = data['code_column'].apply(lambda x: [int(digit) if digit.isdigit() else 0 for digit in x])

# Create a DataFrame from the code_digits Series
code_digits_df = pd.DataFrame(data['code_digits'].tolist(), columns=[f'digit_{i}' for i in range(len(data['code_digits'].iloc[0]))])

# Concatenate the original DataFrame with the code_digits DataFrame
data_encoded = pd.concat([data.drop(columns=['code_column', 'code_digits']), code_digits_df], axis=1)

#Check for NAN values
nan_counts = data_encoded.isna().sum()

while nan_counts.sum() != 0:
    nan_counts = data_encoded.isna().sum()
    # Check if any NaN values exist in the DataFrame
    if nan_counts.sum() == 0:
        print("No NaN values found in the DataFrame.")
    else:
        print("%d NaN values found in the DataFrame. Changing them for 0" % nan_counts.sum())
        data_encoded.fillna(0, inplace=True)


# Convert the DataFrame to a NumPy array
data_features = data_encoded.to_numpy()

# 'X' contains the digit columns and 'y' contains the 'Probability' column
prob_label = data_features[:, 0]   # Select all rows and the first column
code_features = data_features[:, 1:]

train_features, test_features, train_labels, test_labels = train_test_split(code_features, prob_label, test_size=0.1, random_state=42)

normalize = layers.Normalization()

model = tf.keras.Sequential([
    normalize,
    layers.Dense(32, activation='relu', input_shape=(code_features.shape[1],)),
    layers.Dense(64, activation='sigmoid'),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(1)
])


my_callbacks = [
    EarlyStopping(monitor="val_loss", patience=100),
]

model.compile(loss = tf.losses.MeanSquaredLogarithmicError(),
                      optimizer = tf.optimizers.Adam(learning_rate=1e-3))

model.fit(train_features, train_labels, batch_size=30, epochs=1100, validation_split=0.15)

# Define the directory where you want to save the model
base_directory = 'C:/Users/Usuario/Desktop/TFG/ProyectoTFG/Modelos/'

# Change the current working directory to the base directory
os.chdir(base_directory)

# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

# Create a new directory using the formatted datetime string
new_directory = os.path.join(base_directory, dt_string)
os.makedirs(new_directory)

# Save the model inside the new directory
model.save(os.path.join(new_directory, 'Modelo.h5'))

# Evaluate the model on the test set
test_loss = model.evaluate(test_features, test_labels)
print("Test Loss:", test_loss)

# Make predictions on the test set
predictions = model.predict(test_features)

print("###############################################################")
print('Valor actual %f' % test_labels[263])
print("\n\n\n")
print("Valor predicho %f" % predictions[263])
print("###############################################################")

# Plot actual vs predicted probability
plt.figure(figsize=(10, 6))
plt.scatter(test_labels, predictions, color='blue', label='Predictions')
plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'k--', lw=3, color='red', label='Actual')
plt.xlabel('Actual probability')
plt.ylabel('Predicted probability')
plt.title('Actual vs Predicted probability')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(new_directory, 'plot_image.png'))
plt.show()

# Save variables to a text file
with open(os.path.join(new_directory, 'variables.txt'), 'w') as file:
    file.write(f"Test loss: {test_loss}\n")
    file.write(f"Valor real: {test_labels[263]}\n")
    file.write(f"Valor predicho: {predictions[263]}\n")
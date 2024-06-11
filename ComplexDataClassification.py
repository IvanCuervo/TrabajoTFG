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
from tensorflow.keras.callbacks import History
from keras import backend as K


headNames = ["aa_pos", "provean_score", "sift_score", "evm_epistatic_score", "integrated_fitCons_score",
             "LRT_score", "GERP++_RS", "phyloP30way_mammalian", "phastCons30way_mammalian", "SiPhy_29way_logOdds",
             "blosum100", "in_domain", "asa_mean", "aa_psipred_E", "aa_psipred_H", "aa_psipred_C", "bsa_max",
             "h_bond_max", "salt_bridge_max", "disulfide_bond_max", "covalent_bond_max", "solv_ne_abs_max", "mw_delta",
             "pka_delta", "pkb_delta", "pi_delta", "hi_delta", "pbr_delta", "avbr_delta", "vadw_delta", "asa_delta",
             "cyclic_delta", "charge_delta", "positive_delta", "negative_delta", "hydrophobic_delta", "polar_delta",
             "ionizable_delta", "aromatic_delta", "aliphatic_delta", "hbond_delta", "sulfur_delta", "essential_delta",
             "size_delta", "weight", "aa_ref_A", "aa_ref_C", "aa_ref_D", "aa_ref_E", "aa_ref_F", "aa_ref_G", "aa_ref_H",
             "aa_ref_I", "aa_ref_K", "aa_ref_L", "aa_ref_M", "aa_ref_N", "aa_ref_P", "aa_ref_Q", "aa_ref_R", "aa_ref_S",
             "aa_ref_T", "aa_ref_V", "aa_ref_W", "aa_ref_Y", "aa_alt_A", "aa_alt_C", "aa_alt_D", "aa_alt_E", "aa_alt_F",
             "aa_alt_G", "aa_alt_H", "aa_alt_I", "aa_alt_K", "aa_alt_L", "aa_alt_M", "aa_alt_N", "aa_alt_P", "aa_alt_Q",
             "aa_alt_R", "aa_alt_S", "aa_alt_T", "aa_alt_V", "aa_alt_W", "aa_alt_Y"]

data = pd.read_csv(
    "/home/ivan/TrabajoTFG/DatosGenes/VarityR/Classification/VarityRRegressionCSV.csv", delimiter=';', names=headNames
)


# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

# Change integer to floats
for name in headNames:
    if data[name].dtype == 'int64':
        data[name] = data[name].astype(float)

# Convert the DataFrame to a NumPy array
data_features = data.to_numpy()

# Convert label array to binary (0 or 1) based on the percentage
prob_label = data_features[:, 44]

# Convert feature array to a supported numeric data type (e.g., float32)
code_features = np.delete(data_features, 44, axis=1)

# Split data into train, test, and validation sets
train_features, test_features, train_labels, test_labels = train_test_split(code_features, prob_label, test_size=0.1, random_state=42)
train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.1, random_state=42)

train_labels = train_labels.astype(int)
val_labels = val_labels.astype(int)
test_labels = test_labels.astype(int)

# Normalize the data
normalize = tf.keras.layers.experimental.preprocessing.Normalization()
"""
model = tf.keras.Sequential([
    normalize,
    layers.Dense(64, activation='relu', input_shape=(code_features.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
"""

model = tf.keras.Sequential()
model.add(normalize)
model.add(layers.Dense(128, activation='relu', input_shape=(code_features.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Hyperparameters
my_batch_size = 128
my_epoch = 800
my_learning_rate = 1e-4
my_patience = 20

my_callbacks = [
    EarlyStopping(monitor="val_loss", patience=my_patience),
]

model.compile(loss=tf.losses.BinaryCrossentropy(),
              optimizer=tf.optimizers.Adam(learning_rate=my_learning_rate),
              metrics=['Accuracy'])

history = model.fit(train_features, train_labels, batch_size=my_batch_size, epochs=my_epoch, validation_data=(val_features, val_labels),
                    callbacks=my_callbacks)

# Define the directory where you want to save the model
base_directory = '/home/ivan/TrabajoTFG/Modelos/Mayo/Classification/FCNN/Junio'

# Change the current working directory to the base directory
os.chdir(base_directory)

# Create a new directory using the formatted datetime string
new_directory = os.path.join(base_directory, dt_string)
os.makedirs(new_directory)

# Save the model inside the new directory
model.save(os.path.join(new_directory, 'Modelo.h5'))

"""
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_features, test_labels)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
"""

# Make predictions on the test set
predictions = model.predict(test_features)

# Convert predictions to binary (0 or 1)
predictions_binary = (predictions > 0.5).astype(int)

predictions_array = predictions_binary.flatten()

#Baselines function
def mean_baseline(train_data, test_data):
    # Calculate the mean and round to get 0 or 1
    mean_value = round(np.mean(train_data))
    # Predict the mean value for all test data
    predictions = np.full_like(test_data, mean_value)
    return predictions




# Calculate baseline metrics
y_pred_b_avg = mean_baseline(train_labels, test_labels)  # Baseline media

# Define function to calculate accuracy
def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

accuracy_avg = accuracy(test_labels, y_pred_b_avg)
accuracy_model = accuracy(test_labels, predictions_array)


print(f"Modelo             | {'Accuracy':6s} |")
print("-"*47)
print(f"Baseline media     | {accuracy_avg:0.4f} |")
print(f"Mi modelo          | {accuracy_model:0.4f} |")


training_loss = history.history['loss']
validation_loss = history.history['val_loss']
training_accuracy = history.history['Accuracy']
validation_accuracy = history.history['val_Accuracy']

epochs = range(1, len(training_loss) + 1)

plt.figure(figsize=(14, 6))

# Plotting training and validation loss
plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss, 'b', label='Training Loss')
plt.plot(epochs, validation_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, training_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, validation_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(new_directory, 'grafica.png'))

# Save variables to a text file
with open(os.path.join(new_directory, 'variables.txt'), 'w') as file:
    file.write(f"Baseline media: {accuracy_avg}\n")
    file.write(f"Mi modelo: {accuracy_model}\n")
    file.write(f"Batch Size: {my_batch_size}\n")
    file.write(f"Learning Rate: {my_learning_rate}\n")
    file.write(f"Epochs: {my_epoch}\n")
    file.write(f"Patience: {my_patience}\n")

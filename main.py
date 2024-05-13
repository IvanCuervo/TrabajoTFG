import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.callbacks import LearningRateScheduler
from datetime import datetime
from tensorflow.python.keras.callbacks import EarlyStopping
from keras import backend as K

################################################ FUNCIONES ################################################

def mean_absolute_error(y_true, y_pred):
  return K.mean(K.abs(y_pred - y_true))

def root_mean_squared_error(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true)))

def mean_squared_error(y_true, y_pred):
  return K.mean(K.square(y_pred - y_true))

################################################################################################################################################

data = pd.read_csv(
    "/home/ivan/TrabajoTFG/DatosGenes/DatosSinAmbiguoTransformadoCSV.csv",  delimiter=';',
    names=["UnitProtein", "Actual", "Position", "Change", "Pathogenic"])

# Convert the DataFrame to a NumPy array
data_features = data.to_numpy()

# Convert label array to a supported numeric data type (e.g., float32)
prob_label = data_features[:, 4]

# Convert feature array to a supported numeric data type (e.g., float32)
code_features = np.delete(data_features, 4, axis=1)

# Split data into train, test, and validation sets
train_features, test_features, train_labels, test_labels = train_test_split(code_features, prob_label, test_size=0.1, random_state=42)
train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.1, random_state=42)

normalize = tf.keras.layers.experimental.preprocessing.Normalization()

model = tf.keras.Sequential([
    normalize,
    layers.Dense(64, activation='relu', input_shape=(code_features.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

my_callbacks = [
    EarlyStopping(monitor="val_loss", patience=1000),
]

model.compile(loss=tf.losses.MeanSquaredError(), 
              optimizer = tf.optimizers.Adam(learning_rate=1e-3))


history = model.fit(train_features, train_labels, batch_size=30, epochs=1000, validation_data=(val_features, val_labels),
                    callbacks=my_callbacks)

# Define the directory where you want to save the model
base_directory = '/home/ivan/TrabajoTFG/Modelos/Mayo'

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
#model.save(os.path.join(new_directory, 'Modelo.h5'))

"""
# Evaluate the model on the test set
test_loss = model.evaluate(test_features, test_labels)
print("Test Loss:", test_loss)
"""
# Make predictions on the test set
predictions = model.predict(test_features)

################################################ BASELINES ################################################

y_pred_b_rnd = np.random.random(len(test_labels)) #Random values
merged_array = np.concatenate((train_labels, val_labels))
y_pred_b_avg = [np.mean(merged_array)]*len(test_labels) # Baseline media (OJO, LA MEDIA SE OBTIENE DEL CONJUNTO DE ENTRENAMIENTO [O ENTRENAMIENTO+VAL] Y LA LOSS SERÍA SOBRE EL TEST)

mae_rnd = mean_absolute_error(test_labels, y_pred_b_rnd)
rmse_rnd = root_mean_squared_error(test_labels, y_pred_b_rnd)
mse_rnd = mean_squared_error(test_labels, y_pred_b_rnd)

rmse_avg = root_mean_squared_error(test_labels, y_pred_b_avg)
mse_avg = mean_squared_error(test_labels, y_pred_b_avg)
mae_avg = mean_absolute_error(test_labels, y_pred_b_avg)

################################################################################################

print(f"Modelo             | {'RMSE':6s} |")
print("-"*47)
print(f"Baseline aleatorio | {rmse_rnd.numpy():0.4f} |")
print(f"Baseline media     | {rmse_avg.numpy():0.4f} |")
print(f"Mi modelo          | {root_mean_squared_error(test_labels, predictions).numpy():0.4f} |")

# Plot actual vs predicted probability
plt.figure(figsize=(10, 6))
plt.scatter(test_labels, predictions, color='blue', label='Predictions')
plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'k--', lw=3, color='red', label='Actual')
plt.xlabel('Actual probability')
plt.ylabel('Predicted probability')
plt.title('Actual vs Predicted probability')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(new_directory, 'puntos.png'))

training_loss = history.history['loss']

validation_loss = history.history['val_loss']

epochs = range(1, len(training_loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, training_loss, 'b', label='Training Loss')
plt.plot(epochs, validation_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(new_directory, 'grafica.png'))

# Save variables to a text file
with open(os.path.join(new_directory, 'variables.txt'), 'w') as file:
    file.write(f"Baseline aleatorio: {rmse_rnd.numpy()}\n")
    file.write(f"Baseline media: {rmse_avg.numpy()}\n")
    file.write(f"Mi modelo: {root_mean_squared_error(test_labels, predictions).numpy()}\n")

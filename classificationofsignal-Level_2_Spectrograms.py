import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense
import tensorflow_model_optimization as tfmot
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from scipy.signal import find_peaks
df = pd.read_excel('/Users/ahmedatia/ATIA/S_D/SHM/SHM/PhD-Thesis-work/All_numerical&code/Numerical_Work/7_Sorey_input_data_neural_networks /RFCNN.xlsx')
# test=pd.read_excel('/Users/ahmedatia/ATIA/S D/SHM/PhD-Thesis-work/All_numerical&code/Concretesapmodel/Examined_neural_network_Data/Examined_data.xls')
#
df.columns
result = df.head()
# print (result)
Damage_condition = df.loc[:,"Damage_condition"]
Joint = df.loc[:,"Joint"]
#
#print(Damage_condition)
Damage_condition= Damage_condition.to_list()
# #print(Damage_condition)
Earthquake = df.loc[:,"Earthquake"]
Earthquake = Earthquake.to_list()
# test_Acceleration = test.loc[:,'U1']
df2= df.drop(['Earthquake','Joint','StepNum','Damage_condition'], axis=1)
earthequake= df.drop(['Damage_condition','Joint','StepNum','U1'], axis=1)
result = df2.head()
from numpy import transpose

df2 =tf.convert_to_tensor(df2)
# test_Acceleration= tf.convert_to_tensor(test_Acceleration)
# # print(df.dtype)
# df = tf.constant(df.values)
#print
i = 0
j = 901
A3 = []
while j <= 631601:
    indices = range(i, j)
    df_slice = df.iloc[indices, :]
    A = np.transpose(df_slice)
    A3.append(A)
    i += 901
    j += 901

#print(len(A3))
#print(A3[1].shape)
i=0
j=901
A2=[]
while i<631601 and j<631601 :
 A= tf.strided_slice(df2, [i], [j])
 A = transpose(A)
 i = i+901
 j= j+901
 A2.append(A)

list1 = []

for i in range(1, 631601):
    if (i % 901 == 0):
        list1.append(Damage_condition[i])
list2 =[]
for i in range (1, 631601):
    if(i%901==0) :
        list2.append(Earthquake[i])
#print(len(list1))
Joints = []
for i in range(1, 631601):
    if (i % 901 == 0):
        Joints.append(Joint[i])
from keras.utils import to_categorical
x = A2

x= np.array(x)
y = np.array(list1)
random_seed = 42
X_train, X_val, y_train, y_val = model_selection.train_test_split (x , y, test_size=.2 , random_state=random_seed)

classes = np.unique(np.concatenate((y_train, y_val), axis=0))
num_classes = len(np.unique(y_train))
# Count the occurrences of each category
unique_categories, counts = np.unique(y_val, return_counts=True)

# Create a dictionary to store the counts for each category
category_counts = dict(zip(unique_categories, counts))

# Print the counts for each category
for category, count in category_counts.items():
    print(f'{category}: {count}')
label_map = {'IO': 0, 'LS': 1, 'S': 2, 'CP': 3, 'IO\ufeff': 0}

y_train = [label_map.get(label, -1) for label in y_train]
y_val = [label_map.get(label, -1) for label in y_val]
y_train= tf.convert_to_tensor(y_train)
y_val= tf.convert_to_tensor(y_val)
X = x.reshape(x.shape[0], -1)
df = pd.DataFrame(X)
# set the directory path
directory = '/Users/ahmedatia/ATIA/S_D/SHM/SHM/PhD-Thesis-work/All_numerical&code/Numerical_Work/7_Sorey_input_data_neural_networks /RFCNN.xlsx'

# set the file name
filename = 'my_excel_file.xlsx'

# export the dataframe to an Excel file in the specified directory
df.to_excel(directory + filename, index=False)

X = x.reshape(x.shape[0], -1)
directory = '/Users/ahmedatia/ATIA/S_D/SHM/SHM/PhD-Thesis-work/All_numerical&code/Numerical_Work/7_Sorey_input_data_neural_networks /RFCNN.xlsx'

# set the file name
filename = 'my_excel_file_catagory.xlsx'
filename2 = 'Catagory.xlsx'
# export the dataframe to an Excel file in the specified directory
df.to_excel(directory + filename, index=False)

from tensorflow import keras
import librosa.display
import librosa.display
import numpy as np
import librosa
import librosa.display
from PIL import Image  # Make sure to install the Pillow library
fs= 1000
signal = X[45]
import numpy as np

def find_period(signal):
    # Find peaks
    peaks, _ = find_peaks(signal)

    # Calculate differences between peak positions
    differences = np.diff(peaks)

    # Compute the average difference (period)
    period = np.mean(differences)

    return period
period = find_period(signal)
print("Period:", period)

def calculate_max_frequency_from_raw(signal, fs):
    # Calculate the FFT of the signal
    fft_signal = np.fft.fft(signal)

    # Calculate the frequencies corresponding to the FFT
    freqs = np.fft.fftfreq(len(signal), 1/fs)

    # Find peaks in the FFT (excluding DC component)
    peaks, _ = find_peaks(np.abs(fft_signal), height=0)

    # Find the frequency corresponding to the highest peak
    max_frequency_index = peaks[np.argmax(np.abs(fft_signal[peaks]))]
    max_frequency = freqs[max_frequency_index]

    return np.abs(max_frequency)
# Function to compute the spectrogram for a single signal
max_frequency = calculate_max_frequency_from_raw(signal, fs)
print(f'Maximum frequency: {max_frequency:.2f} Hz')

def compute_spectrogram(signal):
    # Compute the Short-Time Fourier Transform (STFT)
    window = 'hamming'
    stft = librosa.stft(signal,n_fft=512, window=window)

    # Convert the magnitude spectrogram to decibels (log scale)
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(12 , 8))
    librosa.display.specshow(spectrogram_db, sr=100, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')

    # Save the current figure as an image file
    plt.savefig('temp_spectrogram.png')

    # Read the saved image file and convert it to a NumPy array
    image_array = np.array(Image.open('temp_spectrogram.png'))

    # Close the figure to avoid displaying it
    plt.close()

    return image_array


# Example usage
spectrogram_images = []

for i in range(len(X)):
    signal = X[i]
    spectrogram_image = compute_spectrogram(signal)
    spectrogram_images.append(spectrogram_image)
import matplotlib.pyplot as plt

# Assuming spectrogram_images_array is a NumPy array containing spectrogram images

# Convert the list of spectrogram images to a 3D NumPy array
spectrogram_images_array = np.array(spectrogram_images)

import matplotlib.pyplot as plt
import os

# Directory where you want to save the image
directory = '/Users/ahmedatia/ATIA/S_D/SHM/SHM/PhD-Thesis-work/All_numerical&code/Numerical_Work/7_Sorey_input_data_neural_networks'

# Ensure the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)

# Filename for the saved image
filename = os.path.join(directory, 'spectrogram_150-2.png')

# Assuming spectrogram_images_array is your array of spectrograms
fig, ax = plt.subplots()
cax = ax.imshow(spectrogram_images_array[150], cmap='viridis')  # Adjust the index as needed

# Increase the font size of the axis labels and ticks
ax.set_xlabel('Time', fontsize=30)
ax.set_ylabel('Frequency', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=12)

# Remove axis
ax.axis('off')

# Adjust layout to remove any padding
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Save the figure
plt.savefig(filename, bbox_inches='tight', pad_inches=0)
plt.close()  # Close the plot to free up memory



# Normalize pixel values to [0, 1]
# spectrogram_images_array = spectrogram_images_array / 255.0

# Split the data into training and validation sets
X_train_image, X_val_image = model_selection.train_test_split(spectrogram_images_array, test_size=0.2, random_state=42)

print (X_train_image.shape)

# def make_model(input_shape):
#     input_layer = keras.layers.Input(input_shape)
#
#     # Convolutional blocks
#     conv1 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(input_layer)
#     conv1 = keras.layers.BatchNormalization()(conv1)
#     conv1 = keras.layers.LeakyReLU()(conv1)
#     conv1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
#     conv1 = keras.layers.Dropout(0.2)(conv1)
#
#     conv2 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(conv1)
#     conv2 = keras.layers.BatchNormalization()(conv2)
#     conv2 = keras.layers.LeakyReLU()(conv2)
#     conv2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
#     conv2 = keras.layers.Dropout(0.2)(conv2)
#
#     # Global Average Pooling
#     gap = keras.layers.GlobalAveragePooling2D()(conv2)
#
#     # Output layer
#     output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)
#
#     # Create and compile the model
#     model = keras.models.Model(inputs=input_layer, outputs=output_layer)
#
#     return model

import tensorflow as tf
X_train_image = X_train_image.reshape(X_train_image.shape[0], 100, 70, 4)
X_val_image = X_val_image.reshape(X_val_image.shape[0], 100, 70, 4)
print("Shape before con2d:", X_train_image.shape)
import tensorflow as tf
from tensorflow.keras import layers, models
import time
input_shape = X_train_image[0].shape
# Define VGGNet Model
def vggnet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    print("Shape before con2d:", inputs.shape)
    # Block 1
    x = layers.Conv2D(64, 3, activation= layers.LeakyReLU(), padding='same')(inputs)
    x = layers.Conv2D(64, 3, activation= layers.LeakyReLU(), padding='same')(x)
    print("Shape before MaxPooling2D:", x.shape)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # After the layer causing the issue
    print("Shape after MaxPooling2D:", x.shape)
    x = layers.Dropout(0.25)(x)
    # Block 2
    x = layers.Conv2D(128, 3, activation= layers.LeakyReLU(), padding='same')(x)
    x = layers.Conv2D(128, 3, activation= layers.LeakyReLU(), padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Block 3
    x = layers.Conv2D(256, 3, activation= layers.LeakyReLU(), padding='same')(x)
    x = layers.Conv2D(256, 3, activation= layers.LeakyReLU(), padding='same')(x)
    x = layers.Conv2D(256, 3, activation= layers.LeakyReLU(), padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    # Block 4
    x = layers.Conv2D(512, 3, activation= layers.LeakyReLU(), padding='same')(x)
    x = layers.Conv2D(512, 3, activation= layers.LeakyReLU(), padding='same')(x)
    x = layers.Conv2D(512, 3, activation= layers.LeakyReLU(), padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Block 5
    x = layers.Conv2D(512, 3, activation= layers.LeakyReLU(), padding='same')(x)
    x = layers.Conv2D(512, 3, activation= layers.LeakyReLU(), padding='same')(x)
    x = layers.Conv2D(512, 3, activation= layers.LeakyReLU(), padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    # Flatten
    x = layers.Flatten()(x)

    # Fully Connected Layers
    x = layers.Dense(4096, activation= layers.LeakyReLU())(x)
    x = layers.Dense(4096, activation= layers.LeakyReLU())(x)

    # Output Layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Define model parameters
input_shape = X_train_image[0].shape
num_classes = 4  # Number of classes

# Build the model
model = vggnet(input_shape, num_classes)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"]
)

# Train the model
epochs = 200
batch_size = 32

# Define callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.001),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1)
]

# Train the model
start_time = time.time()
history = model.fit(
    X_train_image,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=(X_val_image, y_val),
    verbose=1
)
end_time = time.time()

# Evaluate the model
test_loss, test_acc = model.evaluate(X_val_image, y_val)

# Print results
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")
print(f"Training Time: {end_time - start_time} seconds")

# Save the model
model.save('classification_model.h5')

# Plot the model architecture
tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True)

import tensorflow as tf
tf.keras.utils.plot_model(model, to_file='model.png')
#
#
#
#
#
#
#
y_pred_prob = model.predict(X_val_image)

y_pred = y_pred_prob.argmax(axis=-1)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# assume y_true and y_pred are your true and predicted labels
cm = confusion_matrix( y_val, y_pred )

# create heatmap#
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
fig, ax = plt.subplots()

import numpy as np
print(len(y_pred))
# Extract the misclassified signals
misclassified_sequences = []
for i in range(len(y_val)):
        if not np.array_equal(y_pred[i], y_val[i]):

            misclassified_sequences.append(i)


misclassified_signals = np.array( misclassified_sequences)
misclassified_signals_itself = []

for i in range(len(y_val)):
        if (y_pred[i] != y_val[i]):
            misclassified_signals_itself.append(X_val[i])

# Check if the two arrays are equal element-wise
misclassified_sequences_joint = []
misclassified_sequences_earthquake = []
for i in range(len(misclassified_signals_itself)):
    for j in range(len(A2)):
        if np.array_equal(misclassified_signals_itself[i], A2[j]):
            misclassified_sequences_joint.append(Joints[j])
            misclassified_sequences_earthquake.append(Earthquake[j])

print(misclassified_signals_itself)


print(misclassified_sequences_joint)
print(misclassified_sequences_earthquake)
print(len(misclassified_sequences_earthquake))
print(len(misclassified_sequences_joint))


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features = model.predict(X_val)
X_tsne = tsne.fit_transform(features)
# Plot the t-SNE embeddings, colored by the predicted class labels
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_pred)
plt.colorbar()
plt.title('t-SNE Visualization of CNN Classification Results')

# Add feature names to x and y axes
plt.xlabel("TSNE-1")
plt.ylabel(features[1])

plt.show()
# print(result)reate heatmap#
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
fig, ax = plt.subplots()

from sklearn.metrics import precision_score, recall_score, f1_score


precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')

print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1))

# plt.show()
import numpy as np
print(misclassified_sequences_joint)
print(len(misclassified_sequences_joint))
# # Get the indices of the rows where the two arrays are equal
# equal_indices = np.where(equal_arrays)[0]

# Print the corresponding columns of the equal rows
# if len(equal_indices) > 0:
#     equal_columns = array1[equal_indices, :]
#     print(equal_columns)
# else:
#     print("The two arrays are not equal.")
misclassified_sequences_joint = np.array(misclassified_sequences_joint)
print(misclassified_sequences_joint.shape)
floor_number = []
for i in range(len(misclassified_sequences_joint)):
    if misclassified_sequences_joint[i] == 9:
        floor_number.append('Level_base')
    elif misclassified_sequences_joint[i] == 10:
        floor_number.append('Level_1')
    elif misclassified_sequences_joint[i] == 11:
        floor_number.append('Level_2')
    elif misclassified_sequences_joint[i] == 12:
        floor_number.append('Level_3')
    elif misclassified_sequences_joint[i] == 13:
        floor_number.append('Level_4')
    elif misclassified_sequences_joint[i] == 14:
        floor_number.append('Level_5')
    elif misclassified_sequences_joint[i] == 15:
        floor_number.append('Level_6')
    elif misclassified_sequences_joint[i] == 16:
        floor_number.append('Level_7')
print(floor_number)
# # Find the indices where the first column of arr is equal to 4
# indices = np.where(arr[:, 0] == 4)
# count the frequency of each element
import numpy as np
# unique, counts = np.unique(misclassified_sequences_joint, return_counts=True)
#
# # plot the bar chart
# plt.bar(unique, counts)
#
# # set the title and labels
# plt.title("Frequency of elements in array")
# plt.xlabel("Element")
# plt.ylabel("Frequency")
# plt.show()
# #plot the bar chart of misclassified signals

# unique, counts = np.unique(floor_number, return_counts=True)


# # Use the indices to get the corresponding values from the second and third columns to end

unique, counts = np.unique(floor_number, return_counts=True)

# plot the bar chart
plt.bar(unique, counts)

# set the title and labels
plt.title("Number of misclassified signals In each floor")
plt.xlabel("Floor Number")
plt.ylabel("Number mispredicted signals")

# show the plot
plt.show()
import tensorflow as tf
import pandas as pd



# Create a list of layers and their parameters
# Create a list of layers and their parameters
layers = []
for layer in model.layers:
    layer_name = layer.name
    layer_type = type(layer).__name__
    if hasattr(layer, 'kernel_size'):
        kernel_size = layer.kernel_size
        stride = layer.strides
        out_channels = layer.filters
        activation = layer.activation.__name__
    else:
        kernel_size = None
        stride = None
        out_channels = None
        activation = None
    layers.append([layer_name, layer_type, kernel_size, stride, out_channels, activation])

# Create a pandas dataframe from the list of layers
Internal_Arch_2DCNN = pd.DataFrame(layers, columns=['Layer Name', 'Layer Type', 'Kernel Size', 'Stride', 'Out Channels', 'Activation'])

# Print the dataframe
print(Internal_Arch_2DCNN)
# # Use the indices to get the corresponding values from the second and third columns
Internal_Arch_2DCNN.to_excel('vInternal_Arch_2DCNN.xlsx', index=False)

from keras.models import Model
import keras.backend as K
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions


layer_name = "conv1"
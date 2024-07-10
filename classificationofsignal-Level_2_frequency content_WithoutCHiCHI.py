import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
df = pd.read_excel('/Users/ahmedatia/ATIA/S D/SHM/PhD-Thesis-work/All_numerical&code/Numerical_Work/7_Sorey_input_data_neural_networks /RFCNN_Without_CHI_CHI.xlsx')
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

df2= df.drop(['Earthquake','Joint','StepNum','Damage_condition'], axis=1)
result = df2.head()
# print (result)
# print(df2.shape)
from numpy import transpose

df2 =tf.convert_to_tensor(df2)

# # print(df.dtype)
# df = tf.constant(df.values)
#print

i = 0
j = 901
A3 = []

while j <= 561324:
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
# snr_dB = 2.5
while i<561324 and j<561324 :
 A= tf.strided_slice(df2, [i], [j])
 A = transpose(A)
 i = i+901
 j= j+901
 A2.append(A)
# set the directory path

list1 = []
for i in range(1, 561324):
    if (i % 901 == 0):
        list1.append(Damage_condition[i])
list2 =[]
for i in range (1, 561324):
    if(i%901==0) :
        list2.append(Earthquake[i])

#print(len(list1))
Joints = []
for i in range(1, 561324):
    if (i % 901 == 0):
        Joints.append(Joint[i])
# list2= pd.DataFrame(list2)
# directory = '/Users/ahmedatia/ATIA/S D/SHM/PhD-Thesis-work/All_numerical&code/Numerical_Work/7_Sorey_input_data_neural_networks'
#
# # set the file name
# filename = 'earthquake.xlsx'
#
# # export the dataframe to an Excel file in the specified directory
# list2.to_excel(directory + filename, index=False)
x = A2
# print (x)
x= np.array(x)
y = np.array(list1)
# print(list1)

X_train, X_val, y_train, y_val = model_selection.train_test_split (x , y, test_size=.2 )
y_val_2 = y_val
classes = np.unique(np.concatenate((y_train, y_val), axis=0))
num_classes = len(np.unique(y_train))
# print(num_classes)
# idx = np.random.permutation(len(X_train))
# X_train = X_train[idx]
# y_train = y_train[idx]

#print (X_train)
# encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(y_train)
# encoded_Y = encoder.transform(y_train)
# # convert integers to dummy variables (i.e. one hot encoded)
# dummy_y = np_utils.to_categorical(encoded_Y)
label_map = {'IO': 0, 'LS': 1, 'S': 2, 'CP': 3, 'IO\ufeff': 0}

y_train = [label_map.get(label, -1) for label in y_train]
y_val = [label_map.get(label, -1) for label in y_val]
y_train= tf.convert_to_tensor(y_train)
y_val= tf.convert_to_tensor(y_val)
# X = x.reshape(x.shape[0], 0)
# df = pd.DataFrame(X)
#
# # set the directory path
# directory = '/Users/ahmedatia/ATIA/S D/SHM/PhD-Thesis-work/All_numerical&code/Numerical_Work/7_Sorey_input_data_neural_networks'
#
# # set the file name
# filename = 'my_excel_file.xlsx'
#
# # export the dataframe to an Excel file in the specified directory
# df.to_excel(directory + filename, index=False)
#
# X = x.reshape(x.shape[0], -1)
# df = pd.DataFrame(X)
# dt=.01
# df = pd.DataFrame(y)
# Accelration_1 = df.iloc[0]
# n = len(Accelration_1)
# freq = np.fft.fftfreq(n,.01 )
# acc_fft = np.fft.fft(Accelration_1)
# psd = np.abs(acc_fft)**2 / (n * dt)
# plt.figure()
# plt.semilogy(freq[:n//2], psd[:n//2])
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power Spectral Density')
# plt.title('Power Spectral Density of Acceleration')
# plt.grid(True)
# plt.show()


# directory = '/Users/ahmedatia/ATIA/S D/SHM/PhD-Thesis-work/All_numerical&code/Numerical_Work/7_Sorey_input_data_neural_networks'
#
# # set the file name
# filename = 'my_excel_file_catagory.xlsx'
#
# # export the dataframe to an Excel file in the specified directory
# df.to_excel(directory + filename, index=False)
# from tensorflow import keras

# def make_model(input_shape):
#     input_layer = keras.layers.Input(input_shape)
#
#     conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
#     conv1 = keras.layers.BatchNormalization()(conv1)
#     conv1 = keras.layers.ReLU()(conv1)
#     pool1 = keras.layers.MaxPooling1D(pool_size=1)(conv1)
#
#     conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(pool1)
#     conv2 = keras.layers.BatchNormalization()(conv2)
#     conv2 = keras.layers.ReLU()(conv2)
#     pool2 = keras.layers.MaxPooling1D(pool_size=1)(conv2)
#
#     conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(pool2)
#     conv3 = keras.layers.BatchNormalization()(conv3)
#     conv3 = keras.layers.ReLU()(conv3)
#     pool3 = keras.layers.MaxPooling1D(pool_size=1)(conv3)
#
#     conv4 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(pool3)
#     conv4 = keras.layers.BatchNormalization()(conv4)
#     conv4 = keras.layers.ReLU()(conv4)
#     pool4 = keras.layers.MaxPooling1D(pool_size=1)(conv4)
#
#     gap = keras.layers.GlobalAveragePooling1D()(pool4)
#
#     output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)
#
#     return keras.models.Model(inputs=input_layer, outputs=output_layer)


def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.LeakyReLU()(conv1)
    conv1 = keras.layers.Dropout(0.2)(conv1)

    conv2 = keras.layers.Conv1D(filters=128, kernel_size=5, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.LeakyReLU()(conv2)
    conv2 = keras.layers.Dropout(0.2)(conv2)

    conv3 = keras.layers.Conv1D(filters=256, kernel_size=5, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.LeakyReLU()(conv3)
    conv3 = keras.layers.Dropout(0.2)(conv3)

    conv4 = keras.layers.Conv1D(filters=512, kernel_size=5, padding="same")(conv3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.LeakyReLU()(conv4)
    conv4 = keras.layers.Dropout(0.2)(conv4)

    gap = keras.layers.GlobalAveragePooling1D()(conv4)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

# def make_model(input_shape):
#     input_layer = keras.layers.Input(input_shape)
#
#     conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
#     conv1 = keras.layers.BatchNormalization()(conv1)
#     conv1 = keras.layers.ReLU()(conv1)
#
#     conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
#     conv2 = keras.layers.BatchNormalization()(conv2)
#     conv2 = keras.layers.ReLU()(conv2)
#
#     conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
#     conv3 = keras.layers.BatchNormalization()(conv3)
#     conv3 = keras.layers.ReLU()(conv3)
#
#     conv4 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv3)
#     conv4 = keras.layers.BatchNormalization()(conv4)
#     conv4 = keras.layers.ReLU()(conv4)
#     gap = keras.layers.GlobalAveragePooling1D()(conv4)
#
#     output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)
#
#     return keras.models.Model(inputs=input_layer, outputs=output_layer)


# print("Test accuracy", test_acc)
# print("Test loss", test_loss)
# #y_prediction= model.predict(X_val)
# metric = "sparse_categorical_accuracy"
# plt.figure()
# plt.plot(history.history[metric])
# plt.plot(history.history["val_" + metric])
# plt.title("model " + metric)model = make_model(input_shape= X_train.shape[1:])
# epochs = 100
# batch_size = 32
# callbacks = [
#     keras.callbacks.ModelCheckpoint(
#         "best_model.h5", save_best_only=True, monitor="val_loss"
#     ),
#     keras.callbacks.ReduceLROnPlateau(
#         monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
#     ),
#     # keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
#
# ]
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["sparse_categorical_accuracy"],
# )
# history = model.fit(
#     X_train,
#     y_train,
#     batch_size=batch_size,
#     epochs=epochs,
#     callbacks=callbacks,
#     validation_data=(X_val, y_val),
#     verbose=1,
# )
# test_loss, test_acc = model.evaluate(X_val, y_val)
# plt.ylabel(metric, fontsize="large")
# plt.xlabel("epoch", fontsize="large")
# plt.legend(["train", "val"], loc="best")
# plt.show()
# plt.close()
#y_pred_val = model.predict(X_val)
from sklearn.metrics import confusion_matrix
#results = model.evaluate(X_val, y_pred_val, batch_size=32)
# create confusion matrix
# cm = confusion_matrix(y_pred_val, y_val)

#print (y_pred)
# mse = mean_squared_error(y_pred_prob, X_val)
# print(mse)

# # Plotting the predicted vs actual values
# fig, ax = plt.subplots()
# ax.scatter(y_pred, y_val, color='blue', label='Predicted')
# ax.plot([0, max(y_val)], [0, max(y_val)], color='black', linestyle='--')
# ax.set_xlabel('Predicted')
# ax.set_ylabel('Actual')
# ax.set_title('Predicted vs Actual')
# ax.legend()
# ax.text(0.1, 0.9, 'MSE = {:.2f}'.format(mse), ha='center', va='center', transform=ax.transAxes)
#
# plt.show()
# import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt

# assume y_true and y_pred are your true and predicted labels
# cm = confusion_matrix( y_val, y_pred )



# def make_model(input_shape):
#     input_layer = keras.layers.Input(input_shape)
#
#     conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
#     conv1 = keras.layers.BatchNormalization()(conv1)
#     conv1 = keras.layers.ReLU()(conv1)
#
#     conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
#     conv2 = keras.layers.BatchNormalization()(conv2)
#     conv2 = keras.layers.ReLU()(conv2)
#
#     conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
#     conv3 = keras.layers.BatchNormalization()(conv3)
#     conv3 = keras.layers.ReLU()(conv3)
#
#     conv4 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv3)
#     conv4 = keras.layers.BatchNormalization()(conv4)
#     conv4 = keras.layers.ReLU()(conv4)
#     gap = keras.layers.GlobalAveragePooling1D()(conv4)
#
#     output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)
#
#     return keras.models.Model(inputs=input_layer, outputs=output_layer)

model = make_model(input_shape= X_train.shape[1:])
epochs = 1000
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    # keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]

import time
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
start_time = time.time()
history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=(X_val, y_val),
    verbose=1,
)
end_time = time.time()
test_loss, test_acc = model.evaluate(X_val, y_val)

table = [["Training Accuracy", history.history['sparse_categorical_accuracy'][-1]],
         ["Testing Accuracy", test_acc],
         ["Training Time (s)", end_time - start_time]]

print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
model.save('Classificationof signals.h5')
import tensorflow as tf
tf.keras.utils.plot_model(model, to_file='model.png')
# print("Test accuracy", test_acc)
# print("Test loss", test_loss)
# #y_prediction= model.predict(X_val)
# metric = "sparse_categorical_accuracy"
# plt.figure()
# plt.plot(history.history[metric])
# plt.plot(history.history["val_" + metric])
# plt.title("model " + metric)
# plt.ylabel(metric, fontsize="large")
# plt.xlabel("epoch", fontsize="large")
# plt.legend(["train", "val"], loc="best")
# plt.show()
# plt.close()
#y_pred_val = model.predict(X_val)
from sklearn.metrics import confusion_matrix
#results = model.evaluate(X_val, y_pred_val, batch_size=32)
# create confusion matrix
# cm = confusion_matrix(y_pred_val, y_val)
# Train your model and record training and validation accuracy and time

# Make predictions on the validation data
y_pred_prob = model.predict(X_val)
y_pred = y_pred_prob.argmax(axis=-1)
# Compute mean squared error
#print (y_pred)
# mse = mean_squared_error(y_pred_prob, X_val)
# print(mse)

# # Plotting the predicted vs actual values
# fig, ax = plt.subplots()
# ax.scatter(y_pred, y_val, color='blue', label='Predicted')
# ax.plot([0, max(y_val)], [0, max(y_val)], color='black', linestyle='--')
# ax.set_xlabel('Predicted')
# ax.set_ylabel('Actual')
# ax.set_title('Predicted vs Actual')
# ax.legend()
# ax.text(0.1, 0.9, 'MSE = {:.2f}'.format(mse), ha='center', va='center', transform=ax.transAxes)
#
# plt.show()
# import matplotlib.pyplot as plt

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
# for i in range(len(y_val)):
#     ax.scatter(y_pred[i], y_val[i], color='blue', label='Predicted')
# ax.plot([0, max(y_val)], [0, max(y_val)], color='black', linestyle='--')
# ax.set_xlabel('Predicted')
# ax.set_ylabel('Actual')
# ax.set_title('Predicted vs Actual')
# ax.legend()
# ax.text(0.1, 0.9, 'MSE = {:.2f}'.format(mse), ha='center', va='center', transform=ax.transAxes)

# plt.show()
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras


print(len(y_pred))
# Extract the misclassified signals
misclassified_sequences = []
for i in range(len(y_val)):
        if not np.array_equal(y_pred[i], y_val[i]):

            misclassified_sequences.append(i)
# for j in range(len(y_pred)):
#     for i in range(len(y_val)):
#         if not np.array_equal(y_pred[j], y_val[i]):
#             misclassified_sequences.append(i)
# Convert the misclassified signals to a numpy array

misclassified_signals = np.array( misclassified_sequences)
misclassified_signals_itself = []

for i in range(len(y_val)):
        if (y_pred[i] != y_val[i]):
            misclassified_signals_itself.append(X_val[i])

# Check if the two arrays are equal element-wise
misclassified_sequences_joint = []
misclassified_joint = []
misclassified_joint_number_sequence = []
misclassified_joint_Damage_class = []
for i in range(len(misclassified_signals_itself)):
    for j in range(len(A2)):
        if np.array_equal(misclassified_signals_itself[i], A2[j]):
            misclassified_sequences_joint.append(Joints[j])
            misclassified_joint_number_sequence.append(i)

            misclassified_joint_Damage_class.append(list1[j])

print(misclassified_joint_number_sequence)
print(misclassified_sequences_joint)
print(misclassified_joint_Damage_class)
# print(misclassified_signals_itself)
# misclassified_sequences_joint = []
# for i in range(len(misclassified_signals_itself) ** len(A2)):
#     for j in range(len(A2)):
#         if np.array_equal(misclassified_signals_itself[i // len(A2) ** j % len(misclassified_signals_itself)], A2[j]):
#             misclassified_sequences_joint.append(Joints[j])

# Check if the two arrays are equal element-wise_Earthquake
misclassified_sequences_Earthquake = []
for i in range(len(misclassified_signals_itself)):
    for j in range(len(A2)):
        if np.array_equal(misclassified_signals_itself[i], A2[j]):
            misclassified_sequences_Earthquake.append(list2[j])

X_validated_Earthquake = []
X_validated_Joint = []
X_validated_Joint_sequence = []
X_validated_damage_class = []
for i in range(len(X_val)):
    for j in range(len(A2)):
        if np.array_equal(X_val[i], A2[j]):
            X_validated_Earthquake.append(list2[j])
            X_validated_Joint.append(Joints[j])
            X_validated_Joint_sequence.append(i)
            X_validated_damage_class.append(list1[j])
print(X_validated_Earthquake)
print(X_validated_Joint)
print(X_validated_damage_class)
print(X_validated_Joint_sequence)
print(misclassified_sequences_Earthquake)
print(len(misclassified_sequences_Earthquake))
# for i in range(len(A2)):
#     if np.array_equal(misclassified_signals_itself[i], A2[i]):
#
#             misclassified_sequences_joint.append(Joints[i])
# print(misclassified_sequences_joint)
print(len(misclassified_sequences_joint))
# # Get the indices of the rows where the two arrays are equal
# equal_indices = np.where(equal_arrays)[0]

# Print the corresponding columns of the equal rows
# if len(equal_indices) > 0:
#     equal_columns = array1[equal_indices, :]
#     print(equal_columns)
# else:
#     print("The two arrays are not equal.")
#
#
#
# # Find the indices where the first column of arr is equal to 4
# indices = np.where(arr[:, 0] == 4)
#
# # Use the indices to get the corresponding values from the second and third columns
# result = arr[indices, 1:]
#ploting TSNE


# Assume that 'X_test' is a numpy array of shape (num_samples, num_features)
# containing the feature vectors for the test signals, and 'y_pred' is a numpy array
# of shape (num_samples,) containing the predicted class labels for each test signal.
# Also assume that 'feature_names' is a list of length 'num_features' containing the names
# of the features in the same order as the columns of 'X_test'.

# Apply t-SNE to obtain a low-dimensional representation of the features

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
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

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
Internal_Arch_CNN = pd.DataFrame(layers, columns=['Layer Name', 'Layer Type', 'Kernel Size', 'Stride', 'Out Channels', 'Activation'])

# Print the dataframe
print(Internal_Arch_CNN)
# # Use the indices to get the corresponding values from the second and third columns
Internal_Arch_CNN.to_excel('vInternal_Arch_CNN.xlsx', index=False)
# import os
#
# cwd = os.getcwd()
# print(cwd)
# from sklearn.metrics import precision_recall_curve
# precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
# plt.plot(recall, precision, color='b', label='Precision-Recall curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc='lower left')
# plt.show()
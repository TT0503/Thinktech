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
import scipy.io

# Load MATLAB file into a dictionary of NumPy arrays
mat_data01 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/previous_Experimental_data /4 floor steel frame /CD of UBC.experiment 2002 2/data/Shaker/Random/shm01s.mat')
mat_data02 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/previous_Experimental_data /4 floor steel frame /CD of UBC.experiment 2002 2/data/Shaker/Random/shm02s.mat')
mat_data03 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/previous_Experimental_data /4 floor steel frame /CD of UBC.experiment 2002 2/data/Shaker/Random/shm03s.mat')
mat_data04 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/previous_Experimental_data /4 floor steel frame /CD of UBC.experiment 2002 2/data/Shaker/Random/shm04s.mat')
mat_data05 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/previous_Experimental_data /4 floor steel frame /CD of UBC.experiment 2002 2/data/Shaker/Random/shm05s.mat')
mat_data06 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/previous_Experimental_data /4 floor steel frame /CD of UBC.experiment 2002 2/data/Shaker/Random/shm06s.mat')
mat_data07 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/previous_Experimental_data /4 floor steel frame /CD of UBC.experiment 2002 2/data/Shaker/Random/shm07s.mat')
mat_data08 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/previous_Experimental_data /4 floor steel frame /CD of UBC.experiment 2002 2/data/Shaker/Random/shm08s.mat')
mat_data9 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/previous_Experimental_data /4 floor steel frame /CD of UBC.experiment 2002 2/data/Shaker/Random/shm09s.mat')

# Access the NumPy array corresponding to a variable in the MATLAB file


data1 = mat_data01['dasy']
data2 = mat_data02['dasy']
data3 = mat_data03['dasy']
data4 = mat_data04['dasy']
data5 = mat_data05['dasy']
data6 = mat_data06['dasy']
data7 = mat_data07['dasy']
data8 = mat_data08['dasy']
data9 = mat_data9['dasy']
# df2 =tf.convert_to_tensor(data )
#print
# i=0
# j=901
# A1=[]
# while i<631601 and j<631601 :
#  A= tf.strided_slice(df2, [i], [j])
#  A = transpose(A)
#  i = i+901
#  j= j+901
#  A1.append(A)
# Print the NumPy array
#print(data)
#print(data.shape)

A9 =[]
field_names = ['DA01', 'DA02', 'DA03', 'DA04', 'DA05', 'DA06', 'DA07', 'DA08', 'DA09', 'DA10', 'DA11', 'DA12', 'DA13', 'DA14', 'DA15']

for field_name in field_names:
 first_condition = data9[0][0][field_name]
 first_condition = first_condition.T
 A9.append(first_condition)
A9 = np.array(A9)
print(A9)
A9 = np.reshape(A9, (15, -1))
print(A9.shape)
A8 =[]
field_names = ['DA01', 'DA02', 'DA03', 'DA04', 'DA05', 'DA06', 'DA07', 'DA08', 'DA09', 'DA10', 'DA11', 'DA12', 'DA13', 'DA14', 'DA15']

for field_name in field_names:
 first_condition = data8[0][0][field_name]
 first_condition = first_condition.T
 A8.append(first_condition)
A8 = np.array(A8)
print(A8)
A8 = np.reshape(A8, (15, -1))
print(A8.shape)

A7 =[]
field_names = ['DA01', 'DA02', 'DA03', 'DA04', 'DA05', 'DA06', 'DA07', 'DA08', 'DA09', 'DA10', 'DA11', 'DA12', 'DA13', 'DA14', 'DA15']

for field_name in field_names:
 first_condition = data7[0][0][field_name]
 first_condition = first_condition.T
 A7.append(first_condition)
A7 = np.array(A7)
print(A7)
A7 = np.reshape(A7, (15, -1))
print(A7.shape)

A6 =[]
field_names = ['DA01', 'DA02', 'DA03', 'DA04', 'DA05', 'DA06', 'DA07', 'DA08', 'DA09', 'DA10', 'DA11', 'DA12', 'DA13', 'DA14', 'DA15']

for field_name in field_names:
 first_condition = data6[0][0][field_name]
 first_condition = first_condition.T
 A6.append(first_condition)
A6 = np.array(A6)
print(A6)
A6 = np.reshape(A6, (15, -1))
# Create noise array with desired shape

# Calculate the power of A2
power = np.mean(A6 ** 2)

# Calculate the power of the desired noise at 10 dB
noise_power = power / 10

# Generate Gaussian noise with the desired power and shape
noise = np.random.normal(scale=np.sqrt(noise_power), size=(15, 12000))

# Concatenate the noise array to A2
A6 = np.concatenate((A6, noise), axis=1)
print(A6.shape)

A5 =[]
field_names = ['DA01', 'DA02', 'DA03', 'DA04', 'DA05', 'DA06', 'DA07', 'DA08', 'DA09', 'DA10', 'DA11', 'DA12', 'DA13', 'DA14', 'DA15']

for field_name in field_names:
 first_condition = data5[0][0][field_name]
 first_condition = first_condition.T
 A5.append(first_condition)
A5 = np.array(A5)
print(A5)
A5 = np.reshape(A5, (15, -1))
print(A5.shape)

A4 =[]
field_names = ['DA01', 'DA02', 'DA03', 'DA04', 'DA05', 'DA06', 'DA07', 'DA08', 'DA09', 'DA10', 'DA11', 'DA12', 'DA13', 'DA14', 'DA15']

for field_name in field_names:
 first_condition = data4[0][0][field_name]
 first_condition = first_condition.T
 A4.append(first_condition)
A4 = np.array(A4)
print(A4)
A4 = np.reshape(A4, (15, -1))
print(A4.shape)

A3 =[]
field_names = ['DA01', 'DA02', 'DA03', 'DA04', 'DA05', 'DA06', 'DA07', 'DA08', 'DA09', 'DA10', 'DA11', 'DA12', 'DA13', 'DA14', 'DA15']

for field_name in field_names:
 first_condition = data3[0][0][field_name]
 first_condition = first_condition.T
 A3.append(first_condition)
A3 = np.array(A3)
print(A3)
A3 = np.reshape(A3, (15, -1))
# noise = np.random.normal(0, 1, size=(15, 48000))
# A3 = A3 + noise
print(A3.shape)
A2 =[]
field_names = ['DA01', 'DA02', 'DA03', 'DA04', 'DA05', 'DA06', 'DA07', 'DA08', 'DA09', 'DA10', 'DA11', 'DA12', 'DA13', 'DA14', 'DA15']

for field_name in field_names:
 first_condition = data2[0][0][field_name]
 first_condition = first_condition.T
 A2.append(first_condition)
A2 = np.array(A2)
A2 = np.reshape(A2, (15, -1))
print(A2)
# Create noise array with desired shape

# Calculate the power of A2
# power = np.mean(A2 ** 2)
#
# # Calculate the power of the desired noise at 10 dB
# noise_power = power / 10
#
# # Generate Gaussian noise with the desired power and shape
# noise = np.random.normal(scale=np.sqrt(noise_power), size=(15, 36000))
#
# # Concatenate the noise array to A2
# A2 = np.concatenate((A2, noise), axis=1)
print(A2.shape)


A1 =[]
field_names = ['DA01', 'DA02', 'DA03', 'DA04', 'DA05', 'DA06', 'DA07', 'DA08', 'DA09', 'DA10', 'DA11', 'DA12', 'DA13', 'DA14', 'DA15']

for field_name in field_names:
 first_condition = data1[0][0][field_name]
 first_condition = first_condition.T
 A1.append(first_condition)
A1 = np.array(A1)
print(A1)
A1 = np.reshape(A1, (15, -1))
print(A1.shape)


# A6 = []
# for j in range(1, 10):
#     for i in range(1, 16):
#         var_name = "data{}".format(j)
#         first_condition = locals()[var_name][0][0][i].T
#         A6.append(first_condition)
#
#   A6 = np.array(A6).reshape(15, 72000)
# print(A6.shape)
df1 = pd.concat([pd.DataFrame(A1), pd.DataFrame(A2), pd.DataFrame(A3), pd.DataFrame(A4), pd.DataFrame(A5)]  ,ignore_index=True)
df2 = pd.concat([pd.DataFrame(A6), pd.DataFrame(A7) , pd.DataFrame(A8) , pd.DataFrame(A9)], ignore_index=True)
df1 = np.array(df1.values)
df2 = np.array(df2.values)
# df1 =tf.convert_to_tensor(df1)
# df2 =tf.convert_to_tensor(df2)



# Create variables data1 to data9 here



# Read the Excel sheet into a pandas dataframe
df = pd.read_excel('/Users/ahmedatia/ATIA/S D/SHM/previous_Experimental_data /4 floor steel frame /CD of UBC.experiment 2002 2/data/Shaker/Random/Catagorical_Damage .xlsx')

# Extract the values from the dataframe and create a numpy array
array_9x15 = np.array(df.values)
print (array_9x15)
# Reshape the array to have 9 arrays of 15 rows each
y = array_9x15.reshape((135,1))
# df = pd.DataFrame(A)
y1, y2 = np.split(y, [75])
# Print the numpy array
print(array_9x15)


X_train1, X_val1, y_train1, y_val1 = model_selection.train_test_split (df1, y1, test_size=.25 ,shuffle=True, random_state=42, stratify=y1)
assert len(X_train1) == len(y_train1)
classes = np.unique(np.concatenate((y_train1, y_val1), axis=0))
num_classes = len(np.unique(y_train1))
from keras.utils import to_categorical
# for the second model

# label_map = {'H': 0, 'D': 1, 'O': 2}
# y_train = [label.astype(str) for label in y_train]
# y_val = [label.astype(str) for label in y_val]
# y_train= tf.convert_to_tensor(y_train)
# y_val= tf.convert_to_tensor(y_val)
from sklearn.preprocessing import LabelEncoder

y_train1= tf.convert_to_tensor(y_train1)
y_val1= tf.convert_to_tensor(y_val1)


X_train2, X_val2, y_train2, y_val2 = model_selection.train_test_split (df2, y2, test_size=.25 ,random_state=42,stratify=y2 )
assert len(X_train2) == len(y_train2)
classes = np.unique(np.concatenate((y_train2, y_val2), axis=0))
num_classes = len(np.unique(y_train2))
from keras.utils import to_categorical
# for the second model
classes = np.unique(np.concatenate((y_train2, y_val2), axis=0))
num_classes = len(np.unique(y_train2))


# idx = np.random.permutation(len(X_train))
# X_train = X_train[idx]
# y_train = y_train[idx]

label_map = {'H': 0, 'D': 1, 'O': 2}
# encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(y_train)
# encoded_Y = encoder.transform(y_train)
# # convert integers to dummy variables (i.e. one hot encoded)
# dummy_y = np_utils.to_categorical(encoded_Y)

y_train2= tf.convert_to_tensor(y_train2)
y_val2= tf.convert_to_tensor(y_val2)
from tensorflow import keras


X_train1 = X_train1.reshape(X_train1.shape[0], X_train1.shape[1], 1)
X_val1 = X_val1.reshape(X_val1.shape[0], X_val1.shape[1], 1)
X_train2 = X_train2.reshape(X_train2.shape[0], 1, X_train2.shape[1])
X_val2 = X_val2.reshape(X_val2.shape[0], 1 , X_val2.shape[1])
print(len(X_train1), len(X_train2), len(y_train1), len(y_train2), len(X_val1), len(X_val2), len(y_val1), len(y_val2))
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train1 = label_encoder.fit_transform(y_train1)
y_val1 = label_encoder.fit_transform(y_val1)
y_train2 = label_encoder.fit_transform(y_train2)
y_val2 = label_encoder.fit_transform(y_val2)
# def make_model(input_shape1 , input_shape2):
#     input1 = keras.layers.Input(input_shape1)
#     input2 = keras.layers.Input(input_shape2)
#
# # Define the first branch of the model that processes input1
#     conv1_1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input1)
#     conv1_1 = keras.layers.BatchNormalization()(conv1_1)
#     conv1_1 = keras.layers.ReLU()(conv1_1)
#     conv2_1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1_1)
#     conv2_1 = keras.layers.BatchNormalization()(conv2_1)
#     conv2_1 = keras.layers.ReLU()(conv2_1)
#     conv3_1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2_1)
#     conv3_1 = keras.layers.BatchNormalization()(conv3_1)
#     conv3_1 = keras.layers.ReLU()(conv3_1)
#     conv4_1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv3_1)
#     conv4_1 = keras.layers.BatchNormalization()(conv4_1)
#     conv4_1 = keras.layers.ReLU()(conv4_1)
#     gap1 = keras.layers.GlobalAveragePooling1D()(conv4_1)
#
# # Define the second branch of the model that processes input2
#     conv1_2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input2)
#     conv1_2 = keras.layers.BatchNormalization()(conv1_2)
#     conv1_2 = keras.layers.ReLU()(conv1_2)
#     conv2_2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1_2)
#     conv2_2 = keras.layers.BatchNormalization()(conv2_2)
#     conv2_2 = keras.layers.ReLU()(conv2_2)
#     conv3_2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2_2)
#     conv3_2 = keras.layers.BatchNormalization()(conv3_2)
#     conv3_2 = keras.layers.ReLU()(conv3_2)
#     conv4_2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv3_2)
#     conv4_2 = keras.layers.BatchNormalization()(conv4_2)
#     conv4_2 = keras.layers.ReLU()(conv4_2)
#     gap2 = keras.layers.GlobalAveragePooling1D()(conv4_2)
#
# # Merge the outputs of the two branches
#     merged = keras.layers.Concatenate()([gap1, gap2])
#
# # Add additional layers to generate the final output
#     dense1 = keras.layers.Dense(64, activation='relu')(merged)
#     output_layer = keras.layers.Dense(num_classes, activation='softmax')(dense1)
#
# # Define the model with the two inputs and the final output
#     return   keras.models.Model(inputs=[input1, input2], outputs=output_layer)
def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=X_train2.shape[1:])
from tabulate import tabulate
#
# model = make_model(input_shape1= X_train1.shape[1:],input_shape2= X_train2.shape[1:] )
# y_pred_prob1 = model.predict(X_val1)
# y_pred = y_pred_prob1.argmax(axis=-1)
# y_pred_prob2 = model.predict(X_val2)
# y_pred = y_pred_prob1.argmax(axis=-2)

epochs = 1000
batch_size = 32
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
import time
start_time = time.time()
history = model.fit(
    X_train2,
    y_train2,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=(X_val2, y_val2),
    verbose=1,
)
test_loss, test_acc = model.evaluate(X_val2, y_val2)
# history = model.fit([X_train1, X_train2], [y_train1, y_train2], epochs=epochs, batch_size=batch_size, validation_data=([X_val1, X_val2], [y_val1, y_val2]))
end_time = time.time()
# test_loss, test_acc = model.evaluate(X_val1, y_val1)
from keras.utils import to_categorical
print (len(X_train1),len(y_train1))
#HASHED TO TRAIN SAFE
print('X_train1 shape:', X_train1.shape)
print('X_train2 shape:', X_train2.shape)
print('y_train1 shape:', y_train1.shape)
print('y_train2 shape:', y_train2.shape)
print('X_val1 shape:', X_val1.shape)
print('X_val2 shape:', X_val2.shape)
print('y_val1 shape:', y_val1.shape)
print('y_val2 shape:', y_val2.shape)
table = [["Training Accuracy", history.history['sparse_categorical_accuracy'][-1]],
         ["Testing Accuracy", test_acc],
         ["Training Time (s)", end_time - start_time]]
#
print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
y_pred_prob = model.predict(X_val2)
y_pred = y_pred_prob.argmax(axis=-1)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# assume y_true and y_pred are your true and predicted labels
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# assume y_true and y_pred are your true and predicted labels
cm = confusion_matrix( y_val2, y_pred )

# create heatmap#
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
# fig, ax = plt.subplots()
# # Assuming you have two arrays: y_true and y_pred
# # y_true contains the true labels of the samples
# # y_pred contains the predicted labels of the samples
# from sklearn.metrics import precision_score, recall_score, f1_score
# precision = precision_score(y_val, y_pred, average='weighted')
# recall = recall_score(y_val, y_pred, average='weighted')
# f1 = f1_score(y_val, y_pred, average='weighted')
#
# print("Precision: {:.2f}".format(precision))
# print("Recall: {:.2f}".format(recall))
# print("F1-score: {:.2f}".format(f1))
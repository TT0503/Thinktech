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
import pandas as pd

df = pd.read_excel('/Users/ahmedatia/ATIA/S D/SHM/Numerical_Work_NI&code /Data/Steel_Model_Acceleration_.xlsx')
#
df.columns
df = df.iloc[1:]

print(df)
result = df.head()
print (result)
print(df.columns)
# Damage_condition = df.loc[:,"Damage_condition"]
# #
# print(Damage_condition)
# Damage_condition= Damage_condition.to_list()
# # #print(Damage_condition)
# Earthquake = df.loc[:,"Earthquake"]
# Earthquake = Earthquake.to_list()

df2= df.drop(['Joint','StepNum'], axis=1)
result = df2.head()
print (result)
print(df2.shape)
arr = df2.to_numpy()


print(t)
from numpy import transpose

df2 =tf.convert_to_tensor(df2)
#printA2 = []
A2 = []
i = 0
j = 1000

while i < 63066 and j < 64066:
    # get a slice of the array
    A = arr[:, i:j]
    # transpose the slice
    A = A.T
    # update the indices
    i = i + 901
    j = j + 901
    # append the slice to the list
    A2.append(A)
print(df2.shape)
list1 = []
for i in range(1, 631601):
    if (i % 901 == 0):
        list1.append(Damage_condition[i])
list2 =[]
for i in range (1, 631601):
    if(i%901==0) :
        list2.append(Earthquake[i])
print(len(list1))

from keras.utils import to_categorical
x = A2
print (x)
x= np.array(x)
y = np.array(list1)
print(list1)

X_train, X_val, y_train, y_val = model_selection.train_test_split (x , y, test_size=.2 )
y_val_2 = y_val
classes = np.unique(np.concatenate((y_train, y_val), axis=0))
num_classes = len(np.unique(y_train))
print(num_classes)
# idx = np.random.permutation(len(X_train))
# X_train = X_train[idx]
# y_train = y_train[idx]

print (X_train)
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
from tensorflow import keras

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
    conv1 = keras.layers.ReLU()(conv1)
    conv1 = keras.layers.Dropout(0.2)(conv1)

    conv2 = keras.layers.Conv1D(filters=128, kernel_size=5, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv2 = keras.layers.Dropout(0.2)(conv2)

    conv3 = keras.layers.Conv1D(filters=256, kernel_size=5, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    conv3 = keras.layers.Dropout(0.2)(conv3)

    conv4 = keras.layers.Conv1D(filters=512, kernel_size=5, padding="same")(conv3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.ReLU()(conv4)
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

model = make_model(input_shape= X_train.shape[1:])
epochs = 100
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
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=(X_val, y_val),
    verbose=1,
)
test_loss, test_acc = model.evaluate(X_val, y_val)

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
# cm = confusion_matrix( y_val, y_pred )

# create heatmap#
# sns.heatmap(cm, annot=True, cmap="Blues")
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.show()
# fig, ax = plt.subplots()
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



# Extract the misclassified signals
misclassified_sequences = []
for i in range(len(y_val)):
    if (y_pred[i] != y_val[i]):
        misclassified_sequences.append(i)

# Convert the misclassified signals to a numpy array
misclassified_signals = np.array( misclassified_sequences)

print(misclassified_signals)


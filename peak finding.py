import numpy as np
from scipy.signal import find_peaks
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
mat_data01 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/Data_Experimental/Initial_data /Test34/Acc 1.mat')
mat_data02 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/Data_Experimental/Initial_data /Test34/Acc 2.mat')
mat_data03 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/Data_Experimental/Initial_data /Test34/Acc 3.mat')
mat_data04 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/Data_Experimental/Initial_data /Test34/Acc 4.mat')
mat_data05 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/Data_Experimental/Initial_data /Test34/Acc 5.mat')
mat_data06 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/Data_Experimental/Initial_data /Test34/Acc 6.mat')
mat_data07 = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/Data_Experimental/Initial_data /Test34/Acc 7.mat')
Hammer = scipy.io.loadmat('/Users/ahmedatia/ATIA/S D/SHM/Data_Experimental/Initial_data /Test34/hammer.mat')
mat_data01=mat_data01['Data1_Acc_1__1__minus__AI_1__100Hz']
mat_data02=mat_data02['Data1_Acc_2__1__minus__AI_2__100Hz']
mat_data03=mat_data03['Data1_Acc_3__1__minus__AI_3__100Hz']
mat_data04=mat_data04['Data1_Acc_4__1__minus__AI_4__100Hz']
mat_data05=mat_data05['Data1_Acc_5__1__minus__AI_5__100Hz']
mat_data06=mat_data06['Data1_Acc_6__1__minus__AI_6__100Hz']
mat_data07=mat_data07['Data1_Acc_7__1__minus__AI_7__100Hz']
Hammer=Hammer['Data1_hammer__1__minus__AI_8__100Hz']

from numpy import transpose
mat_data01 = transpose(mat_data01)
mat_data02 = transpose(mat_data02)
mat_data03 = transpose(mat_data03)
mat_data04 = transpose(mat_data04)
mat_data05 = transpose(mat_data05)
mat_data06 = transpose(mat_data06)
mat_data07 = transpose(mat_data07)
Hammer = transpose(Hammer)
mat_data01= mat_data01[-1]
mat_data02= mat_data02[-1]
mat_data03= mat_data03[-1]
mat_data04= mat_data04[-1]
mat_data05= mat_data05[-1]
mat_data06= mat_data06[-1]
mat_data07= mat_data07[-1]
Hammer= Hammer[-1]
from scipy.signal import find_peaks
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

# Example usage
fs = 100  # Sample rate (Hz)
t = np.arange(0, 1, 1/fs)  # Time vector


# Calculate the maximum signal frequency
max_frequency = calculate_max_frequency_from_raw(mat_data06, fs)

print("Maximum signal frequency:", max_frequency, "Hz")
# FRF caculation


import numpy as np
from scipy import fft

def calculate_FRF(input_signal, output_signal, fs):
    # Perform FFT on input and output signals
    input_fft = fft.fft(input_signal)
    output_fft = fft.fft(output_signal)

    # Compute the frequency axis
    freq_axis = np.fft.fftfreq(len(input_signal), 1/fs)

    # Compute FRF
    FRF = output_fft / input_fft

    # Compute phase angle of FRF in degrees
    phase_angle_degrees = np.angle(FRF, deg=True)

    return freq_axis, np.abs(FRF), phase_angle_degrees

# Example usage
# Replace Hammer and mat_data01 with your input and output signals
fs = 100  # Sample rate
# Hammer and mat_data01 are your input and output signals
# Example:
# Hammer = ...
# mat_data01 = ...

# Calculate FRF and phase angle
freq_axis, magnitude, phase_angle_degrees = calculate_FRF(Hammer, mat_data01, fs)
peak_indices, _ = find_peaks(magnitude)
local_maxima_freqs = freq_axis[peak_indices]
# Find resonance frequency where phase angle reaches 90 degrees


freq_axis_90 =  np.argmin(np.abs(phase_angle_degrees - 90))

Freq_axis_resonse=[]
for i in range(len(freq_axis_90)):
    for j in range(len(local_maxima_freqs)):
        if freq_axis[i] == freq_axis_90[i] and freq_axis[j] == local_maxima_freqs[j]:
            Freq_axis_resonse.append(freq_axis[j])

# Plot the FRF magnitude and phase angle in degrees
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Plot FRF magnitude
ax1.plot(freq_axis, magnitude, color='tab:red')
ax1.set_ylabel('Magnitude')

# Plot phase angle in degrees
ax2.plot(freq_axis, phase_angle_degrees, color='tab:blue')
ax2.set_ylabel('Phase Angle (degrees)')
ax2.set_xlabel('Frequency (Hz)')

# Highlight resonance frequency on FRF plot
ax1.axvline(freq_axis[Freq_axis_resonse], color='black', linestyle='--', label='Resonance Frequency')

fig.suptitle('Frequency Response Function (FRF) and Phase Angle')
plt.grid(True)
plt.legend()
plt.show()


# -*- coding: utf-8 -*-

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
from scipy.io import wavfile

"""
Baixando um dataset (cat / dog)
Fonte: https://github.com/1fmusic/Audio_cat_dog_classification


git clone https://github.com/1fmusic/Audio_cat_dog_classification
"""

#Sons de gatos

y, sr = librosa.load('./Audio_cat_dog_classification/cats/cat_6.wav')
plt.plot(y);
plt.title('Signal (Cat)');
plt.xlabel('Time (samples)');
plt.ylabel('Amplitude');


y #vetor de caracteristicas


#EXECUTANDO UM PLAYER
# Load the file on an object
data = wavfile.read('./Audio_cat_dog_classification/cats/cat_3.wav')

# Separete the object elements
framerate = data[0]
sounddata = data[1]
time      = np.arange(0,len(sounddata))/framerate

# Show information about the object
print('Sample rate:',framerate,'Hz')
print('Total time:',len(sounddata)/framerate,'s')

# Generate a player for mono sound
Audio(sounddata,rate=framerate)


#Sons de cachorros

y, sr = librosa.load('./Audio_cat_dog_classification/dogs/dog_barking_5.wav')
plt.plot(y);
plt.title('Signal (Dog)');
plt.xlabel('Time (samples)');
plt.ylabel('Amplitude');

y

# Load the file on an object
data = wavfile.read('./Audio_cat_dog_classification/dogs/dog_barking_5.wav')

# Separete the object elements
framerate = data[0]
sounddata = data[1]
time      = np.arange(0,len(sounddata))/framerate

# Show information about the object
print('Sample rate:',framerate,'Hz')
print('Total time:',len(sounddata)/framerate,'s')

# Generate a player for mono sound
Audio(sounddata,rate=framerate)


#Transformada de Fourier

#Espectro de um gato

import numpy as np

y, sr = librosa.load('./Audio_cat_dog_classification/cats/cat_3.wav')
n_fft = 2048
ft = np.abs(librosa.stft(y[:n_fft], hop_length = n_fft+1))
plt.plot(ft);
plt.title('Spectrum (Cat)');
plt.xlabel('Frequency Bin');
plt.ylabel('Amplitude');


ft


#Espectro de um cachorro

y, sr = librosa.load('./Audio_cat_dog_classification/dogs/dog_barking_3.wav')
n_fft = 2048
ft = np.abs(librosa.stft(y[:n_fft], hop_length = n_fft+1))
plt.plot(ft);
plt.title('Spectrum (Dog)');
plt.xlabel('Frequency Bin');
plt.ylabel('Amplitude');


#APRESENTANDO UM SPECTRO DO GATO

y, sr = librosa.load('./Audio_cat_dog_classification/cats/cat_6.wav')
spec = np.abs(librosa.stft(y, hop_length=512))
spec = librosa.amplitude_to_db(spec, ref=np.max)
librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log');
plt.colorbar(format='%+2.0f dB');
plt.title('Spectrogram (Cat)');


#APRESENTANDO UM SPECTRO DO CACHORRO

y, sr = librosa.load('./Audio_cat_dog_classification/dogs/dog_barking_3.wav')
spec = np.abs(librosa.stft(y, hop_length=512))
spec = librosa.amplitude_to_db(spec, ref=np.max)
librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram (Dog)')


#APRESENTANDO UM MEL SPECTRO DE UM GATO

y, sr = librosa.load('./Audio_cat_dog_classification/cats/cat_6.wav')
mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
plt.title('Mel Spectrogram (Cat)');
plt.colorbar(format='%+2.0f dB');


#APRESENTANDO UM MEL SPECTRO DE UM CACHORRO

y, sr = librosa.load('./Audio_cat_dog_classification/dogs/dog_barking_3.wav')
mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
plt.title('Mel Spectrogram (Dog)');
plt.colorbar(format='%+2.0f dB');
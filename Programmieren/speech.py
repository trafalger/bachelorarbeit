# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:26:56 2017

@author: wajid
"""
import matplotlib.pyplot as plt
import wave, sys, pyaudio
import numpy as np
from scipy.io.wavfile import read
import librosa
import librosa.display



#sound = wave.open('001.wav.wav')
#print sound.getsampwidth()
#p = pyaudio.PyAudio()
#chunk = 1024
#stream = p.open(format =
#                p.get_format_from_width(sound.getsampwidth()),
#                channels = sound.getnchannels(),
#                rate = sound.getframerate(),
#                output = True)
#data = sound.readframes(chunk)
#while data != '':
#    stream.write(data)
#    data = sound.readframes(chunk)
# 
#p.terminate()

#Gib die Sampelrate und das Signal als Array zurück
samplerate,data = read("001.wav.wav")

data_fft = fft(data)

plt.plot(data)
plt.ylabel('Amplitude')
plt.show()
#TODO
#normalisieren
#FFT

#MFCC
# y = np.Array(Data) , sr = samplingrate default 22khz
y, sr = librosa.load("001.wav.wav")
librosa.feature.mfcc(y=y, sr=sr)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
librosa.feature.mfcc(S=librosa.power_to_db(S))



plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()

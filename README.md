## AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss

This repository provides a PyTorch implementation of AUTOVC.

AUTOVC is a many-to-many non-parallel voice conversion framework. 

**_To ensure respect for privacy rights and responsible use of our code, we are only releasing a portion of our code to allow users to convert voices among a predefined set of speakers in VCTK. Conversions from and to other voices have been disabled._**


### Audio Demo

The audio demo for AUTOVC can be found [here](https://auspicious3000.github.io/autovc-demo/)

### Dependencies
- Python 3
- Numpy
- PyTorch >= v0.4.1
- TensorFlow >= v1.3 (only for tensorboard)
- librosa
- tqdm
- wavenet_vocoder ```pip install wavenet_vocoder```
  for more information, please refer to https://github.com/r9y9/wavenet_vocoder

### Pre-trained models

| AUTOVC | WaveNet Vocoder |
|----------------|----------------|
| [link](https://drive.google.com/file/d/1SZPPnWAgpGrh0gQ7bXQJXXjOntbh4hmz/view?usp=sharing)| [link](https://drive.google.com/file/d/1Zksy0ndlDezo9wclQNZYkGi_6i7zi4nQ/view?usp=sharing) |


### 0.Converting Mel-Spectrograms

Download pre-trained AUTOVC model, and run the ```conversion.ipynb``` in the same directory.


### 1.Mel-Spectrograms to waveform

Download pre-trained WaveNet Vocoder model, and run the ```vocoder.ipynb``` in the same the directory.


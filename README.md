# Music2Sheet: WAV to Sheet Music Converter

## Overview

Music2Sheet is an AI-based project that converts WAV files to sheet music. Utilizing the MusicNet dataset, this project processes audio data, trains a neural network model, and generates corresponding sheet music. The project leverages deep learning techniques and PyTorch for implementation.

## Features

- **Data Loading and Processing**: Efficient handling of large audio files and labels using memory mapping and interval trees. (Use mmap if >21GB Ram)
- **Model Training**: Uses a neural network to learn the mapping from audio signals to musical notes.
- **Mixed Precision Training**: Incorporates mixed precision training to optimize performance.
- **GPU Acceleration**: Leverages GPU capabilities for faster training. (Use MPS if CUDA not supported)

## Dataset

The project uses the [MusicNet dataset](https://www.kaggle.com/datasets/imsparsh/musicnet-dataset), which consists of classical music recordings along with their corresponding labels.

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- SciPy
- Scikit-learn
- Librosa
- Matplotlib
- IPython
- Intervaltree

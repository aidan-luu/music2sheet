import os
import librosa
import pandas as pd

def load_wav_files(data_dir):
    wav_files = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(data_dir, filename)
            wav_files.append(file_path)
    return wav_files

def load_labels(label_dir):
    labels = {}
    for filename in os.listdir(label_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(label_dir, filename)
            labels[filename.split('.')[0]] = pd.read_csv(file_path)
    return labels

train_data_dir = 'data/musicnet/train_data'
train_label_dir = 'data/musicnet/train_labels'
test_data_dir = 'data/musicnet/test_data'
test_label_dir = 'data/musicnet/test_labels'

train_wav_files = load_wav_files(train_data_dir)
train_labels = load_labels(train_label_dir)
test_wav_files = load_wav_files(test_data_dir)
test_labels = load_labels(test_label_dir)

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return mfccs, chroma

train_features = []
train_targets = []

for wav_file in train_wav_files:
    mfccs, chroma = extract_features(wav_file)
    file_id = os.path.basename(wav_file).split('.')[0]
    target = train_labels[file_id]
    train_features.append((mfccs, chroma))
    train_targets.append(target)

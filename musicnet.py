from __future__ import print_function
from subprocess import call
import torch.utils.data as data
import os,mmap
import os.path
import pickle
import errno
import csv
import numpy as np
import torch
import requests
import urllib3

from intervaltree import IntervalTree
from scipy.io import wavfile

sz_float = 4    # size of a float
epsilon = 10e-8 # fudge factor for normalization

class MusicNet(data.Dataset):
    train_data, train_labels, train_tree = 'train_data', 'train_labels', 'train_tree.pckl'
    test_data, test_labels, test_tree = 'test_data', 'test_labels', 'test_tree.pckl'

    def __init__(self, root, train=True, download=False, refresh_cache=True, mmap=False, normalize=True, window=16384, pitch_shift=0, jitter=0., epoch_size=100000):
        self.refresh_cache = refresh_cache
        self.mmap = mmap
        self.normalize = normalize
        self.window = window
        self.pitch_shift = pitch_shift
        self.jitter = jitter
        self.size = epoch_size
        self.m = 128

        self.root = os.path.expanduser(root.strip())

        if train:
            self.data_path = os.path.join(self.root, self.train_data)
            labels_path = os.path.join(self.root, self.train_labels, self.train_tree)
        else:
            self.data_path = os.path.join(self.root, self.test_data)
            labels_path = os.path.join(self.root, self.test_labels, self.test_tree)

        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)

        self.rec_ids = list(self.labels.keys())
        self.records = dict()
        self.open_files = []

    def __enter__(self):
        for record in os.listdir(self.data_path):
            if not record.endswith('.bin'): continue
            if self.mmap:
                fd = os.open(os.path.join(self.data_path, record), os.O_RDONLY)
                buff = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
                self.records[int(record[:-4])] = (buff, len(buff)//sz_float)
                self.open_files.append(fd)
            else:
                f = open(os.path.join(self.data_path, record))
                self.records[int(record[:-4])] = (os.path.join(self.data_path, record),os.fstat(f.fileno()).st_size//sz_float)
                f.close()

    def __exit__(self, *args):
        if self.mmap:
            for mm in self.records.values():
                mm[0].close()
            for fd in self.open_files:
                os.close(fd)
            self.records = dict()
            self.open_files = []

    def access(self, rec_id, s, shift=0, jitter=0):
        scale = 2.**((shift + jitter) / 12.)

        if rec_id not in self.records:
            print(f"Error: Record ID {rec_id} not found in records.")
            return None, None

        if self.mmap:
            x = np.frombuffer(self.records[rec_id][0][s * sz_float:int(s + scale * self.window) * sz_float], dtype=np.float32).copy()
        else:
            fid, _ = self.records[rec_id]
            with open(fid, 'rb') as f:
                f.seek(s * sz_float, os.SEEK_SET)
                x = np.fromfile(f, dtype=np.float32, count=int(scale * self.window))

        if self.normalize:
            x /= np.linalg.norm(x) + epsilon

        xp = np.arange(self.window, dtype=np.float32)
        x = np.interp(scale * xp, np.arange(len(x), dtype=np.float32), x).astype(np.float32)

        y = np.zeros(self.m, dtype=np.float32)
        for label in self.labels[rec_id][s + scale * self.window / 2]:
            y[label.data[1] + shift] = 1

        return x, y

    def __getitem__(self, index):
        shift = 0
        if self.pitch_shift > 0:
            shift = np.random.randint(-self.pitch_shift, self.pitch_shift)

        jitter = 0.
        if self.jitter > 0:
            jitter = np.random.uniform(-self.jitter, self.jitter)

        rec_id = self.rec_ids[np.random.randint(0, len(self.rec_ids))]
        if rec_id not in self.records:
            print(f"Error: Record ID {rec_id} not found in records.")
            return None

        s = np.random.randint(0, self.records[rec_id][1] - (2.**((shift + jitter) / 12.)) * self.window)
        return self.access(rec_id, s, shift, jitter)

    def __len__(self):
        return self.size

    def process_data(self, path):
        data_dir = os.path.join(self.root, path)
        if not os.path.exists(data_dir):
            print(f"Data directory {data_dir} not found.")
            return
        for item in os.listdir(data_dir):
            if not item.endswith('.wav'):
                continue
            uid = int(item[:-4])
            wav_path = os.path.join(data_dir, item)
            print(f"Processing {wav_path}...")
            try:
                _, data = wavfile.read(wav_path)
                bin_path = os.path.join(data_dir, item[:-4] + '.bin')
                data.tofile(bin_path)
                print(f"Saved binary file to {bin_path}")
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

    def process_labels(self, path):
        trees = dict()
        label_dir = os.path.join(self.root, path)
        if not os.path.exists(label_dir):
            print(f"Label directory {label_dir} not found.")
            return trees
        for item in os.listdir(label_dir):
            if not item.endswith('.csv'):
                continue
            uid = int(item[:-4])
            tree = IntervalTree()
            with open(os.path.join(label_dir, item), 'r') as f:
                reader = csv.DictReader(f, delimiter=',')
                for label in reader:
                    start_time = int(label['start_time'])
                    end_time = int(label['end_time'])
                    instrument = int(label['instrument'])
                    note = int(label['note'])
                    start_beat = float(label['start_beat'])
                    end_beat = float(label['end_beat'])
                    note_value = label['note_value']
                    tree[start_time:end_time] = (instrument, note, start_beat, end_beat, note_value)
            trees[uid] = tree
        print(f"Processed labels in {label_dir}")
        return trees
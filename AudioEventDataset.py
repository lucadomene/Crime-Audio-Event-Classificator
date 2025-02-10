import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import os
from sklearn.preprocessing import LabelEncoder
from glob import glob
import numpy as np

class AudioEventDataset(Dataset):

    def __init__(self, src_dir, frame_size, hop_size, sample_rate):
        super().__init__()
        self.src_dir = src_dir
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate

        audio_paths_absolute = [x for x in glob('{}/**'.format(src_dir), recursive = True) if '.wav' in x]
        audio_paths = []
        for path in audio_paths_absolute:
            audio_paths.append(os.path.relpath(path, src_dir))

        self.paths = audio_paths

        self.classes = sorted(list(set( [x.split('/')[0] for x in self.paths] )))
        self.n_classes = len(self.classes)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)

        print(f"Total samples found: {len(self)}\n")
        print(f"Total classes found: {self.n_classes}")
        for class_item in self.classes:
            print(f"    - {class_item}")
        print('\n')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        
        path = self.paths[index]
       
        audio_data, _ = torchaudio.load(os.path.join(self.src_dir, path), normalize = True)
        framed_audio_data = AudioEventDataset.frame_audio_overlap(audio_data, self.frame_size, self.sample_rate, self.hop_size)
        label = path.split('/')[0]

        label = self.label_encoder.transform([label])[0]
        labels = np.array([np.eye(self.n_classes, dtype = 'float')[label]])

        labels = np.repeat(labels, framed_audio_data.size()[0], axis = 0)

        return framed_audio_data, torch.tensor(labels)

    @staticmethod
    def frame_audio_overlap(audio_tensor, frame_size, sample_rate, hop_size):
        audio = audio_tensor[0]
        frame_size_samples = int(frame_size * sample_rate)
        
        if hop_size:
            hop_size_samples = int(hop_size * sample_rate)
        else: hop_size_samples = frame_size_samples

        frames = torch.empty((0, frame_size_samples), dtype = torch.float32)

        for start in range(0, audio.size()[0], hop_size_samples):
            end = start + frame_size_samples
            frame = audio[start:end]
            if len(frame) < frame_size_samples:
                if len(frame) >= (frame_size_samples / 2):
                    frame = nn.functional.pad(frame, (0, frame_size_samples - frame.size()[0]), 'constant', value = 0)
                else: break
            frame = torch.reshape(frame, (1, -1))
            frames = torch.cat([frames, frame], dim = 0)

        return frames

if __name__ == '__main__':
    from glob import glob
    import os
    src_dir = '/home/ldomeneghetti/Documents/Forensics/audio_classification_pytorch/raw_audio/car_crash'
    audio_paths_absolute = [x for x in glob('{}/**'.format(src_dir), recursive = True) if '.wav' in x]
    audio_paths = []
    for path in audio_paths_absolute:
        audio_paths.append(os.path.relpath(path, src_dir))

    dataset = AudioEventDataset(audio_paths, src_dir, 1, 0.5, 44100)

    item = dataset.__getitem__(0)

    print(item)

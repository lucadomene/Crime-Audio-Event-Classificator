import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms
import math
from torchinfo import summary

class LSTMNetwork(nn.Module):

    def __init__(self, n_classes, sample_rate = 44100, sample_duration = 1.0, print_summary = True):

        super(LSTMNetwork, self).__init__()

        self.melspectrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            n_mels=128,
            hop_length=256,
            f_max= int(sample_rate / 2)
        )
        self.amplitude_to_db = transforms.AmplitudeToDB()

        self.mfcc = transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            melkwargs={"n_fft": 2048, "hop_length": 256, "n_mels": 128, "f_max": int(sample_rate / 2)}
        )

        self.layer_norm_mel = nn.LayerNorm(128)
        self.layer_norm_mfcc = nn.LayerNorm(40)

        self.bidirectional_lstm_mel = nn.LSTM(
            input_size = 128, hidden_size = 256, num_layers = 1, batch_first = True, bidirectional = True
        )

        self.bidirectional_lstm_mfcc = nn.LSTM(
            input_size = 40, hidden_size = 256, num_layers = 1, batch_first = True, bidirectional = True
        )

        self.dense_1_relu = nn.Linear(1024, 512)  # LSTM outputs (512 + 512)
        self.batch_norm_1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.dense_2_relu = nn.Linear(512, 128)
        self.batch_norm_2 = nn.BatchNorm1d(128)
        self.dense_3_relu = nn.Linear(128, 64)
        self.batch_norm_3 = nn.BatchNorm1d(64)

        self.output_layer = nn.Linear(64, n_classes)

        if(print_summary): summary(self, torch.Size([1, 44100]))

    def forward(self, x):
        x_mel = self.melspectrogram(x)
        x_mel = self.amplitude_to_db(x_mel)

        x_mfcc = self.mfcc(x)

        x_mel = x_mel.permute(0, 2, 1)
        x_mel = self.layer_norm_mel(x_mel)

        x_mfcc = x_mfcc.permute(0, 2, 1)
        x_mfcc = self.layer_norm_mfcc(x_mfcc)

        lstm_out_mel, (h_n_mel, _) = self.bidirectional_lstm_mel(x_mel)
        lstm_out_mfcc, (h_n_mfcc, _) = self.bidirectional_lstm_mfcc(x_mfcc)

        x_mel = torch.cat([h_n_mel[-2,:,:], h_n_mel[-1,:,:]], dim=1)
        #x_mel = lstm_out_mel[:, -1, :]
        x_mfcc = torch.cat([h_n_mfcc[-2,:,:], h_n_mfcc[-1,:,:]], dim=1)
        #x_mfcc = lstm_out_mfcc[:, -1, :]

        x = torch.cat([x_mel, x_mfcc], dim=1)

        x = self.dense_1_relu(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)

        x = self.dropout(x) 

        x = self.dense_2_relu(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)
        x = self.dense_3_relu(x)
        x = self.batch_norm_3(x)
        x = self.relu(x)

        logits = self.output_layer(x)

        return logits

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lstm = LSTMNetwork(10, 44100, 1.0).to(device)
    summary(lstm, torch.Size([1, 44100]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from AudioEventDataset import AudioEventDataset
from models import LSTMNetwork

from tqdm import tqdm

import argparse

import os

import json

def collate_audio(batch):
    data_list = []
    label_list = []
    for element in batch:
        data_list.append(element[0])
        label_list.append(element[1])

    data_tensor = torch.cat(data_list, dim = 0)
    label_tensor = torch.cat(label_list, dim = 0)

    return data_tensor, label_tensor


def initialize_train(args):

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    src_dir = args.src_dir
    batch_size = args.batch_size
    frame_size = args.frame_size
    hop_size = args.hop_size
    sample_rate = args.sample_rate
    test_size = args.test_size
    random_state = args.random_state
    epochs = args.epochs
    out_file = args.out_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_dataset = AudioEventDataset(src_dir, frame_size, hop_size, sample_rate)

    assert (test_size > 0 and test_size < 1), "test_size must be strictly between 0 and 1"
    train_dataset, test_dataset = random_split(audio_dataset, [1.0 - test_size, test_size], generator = torch.Generator().manual_seed(random_state))

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_audio)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, collate_fn = collate_audio)

    with open('classes.json', 'w', encoding = 'utf-8') as json_file:
        json.dump({'classes': audio_dataset.classes}, json_file, ensure_ascii = False, indent = 4)

    lstm_model = LSTMNetwork(audio_dataset.n_classes, sample_rate, frame_size).to(device)

    train(lstm_model, train_dataloader, test_dataloader, epochs, device, audio_dataset.classes)

    torch.save(lstm_model.state_dict(), out_file + "_dict.pth")
    torch.save(lstm_model, out_file + ".pth")



def train(model, train_dl, test_dl, epochs, device, classes):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr = 0.001,
                                                    steps_per_epoch = int(len(train_dl)),
                                                    epochs = epochs,
                                                    anneal_strategy = 'linear')

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)


        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"    Train Loss: {train_loss:.4f}        Train Acc:  {train_acc:.4f}")
        print("----------------------------------------------------------------")
        
    test_loss, test_acc = test_model(model, test_dl, criterion, classes)
    print("\nFinal model:")
    print(f"    Train Loss: {train_loss:.4f}        Train Acc:  {train_acc:.4f}")
    print(f"    Test Loss: {test_loss:.4f}        Test Acc:  {test_acc:.4f}")


def train_one_epoch(model, dl, criterion, optimizer, device):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_data = 0

    progress_bar = tqdm(dl, desc = "Training", unit = "batch")

    for data, target in progress_bar:
        data = data.to(device)
        target = target.to(device)

        model.zero_grad()

        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, dim = 1)
        _, correct = torch.max(target, dim = 1)

        total_data += data.size(0)

        partial_correct = torch.sum(predicted == correct).item()
        partial_accuracy = partial_correct / data.size(0)
        partial_loss = loss.item()

        total_correct += partial_correct
        total_loss += partial_loss * data.size(0)

        progress_bar.set_postfix({"Accuracy": partial_accuracy, "Loss": partial_loss})

    epoch_loss = total_loss / total_data
    epoch_acc = total_correct / total_data

    return epoch_loss, epoch_acc



def test_model(model, dl, criterion, classes):

    device = torch.device('cpu')

    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_data = 0

    progress_bar = tqdm(dl, desc = "Testing", unit = "batch")

    correct_labels = torch.empty((0), dtype = int)
    predicted_labels = torch.empty((0), dtype = int)

    with torch.no_grad():
        for batch_index, (data, target) in enumerate(dl):
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            loss = criterion(output, target)

            _, predicted = torch.max(output, dim = 1)
            _, correct = torch.max(target, dim = 1)

            correct_labels = torch.cat([correct_labels, correct])
            predicted_labels = torch.cat([predicted_labels, predicted])

            total_data += data.size(0)

            partial_correct = torch.sum(predicted == correct).item()
            partial_accuracy = partial_correct / data.size(0)
            partial_loss = loss.item()

            total_correct += partial_correct
            total_loss += partial_loss * data.size(0)

            progress_bar.set_postfix({"Accuracy": partial_accuracy, "Loss": partial_loss})

    display = ConfusionMatrixDisplay.from_predictions(correct_labels.tolist(), predicted_labels.tolist(), display_labels = classes, xticks_rotation = 'vertical')
    display.plot()
    plt.show()

    test_loss = total_loss / total_data
    test_acc = total_correct / total_data

    return test_loss, test_acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Audio Event Classification training")

    parser.add_argument('--src-dir', type = str, required = True, help = "directory of source audio files")

    parser.add_argument('--batch-size', type = int, default = 16, help = "batch size")

    parser.add_argument('--frame-size', type = float, default = 1.0, help = "audio frame size in seconds")

    parser.add_argument('--hop-size', type = float, default = 0.5, help = "audio hop size in seconds")

    parser.add_argument('--sample-rate', type = int, default = 44100, help = "sample rate")

    parser.add_argument('--test-size', type = float, default = 0.1, help = "ammount of samples (from 0.0 to 1.0) to be used for testing")

    parser.add_argument('--random-state', type = int, default = 2159017, help = "random state for samples shuffling")

    parser.add_argument('--epochs', type = int, default = 16, help = "epochs to train the network")

    parser.add_argument('--out-file', type = str, default = "best_model", help = "name for the output model file")

    args, _ = parser.parse_known_args()

    initialize_train(args)

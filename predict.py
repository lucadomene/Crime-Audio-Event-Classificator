from AudioEventDataset import AudioEventDataset

import torchaudio
import torch

from models import LSTMNetwork

import argparse

import json

import math

import pygame

def initialize_predict(args):

    audio_file = args.audio_file
    win_hop = args.win_hop
    win_min = args.win_min
    threshold = args.threshold
    model_file = args.model_file
    param_file = args.param_file

    with open(param_file, 'r') as file:
        data_parameters = json.load(file)

    frame_size = data_parameters['frame_size']
    sample_rate = data_parameters['sample_rate']
    classes = data_parameters['classes']

    audio_data, _ = torchaudio.load(audio_file, normalize = True)
    play_audio(audio_file)
    processed_audio = AudioEventDataset.frame_audio_overlap(audio_data, frame_size, sample_rate, win_hop)

    lstm_model = LSTMNetwork(len(classes), sample_rate, frame_size, print_summary = False)
    lstm_model.load_state_dict(torch.load(model_file, weights_only = True))

    predict(lstm_model, processed_audio, classes, threshold, sample_rate, frame_size, win_hop, win_min)


def predict(model, input, classes, threshold, sample_rate, frame_size, hop_size, min_frames):
    model.to('cpu')
    model.eval()

    input = input.to('cpu')

    output = model(input)

    output_softmax = torch.nn.Sigmoid()(output)
    output_normalized = torch.nn.functional.normalize(output_softmax)

    predicted_classes = []
    for tensor in output_normalized:
        value, index = torch.max(tensor, dim = 0)
        if value.item() >= threshold:
            predicted_classes.append(index.item())
        else: predicted_classes.append(-1)

    detected_events = group_contiguous(predicted_classes, min_frames, frame_size, hop_size)

    print("\n##################### DETECTED EVENTS #####################")
    for event in detected_events:
        start_sec = event[1]*hop_size
        end_sec = (event[2])*hop_size + frame_size
        print(f"    - {classes[event[0]]}: start {start_sec:.2f} sec  /  end {end_sec:.2f} sec")
    print("###########################################################\n")


def group_contiguous(input_list, min_contiguous, frame_size, hop_size):
    grouped_list = []
    previous_item = input_list[0]

    count = 0
    start_index = 0
    end_index = 1
    for index, item in enumerate(input_list):
        if item != previous_item:
            if previous_item != -1 and count >= min_contiguous:
                end_index = start_index + count - 1
                grouped_list.append([previous_item, start_index, end_index])
            start_index = index
            previous_item = item
            count = 1
        else: count += 1

    i = 1
    end = len(grouped_list)
    while i < end:
        if grouped_list[i][0] == grouped_list[i-1][0]:
            next_start = grouped_list[i][1]
            previous_end = grouped_list[i-1][2]
            if next_start - previous_end <= math.floor(frame_size/hop_size):
                grouped_list[i][1] = grouped_list[i-1][1]
                del grouped_list[i-1]
                end -= 1
        i += 1

    return grouped_list

def play_audio(filename):

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Audio Event Classification prediction")

    parser.add_argument('--audio-file', type = str, required = True, help = 'source audio file')

    parser.add_argument('--win-hop', type = float, default = 0.2, help = 'detection window hop length')

    parser.add_argument('--win-min', type = int, default = 2, help = 'minimum contiguous windows required to detect an event')

    parser.add_argument('--threshold', type = float, default = 0.8, help = 'value above which a prediction is considered valid')

    parser.add_argument('--model-file', type = str, default = 'lstm_network.pth', help = 'filename of the model to be loaded')

    parser.add_argument('--param-file', type = str, default = 'parameters.json', help = 'filename of the parameters file')

    args, _ = parser.parse_known_args()

    initialize_predict(args)


import pypianoroll # change in outputs.py
from mido import MidiFile
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from BinaryTokenizer import BinaryTokenizer
from tqdm import tqdm
import io
# https://copyprogramming.com/howto/convert-bytes-into-bufferedreader-object-in-python

from BinaryTokenizer import BinaryTokenizer
from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile
# from pathlib import Path
import pandas as pd

# Our tokenizer's configuration
PITCH_RANGE = (21, 109)
BEAT_RES = {(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1}
NB_VELOCITIES = 24
SPECIAL_TOKENS = ["PAD", "MASK", "BOS", "EOS"]
USE_CHORDS = False
USE_RESTS = False
USE_TEMPOS = False
USE_TIME_SIGNATURE = False
USE_PROGRAMS = False
NB_TEMPOS = 32
TEMPO_RANGE = (50, 200)  # (min_tempo, max_tempo)
TOKENIZER_PARAMS = {
    "pitch_range": PITCH_RANGE,
    "beat_res": BEAT_RES,
    "num_velocities": NB_VELOCITIES,
    "special_tokens": SPECIAL_TOKENS,
    "use_chords": USE_CHORDS,
    "use_rests": USE_RESTS,
    "use_tempos": USE_TEMPOS,
    "use_time_signatures": USE_TIME_SIGNATURE,
    "use_programs": USE_PROGRAMS,
    "num_tempos": NB_TEMPOS,
    "tempo_range": TEMPO_RANGE,
}
config = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(config)

# make folder structure
os.makedirs('data', exist_ok=True)
os.makedirs('data/pianorolls', exist_ok=True)
os.makedirs('data/midis', exist_ok=True)
os.makedirs('data/midi_tokens', exist_ok=True)

binaryTokenizer = BinaryTokenizer(num_digits=12)

midifolder = '../data/giantmidi/'
midifiles = os.listdir(midifolder)

# size in beats
segment_size = 64
piece_idx = 0

def make_segment(main_piece, tmp_pianoroll, start_idx, end_idx, piece_idx, transposition_idx, segment_idx):
    new_piece = deepcopy(main_piece)
    new_piece.tracks[0].pianoroll = deepcopy(tmp_pianoroll)
    # trim and binarize
    new_piece.trim(start_idx,end_idx)
    new_piece.binarize()
    # make chroma
    chroma = new_piece.tracks[0].pianoroll[:,:12]
    for i in range(12, 128-12, 12):
        chroma = np.logical_or(chroma, new_piece.tracks[0].pianoroll[:,i:(i+12)])
    chroma[:,-6:] = np.logical_or(chroma[:,-6:], new_piece.tracks[0].pianoroll[:,-6:])
    indexed_chroma = binaryTokenizer.transform( chroma )
    # make name of piece
    piece_name = f'p{piece_idx:05}_t{transposition_idx:02}_s{segment_idx:04}'
    # initialize bytes handle
    b_handle = io.BytesIO()
    # write midi data to bytes handle
    new_piece.write(b_handle)
    # start read pointer from the beginning
    b_handle.seek(0)
    # create a buffered reader to read the handle
    buffered_reader = io.BufferedReader(b_handle)
    # create a midi object from the "file", i.e., buffered reader
    midi_object = MidiFile(file=buffered_reader)
    # close the bytes handle
    b_handle.close()
    # print('midi_object: ', midi_object)
    tmp_tokens = tokenizer.midi_to_tokens( midi_object, apply_bpe_if_possible=True)
    return piece_name, tmp_tokens[0].ids, indexed_chroma

# initialize catalogue
with open('piece_per_idx.txt', 'w') as f:
    print('piece_idx, file_name', file=f)

names = []
chromas = []
tokens = []
is_starting_segment = []
is_ending_segment = []

for midifile in tqdm(midifiles):
    main_piece = pypianoroll.read(midifolder + midifile)
    # keep size to know when to end
    main_piece_size = main_piece.downbeat.shape[0]
    transposition_idx = 0
    with open('piece_per_idx.txt', 'a') as f:
        print(f'{piece_idx:05}, {midifile}', file=f)
    # roll
    for r in tqdm(range(-6,6,1), leave=False):
        # print(f'running for roll: {r}')
        start_idx = 0
        end_idx = segment_size*main_piece.resolution
        tmp_pianoroll = np.roll(main_piece.tracks[0].pianoroll, [0,r])
        segment_idx = 0
        while end_idx < main_piece_size:
            # print(f'running for start: {start_idx} and end: {end_idx}')
            piece_name, tmp_tokens, indexed_chroma = make_segment(main_piece, tmp_pianoroll, start_idx, end_idx, piece_idx, transposition_idx, segment_idx)
            start_idx = end_idx
            end_idx += segment_size*main_piece.resolution
            names.append(piece_name)
            chromas.append(np.array(indexed_chroma))
            tokens.append(np.array(tmp_tokens))
            is_starting_segment.append(segment_idx == 0)
            is_ending_segment.append(False)
            segment_idx += 1
        # end end_idx while
        end_idx = main_piece_size
        start_idx = end_idx - segment_size*main_piece.resolution
        if start_idx > 0:
            piece_name, tmp_tokens, indexed_chroma = make_segment(main_piece, tmp_pianoroll, start_idx, end_idx, piece_idx, transposition_idx, segment_idx)
            names.append(piece_name)
            chromas.append(np.array(indexed_chroma))
            tokens.append(np.array(tmp_tokens))
            is_starting_segment.append(segment_idx == 0)
            is_ending_segment.append(True)
        transposition_idx += 1
    # end transposition range for
    piece_idx += 1
# end midifile for

d = {
    'names': names,
    'chromas': chromas,
    'tokens': tokens,
    'start': is_starting_segment,
    'end': is_ending_segment
}

df = pd.DataFrame.from_dict(d)
df.to_csv('data/test_df.csv', sep='\t')
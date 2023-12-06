from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile
from pathlib import Path

# Creating a multitrack tokenizer configuration, read the doc to explore other parameters
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(config)

path_to_dataset = 'data/maestro-v3.0.0/'
path_to_tokens = 'data/maestro-v3.0.0_noBPE/'
path_to_tokens_bpe = 'data/maestro-v3.0.0_BPE/'
path_to_tokenizer_config = 'data/maestro-v3.0.0_tokenizer.json'
# Tokenize a whole dataset and save it at Json files
midi_paths = list(Path(path_to_dataset).glob("**/*.midi"))

data_augmentation_offsets = [2, 1, 1]  # data augmentation on 2 pitch octaves, 1 velocity and 1 duration values
tokenizer.tokenize_midi_dataset(midi_paths, Path(path_to_tokens),
                                data_augment_offsets=data_augmentation_offsets)

# Constructs the vocabulary with BPE, from the token files
tokenizer.learn_bpe(
    tokens_paths=list(Path(path_to_tokens).glob("**/*.json")),
    start_from_empty_voc=True,
)

# Saving our tokenizer, to retrieve it back later with the load_params method
tokenizer.save_params(Path(path_to_tokenizer_config))

# Applies BPE to the previous tokens
tokenizer.apply_bpe_to_dataset(Path(path_to_tokens), Path(path_to_tokens_bpe))
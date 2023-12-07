import sys
sys.path.insert(0, '..')
from transformer.models import EncoderModel, MLMEncoderWrapper
from torch import randint, rand
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from mlm_pytorch import MLM
from torchsummary import summary

from pathlib import Path
from torch.utils.data import DataLoader
from torchtoolkit.data import create_subsets
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetTok, DataCollator

# load tokenizer
saved_tokenizer_path = '../data_preparation/data/maestro_small_tokenizer.json'
tokenizer = REMI(params=Path(saved_tokenizer_path))

# print(tokenizer.vocab)

src_vocab_size = len(tokenizer.vocab)
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 512
min_seq_len=384
dropout = 0.1
batch_size = 16
epochs = 10

# load data
# path_to_tokens_bpe = '../data_preparation/data/maestro_small_BPE/'
# tokens_paths = list(Path(path_to_tokens_bpe).glob("**/*.json"))
path_to_tokens = '../data_preparation/data/maestro_small_noBPE/'
tokens_paths = list(Path(path_to_tokens).glob("**/*.json"))
dataset = DatasetTok(
    tokens_paths, max_seq_len=max_seq_length, min_seq_len=min_seq_len, one_token_stream=False,
)

subset_train, subset_valid = create_subsets(dataset, [0.3])

collator = DataCollator(tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"], copy_inputs_as_labels=False)
dataloader_test = DataLoader(subset_train, batch_size=batch_size, collate_fn=collator)
# dataloader_test = DataLoader(subset_train, batch_size=batch_size)

# print(subset_train[0])
# btch = next(iter(dataloader_test))
# print(btch)
# print(btch['input_ids'].shape)

# data = randint(0, src_vocab_size, (batch_size, max_seq_length)).cuda()
# print(data.shape)

# prepare trainer
encoderModel = EncoderModel(src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length+2, dropout)
mlmEncoderWrapper = MLMEncoderWrapper(encoderModel)

trainer = MLM(
    mlmEncoderWrapper,
    mask_token_id = 1,          # the token id reserved for masking
    pad_token_id = 0,           # the token id for padding
    mask_prob = 0.15,           # masking probability for masked language modeling
    replace_prob = 0.90,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
    mask_ignore_token_ids = []  # other tokens to exclude from masking, include the [cls] and [sep] here
).cuda()

# optimizer
opt = Adam(trainer.parameters(), lr=3e-4)

for epoch in range(epochs):
    for batch in iter(dataloader_test):
        # print(batch['input_ids'].shape)
        output = mlmEncoderWrapper(batch['input_ids'].cuda())
        loss = trainer(batch)
        loss.backward()
        opt.step()
        opt.zero_grad()

# save
# torch.save(transformer, f'./pretrained-model.pt')
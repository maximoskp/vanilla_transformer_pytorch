# for training, check out this repository:
# https://github.com/lucidrains/mlm-pytorch/tree/master
import sys
sys.path.insert(0, '..')
from transformer.models import EncoderModel, MLMEncoderWrapper
from torch import randint, rand
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from mlm_pytorch import MLM
from torchsummary import summary

src_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1
batch_size = 2

from x_transformers import TransformerWrapper, Encoder
transformer = TransformerWrapper(
    num_tokens = src_vocab_size,
    max_seq_len = max_seq_length,
    attn_layers = Encoder(
        dim = d_model,
        depth = num_layers,
        heads = num_heads
    )
)
print(transformer.parameters())
# not working, out of memory
# summary(transformer.cuda(), (batch_size, max_seq_length, src_vocab_size))

# not working with encoderModel:

encoderModel = EncoderModel(src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
mlmEncoderWrapper = MLMEncoderWrapper(encoderModel)
print(encoderModel.parameters())
# not working:
# RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
# summary(encoderModel.cuda(), (batch_size, max_seq_length, src_vocab_size))

trainer = MLM(
    mlmEncoderWrapper,
    mask_token_id = 2,          # the token id reserved for masking
    pad_token_id = 0,           # the token id for padding
    mask_prob = 0.15,           # masking probability for masked language modeling
    replace_prob = 0.90,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
    mask_ignore_token_ids = []  # other tokens to exclude from masking, include the [cls] and [sep] here
).cuda()

# optimizer

opt = Adam(trainer.parameters(), lr=3e-4)

# one training step (do this for many steps in a for loop, getting new `data` each time)

data = randint(0, src_vocab_size, (batch_size, max_seq_length)).cuda()
output = mlmEncoderWrapper(data)
print(output.shape)

loss = trainer(data)
loss.backward()
opt.step()
opt.zero_grad()
# vanilla_transformer_pytorch
A vanilla encoder-decoder base in pytorch, for experimentation. The main branch is intended to remain in vanilla form and branches from that should include the experimental variations.

Data: in data_preparation, a folder called "data" needs to be created. Within the "data" folder, the Maestro v3 MIDI dataset needs to be downloaded in the default download folder named "maestro-v3.0.0". Maestro v3 MIDI can be downloaded using this link: https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip 

Mainly followed this code:

https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb

But tried to maintain the structure of:

https://github.com/maximoskp/TransformersFromScratch

DONE:
- Vanilla implementation
- Implement model-encoder-decoder transformer.
- MLM model and training with https://github.com/lucidrains/mlm-pytorch/tree/master 

TODO:
- Test vanilla with real data.
- Locking decoder.
- Constraint-mask decoder.
- Split-dictionary encoder-decoder.

Notes to me:

pip list --format=freeze > requirements.txt
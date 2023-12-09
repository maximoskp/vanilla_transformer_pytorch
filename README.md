# vanilla_transformer_pytorch
A vanilla encoder-decoder base in pytorch, for experimentation. The main branch is intended to remain in vanilla form and branches from that should include the experimental variations.

Data: in data_preparation,
- create a folder called ```data```.
- Within the ```data``` folder, put the Maestro v3 MIDI dataset in the ```data/maestro-v3.0.0/``` subfolder.
- Within the ```data``` folder, put the GiantMIDI dataset in the ```data/giantmidi/``` subfolder.

Download the datasets.
- Maestro v3 MIDI: https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip 
- GiantMIDI: https://github.com/bytedance/GiantMIDI-Piano 

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
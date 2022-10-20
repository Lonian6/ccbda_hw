## Training
modify the train data path in line 183.
data augmentation transform can be modified in line 213.

first time training `python train.py`
Continue training `python train.py epoch_number`

## Testing
Check the pth file of cnn_encoder, rnn_decoder, optimizer is in the CRNN_ckpt folder.
`python test.py epoch_number`
the result.csv will generate in current folder.
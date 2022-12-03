## Dataset
- Unzip data.zip to `../mnist`

- Folder structure
    ```
    .
    ├── mnist
    │   ├── 00001.png
    │   ├...
    │   └── 60000.png
    ├── hw3
         ├── Readme.md
         ├── requirements.txt
         ├── mnist.npz
         └── ddpm.py
    ```

## Environment
- Python 3.6 or later version
```sh
pip install -r requirements.txt
```

## Train
```sh
python ddpm.py --batch_size 64 --max_epoch 100 --timesteps 1000
```
`--batch_size decide` how many images the DataLoader read
`--max_epoch` decide how many rpoch to train
`--timesteps` is the number of denoise step

## Generate images
```sh
python ddpm.py --phase test --test_epoch 20 --test_num_images 10000
```
`--test_epoch` decide which pre-trained model to be used in `./models`
`--test_num_images` decide how many images you want to generate

The denoise process step image is `./epoch{test_epoch}.png`

## Count FID score
Install the counting toolkit first
```sh
pip install pytorch_gan_metrics
pip install torch torchvision pytorch_gan_metrics
```
Conducting in hw3 folder, run the code and get the score 
```
python -m pytorch_gan_metrics.calc_metrics \
--path ./plots \
--stats ../mnist.npz
```

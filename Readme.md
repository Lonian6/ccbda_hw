## Dataset
- Unzip data.zip to `../data`

- Folder structure
    ```
    .
    ├── data
    │   ├── unlabeled/
    │   └── test/
    ├── 0717072
	 ├── config.py
         ├── Readme.md
         ├── requirements.txt
         ├── loaddataset.py
         ├── net.py
         ├── test.py
         ├── trainstage1.py
         └── trainstage2.py
    ```

## Environment
- Python 3.6 or later version
    ```sh
    pip install -r requirements.txt
    ```

## Train
### Unsupervised learning
```sh
python trainstage1.py --batch_size 256 --max_epoch 300
```
### Supervised finetune
must specify the pre-trained trainstage1 model in ./pth
```sh
python trainstage2.py --batch_size 256 --max_epoch 300 --pre_model ./pth/model_stage1_epoch300.pth
```

## Test KNN and generate presentation of unlabeled, test data
must specify the pre-trained trainstage1 model in ./pth
```sh
python test.py --batch_size 256 --pre_model ./pth/model_stage2_epoch300.pth
```
The presentation file of unlabeled data is `0717072.npy`.

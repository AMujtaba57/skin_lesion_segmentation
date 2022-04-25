# Skin Lesion Segmentation

Implementation DoubleU-Net for lesion boundary Segmentation (ISIC 2018-task 1).

## Preprequisites

* Install python 3.
* Install dependencies using requirements.txt by :  ```!pip install -r requirements.txt```.
* Download data from [ISIC2018_task1 Lesion Boundary Segmentation Challenge](1.	https://challenge.isic-archive.com/data/).

## Architecture
### 1,Double-net

DoubleU-Net includes two sub-networks, look alike two U-Net concatenated.

Input is fed into modified U-Net and then generate Output1. Output1 has the same size as input image.
The sub-network 2 is for fine-grained proposal. It was built from scratch with the same idea as U-Net. However, in the decoder of sub-network 2, skip_connection from encoder1 is fed into.

At the end the Output1 and Output2 was conatenated in channel axis. So we can get one of those for prediction.
In original paper, author showed that Output1 and Output2 had the same result.

![DoubleU-Net architecture](graph/DoubleU-net_Architecture.png).

## Training

### Data

There are two common ways to augment data:

- Offline augmentation: Generate augmented images before training.

- Online augmentation: Generate augmented images during training progress.

To reduce training time, I chosen the first way.

Download raw data from [https://challenge.isic-archive.com/data/].

Your directory structure will be:

```
Unet-and-double-Unet-implementation
├──data_augmented
│    ├── mask/
│    ├── image/
├──validation
│    ├── mask/
│    ├── image/
├──image
│    ├── demo2.png
│    ├── demo3.png
│    ├── DoubleU-net_Architecture.png
│    ├── Unet_Architecture.png
├──README.md
├──data.py
├──metrics.py
├──model.py
├──predict.py
├──requirements.txt
├──train.py
├──utils.py

###
```

Train your model:

```
python train.py

```
Evaluate and Predict the Model:

```
python predict.py

```
Calculate Dice Similarity of One Image with corresposding Mask:

```
python dice_similarity.py

```
Calculate the Dice Similarity of All 60-Images with graph:

```
dice.ipynb

```
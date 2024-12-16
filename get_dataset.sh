#!/bin/bash
curl -L -o datasets/hammad_dataset.zip https://www.kaggle.com/api/v1/datasets/download/hammadjavaid/6992-labeled-meme-images-dataset
unzip datasets/hammad_dataset.zip  -d datasets/hammad_dataset
mv datasets/hammad_dataset/images/images/* datasets/hammad_dataset/
rm -r datasets/hammad_dataset/images/
rm datasets/hammad_dataset.zip && rm datasets/hammad_dataset/labels.csv

gdown https://drive.google.com/uc?id=1j6YG3skamxA1-mdogC1kRjugFuOkHt_A -O datasets/deephumor.zip
unzip datasets/deephumor.zip -d datasets/deephumor
rm datasets/deephumor.zip
mv datasets/deephumor/memes900k/images/* datasets/deephumor/
rm -r datasets/deephumor/memes900k/

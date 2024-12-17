#!/bin/bash
curl -L -o datasets/hammad_dataset.zip https://www.kaggle.com/api/v1/datasets/download/hammadjavaid/6992-labeled-meme-images-dataset
unzip -q datasets/hammad_dataset.zip  -d datasets/hammad_dataset
mv datasets/hammad_dataset/images/images/* datasets/hammad_dataset/
rm -r datasets/hammad_dataset/images/
rm datasets/hammad_dataset.zip && rm datasets/hammad_dataset/labels.csv

gdown https://drive.google.com/uc?id=1j6YG3skamxA1-mdogC1kRjugFuOkHt_A -O datasets/deephumor.zip
unzip -q datasets/deephumor.zip -d datasets/deephumor
rm datasets/deephumor.zip
mv datasets/deephumor/memes900k/images/* datasets/deephumor/
rm -r datasets/deephumor/memes900k/

# curl -L -o datasets/cats_dataset.zip  https://www.kaggle.com/api/v1/datasets/download/vekosek/cats-from-memes
# unzip -q datasets/cats_dataset.zip  -d datasets/cats_dataset/
# mv datasets/cats_dataset/cats_from_memes/* datasets/cats_dataset/
# rm -r datasets/cats_dataset/cats_from_memes/
# rm datasets/cats_dataset.zip

# curl -L -o datasets/maths_dataset.zip https://www.kaggle.com/api/v1/datasets/download/abdelghanibelgaid/mathematical-mathematics-memes
# unzip -q datasets/maths_dataset.zip -d datasets/maths_dataset/
# rm datasets/maths_dataset.zip
curl -L -o datasets/memotion_dataset.zip https://www.kaggle.com/api/v1/datasets/download/williamscott701/memotion-dataset-7k
unzip -q datasets/memotion_dataset.zip -d datasets/memotion_dataset/
mv datasets/memotion_dataset/memotion_dataset_7k/images/* datasets/memotion_dataset/
rm -r datasets/memotion_dataset/memotion_dataset_7k/
rm datasets/memotion_dataset.zip

curl -L -o datasets/russian_dataset.zip https://www.kaggle.com/api/v1/datasets/download/lexusbedra/memes-in-russian-picture-dataset
unzip -q datasets/russian_dataset.zip -d datasets/russian_dataset
rm datasets/russian_dataset.zip

# Design an autonomous car

Complete the 'Image Segmentation' part of an embedded computer vision system for autonomous vehicles.

---

## Technologies
- Python
- Tensorflow
- Keras
- FastApi
- Streamlit
- Azure Web App
- Git

## Deep Learning models 
- Unet mini
- VGG16
- VGG16 + Data augmentation

## Metrics to evaluate the models
- dice_coeff
- jaccard
- accuracy

---

## Clone repository and install dependancies

```
git clone git@github.com:ahmedaao/design-autonomous-car.git
cd design-autonomous-car
pip install -r requirements.txt
pip install . (or pip install --editable .) # Install modules from package src/
```
---

## Data Source

Input Images and output masks are dowloaded here: [CITYSCAPES](https://www.cityscapes-dataset.com/dataset-overview/)  
You have to dowload 2 zip files:    
1. P8_Cityscapes_leftImg8bit_trainvaltest.zip for original images
2. P8_Cityscapes_gtFine_trainvaltest.zip for masks

You have to manually unzip these 2 zip files:

```
mkdir -p data/raw
unzip P8_Cityscapes_leftImg8bit_trainvaltest.zip -d /path/to/data/raw
unzip P8_Cityscapes_gtFine_trainvaltest.zip -d /path/to/data/raw
```

# TODO : Complete this README.md file
# YouTube-Video-Quality-detector

This project aims to create a CNN which can detect a YouTube video's quality based on its thumbnail.

The neural network will be trained based on thumbnails scraped from various YouTube channels, using the YouTube API.

## Usage

Set train = True to train.

Modify image path in "paths" variable

Commented out code: #1 Spliting image into train/test, #2 Data augmentation (elaborated later)

Processed image size and other adjusted in convert-image function, and tensor input shape.

Run main.py

## Results

The neural network peaked at around 50% validation accuracy, with validation loss increasing in each epoch.
However, the actual accuracy managed to get up to 99.9% at the 20th epoch, which suggests overfitting. 

## Probable reasons

Due to YouTube's API limit, not enough training data was scraped, ending up with around 
2000 original images across all three categories scraped from several youtube channels. 

#### Data augmentation: 

A Keras feature where a single image can be turned into multiple similar (augmented) images. 

Using data augmentation, the initial training set was multiplied by 20 times, resulting in
around 40,000 total images across 3 categories. 

However, the lack of original images, even with data augmentation, still strongly contributed to the 
inability to gain more accuracy and overfitting.




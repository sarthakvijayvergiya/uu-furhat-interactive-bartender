'''
(ii) use Machine Learning (ML) techniques to process the features into 
a high-level representation of the user's behavior 
(ie, affective states: valence or arousal, or both) 
that can be used by the second sub-system.
You are free to choose which facial features to extract, 
which ML techniques to apply, which affective state to model,
and whether to automatically detect the affective state of one user 
or a group of users. In order to train the ML model for 
the automatic detection of affective states, you may choose
either the MultiEmoVA dataset (if you choose to automatically 
detect the affective state of a group of users in your scenario) 
or the DiffusionFER dataset (if you choose to automatically detect
the affective state of one single user in your scenario). 
Regardless of your choice of dataset, you will need to extract 
your own features. Images and labels for these datasets will be 
made available in Studium.

valence - positive or negative emotion
arousal - energy of emotion (low or high)
between -1<e<1 where 0 = neutral emotion

create database:
1) detect faces from images to get AU's
2) add columns for valence and arousal using datasheet
'''

import cv2
from feat import Detector
import numpy as np
import os
import pandas as pd

# face detection
detector = Detector('retinaface')

image_dir = ['original/neutral',
             'original/happy',
             'original/sad',
             'original/surprise',
             'original/fear',
             'original/disgust',
             'original/angry'
             ]

dfs = []

for dir in image_dir:

    image_paths=[]

    for file in os.listdir(dir):
        if file.lower().endswith('.png'):
            image_path = os.path.join(dir, file)
            image_paths.append(image_path)
    
    det = detector.detect_image(image_paths, batch_size=1)
    det.to_csv("res.csv", index=False)
    df = pd.read_csv('res.csv')

    df['input'] = df['input'].str.replace('original/', '')


    dfs.append(df)


df2 = pd.read_csv('dataset_sheet.csv')
df2['subDirectory_filePath'] = df2['subDirectory_filePath'].str.replace('DiffusionEmotion_S_cropped/', '')


result = pd.concat(dfs, ignore_index=True)
print('concat df cols \n',list(result.columns)[0:])

merged_df = pd.merge(result, df2, left_on='input', right_on='subDirectory_filePath', how='left')
print('merged df cols for val and arou \n',list(merged_df.columns)[0:])
merged_df.to_csv('data.csv')






"====================================================="


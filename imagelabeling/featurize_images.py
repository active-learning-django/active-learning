import os
import numpy as np
#import imageio
import pandas as pd
from pathlib import Path
import cv2
###
from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input
from keras.models import Model

def feat_data():
    data_dir = Path('./www')
    labeled_dir = data_dir / 'LABELED'
    unlabeled_dir = data_dir / 'UNLABELED'
    normal_cases_dir = labeled_dir / 'NORMAL'
    abnormal_cases_dir = labeled_dir / 'ABNORMAL'
    normal_cases = normal_cases_dir.glob('*.jpeg')
    abnormal_cases = abnormal_cases_dir.glob('*.jpeg')
    unlabeled_cases = unlabeled_dir.glob('*.jpeg')
    data = []

    # Go through all the normal cases. The label for these cases will be 0
    for img in normal_cases:
        imgx = cv2.imread(str(img))
        if imgx.shape[2] ==3:
            data.append((img,0))

    # Go through all the abnormal cases. The label for these cases will be 1
    for img in abnormal_cases:
        imgx = cv2.imread(str(img))
        if imgx.shape[2] ==3:
            data.append((img, 1))

    # Go through all the abnormal cases. The label for these cases will be 'NA'
    for img in unlabeled_cases:
        imgx = cv2.imread(str(img))
        if imgx.shape[2] ==3:
            data.append((img, np.nan))

    # Get a pandas dataframe from the data we have in our list
    data = pd.DataFrame(data, columns=['image', 'label'],index=None)

    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    base_model = DenseNet121(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1000').output)
    fc_features =np.zeros((data.shape[0],1000))
    for i in range(data.shape[0]):
        img_path = str(data['image'][i])
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        fc_features[i]=model.predict(x)

    df = pd.DataFrame(fc_features, columns=['X'+str(i) for i in range(fc_features.shape[1])],index=None)
    final_data = pd.concat([df, data], axis= 1)
    final_data['image'] = final_data['image'].astype(str)
    final_data.to_csv('final_data_test.csv')
    return final_data
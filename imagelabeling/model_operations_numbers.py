print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Revised 3/8/2020 by Nick Kebbas
# License: BSD 3 clause

# thread
import threading
#os for directory  walking
import os
# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

# Classifier Stuff
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier

# multilabel classification stuff
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

# classification report for when we use model on test set
from sklearn.metrics import classification_report

# export model
from sklearn.externals import joblib


# Import datasets, classifiers and performance metrics
from sklearn.model_selection import train_test_split

# Import keras stuff to do the image recognition/loading stuff
from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input
from keras.models import Model
from keras import backend as K



class ModelOperationsNumbers:
    # this seems to go through the images and featurize them, then builds the model
    def train_model(self, model_name):
        print("training model " + model_name)

        final_data = self.process_images(model_name, 'final_data_test_' + model_name + '.csv', 1)

        X_train, X_test, y_train, y_test = train_test_split(final_data.drop('label', axis=1), final_data['label'], test_size=0.20, random_state=0)

        # get rid of NA values
        final_data.fillna(final_data.mean(), inplace=True)
        # final_data = final_data.reset_index()
        # we create our test and training sets

        # set baseline then randomforest classifier
        #classifier = RandomForestClassifier()
        #model = classifier.fit(X_train, y_train)


        # the classifier is fit on a 1d array of multiclass labels and
        # the predict() method therefore provides corresponding multiclass predictions.
        classifier = OneVsRestClassifier(estimator=SVC(random_state=0))
        model = classifier.fit(X_train, y_train)
        y_pred_best = model.predict(X_test)
        print("model score: %.3f" % model.score(X_test, y_test))
        print(classification_report(y_test, y_pred_best))

        # now need to figure out how to save this to the database and associate
        # with the images on the site

        # now also check the unlabeled images and figure assign what they are

        # for now, just dump the model into a file, if model.joblib does not exist
        joblib.dump(model, 'model_joblibs/model_' + model_name + '.joblib')
        K.clear_session()

    def process_images(self, ml_model_name, export_file_name, label):
        # check if there's a directory first because the zip file module functionality
        data_dir = Path('./media/ml_model_images/' + ml_model_name)
        categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        # get the 0 through 9 digit images
        zero_cases = data_dir.glob('**/0_*.jpg')
        one_cases = data_dir.glob('**/1_*.jpg')
        two_cases = data_dir.glob('**/2_*.jpg')
        three_cases = data_dir.glob('**/3_*.jpg')
        four_cases = data_dir.glob('**/4_*.jpg')
        five_cases = data_dir.glob('**/5_*.jpg')
        six_cases = data_dir.glob('**/6_*.jpg')
        seven_cases = data_dir.glob('**/7_*.jpg')
        eight_cases = data_dir.glob('**/8_*.jpg')
        nine_cases = data_dir.glob('**/9_*.jpg')
        data = []

        # Go through all the normal cases. The label for these cases will be 0
        if label:
            for img in zero_cases:
                print("zero cases")
                imgx = cv2.imread(str(img))
                if imgx.shape[2] == 3:
                    data.append((img, 0))

            # Go through all the abnormal cases. The label for these cases will be 1
            for img in one_cases:
                imgx = cv2.imread(str(img))
                if imgx.shape[2] == 3:
                    data.append((img, 1))

            for img in two_cases:
                imgx = cv2.imread(str(img))
                if imgx.shape[2] == 3:
                    data.append((img, 2))

            # Go through all the abnormal cases. The label for these cases will be 1
            for img in three_cases:
                imgx = cv2.imread(str(img))
                if imgx.shape[2] == 3:
                    data.append((img, 3))

            for img in four_cases:
                imgx = cv2.imread(str(img))
                if imgx.shape[2] == 3:
                    data.append((img, 4))

            # Go through all the abnormal cases. The label for these cases will be 1
            for img in five_cases:
                imgx = cv2.imread(str(img))
                if imgx.shape[2] == 3:
                    data.append((img, 5))

            for img in six_cases:
                imgx = cv2.imread(str(img))
                if imgx.shape[2] == 3:
                    data.append((img, 6))

            for img in seven_cases:
                imgx = cv2.imread(str(img))
                if imgx.shape[2] == 3:
                    data.append((img, 7))

            for img in eight_cases:
                imgx = cv2.imread(str(img))
                if imgx.shape[2] == 3:
                    data.append((img, 8))

            for img in nine_cases:
                imgx = cv2.imread(str(img))
                if imgx.shape[2] == 3:
                    data.append((img, 9))


        # if we're not labeling it, return dataframe of image features from when we trained it
        else:
            data = pd.read_csv(export_file_name)
            return data

        # Get a pandas dataframe from the data we have in our list
        data = pd.DataFrame(data, columns=['image', 'label'], index=None)

        # Shuffle the data
        data = data.sample(frac=1).reset_index(drop=True)

        base_model = DenseNet121(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1000').output)
        fc_features = np.zeros((data.shape[0], 1000))
        for i in range(data.shape[0]):
            img_path = str(data['image'][i])
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            fc_features[i] = model.predict(x)

        df = pd.DataFrame(fc_features, columns=['X' + str(i) for i in range(fc_features.shape[1])], index=None)
        final_data = pd.concat([df, data], axis=1)
        final_data['image'] = final_data['image'].astype(str)
        final_data.to_csv(export_file_name, index=False)
        # return final_data
        print(final_data.head(10))
        # now need to train the model based on the featurized images
        # drop the image column because we don't need it
        final_data = final_data.drop(columns='image')
        return final_data

    # idea here is to make predictions with the saved ml_model
    # and save to the corresponding image_label in database
    def predict_with_model(self, model_name):
        print("Predicting")
        final_data = self.process_images(model_name, 'final_data_test_' + model_name + '.csv', 0)
        K.clear_session()
        model = joblib.load('model_joblibs/model_' + model_name + '.joblib')

        # final_data = final_data.reset_index()
        # we create our test and training sets

        # use all but last column to make prediction, since the last column is the label we're predicting
        features = final_data[final_data.columns[:-1]]
        print(model.predict(features))
        print(model.predict_proba(features))
        K.clear_session()

    def launch_training(self, model_name):
        # launch the thread that does the featurizing and model building
        t = threading.Thread(target=self.train_model(model_name), args=(), kwargs={})
        t.setDaemon(True)
        t.start()

    # launch the thread that does the featurizing and model building
    def launch_predicting(self, model_name):
        t = threading.Thread(target=self.predict_with_model(model_name), args=(), kwargs={})
        t.setDaemon(True)
        t.start()

    def __str__(self):
        return self

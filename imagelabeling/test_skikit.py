print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Revised 3/8/2020 by Nick Kebbas
# License: BSD 3 clause

# thread
import threading

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

# classification report for when we use model on test set
from sklearn.metrics import classification_report

# export model
from sklearn.externals import joblib


# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Import keras stuff to do the image recognition/loading stuff
from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input
from keras.models import Model


class Test_Skikit:
    # this seems to go through the images and featurize them, then builds the model
    def train_model(self):
        final_data = self.process_images('LABELED/NORMAL', 'LABELED/ABNORMAL', 'final_data_test.csv')

        X_train, X_test, y_train, y_test = train_test_split(final_data.drop('label', axis=1), final_data['label'], test_size=0.20, random_state=0)

        # get rid of NA values
        final_data.fillna(final_data.mean(), inplace=True)
        # final_data = final_data.reset_index()
        # we create our test and training sets

        # set baseline then randomforest classifier
        baseline = DummyClassifier(strategy='most_frequent', random_state=0).fit(X_train, y_train)
        y_pred = baseline.predict(X_test)
        classifier = RandomForestClassifier()
        model = classifier.fit(X_train, y_train)
        print("model score: %.3f" % model.score(X_test, y_test))

        y_pred_best = model.predict(X_test)
        print(classification_report(y_test, y_pred_best))

        # now need to figure out how to save this to the database and associate
        # with the images on the site

        # now also check the unlabeled images and figure assign what they are

        # for now, just dump the model into a file, if model.joblib does not exist
        joblib.dump(model, 'model.joblib')

    def process_images(self, directory_1, directory_2, export_file_name):
        data_dir = Path('./media/images/mini')
        normal_labeled_dir = data_dir / directory_1
        abnormal_labeled_dir = data_dir / directory_2
        normal_cases = normal_labeled_dir.glob('*.jpeg')
        abnormal_cases = abnormal_labeled_dir.glob('*.jpeg')
        data = []
        # Go through all the normal cases. The label for these cases will be 0
        for img in normal_cases:
            imgx = cv2.imread(str(img))
            if imgx.shape[2] == 3:
                data.append((img, 0))

        # Go through all the abnormal cases. The label for these cases will be 1
        for img in abnormal_cases:
            imgx = cv2.imread(str(img))
            if imgx.shape[2] == 3:
                data.append((img, 1))

        # Go through all the abnormal cases. The label for these cases will be 'NA'
        # this causes an error
        # for img in unlabeled_cases:
        #     imgx = cv2.imread(str(img))
        #     if imgx.shape[2] ==3:
        #         data.append((img, np.nan))

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
        final_data.to_csv(export_file_name)
        # return final_data
        print(final_data.head(10))
        # now need to train the model based on the featurized images
        # drop the image column because we don't need it
        final_data = final_data.drop(columns='image')
        return final_data

    def launch_process(self):
        # launch the thread that does the featurizing and model building
        print("name is main")
        t = threading.Thread(target=self.train_model, args=(), kwargs={})
        t.setDaemon(True)
        t.start()

    def predict_with_model(self):
        print("Predicting")
        final_data = self.process_images('UNLABELED/NORMAL', 'UNLABELED/ABNORMAL', 'final_data_predicted.csv')
        model = joblib.load('model.joblib')

        X_train, X_test, y_train, y_test = train_test_split(final_data.drop('label', axis=1), final_data['label'],
                                                            test_size=0.20, random_state=0)

        # get rid of NA values
        final_data.fillna(final_data.mean(), inplace=True)
        # final_data = final_data.reset_index()
        # we create our test and training sets

        y_pred_best = model.predict(X_test)
        print(y_pred_best)



    def __str__(self):
        return self

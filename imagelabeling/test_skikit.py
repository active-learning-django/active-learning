print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Revised 3/8/2020 by Nick Kebbas
# License: BSD 3 clause

# TODO: Fix this once i understand python
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
from sklearn.tree import DecisionTreeClassifier

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
    # this seems to go through the images and featurize them, as far as I can tell
    def featurize_images():
        data_dir = Path('./media/images/mini')
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
        final_data.to_csv('final_data_test.csv')
        # return final_data
        print(final_data.head())
        # now need to train the model based on the featurized images

        # we create our test and training sets
        X_train, X_test, y_train, y_test = train_test_split(final_data.drop('label', axis=1), final_data['label'], test_size=0.20,
                                                            random_state=0)
       # test which classifier is best
        baseline = DummyClassifier(strategy='most_frequent', random_state=0).fit(X_train, y_train)
        y_pred = baseline.predict(X_test)
        print(round(accuracy_score(y_test, y_pred), 4))

        # classifier = DecisionTreeClassifier()
        # model = classifier.fit(X_train, y_train)
        # print(classifier)
        # print("model score: %.3f" % model.score(X_test, y_test))

    def runModel (self):
        print("This crashes the project")
        # The digits datasetskikit

        digits = datasets.load_digits()
        print(type(digits))

        # The data that we are interested in is made of 8x8 images of digits, let's
        # have a look at the first 4 images, stored in the `images` attribute of the
        # dataset.  If we were working from image files, we could load them using
        # matplotlib.pyplot.imread.  Note that each image must have the same size. For these
        # images, we know which digit they represent: it is given in the 'target' of
        # the dataset.
        _, axes = plt.subplots(2, 4)
        images_and_labels = list(zip(digits.images, digits.target))
        for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
            ax.set_axis_off()
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            ax.set_title('Training: %i' % label)

        # To apply a classifier on this data, we need to flatten the image, to
        # turn the data in a (samples, feature) matrix:
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))

        # Create a classifier: a support vector classifier
        classifier = svm.SVC(gamma=0.001)

        # Split data into train and test subsets
        X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=0.5, shuffle=False)

        # We learn the digits on the first half of the digits
        classifier.fit(X_train, y_train)

        # Now predict the value of the digit on the second half:
        predicted = classifier.predict(X_test)

        images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
        for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
            ax.set_axis_off()
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            ax.set_title('Prediction: %i' % prediction)

        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(y_test, predicted)))
        disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
        disp.figure_.suptitle("Confusion Matrix")
        print("Confusion matrix:\n%s" % disp.confusion_matrix)
        plt.show()

    # launch the thread i guess
    print("name is main")
    t = threading.Thread(target=featurize_images, args=(), kwargs={})
    t.setDaemon(True)
    t.start()
    def __str__(self):
        return self.runModel(self)

    # runModel("")

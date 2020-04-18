import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV

# multiclass attempts

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsOneClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.externals import joblib



class Calculation:
    def readCSV(self):
        data = pd.read_csv(self)
        # data.drop(['Unnamed: 0'], axis=1, inplace=True)
        data['dif'] = 0
        data['probability'] = 0
        data.dropna(inplace=True)
        # data.replace('',np.nan, regex=True)
        return data

    def ridge_regression(self):

        X = self.drop(['label', "image", 'dif', 'probability'], axis=1)

        # actual values of labels
        y = self['label'].values
        for i in y:
            print(i)


        classifications = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
        print("classifications ===================")
        for i in classifications:
            print(i)
        # this is the multiclass part
        # https://stackoverflow.com/questions/61222686/scikit-learns-ridge-clasifiers-working-for-multi-class-not-clear

        # probability result
        self['probability'] = classifications



        # result = [data, data['probability'], data['dif'],score]

        return self

    def readConfusionMatrix(self):

        X = self.drop(['label', "image", 'dif', 'probability'], axis=1)

        # actual values of labels
        y = self['label'].values
        for i in y:
            print(i)


        classifications = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
        print("classifications ===================")
        for i in classifications:
            print(i)
        # this is the multiclass part
        # https://stackoverflow.com/questions/61222686/scikit-learns-ridge-clasifiers-working-for-multi-class-not-clear

        # probability result
        self['probability'] = classifications



        # result = [data, data['probability'], data['dif'],score]

        return self

    def findprob(self):
        return self['probability']

    def ROC(self):
        #self == dataframe
        prediction = self['probability']
        actual = self['label'].values
        confusion_matrix(actual, prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def concateData(self):

        # sort the data to
        self.sort_values(by='dif', ascending=False)

        # find find the lowest difference
        low_diff = self[self['dif'] < 0.3]
        # if (len(low_diff) < 2):

        newdf = pd.concat([self, low_diff], axis=0)

        return newdf

    def outputCSV(self, model_name):
        df = pd.DataFrame(self)
        df.to_csv('final_data_' + model_name + '.csv', index=False)

        return df

    def outputJSON(self, model_name):
        df = pd.DataFrame(self)
        df.to_json('output_' + model_name + '.json', orient='records')

        return df



if __name__ == '__main__':
    # unittest.main()

    path = '../final_data_test.csv'
    # path = '/Users/maggie/Desktop/active-learning/final_data_test_Test-951.csv'
    data = pd.read_csv(path)
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data['dif'] = 0
    data['probability'] = 0
    # # print(data)
#     #
#
#
#
#     # #read file & run ridge regression
    result = Calculation.ridge_regression(data)
    final = Calculation.concateData(result)
    # print(type(final))

    result2 = Calculation.ridge_regression(final)
    final2 = Calculation.concateData(result2)
    print(final2)
    print(final2.loc[1:3,])
#     # # print(result)
#     finaldf = result
#     # #
#     for i in range(2,4):
#         tempdf = Calculation.ridge_regression(result)
#         finaldf = Calculation.concateData(tempdf)
#     prob = Calculation.outputCSV(finaldf)
#     print(prob)
    #
    # re2 = Calculation.ridge_regression(finaldf)
    # prob2 = Calculation.outputCSV(re2)
    # print(prob2)
    #     print(len(finaldf))
    # #create ROC curve
    # Calculation.ROC(result)
    #
    # add uncertain cases to original data
    # print(Calculation.concateData(result[0]))

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.externals import joblib

class Model:
    # self == file path
    def readCSV(self):
        data = pd.read_csv(self)
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
        data['dif'] = 0
        data['probability'] = 0
        return data

    # self == data frame
    def ridge_regression(self):
        # data = pd.read_csv(self)
        # data.drop(['Unnamed: 0'], axis=1, inplace=True)
        # data['dif'] = 0
        # data['probability'] = 0
        data = self
        # seperate x,y
        X = data.drop(['label', "image", 'dif', 'probability'], axis=1)
        y = data['label'].values.reshape(-1, 1)

        # split train_test
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
        # stratify = y is used when data is not enough or biased

        # find best alpha
        alpha_list = 10 ** np.linspace(10, -2, 100) * 0.5

        # run model with all alpha
        ridgecv = RidgeCV(alphas=alpha_list, scoring='neg_mean_squared_error', normalize=True)
        ridgecv.fit(train_X, train_y)

        # the best shrinkage
        best_alpha = ridgecv.alpha_

        # the best model
        ridge_refit = Ridge(alpha=best_alpha, normalize=True)
        ridge_refit.fit(train_X, train_y)
        ridge_refit.predict(test_X)

        #save model to Joblib Module
        joblib_file = "joblib_model.pkl"
        joblib.dump(ridge_refit, joblib_file)

        # Load from file
        joblib_model = joblib.load(joblib_file)

        # run best model with a X data
        # prob = ridge_refit.predict(X)

        prob = joblib_model.predict(X)
        print(prob)
        # score = ridge_refit.score(X, y)
        score = joblib_model.score(X, y)
        print("R^2 score: ", score)

        # probability result
        data['probability'] = prob

        # calculate absolute value
        data['dif'] = abs(data['probability'] - 0.5)

        result = [data, data['probability'], data['dif']]

        return result

    def ROC(self):
        #self == dataframe
        prediction = self['probability']
        actual = self['label'].values.reshape(-1, 1)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, prediction)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('../media/ROC_Curve/ROC.png')

    def concateData(self):

        # sort the data to
        self.sort_values(by='dif', ascending=False)

        # find find the lowest difference
        low_diff = self[self['dif'] < 0.3]

        self = pd.concat([self,low_diff],axis = 0)
        return self


if __name__ == '__main__':
    path = '../final_data_test.csv'
    data = Model.readCSV(path)
    # print(Model.readCSV(path))

    #read file & run ridge regression
    result = Model.ridge_regression(data)

    #create ROC curve
    Model.ROC(data)
    #
    # add uncertain cases to original data
    print(Model.concateData(data))

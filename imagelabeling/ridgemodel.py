import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV

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
        y = self['label'].values.reshape(-1, 1)

        # split train_test
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
        # stratify = y is used when data is not enough or biased

        # find best alpha
        alpha_list = 10 ** np.linspace(2, -5, 22) * 0.5
        print(alpha_list)

        # run model with all alpha
        ridgecv = RidgeCV(alphas=alpha_list, scoring='neg_mean_squared_error', normalize=True)
        ridgecv.fit(train_X, train_y)


        # the best shrinkage
        best_alpha = ridgecv.alpha_

        # the best model
        ridge_refit = Ridge(alpha=best_alpha, normalize=True)
        ridge_refit.fit(train_X, train_y)
        ridge_refit.predict(test_X)

        # #save model to Joblib Module
        joblib_file = "joblib_model.pkl"
        joblib.dump(ridge_refit, joblib_file)
        #
        # # Load from file
        # joblib_model = joblib.load(joblib_file)

        # run best model with a X data
        prob = ridge_refit.predict(X)

        # prob = joblib_model.predict(X)
        # print(prob)
        score = ridge_refit.score(X, y)
        # score = joblib_model.score(X, y)
        # print("R^2 score: ", score)

        # probability result
        self['probability'] = prob

        # calculate absolute value
        self['dif'] = abs(self['probability'] - 0.5)

        # result = [data, data['probability'], data['dif'],score]

        return self

    def findprob(self):
        return self['probability']

    def ROC(self):
        #self == dataframe
        prediction = self['probability']
        actual = self['label'].values.reshape(-1, 1)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, prediction)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plt.savefig('../media/ROC_Curve/ROC.png')
        return plt

    def concateData(self):

        # sort the data to
        self.sort_values(by='dif', ascending=False)

        # find find the lowest difference
        low_diff = self[self['dif'] < 0.3]
        # if (len(low_diff) < 2):

        newdf = pd.concat([self,low_diff],axis = 0)

        return newdf

    def outputCSV(self, model_name):
        df = pd.DataFrame(self)
        df.to_csv('final_data_test_' + model_name + '.csv', index=False)

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

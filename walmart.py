import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Imputer
from sklearn import model_selection as ms
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_input():
    train_data_location = "train.csv"
    x_train= pd.read_csv(train_data_location, index_col=False)
    test_data_location = "test.csv"
    x_test = pd.read_csv(test_data_location, index_col=False)
    submission=pd.read_csv('sample_submission.csv', index_col=False)
    y_train = x_train['TripType'].values
    temp_visit_test=x_test['VisitNumber'].copy()
    temp_visit_train=x_train['VisitNumber'].copy()
    # del x_test['VisitNumber']
    # del x_train['VisitNumber']
    del x_train['TripType']
    return x_train, y_train, x_test, submission, temp_visit_test


class PreProcessing:

    def __init__(self, x_train, x_test):
        self.x_train = x_train
        self.x_test = x_test
        self.col_names = self.x_train.columns
        self.num_cols = self.x_train.select_dtypes(include=[np.number]).columns
        self.cat_cols = None

    def find_cat_cols(self):
        cat_cols=list(set(self.col_names) - set(self.num_cols))
        self.cat_cols=cat_cols
        print(self.cat_cols)

    def label(self):
        for col in self.cat_cols:
            le = LabelEncoder()
            self.x_train[col] = le.fit_transform(self.x_train[col].astype(str))
            self.x_test[col]=le.transform(self.x_test[col].astype(str))

    def add_features(self):
        self.x_train['average_count'] = self.x_train['ScanCount'].groupby(self.x_train['VisitNumber']).transform('mean')
        self.x_test['average_count'] = self.x_test['ScanCount'].groupby(self.x_test['VisitNumber']).transform('mean')

    def one_hot(self):
        enc=OneHotEncoder(categorical_features= [self.col_names.get_loc(c) for c in self.cat_cols])
        self.x_train=enc.fit_transform(self.x_train)
        self.x_test=enc.transform(self.x_test)

    def impute_missing(self):
        for col in list(self.col_names):
            if col in self.cat_cols:
                imp = Imputer(strategy='most_frequent')
                self.x_train[col]=imp.fit_transform(self.x_train[col].as_matrix().reshape(-1, 1))
                self.x_test[col]=imp.transform(self.x_test[col].as_matrix().reshape(-1, 1))
            else:
                imp=Imputer(strategy='mean')
                self.x_train[col]=imp.fit_transform(self.x_train[col].as_matrix().reshape(-1, 1))
                self.x_test[col]=imp.transform(self.x_test[col].as_matrix().reshape(-1, 1))

    def low_var(self):
          var=VarianceThreshold()
          self.x_train = var.fit_transform(self.x_train)
          self.x_test = var.transform(self.x_test)

    def preprocess(self):
          self.find_cat_cols()
          self.label()
          self.impute_missing()
          self.add_features()
          self.one_hot()
          self.low_var()
          return self.x_train, self.x_test


class FeatureEng:
    def __init__(self, x_train, x_test):
        self.x_train = pd.DataFrame(x_train.toarray())
        self.x_test=pd.DataFrame(x_test.toarray())


    def feature_selection(self):
        estimator=RandomForestClassifier()
        fs = RFE(estimator=estimator)
        self.x_train = fs.fit_transform(self.x_train)
        self.x_test=fs.transform(self.x_test)

    def pca(self):
        pca = PCA()
        self.x_train= pca.fit_transform(self.x_train)
        self.x_test=pca.transform(self.x_test)

    def process(self):
        self.pca()
        return self.x_train, self.x_test


# Train a baseline model, preprocessing and feature engineering steps could be added or removed in the tweaking process.
def main():
     x_train, y_train, x_test, submission, temp_visit_test = get_input()
     x_train, x_test = PreProcessing(x_train, x_test).preprocess()
     x_train, x_test = FeatureEng(x_train, x_test).process()
     x_val, x_test_val, y_val, y_test_val = ms.train_test_split(x_train, y_train, test_size=0.1)
     estimators = [('clf', LogisticRegression())]
     pipe = Pipeline(estimators)
     pipe.fit(x_val, y_val)
     y_pred=pipe.predict_proba(x_test)
     y_pred=pd.DataFrame(y_pred)
     y_pred.insert(loc=0, column='triptype',value=temp_visit_test)
     print('current score for logistic regression is:', pipe.score(x_test_val, y_test_val))
     header=submission.columns
     y_pred.to_csv("LogReg.csv", header=header,index=False)  # saving the predictions in Kaggle asked format


if __name__ == "__main__":
    main()





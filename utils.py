import numpy as np
import pandas as pd
from functools import reduce

from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler, LabelEncoder
from sklearn.preprocessing import  MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator

import re


class DateFormatter(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xdate = X.apply(pd.to_datetime)
        print(f"{', '.join(list(X.columns))} deserve to have a date...")
        return Xdate
    
    
class DFImputer(TransformerMixin):
    # Imputer but for pandas DataFrames

    def __init__(self, strategy='median', fill_value=None, missing_values=np.nan):
        print(f"Hey there! Today, I'm gonna fill your voids & nulls.... My strategy is gonna be {strategy} and my fillvalue is {'nothing' if fill_value is None else fill_value}")
        self.imp = SimpleImputer(strategy=strategy, missing_values=missing_values, fill_value = fill_value)
        
    def fit(self, X, y=None):
        self.imp.fit(X)
        print(f"Did someone just fit my imputer?")
        return self

    def transform(self, X, y=None):
        # assumes X is a DataFrame
        Ximp = self.imp.transform(X)
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        return Xfilled
    
class DFLabelEncoder(TransformerMixin):
    def __init__(self):
        print(f"Today I'm ur numbers man. Gonna convert ya strings into numbers...")

    def fit(self , df, y=None):
        maps_={}
        for col in df:
            y = df[col]
            uni = np.unique(y)
            map_ = {}
            for c in uni:
                map_[c] = len(map_)
            maps_[col] = map_
        self.maps_ = maps_
        return self
    
    def transform(self , df, y=None):
        ndf = df.copy()
        for col in df:
            ny = []
            map_= self.maps_[col]
            for c in np.array(df[col]):
                if c in self.maps_[col]:
                    ny.append(self.maps_[col][c])
                else:
                    ny.append(-1)
            ndf[col] = ny
        return ndf
    
    
class DummyTransformer(TransformerMixin):

    def __init__(self):
        self.dv = None

    def fit(self, X, y=None):
        # assumes all columns of X are strings
        print(f"Working on Dummyifying a few guys. Hold tight...")
        Xdict = X.to_dict('records')
        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(Xdict)
        return self

    def transform(self, X, y=None):
        # assumes X is a DataFrame
        Xdict = X.to_dict('records')
        Xt = self.dv.transform(Xdict)
        cols = self.dv.get_feature_names()
        Xdum = pd.DataFrame(Xt, index=X.index, columns=cols)
        # drop column indicating NaNs
        nan_cols = [c for c in cols if '=' not in c]
        Xdum = Xdum.drop(nan_cols, axis=1)
        return Xdum
      
    def get_feature_names(self):
        return self.dv.get_feature_names()
    
class DateTransformer(TransformerMixin):

    def __init__(self):
        self.cols = []

    def fit(self, X, y=None):
        # assumes all columns of X are strings
        return self
      
    def get_feature_names(self):
        return self.cols

    def transform(self, X, y=None):
        # assumes X is a DataFrame
        print(f"Parsing dates to additional features...")
        Xc = X.copy()
        for fldname in Xc.columns:
            targ_pre = re.sub('[Dd]ate$', '', fldname)
            attr = ['Year', 'Month', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
            for n in attr: 
                Xc[targ_pre + n] = getattr(Xc[fldname].dt, n.lower())
            #df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9      
            Xc.drop(fldname, axis=1, inplace=True)
        self.cols = list(Xc.columns)
        return Xc
    
class ClipTransformer(TransformerMixin):

    def __init__(self, a_min, a_max):
        self.a_min = a_min
        self.a_max = a_max

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X, y=None):
        # assumes X is a DataFrame
        Xclip = np.clip(X, self.a_min, self.a_max)
        return Xclip
    
   
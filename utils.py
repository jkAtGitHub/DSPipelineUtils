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
import warnings
import re

class StringTransformer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        print(f"Let's consider the values in {', '.join(list(X.columns))} columns as text values for a while...")

        Xstr = X.applymap(str)
        return Xstr
              
class DateDiffer(TransformerMixin):
    
    def __init__(self, col_name):
        self.col_name = col_name
              
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        beg_col = X.columns[0:-1]
        end_col = X.columns[1:]
        Xbeg = X[beg_col].values
        Xend = X[end_col].values
        Xd = (Xend - Xbeg) / np.timedelta64(1, 'D')
        Xdiff = pd.DataFrame(Xd, index=X.index, columns=[self.col_name])
        return Xdiff  

    def get_feature_names(self):
        return [self.col_name]              
              
class ColDiffer(TransformerMixin):
              
    def __init__(self, col_name):
        self.col_name = col_name
                 
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        beg_col = X.columns[0:-1]
        end_col = X.columns[1:]
        Xbeg = X[beg_col].values
        Xend = X[end_col].values
        Xd = (Xend - Xbeg)
        Xdiff = pd.DataFrame(Xd, index=X.index, columns=[self.col_name])
        return Xdiff   
              
    def get_feature_names(self):
        return [self.col_name]
    
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
        self.strategy = strategy
        self.fill_value = fill_value
        self.imp = SimpleImputer(strategy=strategy, missing_values=missing_values, fill_value = fill_value)
        
    def fit(self, X, y=None):
        self.imp.fit(X)
        print(f"Hey there! Today, I'm gonna fill your voids & nulls...\nin {', '.join(list(X.columns))} columns with the {self.strategy} value")
        return self

    def transform(self, X, y=None):
        # assumes X is a DataFrame
        Ximp = self.imp.transform(X)
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        return Xfilled
    
class DFLabelEncoder(TransformerMixin):        

    def fit(self , df, y=None):
        print(f"Today I'm ur Numbers man. Gonna convert ya strings in {', '.join(list(df.columns))} to numbers...")
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
        print(f"Working on Dummyifying {', '.join(list(df.columns))}. Hold tight...")
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

def make_date(df, date_field):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)
              
def add_datepart(df, field_name, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    make_date(df, field_name)
    field = df[field_name]
    prefix = re.sub('[Dd]ate$', '', field_name)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    # Pandas removed `dt.week` in v1.1.10
    week = field.dt.isocalendar().week.astype(field.dt.day.dtype) if hasattr(field.dt, 'isocalendar') else field.dt.week
    for n in attr: df[prefix + n] = getattr(field.dt, n.lower()) if n != 'Week' else week
    mask = ~field.isna()
    df[prefix + 'Elapsed'] = np.where(mask,field.values.astype(np.int64) // 10 ** 9,None)
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df
              
class DateTransformer(TransformerMixin):

    def __init__(self, time = True):
        self.cols = []
        self.time = time

    def fit(self, X, y=None):
        # assumes all columns of X are strings
        print(f"I specialize in dealing with dates ;)")
        return self
      
    def get_feature_names(self):
        return self.cols

    def transform(self, X, y=None):
        # assumes X is a DataFrame
        print(f"Parsing dates to additional features...")
        Xc = X.copy()
        for fldname in Xc.columns:
            add_datepart(Xc, fldname, time= self.time)
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
              
              
def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names

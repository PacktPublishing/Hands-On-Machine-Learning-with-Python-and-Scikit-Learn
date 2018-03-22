from __future__ import print_function, division

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import BaggingRegressor
from sklearn.externals import six

import numpy as np
import pandas as pd

__all__ = [
    'BaggedRegressorImputer',
    'CustomPandasTransformer',
    'DummyEncoder'
]

class CustomPandasTransformer(BaseEstimator, TransformerMixin):
    def _validate_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame, but got type=%s" 
                            % type(X))
        return X
    
    @staticmethod
    def _validate_columns(X, cols):
        scols = set(X.columns)  # set for O(1) lookup
        if not all(c in scols for c in cols):
            raise ValueError("all columns must be present in X")
            

class DummyEncoder(CustomPandasTransformer):
    """A custom one-hot encoding class that handles previously unseen
    levels and automatically drops one level from each categorical
    feature to avoid the dummy variable trap.
    
    Parameters
    ----------
    columns : list
        The list of columns that should be dummied
        
    sep : str or unicode, optional (default='_')
        The string separator between the categorical feature name
        and the level name.
        
    drop_one_level : bool, optional (default=True)
        Whether to drop one level for each categorical variable.
        This helps avoid the dummy variable trap.
        
    tmp_nan_rep : str or unicode, optional (default="N/A")
        Each categorical variable adds a level for missing values
        so test data that is missing data will not break the encoder
    """
    def __init__(self, columns, sep='_', drop_one_level=True, 
                 tmp_nan_rep='N/A'):
        self.columns = columns
        self.sep = sep
        self.drop_one_level = drop_one_level
        self.tmp_nan_rep = tmp_nan_rep
        
    def fit(self, X, y=None):
        # validate the input, and get a copy of it
        X = self._validate_input(X).copy()
        
        # load class attributes into local scope
        tmp_nan = self.tmp_nan_rep
        
        # validate all the columns present
        cols = self.columns
        self._validate_columns(X, cols)
                
        # begin fit
        # for each column, fit a label encoder
        lab_encoders = {}
        for col in cols:
            vec = [tmp_nan if pd.isnull(v) 
                   else v for v in X[col].tolist()]
            
            # if the tmp_nan value is not present in vec, make sure it is
            # so the transform won't break down
            svec = list(set(vec))
            if tmp_nan not in svec:
                svec.append(tmp_nan)
            
            le = LabelEncoder()
            lab_encoders[col] = le.fit(svec)
            
            # transform the column, re-assign
            X[col] = le.transform(vec)
            
        # fit a single OHE on the transformed columns - but we need to ensure
        # the N/A tmp_nan vals make it into the OHE or it will break down later.
        # this is a hack - add a row of all transformed nan levels
        ohe_set = X[cols]
        ohe_nan_row = {c: lab_encoders[c].transform([tmp_nan])[0] for c in cols}
        ohe_set = ohe_set.append(ohe_nan_row, ignore_index=True)
        ohe = OneHotEncoder(sparse=False).fit(ohe_set)
        
        # assign fit params
        self.ohe_ = ohe
        self.le_ = lab_encoders
        self.cols_ = cols
        
        return self
    
    def transform(self, X):
        check_is_fitted(self, 'ohe_')
        X = self._validate_input(X).copy()
        
        # fit params that we need
        ohe = self.ohe_
        lenc = self.le_
        cols = self.cols_
        tmp_nan = self.tmp_nan_rep
        sep = self.sep
        drop = self.drop_one_level
        
        # validate the cols and the new X
        self._validate_columns(X, cols)
        col_order = []
        drops = []
        
        for col in cols:
            # get the vec from X, transform its nans if present
            vec = [tmp_nan if pd.isnull(v) 
                   else v for v in X[col].tolist()]
            
            le = lenc[col]
            vec_trans = le.transform(vec)  # str -> int
            X[col] = vec_trans
            
            # get the column names (levels) so we can predict the 
            # order of the output cols
            le_clz = le.classes_.tolist()
            classes = ["%s%s%s" % (col, sep, clz) for clz in le_clz]
            col_order.extend(classes)
            
            # if we want to drop one, just drop the last
            if drop and len(le_clz) > 1:
                drops.append(classes[-1])
                
        # now we can get the transformed OHE
        ohe_trans = pd.DataFrame.from_records(data=ohe.transform(X[cols]), 
                                              columns=col_order)
        
        # set the index to be equal to X's for a smooth concat
        ohe_trans.index = X.index
        
        # if we're dropping one level, do so now
        if drops:
            ohe_trans = ohe_trans.drop(drops, axis=1)
        
        # drop the original columns from X
        X = X.drop(cols, axis=1)
        
        # concat the new columns
        X = pd.concat([X, ohe_trans], axis=1)
        return X
    

class BaggedRegressorImputer(CustomPandasTransformer):
    """Fit bagged regressor models for each of the impute columns in order
    to impute the missing values.
    
    Parameters
    ----------
    impute_cols : list
        The columns to impute
        
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.
        
    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
    """
    def __init__(self, impute_cols, base_estimator=None, n_estimators=10, 
                 max_samples=1.0, max_features=1.0, bootstrap=True, 
                 bootstrap_features=False, n_jobs=1,
                 random_state=None, verbose=0):
        
        self.impute_cols = impute_cols
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
    def fit(self, X, y=None):
        # validate that the input is a dataframe
        X = self._validate_input(X)  # don't need a copy this time
        
        # validate the columns exist in the dataframe
        cols = self.impute_cols
        self._validate_columns(X, cols)
        
        # this dictionary will hold the models
        regressors = {}
        
        # this dictionary maps the impute column name(s) to the vecs
        targets = {c: X[c] for c in cols}
        
        # drop off the columns we'll be imputing as targets
        X = X.drop(cols, axis=1)  # these should all be filled in (no NaN)
        
        # iterate the column names and the target columns
        for k, target in six.iteritems(targets):
            # split X row-wise into train/test where test is the missing
            # rows in the target
            test_mask = pd.isnull(target)
            train = X.loc[~test_mask]
            train_y = target[~test_mask]
            
            # fit the regressor
            regressors[k] = BaggingRegressor(
                base_estimator=self.base_estimator,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                n_jobs=self.n_jobs, 
                random_state=self.random_state,
                verbose=self.verbose, oob_score=False,
                warm_start=False).fit(train, train_y)
            
        # assign fit params
        self.regressors_ = regressors
        return self
        
    def transform(self, X):
        check_is_fitted(self, 'regressors_')
        X = self._validate_input(X).copy()  # need a copy
        
        cols = self.impute_cols
        self._validate_columns(X, cols)
        
        # fill in the missing
        models = self.regressors_
        for k, model in six.iteritems(models):
            target = X[k]
            
            # split X row-wise into train/test where test is the missing
            # rows in the target
            test_mask = pd.isnull(target)
            
            # if there's nothing missing in the test set for this feature, skip
            if test_mask.sum() == 0:
                continue
            test = X.loc[test_mask].drop(cols, axis=1)  # drop impute cols
            
            # generate predictions
            preds = model.predict(test)
            
            # impute!
            X.loc[test_mask, k] = preds
            
        return X

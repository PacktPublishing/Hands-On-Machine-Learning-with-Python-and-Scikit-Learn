
from __future__ import print_function, division, absolute_import

from sklearn.externals import six

import pandas as pd
import numpy as np

__all__ = [
    'get_grid_results_table'
]


def get_grid_results_table(search):
    """Get the grid results from a fit ``RandomizedSearchCV``.
    
    Parameters
    ----------
    search : RandomizedSearchCV
        The pre-fit grid search.
        
    Returns
    -------
    res : pd.DataFrame
        The results dataframe
    """
    # the search results
    res = search.cv_results_
    
    # unpack the dict
    dct = {k: res[k] for k in 
           ('mean_fit_time', 'std_fit_time', 
            'mean_score_time', 'std_score_time',
            'mean_test_score', 'std_test_score')}
    
    prefix = "param_"
    for k, v in six.iteritems(res):
        if k.startswith(prefix):
            key = k.split(prefix)[-1]
            dct[key] = v.data
            
    return pd.DataFrame.from_dict(dct)

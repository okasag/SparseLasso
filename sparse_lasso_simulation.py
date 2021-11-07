"""
Simulation Exercise for SparseLasso

by Gabriel Okasa
"""

# =============================================================================
# import modules
# =============================================================================

# pandas, numpy and sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn import linear_model

# import SparseLasso
from sparse_lasso import SparseLogisticRegressionCV, SparseLassoCV

# =============================================================================
# define function for lasso comparison via simulation
# =============================================================================

# compare an average performance between LassoCV and SparseLassoCV
def compare_lasso(lasso_model,
                  n_lambdas=100,
                  n_folds=10,
                  n_sim=1000,
                  n_samples=1000,
                  n_features=100,
                  n_informative=10,
                  noise=10,
                  verbose=True):
    """
    Compare Lasso regularizations with cross-validation tuning for the
    penalty choice using optimal lambda minimizing the CV error
    and using lambda according to the 1 standard error rule.

    Parameters
    ----------
    lasso_model : TYPE: string
        DESCRIPTION: what model should be compared?
        Options are {'linear', 'logistic'}.
    n_lambdas : TYPE: int
        DESCRIPTION: number of lambda tuning values for cross-validation.
        Default is 100.
    n_lfolds : TYPE: int
        DESCRIPTION: number of folds for cross-validation.
        Default is 10.
    n_sim : TYPE: int
        DESCRIPTION: number of simulation replications.
        Default is 1000.
    n_samples : TYPE: int
        DESCRIPTION: number of observations in the simulated dataset.
        Default is 1000.
    n_features : TYPE: int
        DESCRIPTION: number of features in the simulated dataset.
        Default is 100.
    n_informative : TYPE: int
        DESCRIPTION: number of informative features in the simulated dataset.
        Default is 10.
    noise : TYPE: float
        DESCRIPTION: noise in the DGP for the simulated dataset.
        Default is 10.
    verbose : TYPE: bool
        DESCRIPTION: should be the results printed to console?
        Default is True.

    Returns
    -------
    result: Simulated parameters.
    """
    
    # prepare output for number of non-zero coefficients
    n_coefs_lasso_min = []
    n_coefs_lasso_1se = []
    # prepare output for number of not matching variables
    n_vars_lasso_min = []
    n_vars_lasso_1se = []
    
    # start simulating the performance
    for sim_idx in range(n_sim):
        
        # simulate some example data for a sparse model
        if lasso_model == 'linear':
            # use regression DGP
            X, y, coef = make_regression(n_samples=n_samples,
                                         n_features=n_features, 
                                         n_informative=n_informative,
                                         noise=noise,
                                         coef=True,
                                         random_state=sim_idx)
            
            # estimate standard LassoCV with optimal lambda minimizing error
            lasso_min = linear_model.LassoCV(fit_intercept=False,
                                             n_alphas=n_lambdas,
                                             cv=n_folds).fit(X=X,
                                                             y=y)
            # get the coefficients
            lasso_min_coef = lasso_min.coef_
            
            # estimate SparseLassoCV with lambda using 1 standard error rule
            lasso_1se = SparseLassoCV(fit_intercept=False,
                                      n_alphas=n_lambdas,
                                      cv=n_folds).fit(X=X,
                                                      y=y) 
            # get the coefficients
            lasso_1se_coef = lasso_1se.coef_
        
        elif lasso_model == 'logistic':
            # use logistic DGP modification
            X, y, coef = make_regression(n_samples=n_samples,
                                         n_features=n_features, 
                                         n_informative=n_informative,
                                         noise=0,
                                         coef=True,
                                         random_state=sim_idx)
            # get the x*beta dot product and feed it into sigmoid
            pi = 1/(1 + np.exp(- np.dot(X, coef/100)))
            # re-generate a binary response
            y = np.random.binomial(1, pi)
            
            # LogisticRegressionCV with optimal lambda minimizing error
            lasso_min = linear_model.LogisticRegressionCV(penalty='l1',
                                                          solver='liblinear',
                                                          fit_intercept=False,
                                                          Cs=n_lambdas,
                                                          cv=n_folds).fit(X=X,
                                                                          y=y)
            # get the coefficients
            lasso_min_coef = lasso_min.coef_[0]
            
            # SparseLogisticRegressionCV with lambda using 1 SE rule
            lasso_1se = SparseLogisticRegressionCV(penalty='l1',
                                                   solver='liblinear',
                                                   fit_intercept=False,
                                                   Cs=n_lambdas,
                                                   cv=n_folds).fit(X=X,
                                                                   y=y)
            # get the coefficients
            lasso_1se_coef = lasso_1se.coef_[0]
        
        else:
            # raise an error
            raise ValueError("Wrong input for 'lasso_model' argument."
                             "Must be one of 'linear' or 'logistic'.")
    
        
        # get the number of non-zero coefficients and save the output
        n_coefs_lasso_min.append(np.sum((lasso_min_coef != 0) * 1))
        # get the number of non-matching variables
        n_vars_lasso_min.append(
            len(np.setxor1d(np.where(lasso_min_coef != 0),
                            np.where(coef != 0))))
        
        
        # get the number of non-zero coefficients and save the output
        n_coefs_lasso_1se.append(np.sum((lasso_1se_coef != 0) * 1))
        # get the number of non-matching variables
        n_vars_lasso_1se.append(
            len(np.setxor1d(np.where(lasso_1se_coef != 0),
                            np.where(coef != 0))))
    
    # check if the results should be printed to the console
    if verbose:
        # compare the results on average for number of selected variables
        print('Average number of selected features', '\n\n',
              'Lasso with optimal tuning:', round(np.mean(n_coefs_lasso_min), 2),
              '\n',
              'Lasso with 1 SE rule:     ', round(np.mean(n_coefs_lasso_1se), 2),
              '\n',
              'True features:            ', len(np.where(coef!=0)[0]), '\n')
        
        # compare the results on average for number of non-matching variables
        print('Average number of non-matching features', '\n\n',
              'Lasso with optimal tuning:', round(np.mean(n_vars_lasso_min), 2),
              '\n',
              'Lasso with 1 SE rule:     ', round(np.mean(n_vars_lasso_1se), 2),
              '\n')
    
    # wrap the results
    results = {'coef_lasso_1se': n_coefs_lasso_1se,
               'coef_lasso_min': n_coefs_lasso_min,
               'vars_lasso_1se': n_vars_lasso_1se,
               'vars_lasso_min': n_vars_lasso_min}
    # reutrn the results
    return results

# =============================================================================
# Simulation for Linear Model
# =============================================================================

# compare the Lasso for Linear Model with default parameters
linear_lasso = compare_lasso('linear')

# =============================================================================
# Simulation for Logistc Model
# =============================================================================

# compare the Lasso for Logistic Model with default parameters
logistic_lasso = compare_lasso('logistic')

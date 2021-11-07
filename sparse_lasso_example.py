"""
Practical Example for SparseLasso Usage

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
# compare LassoCV and SparseLassoCV for variable selection
# =============================================================================

# simulate some example data for a sparse linear model
# use regression DGP
X, y, coef = make_regression(n_samples=1000,
                             n_features=100, 
                             n_informative=10,
                             noise=10,
                             coef=True,
                             random_state=0)

# estimate standard LassoCV with optimal lambda minimizing error
lasso_min = linear_model.LassoCV(n_alphas=100, cv=10).fit(X=X, y=y)

# estimate SparseLassoCV with lambda using 1 standard error rule
lasso_1se = SparseLassoCV(n_alphas=100, cv=10).fit(X=X, y=y) 

# get the number of non-zero coefficients and save the output
print('Lasso Min Number of Selected Variables:     ',
      np.sum((lasso_min.coef_ != 0) * 1), '\n',
      'Lasso 1se Number of Selected Variables:     ',
      np.sum((lasso_1se.coef_ != 0) * 1), '\n')

# get the number of non-matching variables
print('Lasso Min Number of Non-Matching Variables: ',
      len(np.setxor1d(np.where(lasso_min.coef_ != 0),
                      np.where(coef != 0))), '\n',
      'Lasso 1se Number of Non-Matching Variables: ',
      len(np.setxor1d(np.where(lasso_1se.coef_ != 0),
                      np.where(coef != 0))), '\n\n')

# =============================================================================
# compare LogisticRegressionCV and SparseLogisticRegressionCV
# =============================================================================

# simulate some example data for a sparse logistic model
# use logistic DGP modification
X, y, coef = make_regression(n_samples=1000,
                             n_features=100, 
                             n_informative=10,
                             noise=0,
                             coef=True,
                             random_state=0)
# get the x*beta dot product and feed it into sigmoid
pi = 1/(1 + np.exp(- np.dot(X, coef/100)))
# re-generate a binary response
y = np.random.binomial(1, pi)

# LogisticRegressionCV with optimal lambda minimizing error
lasso_min = linear_model.LogisticRegressionCV(penalty='l1',
                                              solver='liblinear',
                                              Cs=100,
                                              cv=10).fit(X=X, y=y)

# SparseLogisticRegressionCV with lambda using 1 SE rule
lasso_1se = SparseLogisticRegressionCV(penalty='l1',
                                       solver='liblinear',
                                       Cs=100,
                                       cv=10).fit(X=X, y=y)
        
# get the number of non-zero coefficients and save the output
print('Lasso Min Number of Selected Variables:     ',
      np.sum((lasso_min.coef_[0] != 0) * 1), '\n',
      'Lasso 1se Number of Selected Variables:     ',
      np.sum((lasso_1se.coef_[0] != 0) * 1), '\n')

# get the number of non-matching variables
print('Lasso Min Number of Non-Matching Variables: ',
      len(np.setxor1d(np.where(lasso_min.coef_[0] != 0),
                      np.where(coef != 0))), '\n',
      'Lasso 1se Number of Non-Matching Variables: ',
      len(np.setxor1d(np.where(lasso_1se.coef_[0] != 0),
                      np.where(coef != 0))), '\n\n')

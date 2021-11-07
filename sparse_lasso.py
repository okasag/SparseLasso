"""
SparseLasso: Sparse Solutions for the Lasso

by Gabriel Okasa

SparseLasso provides a Scikit-Learn based estimation of the Lasso with
cross-validation tuning for the penalty choice using the 'one standard error'
rule to yield sparse solutions. The 'one standard error' rule recognizes the
fact that the cross-validation path is estimated with error and selects the
more parsimonious model (see Hastie, Tibshirani and Friedman, 2009). This rule
thus chooses the largest possible penalty which is still within the one
standard error of the cross-validation optimal value. Given that the Lasso
often selects too many variables in practice, the one standard error rule
provides a practical solution to yield sparser models. The software
implementation of this rule is readily available in the R-package 'glmnet'
(Friedman, Hastie and Tibshirani, 2010), however, it is absent from the
Scikit-Learn module. SparseLasso provides estimation of the penalized linear
and logistic model based on Scikit-Learn's LassoCV and LogisticRegressionCV,
respectively and thus accepts the standard Scikit-Learn arguments.
"""

# =============================================================================
# import modules
# =============================================================================

# numpy and sklearn
import numpy as np
# import the regression mixin class to inherit from
from sklearn.base import RegressorMixin
# import the linear model for LassoCV and Lasso classes
from sklearn import linear_model

# =============================================================================
# Class Definition for SparseLassoCV
# =============================================================================

# define own class for sparse lasso (sklearn inheritance)
class SparseLassoCV(RegressorMixin):
    # initialize using lazy arguments
    def __init__(self, *args, **kwargs):
        # lassoCV as input model
        self.modelCV = linear_model.LassoCV(*args, **kwargs)
        # linear_model for re-estimating the Lasso with internal parameters
        self.model = linear_model

    # define the fit method
    def fit(self, X, y):
        # initialize variables
        self.X = X
        self.y = y
        # fith the CV model
        self.modelCV.fit(X, y)
        
        # add new 1se rule
        # get the cross-validation lambda
        cv_lambdas = self.modelCV.alphas_
        # get the cross-validation scores
        cv_scores = self.modelCV.mse_path_

        # average over the folds
        cv_scores_mean = np.mean(cv_scores, axis=1)
        # get standard error over the folds
        cv_scores_se = np.std(cv_scores, axis=1)/np.sqrt(len(cv_scores[0]))

        # take the value of the cv score corresponding to the optimal lambda
        # minimizing the cv error (np.argmin(cv_scores_mean))
        # scikit-learn is minimizing the error in linear case
        lambda_min_index = np.argmin(cv_scores_mean)
        # take the value of the optimal lambda (alpha_)
        lambda_min_value = cv_lambdas[lambda_min_index]

        # get the value of standard error of the minimum lambda
        lambda_min_se_value = cv_scores_se[lambda_min_index]
        # take the largest possible value of lambda (i.e. the highest possible
        # penalization) such that the cross-validation score, i.e. error (the
        # lower the better) is still lower than the optimal minimum score
        # increased by its standard error, i.e. still within the 1se
        # linear lasso uses the standard penalty so larger means higher penalty
        lambda_1se_value = np.max(
            cv_lambdas[cv_scores_mean < 
                       (np.min(cv_scores_mean) + lambda_min_se_value)]) 
        
        # reestimate lasso with new lambda
        lasso = self.model.Lasso(alpha=lambda_1se_value).fit(X, y)
    
        # return the sparse model
        return lasso

# =============================================================================
# Class Definition for SparseLogisticRegressionCV
# =============================================================================

# define own class for sparse logit lasso (sklearn inheritance)
class SparseLogisticRegressionCV(RegressorMixin):
    # initialize using lazy arguments
    def __init__(self, *args, **kwargs):
        # lassoCV as input model
        self.modelCV = linear_model.LogisticRegressionCV(*args, **kwargs)
        # linear_model for re-estimating the Logit with internal parameters
        self.model = linear_model

    # define the fit method
    def fit(self, X, y):
        # initialize variables
        self.X = X
        self.y = y
        # fith the CV model
        self.modelCV.fit(X, y)
        
        # add new 1se rule
        # get the cross-validation lambda
        cv_lambdas = self.modelCV.Cs_
        # get the cross-validation scores
        cv_scores = self.modelCV.scores_[1]
        
        # average over the folds
        cv_scores_mean = np.mean(cv_scores, axis=0)
        # get standard error over the folds
        cv_scores_se = np.std(cv_scores, axis=0)/np.sqrt(len(cv_scores))

        # take the value of the cv score corresponding to the optimal lambda
        # maximizing the score (np.argmax(cv_scores_mean))
        # scikit-learn maximizes the score instead of minimizing the error
        lambda_min_index = np.argmax(cv_scores_mean)
        # take the value of the optimal lambda (C_)
        lambda_min_value = cv_lambdas[lambda_min_index]

        # get the value of standard error of the minimum lambda
        lambda_min_se_value = cv_scores_se[lambda_min_index]
        # take the smallest possible value of lambda (i.e. the highest possible
        # penalization) such that the cross-validation score (the higher the
        # better) is still higher than the optimal maximum score reduced by its
        # standard error, i.e. still within the 1se
        # logit lasso uses the inverse penalty so smaller means higher penalty
        lambda_1se_value = np.min(
            cv_lambdas[cv_scores_mean >
                       (np.max(cv_scores_mean) - lambda_min_se_value)]) 

        # reestimate logit lasso with new lambda
        lasso = self.model.LogisticRegression(penalty='l1', solver='liblinear',
                                              C=lambda_1se_value).fit(X, y)
  
        # return the sparse model
        return lasso

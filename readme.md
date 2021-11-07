# `SparseLasso`: Sparse Solutions for the Lasso

## Introduction

SparseLasso provides a Scikit-Learn based estimation of the Lasso with
cross-validation tuning for the penalty choice using the 'one standard error'
rule to yield sparse solutions. The 'one standard error' rule recognizes the
fact that the cross-validation path is estimated with error and selects the
more parsimonious model (see [Hastie, Tibshirani and Friedman, 2009](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)). This rule thus chooses the largest possible penalty
which is still within the one standard error of the cross-validation optimal value.
Given that the Lasso often selects too many variables in practice, the one standard
error rule provides a practical solution to yield sparser models. The software
implementation of this rule is readily available in the R-package 'glmnet'
([Friedman, Hastie and Tibshirani, 2010](https://www.jstatsoft.org/article/view/v033i01)), however, it is absent from the Scikit-Learn module ([Pedregosa et al., 2011](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)). SparseLasso provides estimation of the penalized linear
and logistic model based on Scikit-Learn's LassoCV and LogisticRegressionCV,
respectively and thus accepts the standard Scikit-Learn arguments.

## Installation

`SparseLasso` module relies on Python 3 and is based on the `scikit-learn` module.
The required modules can be installed by navigating to the root of this project and
executing the following command: `pip install -r requirements.txt`.

## Example

The example below demonstrates the basic usage of the `SparseLasso` module.

```python
# import modules
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoCV

# import SparseLasso
from sparse_lasso import SparseLassoCV

# simulate some example data for the linear model
X, y, coef = make_regression(n_samples=1000,
                             n_features=100, 
                             n_informative=10,
                             noise=10,
                             coef=True,
                             random_state=0)

# estimate standard LassoCV with optimal lambda minimizing error
lasso_min = LassoCV(n_alphas=100, cv=10).fit(X=X, y=y)

# estimate SparseLassoCV with lambda using 1 standard error rule
lasso_1se = SparseLassoCV(n_alphas=100, cv=10).fit(X=X, y=y)

# compare the penalty values
print('Lasso Min Penalty: ', round(lasso_min.alpha_, 2), '\n',
      'Lasso 1se Penalty: ', round(lasso_1se.alpha, 2), '\n')

# compare the number of selected features
print('Lasso Min Number of Selected Variables:     ',
      np.sum((lasso_min.coef_ != 0) * 1), '\n',
      'Lasso 1se Number of Selected Variables:     ',
      np.sum((lasso_1se.coef_ != 0) * 1), '\n')
```

For a more detailed example see the `sparse_lasso_example.py`
as well as the `sparse_lasso_simulation.py` for a simulation
exercise comparing the optimal cross-validation penalty choice
with the one standard error rule for variable selection.

## References

- Hastie, Trevor, Robert Tibshirani, and J H. Friedman. The Elements of Statistical Learning: Data Mining, Inference, and Prediction. , 2009. Print.
- Friedman, Jerome, Trevor Hastie, and Rob Tibshirani. "Regularization paths for generalized linear models via coordinate descent." Journal of statistical software 33.1 (2010): 1.
- Pedregosa, Fabian, et al. "Scikit-learn: Machine learning in Python." the Journal of machine Learning research 12 (2011): 2825-2830.
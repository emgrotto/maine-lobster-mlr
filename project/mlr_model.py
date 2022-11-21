import numpy as np
import pandas as pd


def create_design_matrix(cei, sea_max, sea_min, sea_range):
    number_obs = len(cei)
    print(f"\nCreating the Desgin matrix for the {number_obs} observations")

    ones = [1 for i in range(number_obs)]

    sea_df = pd.DataFrame(list(zip(ones, cei, sea_max, sea_min, sea_range)),
                columns =['ones', 'cei', 'sea_max', 'sea_min', 'sea_range'])
    
    n,p = np.shape(sea_df)

    return sea_df, p-1


def create_single_design_matrix(predictor):
    number_obs = len(predictor)
    print(f"\nCreating the Desgin matrix for the {number_obs} observations")

    ones = [1 for i in range(number_obs)]

    sea_df = pd.DataFrame(list(zip(ones, predictor)),
                columns =['ones', 'predictor'])

    n,p = np.shape(sea_df)

    return sea_df, p-1


def predict_from_design(X, y): 

    print('\nCalculating parameter estimates')
    Xt = np.transpose(X)
    XtX = np.matmul(Xt,X)
    Xty = np.matmul(Xt,y)
    XtXinv = np.linalg.inv(XtX)
    beta_hat = np.matmul(XtXinv,Xty)
    print(beta_hat)
    
    print('\nPredicting Lobster Landings')
    y_hat = np.matmul(X,beta_hat)
    print(y_hat)
    
    print('\nCalculating residuals')
    r = y-y_hat
    print(r)

    print('\nVerifying Orthogonality')
    orth = np.matmul(r, X)
    print(orth)

    return beta_hat, y_hat, r, XtXinv


def results_insights(XtXinv, residuals, target, beta_hat, n_preds):

    n_obs = len(target)

    print('\nCalculating sigma squared estimates')
    sigma2_hat = np.dot(residuals,residuals)/(n_obs-(n_preds+1))
    print(sigma2_hat)

    print('\nCalculating z scores')
    beta_hat_var = sigma2_hat*np.diag(XtXinv)
    zscore = beta_hat/np.sqrt(beta_hat_var)
    print(zscore)

    print('\nCalculating total variability in target variable')
    s2 = np.var(target)
    print(s2)

    print('\nCalculating Explained variability')
    R2 = 1-(n_obs-(n_preds+1))*sigma2_hat/((n_obs-1)*s2)
    print(R2)

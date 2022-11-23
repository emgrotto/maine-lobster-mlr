import numpy as np
import pandas as pd


def create_design_matrix(x1, x2, x3, x4):
    number_obs = len(x1)
    print(f"\nCreating the Desgin matrix for the {number_obs} observations")

    ones = [1 for i in range(number_obs)]

    sea_df = pd.DataFrame(list(zip(ones, x1, x2, x3, x4)),
                columns =['ones', 'x1', 'x2', 'x3', 'x4'])
    
    n,p = np.shape(sea_df)

    return sea_df, p-1, n


def predict_from_design(design_matrix, target): 

    X = design_matrix
    y = target

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


def predict_naive_model(design_matrix, beta_hat, target):
    print('\npredicting using the naive model')

    X = design_matrix
    y = target

    print('\nRestricting coefficients for all predictors')
    naive_beta_hat = beta_hat
    for beta_i in range(1, naive_beta_hat.size):
        naive_beta_hat[beta_i] = 0
    print(naive_beta_hat)

    print('\nPredicting Naive Lobster Landings')
    naive_y_hat = np.matmul(X,naive_beta_hat)
    print(naive_y_hat)
    
    print('\nCalculating residuals')
    naive_r = y-naive_y_hat
    print(naive_r)

    print('\nVerifying Orthogonality')
    orth = np.matmul(naive_r, X)
    print(orth)

    return naive_beta_hat, naive_y_hat, naive_r


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

    return sigma2_hat, zscore, s2, R2


def f_test(naive_residuals, model_residuals, naive_p, model_p, n_observations):
    print('\nPerforming F test')

    naive_RSS = np.dot(naive_residuals, naive_residuals)
    model_RSS = np.dot(model_residuals, model_residuals)

    numerator = (naive_RSS - model_RSS)/(model_p - naive_p)
    denomenator = (model_RSS)/(n_observations - model_p)

    F = numerator/denomenator
    print(F)

    return F


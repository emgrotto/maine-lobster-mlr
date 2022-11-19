import numpy as np
import pandas as pd

def create_design_matrix(cei, sea_max, sea_min, sea_range):
    number_obs = len(cei)
    print(f"\nCreating the Desgin matrix for the {number_obs} observations")

    ones = [1 for i in range(number_obs)]

    sea_df = pd.DataFrame(list(zip(ones, cei, sea_max, sea_min, sea_range)),
                columns =['ones', 'cei', 'sea_max', 'sea_min', 'sea_range'])

    return sea_df

def create_single_design_matrix(predictor):
    number_obs = len(predictor)
    print(f"\nCreating the Desgin matrix for the {number_obs} observations")

    ones = [1 for i in range(number_obs)]

    sea_df = pd.DataFrame(list(zip(ones, predictor)),
                columns =['ones', 'predictor'])

    return sea_df

def predict_from_design(X, y): 

    print('\nComputing parameter estimates')
    Xt = np.transpose(X)
    XtX = np.matmul(Xt,X)
    Xty = np.matmul(Xt,y)
    XtXinv = np.linalg.inv(XtX)
    beta_hat = np.matmul(XtXinv,Xty)
    print(beta_hat)
    
    print('\nPredicting Lobster Landings')
    y_hat = np.matmul(X,beta_hat)
    print(y_hat)
    
    print('\nComputing residuals')
    r = y-y_hat
    print(r)

    print('\nVerifying Orthogonality')
    orth = np.matmul(r, X)
    print(orth)

    return beta_hat, y_hat, r

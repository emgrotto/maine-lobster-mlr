from etl import *
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def main():
    years, lobsters, water_temp = extract_lobster_data()
    cei                         = extract_cei_data()
    sea_avg_max, sea_avg_min, sea_avg_avg, sea_max_max, sea_min_min, sea_max_range = extract_sea_surface_temperature_data()
    number_obs = len(years)

    print(f"\nCreating the Desgin matrix for the {number_obs} observations")
    ones = [1 for i in range(number_obs)]
    design_matrix_full = pd.DataFrame(list(zip(ones, cei, sea_avg_max, sea_avg_min, sea_max_range)),
                columns =['ones', 'cei', 'sea_avg_max', 'sea_avg_min', 'sea_max_range'])
    n_obs,n_preds = np.shape(design_matrix_full)
    print(design_matrix_full)

    # print('\nnormalizing the design matrix')
    # design_matrix_norm = (design_matrix_full - design_matrix_full.mean()) / design_matrix_full.std()
    # print(design_matrix_norm)

    # design_matrix_full = design_matrix_norm
    # design_matrix_full['ones'] = ones

    _, eigenvalues, _ = np.linalg.svd(design_matrix_full.values)

    print(f'Design matrix is positive definite: \n {eigenvalues}')

    print('\nPlotting the scatter matrix for the design matrix')
    scatter_matrix(design_matrix_full.iloc[:, 1:n_preds+1], alpha = 0.6, figsize = (7.5, 7.5), diagonal = 'kde')
    plt.savefig('paper_resources/scatter_matrix.png')

    X = design_matrix_full.copy()
    y = lobsters

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
    residuals = y-y_hat
    print(residuals)

    print('\nVerifying Orthogonality')
    orth = np.matmul(residuals, X)
    print(orth)

    print('\nPlotting the residuals')
    fig, ax = plt.subplots()
    scatter = ax.scatter(years, residuals, s=2, c="b")
    scatter.set_label('Residuals')
    ax.legend()

    year_2004_index = years.index(2004) # in 2004, lobster reporting became mandatory
    years_to_2004 = years[:year_2004_index+1]
    residuals_to_2004 = residuals[:year_2004_index+1]
    a1, b1 = np.polyfit(years_to_2004, residuals_to_2004, 1)
    years_to_2004_array = np.array(years_to_2004)
    line1, = ax.plot(years_to_2004_array, a1*years_to_2004_array+b1, c='m')
    line1.set_label('Best fit for years up to 2004')
    ax.legend()

    years_since_2004 = years[year_2004_index+1:]
    residuals_since_2004 = residuals[year_2004_index+1:]
    a2, b2 = np.polyfit(years_since_2004, residuals_since_2004, 1)
    years_since_2004_array = np.array(years_since_2004)
    line2, = ax.plot(years_since_2004_array, a2*years_since_2004_array+b2, c='tab:orange')
    line2.set_label('Best fit for years after 2004')
    ax.legend()

    ax.set(xlabel='Years', ylabel='Residuals')
    plt.savefig('paper_resources/mlr_residuals.png')

    print('\nCalculating sigma squared estimates')
    sigma2_hat = np.dot(residuals,residuals)/(n_obs-(n_preds+1)) # using residuals as population
    print(sigma2_hat)

    print('\nCalculating z scores') 
    beta_hat_var = sigma2_hat*np.diag(XtXinv) # np.diag returns just the diagonals of a Matrix)
    zscore = beta_hat/np.sqrt(beta_hat_var)
    print(zscore)

    print('\nCalculating total variability in target variable')
    s2 = np.var(lobsters)
    print(s2)

    print('\nCalculating Explained variability')
    R2 = 1-(n_obs-(n_preds+1))*sigma2_hat/((n_obs-1)*s2)
    print(R2)

    ### Data Reduction 
    Xtilde = X.copy().iloc[:,1:]
    print('\nMeans of the design matrix')
    print(Xtilde.mean())
    print('\nCentering the design matrix')
    Xtilde = Xtilde - Xtilde.mean()
    print(Xtilde)

    print('\nVerifying centering')
    print(Xtilde.sum())

    print('\nCalcultaing the Sample Covariance matrix')
    S = (np.matmul(np.transpose(Xtilde), Xtilde))/(n_obs - 1)
    print(S)

    print('\nComputing the singular value decomposision')
    U,d,Vt = np.linalg.svd(Xtilde,full_matrices=False)
    print(Vt)

    # Xtilde-np.matmul(np.matmul(U,np.diag(d)),Vt)

    print('\nPlotting a scree plot')
    d_index = [i for i in range(len(d))]
    fig, ax = plt.subplots()
    ax.scatter(d_index, d, s=25, c="b")
    ax.set(xlabel='', ylabel='',
        title='Scree Plot of Singular Values')
    plt.savefig('paper_resources/scree_plot.png')


    print('\nComputing the singular vectors')
    P = np.matmul(Xtilde,np.transpose(Vt))
    print(P)
    print(d)

    # Note:
    # np.matmul(U,np.diag(d))-np.matmul(Xtilde,np.transpose(Vt))
    # Note:
    # np.corrcoef(np.transpose(P))

    pstar = 2 
    print(f'\nUsing {pstar} principle components')
    P = P.iloc[:,:pstar]
    print(P)

    print('\ncompute least squares estimate of gamma')
    ytilde = y-np.mean(y)
    Pt = np.transpose(P)
    PtP = np.matmul(Pt,P)
    Ptytilde = np.matmul(Pt,ytilde)
    PtPinv = np.linalg.inv(PtP)
    gamma_hat = np.matmul(PtPinv,Ptytilde)
    print(gamma_hat)

    print('\nPredicting centered target values')
    yhat_tilde = np.matmul(P,gamma_hat)
    print(yhat_tilde)
    # residuals
    rtilde = ytilde-yhat_tilde

    print('\nUnexplained variability in centered data')
    sig2_hat_tilde = np.dot(rtilde,rtilde)/(n_obs-(pstar+1))
    print(sig2_hat_tilde)

    print('\nCalculating total variability in centered target variable')
    s2tilde = np.var(ytilde)
    print(s2tilde)

    print('\nexplained variability using pstar principal comps')
    R2_pc = 1-(n_obs-(pstar+1))*sig2_hat_tilde/((n_obs-1)*s2tilde)
    print(R2_pc)

    print('\nCalculating z scores')
    gamma_hat_var = sig2_hat_tilde*np.diag(PtPinv)
    zscore_gamma = gamma_hat/np.sqrt(gamma_hat_var)
    print(zscore_gamma)


if __name__ == "__main__":
    main()

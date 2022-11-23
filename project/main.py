from etl import *
from mlr_model import * 
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def scatterplot_matrix(design_matrix):
    scatter = scatter_matrix(design_matrix, alpha = 0.6, figsize = (7.5, 7.5), diagonal = 'kde')
    plt.savefig('paper_resources/scatter_matrix.png')

def main():
    years, lobsters, water_temp = extract_lobster_data()
    cei                         = extract_cei_data()

    print('\ntarget variable')
    print(lobsters)

    sea_avg_max, sea_avg_min, sea_avg_avg, sea_max_max, sea_min_min, sea_max_range = extract_sea_surface_temperature_data()

    design_matrix_full, n_preds, n_obs = create_design_matrix(cei, sea_avg_max, sea_avg_min, sea_max_range)
    scatterplot_matrix(design_matrix_full.iloc[:, 1:n_preds+1])
    print(design_matrix_full)

    beta_hat, y_hat, residuals, XtXinv = predict_from_design(design_matrix_full, lobsters)
    fig, ax = plt.subplots()
    ax.scatter(years, residuals, s=2, c="r")
    ax.set(xlabel='years', ylabel='mlr residuals',
        title='residuals from mlr by year')
    plt.savefig('paper_resources/mlr_residuals.png')

    results_insights(XtXinv, residuals, lobsters, beta_hat, n_preds)
    naive_beta_hat, naive_y_hat, naive_r = predict_naive_model(design_matrix_full, beta_hat, lobsters)
    fig, ax = plt.subplots()
    ax.scatter(years, naive_r, s=2, c="r")
    ax.set(xlabel='years', ylabel='naive residuals',
        title='residuals from naive model by year')
    plt.savefig('paper_resources/naive_residuals.png')

    F = f_test(naive_r, residuals, 1, n_preds+1, n_obs) # -> are these ps correct, looking from https://en.wikipedia.org/wiki/F-test


if __name__ == "__main__":
    main()

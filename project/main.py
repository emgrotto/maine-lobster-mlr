from etl import extract_lobster_data, extract_cei_data, extract_sea_surface_temperature_data
from mlr_model import create_design_matrix, create_single_design_matrix, predict_from_design, results_insights

def main():
    years, lobsters, water_temp = extract_lobster_data()
    cei                         = extract_cei_data()

    sea_avg_max, sea_avg_min, sea_avg_avg, sea_max_max, sea_min_min, sea_max_range = extract_sea_surface_temperature_data()

    design_matrix_full, n_preds = create_design_matrix(cei, sea_avg_max, sea_avg_min, sea_avg_avg)
    design_matrix, n_preds = create_single_design_matrix(water_temp)
    print(design_matrix_full)

    beta_hat, y_hat, residuals, XtXinv = predict_from_design(design_matrix_full, lobsters)

    results_insights(XtXinv, residuals, lobsters, beta_hat, n_preds)

    '''
    print(f'\n{years}')
    print(f'\n{lobsters}')
    print(f'\n{cei}')
    print(f'\n{sea_max}')
    print(f'\n{sea_min}')
    print(f'\n{sea_avg}')
    '''


if __name__ == "__main__":
    main()

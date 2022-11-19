from etl import extract_lobster_data, extract_cei_data, extract_sea_surface_temperature_data
from mlr_model import create_design_matrix, create_single_design_matrix, predict_from_design

def main():
    years, lobsters, water_temp = extract_lobster_data()
    cei                         = extract_cei_data()

    sea_avg_max, sea_avg_min, sea_avg_avg, sea_max_max, sea_min_min, sea_max_range = extract_sea_surface_temperature_data()

    # design_matrix_full = create_design_matrix(cei).to_numpy()
    design_matrix = create_single_design_matrix(water_temp)
    print(design_matrix)

    beta_hat, y_hat, r = predict_from_design(design_matrix, lobsters)

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

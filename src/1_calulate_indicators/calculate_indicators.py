from data_process_utils import *


def get_factors(datastore_dir):
    df = pd.read_json(datastore_dir, encoding="utf-8", orient='records', lines=True)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_kdj(df)
    df = calculate_returns(df)
    df = calculate_vwap(df)
    df = add_mean_reversion_alpha(df)
    df = add_momentum_alpha(df)
    df = alpha002(df)
    df = alpha006(df)
    df = alpha009(df)
    df = alpha012(df)
    df = alpha021(df)
    df = alpha023(df)
    df = alpha024(df)
    df = alpha028(df)
    df = alpha032(df)
    df = alpha041(df)
    df = alpha046(df)
    df = alpha049(df)
    df = alpha051(df)
    df = alpha053(df)
    df = alpha054(df)
    df = alpha101(df)
    return df


def update_datastore_with_indicator(dataset):
    datastore_dir = "../../data/processed_data/data_store/"+dataset+"_data_store.json"
    df = get_factors(datastore_dir)
    df.to_json(datastore_dir, orient='records', lines=True)
    return 0


if __name__ == "__main__":
    update_datastore_with_indicator('cikm18')

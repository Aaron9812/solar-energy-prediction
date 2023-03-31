import pandas as pd

def reduce_dataframe(df: pd.DataFrame)-> pd.DataFrame:
    # Drop the 'Unnamed: 51' column
    df = df.drop('Unnamed: 51', axis=1)
    # drop indComponentUat0, WindComponentVat0, WindComponentUat100, and WindComponentVat100
    df = df.drop(['WindComponentUat0', 'WindComponentVat0', 'WindComponentUat100', 'WindComponentVat100'], axis=1)
    # drop clearsky_diffuse, clearsky_direct, clearsky_global, clearsky_diffuse_agg, clearsky_direct_agg, and clearsky_global_agg
    df = df.drop(['clearsky_diffuse', 'clearsky_direct', 'clearsky_global', 'clearsky_diffuse_agg', 'clearsky_direct_agg', 'clearsky_global_agg'], axis=1)
    # drop RelativeHumidityAt1000 and RelativeHumidityAt950
    df = df.drop(['RelativeHumidityAt1000', 'RelativeHumidityAt950'], axis=1)
    # drop PotentialVorticityAt950
    df = df.drop(['PotentialVorticityAt950'], axis=1)
    # drop all columns that include cos, sin and math 
    df_reduced = df.drop([col for col in df.columns if 'cos' in col.lower() or 'sin' in col.lower() or 'math' in col.lower()], axis=1)
    return df_reduced



def split_data(df_reduced: pd.DataFrame, split_rate: int):
    # Remove the target column from the dataset to create the feature matrix X
    X = df_reduced.drop('power_normed', axis=1)

    # Set the target variable y to be the 'power_normed' column
    y = df_reduced['power_normed']

    # Calculate the index for the split
    split_index = int(len(X) * split_rate)

    # Split the data into training and testing sets
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, y_train, X_test, y_test

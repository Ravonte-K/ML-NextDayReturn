import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PATH = "/Users/ravontekriegler/Documents/GitHub/ML-NextDayReturn/data/raw/spy.csv"
TEST_SIZE = 0.2

def get_data(path: str) -> pd.DataFrame :
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

def findavg(df):
    avg_10 = df["Close"].shift(1).rolling(10).mean()
    yesterday_close = df["Close"].shift(1)

    pct_from_avg = (yesterday_close - avg_10) / avg_10
    estimated_price = df["Close"] * (1 + pct_from_avg)

    return estimated_price

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['avgvolume5'] = df['Volume'].shift(1).rolling(window=5).mean()
    # df["SPYRETURN"] = 0.001 * pd.Series(range(len(df)), index=df.index)
    df["momentum"] = ((df["Close"].shift(1) - df["Close"].shift(10)) / df["Close"].shift(1))
    df['VOLUMEVS'] = (df["Volume"].shift(1) / df["avgvolume5"])
    df["avg_10"] = df["Close"].shift(1).rolling(10).mean()
    df["pct_from_avg"] = (df["Close"].shift(1) - df["avg_10"]) / df["avg_10"]
    df['useave'] = df["Close"] * (1 + df["pct_from_avg"])

# TURN TARGET INTO MONTH OUT UP OR DOWN AND CONSIDER MAKING A PRICE PREDICTION TARGET 
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)



    df = df.dropna().reset_index(drop=True)
    print(df.head(20))
    return df


def split_data(df: pd.DataFrame):
    feature_columns = [
        # 'SPYRETURN',
        'momentum',
        'VOLUMEVS',
        'useave'
    ]
    X = df[feature_columns]
    y = df["target"]

    split_index = int(len(df) * (1 - TEST_SIZE))
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test, feature_columns

def walkfoward_training(df: pd.DataFrame):
    feature_columns = [
        # 'SPYRETURN',
        'momentum',
        'VOLUMEVS',
        'useave'
    ]
    X = df[feature_columns]
    y = df['target']
    n = len(df)
    train = int(n*0.15)
    test = int(n*.05)
    start = 0
    accuracies = []

    if train == 0 or test == 0:
        raise ValueError("Not enough rows for 15% train and 5% test windows.")

    while start + train + test <= n:
        train_start = start
        train_end = start + train
        test_end = train_end + test

        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]

        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        print(f"train: {train_start}:{train_end}, test: {train_end}:{test_end}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        accuracy = (y_pred == y_test).mean()
        accuracies.append(accuracy)
        print(f"fold accuracy: {accuracy:.2%}")

        start += test

    if accuracies:
        print(f"Average walk-forward accuracy: {sum(accuracies) / len(accuracies):.2%}")

    return accuracies


def main():
    df = get_data(PATH)
    df = make_features(df)

    X_train, X_test, y_train, y_test, feature_columns = split_data(df)
    walkfoward_training(df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # spy_idx = feature_columns.index("SPYRETURN")
    # X_train_scaled[:, spy_idx] *= 2
    # X_test_scaled[:, spy_idx] *= 2

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_accuracy = (y_train_pred == y_train).mean()
    test_accuracy = (y_test_pred == y_test).mean()
    base = (y_test == 1).mean()
    otherb = y_test.mean()
    
    print(otherb)
    print(float(base))
    print(f"Train direction accuracy: {train_accuracy:.2%}")
    print(f"Test direction accuracy: {test_accuracy:.2%}")
    print(test_accuracy)
    print(f'The model is {test_accuracy - base} better than always guessing up')


if __name__ == "__main__":
    main()

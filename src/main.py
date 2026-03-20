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

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['avgvolume5'] = df['Volume'].shift(1).rolling(window=5).mean()
    # df["SPYRETURN"] = 0.001 * pd.Series(range(len(df)), index=df.index)
    df["momentum"] = ((df["Close"].shift(1) - df["Close"].shift(10)) / df["Close"].shift(1))
    df['VOLUMEVS'] = (df["Volume"].shift(1) / df["avgvolume5"])
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna().reset_index(drop=True)
    print(df.head(20))
    return df


def split_data(df: pd.DataFrame):
    feature_columns = [
        # 'SPYRETURN',
        'momentum',
        'VOLUMEVS'
    ]
    X = df[feature_columns]
    y = df["target"]

    split_index = int(len(df) * (1 - TEST_SIZE))
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test, feature_columns



def main():
    df = get_data(PATH)
    df = make_features(df)

    X_train, X_test, y_train, y_test, feature_columns = split_data(df)

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
    base = (1 == y_test).mean()
    otherb = y_test.mean()
    
    print(otherb)
    print(float(base))
    print(f"Train direction accuracy: {train_accuracy:.2%}")
    print(f"Test direction accuracy: {test_accuracy:.2%}")
    print(test_accuracy)
    print(f'The model is {test_accuracy - base} better than always guessing up')


if __name__ == "__main__":
    main()

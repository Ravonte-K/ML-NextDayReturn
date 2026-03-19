import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


PATH = "/Users/ravontekriegler/Documents/GitHub/ML-NextDayReturn/data/raw/sp500.csv"
TICKER = "GOOG"  
TEST_SIZE = 0.2  


def load_ticker_data(path: str, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Ticker"] == ticker].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    
    # df["prev_return"] = df["Close"].pct_change()
    df["prev_volume_change"] = df["Volume"].pct_change()
    df["prev_volume_change"] = df['prev_volume_change']
    df["momentum"] = df["Close"].shift(1) / df["Close"].shift(10) - 1
    df["intraday_return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]

    
    df["target"] = df["Close"].shift(-1) / df["Close"] - 1

    df = df.dropna()
    return df


def split_data(df: pd.DataFrame):
    feature_columns = [
        # "prev_return",
        "prev_volume_change",
        "momentum",
        "intraday_return",
        "daily_range",
    ]
    X = df[feature_columns]
    y = df["target"]

    split_index = int(len(df) * (1 - TEST_SIZE))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    test_dates = df["Date"].iloc[split_index:]

    return X_train, X_test, y_train, y_test, test_dates, feature_columns


def print_results(y_train, y_train_pred, y_test, y_test_pred, model, feature_columns) -> None:
    test_direction_accuracy = ((y_test > 0) == (y_test_pred > 0)).mean()

    print(f"\nTicker: {TICKER}")
    print(f"Train rows: {len(y_train)}")
    print(f"Test rows: {len(y_test)}")

    print("\nTraining Results:")
    print(f"  R²: {r2_score(y_train, y_train_pred):.4f}")
    print(f"  MAE: {mean_absolute_error(y_train, y_train_pred):.6f}")

    print("\nTest Results:")
    print(f"  R²: {r2_score(y_test, y_test_pred):.4f}")
    print(f"  MAE: {mean_absolute_error(y_test, y_test_pred):.6f}")
    print(f"  Direction Accuracy: {test_direction_accuracy:.2%}")

    print("\nCoefficients:")
    for name, coef in zip(feature_columns, model.coef_):
        print(f"  {name}: {coef:.6f}")


def plot(test_dates, y_test, y_test_pred) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(test_dates, y_test, label="Actual Return")
    plt.plot(test_dates, y_test_pred, label="Predicted Return")
    plt.title(f"{TICKER} Actual vs Predicted Returns")
    plt.xlabel("Date")
    plt.ylabel("Next-Day Return")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = load_ticker_data(PATH, TICKER)
    df = add_features(df)
    X_train, X_test, y_train, y_test, test_dates, feature_columns = split_data(df)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print_results(y_train, y_train_pred, y_test, y_test_pred, model, feature_columns)
    plot(test_dates, y_test, y_test_pred)


if __name__ == "__main__":
    main()

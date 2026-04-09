import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

PATH = "/Users/ravontekriegler/Documents/GitHub/ML-NextDayReturn/data/raw/spy.csv"
TEST_SIZE = 0.2
HORIZON_DAYS = 10
NO_TRADE_BAND = 0.005
SHOW_PLOTS = True

DIRECTION_FEATURES = [
    "return_5d",
    "momentum_20",
    "ma_gap_50",
]

RETURN_FEATURES = [
    "return_5d",
    "momentum_20",
    "momentum_60",
    "ma_gap_50",
]


def get_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_5d"] = (df["Close"].shift(1) / df["Close"].shift(6)) - 1
    df["momentum_20"] = df["Close"].shift(1) / df["Close"].shift(21) - 1
    df["momentum_60"] = df["Close"].shift(1) / df["Close"].shift(61) - 1
    df["avg_50"] = df["Close"].shift(1).rolling(50).mean()
    df["ma_gap_50"] = (df["Close"].shift(1) - df["avg_50"]) / df["avg_50"]

    future_return = df["Close"].shift(-HORIZON_DAYS) / df["Close"] - 1
    df["target"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[future_return > NO_TRADE_BAND, "target"] = 1
    df.loc[future_return < -NO_TRADE_BAND, "target"] = 0
    df["target_return"] = future_return

    df = df.dropna().reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame):
    split_index = int(len(df) * (1 - TEST_SIZE))

    X_dir = df[DIRECTION_FEATURES]
    y_dir = df["target"].astype(int)
    X_ret = df[RETURN_FEATURES]
    y_ret = df["target_return"]
    close = df["Close"]

    X_dir_train = X_dir.iloc[:split_index]
    X_dir_test = X_dir.iloc[split_index:]
    y_dir_train = y_dir.iloc[:split_index]
    y_dir_test = y_dir.iloc[split_index:]

    X_ret_train = X_ret.iloc[:split_index]
    X_ret_test = X_ret.iloc[split_index:]
    y_ret_train = y_ret.iloc[:split_index]
    y_ret_test = y_ret.iloc[split_index:]
    close_test = close.iloc[split_index:]
    test_dates = df["Date"].iloc[split_index:]

    return (
        X_dir_train,
        X_dir_test,
        y_dir_train,
        y_dir_test,
        X_ret_train,
        X_ret_test,
        y_ret_train,
        y_ret_test,
        close_test,
        test_dates,
    )


def tune_direction_threshold(X_train: pd.DataFrame, y_train: pd.Series) -> float:
    val_start = int(len(X_train) * 0.8)
    if val_start <= 0 or val_start >= len(X_train):
        return 0.5

    X_subtrain = X_train.iloc[:val_start]
    y_subtrain = y_train.iloc[:val_start]
    X_val = X_train.iloc[val_start:]
    y_val = y_train.iloc[val_start:]

    scaler = StandardScaler()
    X_subtrain_scaled = scaler.fit_transform(X_subtrain)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_subtrain_scaled, y_subtrain)
    y_val_prob = model.predict_proba(X_val_scaled)[:, 1]

    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in [i / 100 for i in range(35, 96)]:
        y_val_pred = (y_val_prob >= threshold).astype(int)
        accuracy = (y_val_pred == y_val).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold


def walkfoward_training(df: pd.DataFrame):
    X = df[DIRECTION_FEATURES]
    y = df["target"].astype(int)
    n = len(df)
    train = int(n * 0.15)
    test = int(n * 0.05)
    start = 0
    accuracies = []
    edges = []

    if train == 0 or test == 0:
        raise ValueError("Not enough rows for 15% train and 5% test windows.")

    while start + train + test <= n:
        train_end = start + train
        test_end = train_end + test

        X_train = X.iloc[start:train_end]
        y_train = y.iloc[start:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        threshold = tune_direction_threshold(X_train, y_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train_scaled, y_train)
        y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_test_prob >= threshold).astype(int)

        accuracy = (y_pred == y_test).mean()
        base = (y_test == 1).mean()
        edge = accuracy - base

        accuracies.append(accuracy)
        edges.append(edge)

        print(f"train: {start}:{train_end}, test: {train_end}:{test_end}")
        print(f"fold threshold: {threshold:.2f}")
        print(f"fold accuracy: {accuracy:.2%}")
        print(f"fold edge vs always-up: {edge:.2%}")

        start += test

    if accuracies:
        print(f"Average walk-forward accuracy: {sum(accuracies) / len(accuracies):.2%}")
        print(f"Average walk-forward edge vs always-up: {sum(edges) / len(edges):.2%}")

    return accuracies


def walkfoward_return_training(df: pd.DataFrame):
    X = df[RETURN_FEATURES]
    y = df["target_return"]
    close = df["Close"]
    n = len(df)
    train = int(n * 0.15)
    test = int(n * 0.05)
    start = 0
    maes = []
    return_edges = []
    price_edges = []

    if train == 0 or test == 0:
        raise ValueError("Not enough rows for 15% train and 5% test windows.")

    while start + train + test <= n:
        train_end = start + train
        test_end = train_end + test

        X_train = X.iloc[start:train_end]
        y_train = y.iloc[start:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]
        close_test = close.iloc[train_end:test_end]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_mae = mean_absolute_error(y_test, y_pred)
        baseline_return_mae = y_test.abs().mean()
        return_edge = baseline_return_mae - fold_mae

        y_test_price = close_test * (1 + y_test)
        y_test_price_pred = close_test * (1 + y_pred)
        baseline_price_mae = mean_absolute_error(y_test_price, close_test)
        price_mae = mean_absolute_error(y_test_price, y_test_price_pred)
        price_edge = baseline_price_mae - price_mae

        maes.append(fold_mae)
        return_edges.append(return_edge)
        price_edges.append(price_edge)

        print(f"return fold MAE: {fold_mae:.4f}")
        print(f"return fold edge vs zero-return MAE: {return_edge:.4f}")
        print(f"price fold edge vs naive MAE: {price_edge:.4f}")

        start += test

    if maes:
        print(f"Average walk-forward return MAE: {sum(maes) / len(maes):.4f}")
        print(f"Average walk-forward return edge: {sum(return_edges) / len(return_edges):.4f}")
        print(f"Average walk-forward price edge: {sum(price_edges) / len(price_edges):.4f}")

    return maes


def plot_results(
    y_dir_test: pd.Series,
    y_dir_test_pred: pd.Series,
    y_ret_test: pd.Series,
    y_ret_test_pred: pd.Series,
    close_test: pd.Series,
    test_dates: pd.Series,
):
    import matplotlib.pyplot as plt


    plt.figure(figsize=(6, 6))
    plt.scatter(y_ret_test, y_ret_test_pred, alpha=0.5)
    mn = min(y_ret_test.min(), y_ret_test_pred.min())
    mx = max(y_ret_test.max(), y_ret_test_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("Actual Return")
    plt.ylabel("Predicted Return")
    plt.title(f"{HORIZON_DAYS}-day Return: Actual vs Predicted")
    plt.show()

    actual_price = (close_test * (1 + y_ret_test)).reset_index(drop=True)
    pred_price = (close_test * (1 + y_ret_test_pred)).reset_index(drop=True)

    plt.figure(figsize=(10, 4))
    plt.plot(test_dates.reset_index(drop=True), actual_price, label="Actual Future Price")
    plt.plot(test_dates.reset_index(drop=True), pred_price, label="Predicted Future Price")
    plt.plot(
        test_dates.reset_index(drop=True),
        close_test.reset_index(drop=True),
        label="Naive Baseline Price",
        linestyle="--",
    )
    plt.legend()
    plt.title(f"{HORIZON_DAYS}-day Implied Price on Test Set")
    plt.tight_layout()
    plt.show()


def main():
    df = get_data(PATH)
    df = make_features(df)

    (
        X_dir_train,
        X_dir_test,
        y_dir_train,
        y_dir_test,
        X_ret_train,
        X_ret_test,
        y_ret_train,
        y_ret_test,
        close_test,
        test_dates,
    ) = split_data(df)

    walkfoward_training(df)
    walkfoward_return_training(df)

    direction_threshold = tune_direction_threshold(X_dir_train, y_dir_train)
    scaler = StandardScaler()
    X_dir_train_scaled = scaler.fit_transform(X_dir_train)
    X_dir_test_scaled = scaler.transform(X_dir_test)

    direction_model = LogisticRegression(max_iter=2000)
    direction_model.fit(X_dir_train_scaled, y_dir_train)

    y_dir_train_prob = direction_model.predict_proba(X_dir_train_scaled)[:, 1]
    y_dir_test_prob = direction_model.predict_proba(X_dir_test_scaled)[:, 1]
    y_dir_train_pred = (y_dir_train_prob >= direction_threshold).astype(int)
    y_dir_test_pred = (y_dir_test_prob >= direction_threshold).astype(int)

    train_accuracy = (y_dir_train_pred == y_dir_train).mean()
    test_accuracy = (y_dir_test_pred == y_dir_test).mean()
    direction_base = (y_dir_test == 1).mean()
    direction_edge = test_accuracy - direction_base

    print(f"Direction threshold: {direction_threshold:.2f}")
    print(f"Train {HORIZON_DAYS}-day direction accuracy: {train_accuracy:.2%}")
    print(f"Test {HORIZON_DAYS}-day direction accuracy: {test_accuracy:.2%}")
    print(f"Always-up baseline accuracy: {direction_base:.2%}")
    print(f"Model edge vs always-up: {direction_edge:.2%}")
    if direction_edge >= 0:
        print("The direction model beat the always-up baseline.")
    else:
        print("The direction model did not beat the always-up baseline.")

    return_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
    )
    return_model.fit(X_ret_train, y_ret_train)
    y_ret_train_pred = return_model.predict(X_ret_train)
    y_ret_test_pred = return_model.predict(X_ret_test)

    train_return_mae = mean_absolute_error(y_ret_train, y_ret_train_pred)
    test_return_mae = mean_absolute_error(y_ret_test, y_ret_test_pred)
    train_return_r2 = r2_score(y_ret_train, y_ret_train_pred)
    test_return_r2 = r2_score(y_ret_test, y_ret_test_pred)
    baseline_test_return_mae = y_ret_test.abs().mean()
    return_edge = baseline_test_return_mae - test_return_mae

    y_test_price = close_test * (1 + y_ret_test)
    y_test_price_pred = close_test * (1 + y_ret_test_pred)
    baseline_test_price_mae = mean_absolute_error(y_test_price, close_test)
    test_price_mae = mean_absolute_error(y_test_price, y_test_price_pred)
    price_edge = baseline_test_price_mae - test_price_mae

    print(f"Train {HORIZON_DAYS}-day return MAE: {train_return_mae:.4f}")
    print(f"Test {HORIZON_DAYS}-day return MAE: {test_return_mae:.4f}")
    print(f"Train {HORIZON_DAYS}-day return R2: {train_return_r2:.4f}")
    print(f"Test {HORIZON_DAYS}-day return R2: {test_return_r2:.4f}")
    print(f"Zero-return baseline MAE: {baseline_test_return_mae:.4f}")
    print(f"Model edge vs zero-return MAE: {return_edge:.4f}")
    print(f"Test {HORIZON_DAYS}-day implied price MAE: {test_price_mae:.4f}")
    print(f"Naive price baseline MAE: {baseline_test_price_mae:.4f}")
    print(f"Model edge vs naive baseline MAE: {price_edge:.4f}")
    if price_edge >= 0:
        print("The price model beat the naive baseline.")
    else:
        print("The price model did not beat the naive baseline.")

    if SHOW_PLOTS:
        plot_results(
            y_dir_test=y_dir_test,
            y_dir_test_pred=y_dir_test_pred, # type: ignore
            y_ret_test=y_ret_test,
            y_ret_test_pred=y_ret_test_pred, # type: ignore
            close_test=close_test,
            test_dates=test_dates,
        )


if __name__ == "__main__":
    main()

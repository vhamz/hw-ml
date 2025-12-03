import marimo

__generated_with = "0.17.8"
app = marimo.App(layout_file="layouts/seminar2.slides.json")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    IMPORTING LIBRARIES
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    import seaborn as sns
    import matplotlib.pyplot as plt
    return (
        LinearRegression,
        StandardScaler,
        mean_absolute_error,
        mean_squared_error,
        np,
        pd,
        plt,
        r2_score,
        sns,
        train_test_split,
    )


@app.cell
def _(pd):
    calendar_df = pd.read_csv('Calendar.csv')
    trip_df = pd.read_csv('trip.csv')
    return calendar_df, trip_df


@app.cell
def _(calendar_df):
    calendar_df
    return


@app.cell
def _(trip_df):
    trip_df
    return


@app.cell
def _(calendar_df, trip_df):
    print(f'Calendar data shape {calendar_df.shape}')
    print(f'Trip data shape {trip_df.shape}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PRE PROCESSING/FEATURE ENGINEERING
    """)
    return


@app.cell
def _(trip_df):
    print('TRIP_DF INFO:')
    print(f"\nData size: {trip_df.shape}")
    print(f"\nData type:\n{trip_df.dtypes}")
    print(f"\nMissing values:\n{trip_df.isnull().sum()}")
    print(f"\nDuplicated data':\n{trip_df.duplicated().sum()}")
    return


@app.cell
def _(calendar_df):
    print('CALENDAR_DF INFO:')
    print(f"\nData size: {calendar_df.shape}")
    print(f"\nData type:\n{calendar_df.dtypes}")
    print(f"\nMissing values:\n{calendar_df.isnull().sum()}")
    print(f"\nDuplicated data':\n{calendar_df.duplicated().sum()}")
    return


@app.cell
def _(trip_df):
    trip_df_1 = trip_df.dropna(ignore_index=True)
    print(trip_df_1.isnull().sum())
    return (trip_df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Feature engineering
    """)
    return


@app.cell
def _(pd, trip_df_1):
    trip_df_1['call_time'] = pd.to_datetime(trip_df_1['call_time'])
    trip_df_1['finish_time'] = pd.to_datetime(trip_df_1['finish_time'])
    trip_df_1[['call_time', 'finish_time']].head()
    return


@app.cell
def _(trip_df_1):
    trip_df_1.info()
    return


@app.cell
def _(trip_df_1):
    trip_df_1['trip_date'] = trip_df_1['call_time'].dt.date
    trip_df_1['trip_duration_min'] = (trip_df_1['finish_time'] - trip_df_1['call_time']).dt.total_seconds() / 60
    trip_df_1['call_hour'] = trip_df_1['call_time'].dt.hour
    trip_df_1[['trip_date', 'trip_duration_min', 'call_hour']].head()
    return


@app.cell
def _(trip_df_1):
    (trip_df_1['finish_time'] < trip_df_1['call_time']).sum()
    return


@app.cell
def _(trip_df_1):
    trip_df_1['trip_duration_min'].describe()
    return


@app.cell
def _(calendar_df, pd):
    calendar_df['calendar_date'] = pd.to_datetime(calendar_df['calendar_date']).dt.date
    calendar_df.head()
    return


@app.cell
def _(calendar_df, trip_df_1):
    merged_df = trip_df_1.merge(calendar_df, left_on='trip_date', right_on='calendar_date', how='left')
    merged_df[['trip_date', 'week_day', 'holiday']].head()
    return (merged_df,)


@app.cell
def _(merged_df):
    merged_df.isnull().sum()
    return


@app.cell
def _(merged_df):
    final_df = merged_df[[
        'trip_distance',
        'trip_duration_min',
        'call_hour',
        'week_day',
        'holiday',
        'surge_rate',
        'trip_fare'
    ]].copy()

    final_df.head()
    return (final_df,)


@app.cell
def _(final_df):
    final_df.info()
    return


@app.cell
def _(final_df):
    import plotly.express as px
    numeric_cols = ['trip_distance', 'trip_duration_min', 'surge_rate', 'trip_fare']
    for _col in numeric_cols:
        px.histogram(final_df, x=_col).show()
    return (numeric_cols,)


@app.cell
def _(final_df, numeric_cols, plt, sns):
    for _col in numeric_cols:
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=final_df[_col])
        plt.title(f'Outliers check: {_col}')
        plt.show()
    return


@app.cell
def _(final_df):
    # lowest point
    min_duration = 3
    max_duration = 90
    # highest point
    final_df_1 = final_df[(final_df['trip_duration_min'] >= min_duration) & (final_df['trip_duration_min'] <= max_duration)].reset_index(drop=True)
    return (final_df_1,)


@app.cell
def _(final_df_1, pd):
    final_df_2 = pd.get_dummies(final_df_1, columns=['week_day'], drop_first=True)
    return (final_df_2,)


@app.cell
def _(final_df_2):
    final_df_2.head(100)
    return


@app.cell
def _(final_df_2):
    final_df_2['trip_distance'].describe()
    return


@app.cell
def _(final_df_2):
    final_df_2.dtypes
    return


@app.cell
def _(final_df_2):
    clean_df = final_df_2.copy()

    clean_df = clean_df[clean_df["trip_distance"] > 0]

    clean_df = clean_df[clean_df["surge_rate"] >= 0]

    clean_df = clean_df[clean_df["trip_distance"] < 60]

    clean_df = clean_df.reset_index(drop=True)
    return (clean_df,)


@app.cell
def _(clean_df):
    clean_df.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    MODEL
    """)
    return


@app.cell
def _(clean_df, train_test_split):
    X = clean_df.drop('trip_fare', axis=1)
    y = clean_df['trip_fare']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _(StandardScaler, X_test, X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, X_train_scaled, scaler


@app.cell
def _(LinearRegression, X_train_scaled, y_train):
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    return (lr,)


@app.cell
def _(X_test_scaled, lr, pd, y_test):
    test_pred = lr.predict(X_test_scaled)

    test_results = pd.DataFrame({
        'actual_trip_fare': y_test,
        'predicted_trip_fare': test_pred,
        'residual': y_test - test_pred
    })

    test_results.head()
    return (test_pred,)


@app.cell
def _(
    mean_absolute_error,
    mean_squared_error,
    np,
    r2_score,
    test_pred,
    y_test,
):
    _mae = mean_absolute_error(y_test, test_pred)
    _mse = mean_squared_error(y_test, test_pred)
    _rmse = np.sqrt(_mse)
    _r2 = r2_score(y_test, test_pred)
    print('MAE :', _mae)
    print('MSE :', _mse)
    print('RMSE:', _rmse)
    print('R²  :', _r2)
    return


@app.cell
def _(plt, test_pred, y_test):
    plt.figure(figsize=(8, 8))
    _max_val = 50
    # Ограничим оси чтобы избавиться от выбросов
    plt.scatter(test_pred, y_test, alpha=0.3, s=10, label='Test data')
    plt.plot([0, _max_val], [0, _max_val], color='red', linewidth=2, label='Ideal')
    plt.xlim(0, _max_val)
    plt.ylim(0, _max_val)
    plt.xlabel('Predicted trip_fare')
    plt.ylabel('Actual trip_fare')
    plt.title('Actual vs Predicted (Linear Regression)')
    plt.legend()
    plt.grid(True)
    # линия идеальных предсказаний
    plt.show()
    return


@app.cell
def _(X_train, lr, pd):
    coef_df = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': lr.coef_
    }).sort_values(by='coefficient', key=abs, ascending=False)

    coef_df
    return (coef_df,)


@app.cell
def _(coef_df, plt):
    plt.figure(figsize=(8,6))
    plt.barh(coef_df['feature'], coef_df['coefficient'])
    plt.title("Feature Coefficients (Linear Regression)")
    plt.xlabel("Coefficient value")
    plt.gca().invert_yaxis()
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    distance_slider = mo.ui.slider(1, 60, step=1, value=10, label="Trip distance (km)")
    duration_slider = mo.ui.slider(1, 60, step=1, value=15, label="Trip duration (min)")
    hour_slider     = mo.ui.slider(0, 23, step=1, value=10, label="Call hour")
    surge_slider    = mo.ui.slider(0.0, 3.0, step=0.1, value=1.0, label="Surge rate (x)")
    return distance_slider, duration_slider, hour_slider, mo, surge_slider


@app.cell
def _(
    X_train,
    distance_slider,
    duration_slider,
    hour_slider,
    lr,
    mo,
    pd,
    scaler,
    surge_slider,
):
    mean_features = X_train.mean().copy()

    mean_features["trip_distance"] = distance_slider.value
    mean_features["trip_duration_min"] = duration_slider.value
    mean_features["call_hour"] = hour_slider.value
    mean_features["surge_rate"] = surge_slider.value

    df_input = pd.DataFrame([mean_features])
    scaled = scaler.transform(df_input)
    pred = lr.predict(scaled)[0]

    mo.md(f"""
    ### Predicted price: **${pred:.2f}**

    - Distance: **{distance_slider.value} km**
    - Duration: **{duration_slider.value} min**
    - Time: **{hour_slider.value}:00**
    - Surge: **x{surge_slider.value}**
    """)
    return


@app.cell
def _(distance_slider):
    distance_slider
    return


@app.cell
def _(surge_slider):
    surge_slider
    return


@app.cell
def _(hour_slider):
    hour_slider
    return


@app.cell
def _(duration_slider):
    duration_slider
    return


if __name__ == "__main__":
    app.run()

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta

# Load your dataset
df = pd.read_csv("airline_lowest_price_analysis.csv")
df['TravelDate'] = pd.to_datetime(df['TravelDate'])
df['Year'] = df['TravelDate'].dt.year
df['Month'] = df['TravelDate'].dt.month
df['Day'] = df['TravelDate'].dt.day
df['SeasonEncoded'] = df['Season'].map({'Summer': 0, 'Fall': 1})

# Compute average best days before travel
avg_days_df = df.groupby(['Airline', 'Season'])['BestDaysBeforeTravel'].mean().reset_index()
avg_days_df.to_csv("avg_best_days_by_airline_season.csv", index=False)

# One-hot encode Airline
df = pd.get_dummies(df, columns=['Airline'])

# Prepare features and target
features = ['BestDaysBeforeTravel', 'SeasonEncoded', 'Month', 'Day'] + [col for col in df.columns if col.startswith('Airline_')]
X = df[features]
y = df['LowestPrice']

# Split into train and test
X_train = X[df['Year'] <= 2022]
y_train = y[df['Year'] <= 2022]
X_test = X[df['Year'].isin([2023, 2024])]
y_test = y[df['Year'].isin([2023, 2024])]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"MAE on 2023–2024 Test Data: {mae:.2f}")

# Predict for 2025 and 2026
future_years = [2025, 2026]
seasons = [('Summer', 0, (5, 15), (6, 15)), ('Fall', 1, (8, 23), (9, 23))]
all_airlines = avg_days_df['Airline'].unique()
airline_cols = [col for col in df.columns if col.startswith('Airline_')]

future_rows = []

for year in future_years:
    for season, season_code, (start_m, start_d), (end_m, end_d) in seasons:
        start_date = datetime(year, start_m, start_d)
        end_date = datetime(year, end_m, end_d)
        for single_date in pd.date_range(start_date, end_date):
            for airline in all_airlines:
                avg_days = avg_days_df.query("Airline == @airline and Season == @season")["BestDaysBeforeTravel"].values
                if len(avg_days) == 0:
                    continue
                row = {
                    "BestDaysBeforeTravel": avg_days[0],
                    "SeasonEncoded": season_code,
                    "Month": single_date.month,
                    "Day": single_date.day,
                }
                for col in airline_cols:
                    row[col] = 1 if col == f"Airline_{airline}" else 0
                row_df = pd.DataFrame([row])
                future_price = model.predict(row_df)[0]
                future_rows.append({
                    "Airline": airline,
                    "Season": season,
                    "TravelDate": single_date.date(),
                    "BestDaysBeforeTravel": int(avg_days[0]),
                    "PredictedPrice": round(future_price, 2)
                })

# Save predictions
df_future = pd.DataFrame(future_rows)
df_future.to_csv("predicted_prices_2025_2026.csv", index=False)

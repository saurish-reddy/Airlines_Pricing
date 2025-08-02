import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load the cleaned CSV file
df = pd.read_csv("completely_sorted_final.csv", parse_dates=["TravelDate", "BookingDate"])

# 1. ANALYSIS: For each airline, for each date, find the DaysBeforeTravel when price is lowest
def analyze_lowest_price_days(df):
    results = []
    for airline in df['Airline'].unique():
        for season in ['IsSummer', 'IsFall']:
            seasonal_df = df[(df[season]) & (df['Airline'] == airline)]
            grouped = seasonal_df.groupby('TravelDate')
            for travel_date, group in grouped:
                cheapest = group.loc[group['PriceUSD'].idxmin()]
                results.append({
                    'Airline': airline,
                    'Season': 'Summer' if season == 'IsSummer' else 'Fall',
                    'TravelDate': travel_date,
                    'BestDaysBeforeTravel': cheapest['DaysBeforeTravel'],
                    'LowestPrice': cheapest['PriceUSD']
                })
    return pd.DataFrame(results)

analysis_result = analyze_lowest_price_days(df)

# Save analysis to file
analysis_result.to_csv("airline_lowest_price_analysis.csv", index=False)

# 2. ML Model: Train on 2015-2022, test on 2023-2024, predict 2025-2026

# Backup airline-route info before get_dummies
route_info = df[['Airline', 'Origin', 'Destination']].drop_duplicates()

# One-hot encoding
df_ml = pd.get_dummies(df.copy(), columns=['Airline', 'Origin', 'Destination'], drop_first=True)

# Split into train/test
train_df = df_ml[df_ml['Year'] <= 2022]
test_df = df_ml[df_ml['Year'].isin([2023, 2024])]

X_train = train_df.drop(columns=['PriceUSD', 'TravelDate', 'BookingDate'])
y_train = train_df['PriceUSD']
X_test = test_df.drop(columns=['PriceUSD', 'TravelDate', 'BookingDate'])
y_test = test_df['PriceUSD']

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))
print("R^2 Score:", r2_score(y_test, preds))

# Predict for 2025–2026
def generate_future_data(route_info, year):
    rows = []
    for season, start, end in [('IsSummer', '05-15', '06-15'), ('IsFall', '08-23', '09-23')]:
        dates = pd.date_range(f"{year}-{start}", f"{year}-{end}")
        for _, row in route_info.iterrows():
            for travel_date in dates:
                for dbt in range(60, 0, -1):
                    booking_date = travel_date - pd.Timedelta(days=dbt)
                    rows.append({
                        'Airline': row['Airline'],
                        'Origin': row['Origin'],
                        'Destination': row['Destination'],
                        'TravelDate': travel_date,
                        'BookingDate': booking_date,
                        'DaysBeforeTravel': dbt,
                        'Year': travel_date.year,
                        'Month': travel_date.month,
                        'Day': travel_date.day,
                        'IsSummer': season == 'IsSummer',
                        'IsFall': season == 'IsFall',
                    })
    return pd.DataFrame(rows)

future_df = pd.concat([
    generate_future_data(route_info, 2025),
    generate_future_data(route_info, 2026)
])

future_encoded = pd.get_dummies(future_df.copy(), columns=['Airline', 'Origin', 'Destination'], drop_first=True)

# Align with training columns
for col in X_train.columns:
    if col not in future_encoded.columns:
        future_encoded[col] = 0
future_encoded = future_encoded[X_train.columns]

# Predict
future_df['PredictedPriceUSD'] = model.predict(future_encoded)

# Save results
future_df.to_csv("predicted_prices_2025_2026.csv", index=False)

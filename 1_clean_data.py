import pandas as pd

# Load CSV
df = pd.read_csv("student_flight_fares_10yrs_full.csv")

# Drop rows with any blank or missing values
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert TravelDate to datetime
df['TravelDate'] = pd.to_datetime(df['TravelDate'], errors='coerce')

# Drop rows where TravelDate couldn't be parsed
df = df.dropna(subset=['TravelDate'])

# Extract Year, Month, Day
df['Year'] = df['TravelDate'].dt.year
df['Month'] = df['TravelDate'].dt.month
df['Day'] = df['TravelDate'].dt.day

# Add IsSummer and IsFall
df['IsSummer'] = df['Month'].isin([5, 6])
df['IsFall'] = df['Month'].isin([8, 9])

# Save to new CSV
df.to_csv("completely_sorted_final.csv", index=False)

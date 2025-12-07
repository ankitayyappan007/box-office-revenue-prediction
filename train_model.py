import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_csv("tmdb_5000_movies.csv")

# Select useful features and target
features = ["budget", "popularity", "runtime", "vote_average"]
target = "revenue"

# Drop rows with missing values
data = data.dropna(subset=features + [target])

#  Scale budget and revenue to millions
data["budget"] = data["budget"] / 1e6
data["revenue"] = data["revenue"] / 1e6

#  Log-transform revenue to stabilize extreme values
data["log_revenue"] = np.log1p(data["revenue"])

X = data[features]
y = data["log_revenue"]

# Scale feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X_scaled, y, epochs=50, batch_size=32, verbose=1)

# Save model and scaler
model.save("movie_model.keras")
joblib.dump(scaler, "scaler.pkl")

print(" Model trained and saved successfully (log-scaled revenue in millions).")

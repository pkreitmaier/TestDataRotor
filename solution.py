#######################################################
# set up session
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# Set the absolute path to the input dataset before running the script.
# Example:
# DATA_FILE = Path(r"C:\Users\YourName\Data\TestDataRotor.csv")
# --------------------------------------------------------------------
#######################################################
# load data
def load_data(filepath):
    """Load dataset."""
    return pd.read_csv(filepath)

df = load_data(DATA_FILE)
#######################################################
# task 1: Label all data points where velocity ≥ 120 km/h.
df["highSpeed"] = df["velocity"].ge(120).astype("boolean")
df.loc[df["velocity"].isna(), "highSpeed"] = pd.NA
#######################################################
# task 2: Plot an example of a consecutive time series of one vehicle where velocity ≥ 120 km/h at least once. 
# Plot the values of velocity and temperatureRotorBack in the same graph.

exampleTripId = (
    df.groupby("tripId")["velocity"]
      .max()
      .loc[lambda x: x >= 120]
      .index[0] # just use the first tripId in this lsit
)

exampleTrip = (
    df[df["tripId"] == exampleTripId]
      .sort_values("timeUnix")
)

# Plot
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(exampleTrip["timeUnix"], exampleTrip["velocity"],
         color="blue", label="Velocity")
ax1.axhline(120, color="black", linestyle=":", label="120 km/h")
ax1.set_xlabel("Time")
ax1.set_ylabel("Velocity [km/h]", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(exampleTrip["timeUnix"], exampleTrip["temperatureRotorBack"],
         color="darkorange", label="Rear rotor temperature")
ax2.set_ylabel("Rear rotor temperature [°C]", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")

fig.legend(loc="upper left")
plt.title(f"Task2: tripId {exampleTripId}")
plt.tight_layout()
plt.savefig("plots/task2_trip.png", dpi=300)
plt.show()
#######################################################
# task 3: Calculate the average temperatureRotorBack within the two labeled groups 
# (velocity ≥ 120 km/h and velocity < 120 km/h) and compare the values.

print("##############################")
print("Task 3: Average rear rotor temperature")

mean_temp = (
    df.dropna(subset=["velocity", "temperatureRotorBack"])
      .groupby(df["velocity"] >= 120)["temperatureRotorBack"]
      .mean()
)
print("Average rear rotor temperature:")
print(f"Velocity < 120 km/h:  {mean_temp[False]:.2f} C")
print(f"Velocity >= 120 km/h: {mean_temp[True]:.2f} C")
print(f"Mean rear rotor temperature  {mean_temp[True] - mean_temp[False]:.2f} higher at velocity >= 120")

# Average rear rotor temperature:
# Velocity < 120 km/h:  68.79 C
# Velocity >= 120 km/h: 92.81 C
# Mean rear rotor temperature  24.02 higher at velocity >= 120
#######################################################
# task 4: Calculate the correlation between velocity and temperatureRotorBack as well as between velocity and temperatureRotorFront.
print("##############################")
print("Task 4: Correlation analysis")

corr_back = df["velocity"].corr(df["temperatureRotorBack"], method="pearson")
corr_front = df["velocity"].corr(df["temperatureRotorFront"])

print(f"Correlation between velocity and rear rotor temperature:  {corr_back:.3f}")
# Pearson correlation between velocity and rear rotor temperature:  0.684
print(f"Correlation between velocity and front rotor temperature: {corr_front:.3f}")
# Pearson correlation between velocity and front rotor temperature: 0.20
#######################################################
# task 5: Plot histograms of temperatureRotorBack for velocity ≥ 120 km/h and velocity < 120 km/h. 
# Use a bin size of 5 for the histograms.

bins = np.arange(25, 110, 5)

high_speed = df.loc[df["velocity"] >= 120, "temperatureRotorBack"].dropna()
low_speed = df.loc[df["velocity"] < 120, "temperatureRotorBack"].dropna()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Top histogram
counts1, _, _ = ax1.hist(high_speed, bins=bins, color="darkorange")
ax1.set_title("Task5: Velocity ≥ 120 km/h")
ax1.set_ylabel("Count")

# Bottom histogram
counts2, _, _ = ax2.hist(low_speed, bins=bins, color="blue")
ax2.set_title("Task5: Velocity < 120 km/h")
ax2.set_xlabel("Rear rotor temperature [°C]")
ax2.set_ylabel("Count")

# Same y-axis scale
max_count = max(counts1.max(), counts2.max())
ax1.set_ylim(0, max_count)
ax2.set_ylim(0, max_count)

plt.tight_layout()
plt.savefig("plots/task5_rotor_temperature.png", dpi=300)
plt.show()
#######################################################
# Task 6a: Trip with the longest duration
print("##############################")
print("Task 6a: Longest trip")

tripDuration = (
    df.groupby("tripId")["timeUnix"]
      .agg(lambda x: x.max() - x.min())
)

longestTripId = tripDuration.idxmax()
longestDuration = tripDuration.max()

print(f"Longest trip: {longestTripId}")
print(f"Duration: {longestDuration} seconds")
# Longest trip: 8960253-16015987-0
# Duration: 1080000 seconds
#######################################################
# Task 6b: Trip with the highest average velocity
print("##############################")
print("Task 6b: Highest average velocity")

meanVelocity = df.groupby("tripId")["velocity"].mean()

fastestTripId = meanVelocity.idxmax()
highestMeanVelocity = meanVelocity.max()

print(f"Trip with highest average velocity: {fastestTripId}")
print(f"Average velocity: {highestMeanVelocity:.2f} km/h")
# Trip with highest average velocity: 8960253-16015987-0
# Average velocity: 119.52 km/h
#######################################################
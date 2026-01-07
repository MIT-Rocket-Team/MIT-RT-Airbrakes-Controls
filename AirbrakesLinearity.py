# Check Linearity of Airbrake Deployment
# /Users/sydneyparke/Documents/RT-MPC/zephtestdata.csv

import csv
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Parameters
# -----------------------------
min_area = 0.0
max_area = 0.00165
width = 0.066
mass = 37.65
CdA_r = 0.00958
Cdperp = 1.28
Cdedge = 0.01
num_airbrake_points = 10  # number of deployments to scan
burnout_time = 4.0        # seconds
trim_points = 100          # points to trim at start/end

# -----------------------------
# Load CSV
# -----------------------------
time_data = []
alt_data = []
vz_data = []
aoa_data = []

with open('zephtestdata.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        t, alt, vz, aoa = map(float, row[:4])  # ignore extra columns
        time_data.append(t)
        alt_data.append(alt)
        vz_data.append(vz)
        aoa_data.append(aoa)

# -----------------------------
# Filter post-burnout and pre-apogee
# -----------------------------
time_data = np.array(time_data)
alt_data = np.array(alt_data)
vz_data = np.array(vz_data)
aoa_data = np.array(aoa_data)

# post-burnout
mask = time_data >= burnout_time
time_data = time_data[mask]
alt_data = alt_data[mask]
vz_data = vz_data[mask]
aoa_data = aoa_data[mask]

# until apogee
apogee_index = np.argmax(alt_data)
time_data = time_data[:apogee_index+1]
alt_data = alt_data[:apogee_index+1]
vz_data = vz_data[:apogee_index+1]
aoa_data = aoa_data[:apogee_index+1]

# trim start/end points to remove outliers
if len(time_data) > 2*trim_points:
    time_data = time_data[trim_points:-trim_points]
    alt_data = alt_data[trim_points:-trim_points]
    vz_data = vz_data[trim_points:-trim_points]
    aoa_data = aoa_data[trim_points:-trim_points]

# -----------------------------
# Airbrake CdA calculation function
# -----------------------------
def airbrake_CdA(fraction, aoa_deg):
    aoa_rad = np.radians(aoa_deg)
    Cd_airbrake = Cdperp * np.cos(aoa_rad) + Cdedge * np.sin(aoa_rad)
    area = min_area + fraction * (max_area - min_area)
    return Cd_airbrake * area

# -----------------------------
# Predict apogee function
# -----------------------------
def predict_apogee(vz, alt, aoa_deg, mass, CdA_r, airbrake_fraction):
    g = 9.80665
    CdA_total = CdA_r + airbrake_CdA(airbrake_fraction, aoa_deg)
    vz_curr = vz
    alt_curr = alt
    dt = 0.02
    while vz_curr > 0:
        rho = 1.225  # assume sea level density for simplicity
        drag = 0.5 * rho * CdA_total * vz_curr**2 / mass
        vz_curr -= (drag + g) * dt
        alt_curr += vz_curr * dt
    return alt_curr

# -----------------------------
# Airbrake scan
# -----------------------------
deployment_fractions = np.linspace(0, 1, num_airbrake_points)
baseline_apogees = []
predicted_apogees_all = []

for vz, alt, aoa in zip(vz_data, alt_data, aoa_data):
    apogees = [predict_apogee(vz, alt, aoa, mass, CdA_r, f) for f in deployment_fractions]
    predicted_apogees_all.append(apogees)
    baseline_apogees.append(apogees[0])

predicted_apogees_all = np.array(predicted_apogees_all)
baseline_apogees = np.array(baseline_apogees)

# -----------------------------
# Compute delta apogee for R²
# -----------------------------
delta_apogees_all = predicted_apogees_all - baseline_apogees[:, np.newaxis]

linear_r2 = []
quad_r2 = []

for deltas in delta_apogees_all:
    # Linear fit
    lin_fit = np.polyfit(deployment_fractions, deltas, 1)
    lin_pred = np.polyval(lin_fit, deployment_fractions)
    r2_lin = 1 - np.sum((deltas - lin_pred)**2) / np.sum((deltas - np.mean(deltas))**2)
    linear_r2.append(r2_lin)
    # Quadratic fit
    quad_fit = np.polyfit(deployment_fractions, deltas, 2)
    quad_pred = np.polyval(quad_fit, deployment_fractions)
    r2_quad = 1 - np.sum((deltas - quad_pred)**2) / np.sum((deltas - np.mean(deltas))**2)
    quad_r2.append(r2_quad)

linear_r2 = np.array(linear_r2)
quad_r2 = np.array(quad_r2)

# -----------------------------
# Heatmaps: velocity vs altitude with color = R²
# -----------------------------
plt.figure(figsize=(10, 6))
plt.scatter(vz_data, alt_data, c=linear_r2, cmap='viridis', vmin=0, vmax=1)
plt.colorbar(label='Linear fit R²')
plt.xlabel('Vertical velocity (m/s)')
plt.ylabel('Altitude (m)')
plt.title('Heatmap of linear R² for all flight points')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(vz_data, alt_data, c=quad_r2, cmap='plasma', vmin=0, vmax=1)
plt.colorbar(label='Quadratic fit R²')
plt.xlabel('Vertical velocity (m/s)')
plt.ylabel('Altitude (m)')
plt.title('Heatmap of quadratic R² for all flight points')
plt.grid(True)
plt.show()

# -----------------------------
# Representative points: 5 equally spaced in dataset
# -----------------------------
indices = np.linspace(0, len(vz_data)-1, 5, dtype=int)
for idx, i in enumerate(indices):
    plt.figure()
    plt.plot(deployment_fractions, delta_apogees_all[i, :], 'o-', label=f'Flight point {i}')
    plt.xlabel('Airbrake deployment fraction')
    plt.ylabel('Δ Apogee (m)')
    plt.title(f'Predicted apogee change vs deployment (flight point {i})')
    # Linear & quadratic fit lines
    lin_fit = np.polyfit(deployment_fractions, delta_apogees_all[i, :], 1)
    quad_fit = np.polyfit(deployment_fractions, delta_apogees_all[i, :], 2)
    plt.plot(deployment_fractions, np.polyval(lin_fit, deployment_fractions), '--', label='Linear fit')
    plt.plot(deployment_fractions, np.polyval(quad_fit, deployment_fractions), ':', label='Quadratic fit')
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------------
# Plot R² vs time, velocity, altitude
# -----------------------------
plt.figure(figsize=(12, 8))
plt.subplot(3,1,1)
plt.scatter(time_data, linear_r2, c='blue', s=10)
plt.ylabel('Linear R²')
plt.title('Linear R² over flight conditions')
plt.grid(True)

plt.subplot(3,1,2)
plt.scatter(vz_data, linear_r2, c='green', s=10)
plt.ylabel('Linear R²')
plt.title('Linear R² vs vertical velocity')
plt.grid(True)

plt.subplot(3,1,3)
plt.scatter(alt_data, linear_r2, c='red', s=10)
plt.ylabel('Linear R²')
plt.xlabel('Altitude (m)')
plt.title('Linear R² vs altitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Summary statistics
# -----------------------------
print(f"Linear R²: min {linear_r2.min():.3f}, max {linear_r2.max():.3f}, mean {linear_r2.mean():.3f}")
print(f"Quadratic R²: min {quad_r2.min():.3f}, max {quad_r2.max():.3f}, mean {quad_r2.mean():.3f}")

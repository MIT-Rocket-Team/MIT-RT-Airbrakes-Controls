import numpy as np
import matplotlib.pyplot as plt

# Rocket and environment parameters
m = 5.8967  # kg
g = 9.81  # m/s^2
rho = 1.225  # kg/m^3
v_max = 70  # m/s
CdA_rocket = 0.00453  # m^2

# Airbrake parameters
Cd_airbrake = 1.28
airbrake_area = np.linspace(0.002, 0.01, 50)  # m^2

# Apogee calculation
apogee_reduction = []
for A in airbrake_area:
    CdA_total = CdA_rocket + Cd_airbrake * A
    F_D = 0.5 * rho * CdA_total * v_max**2
    delta_h = F_D / (m * g) * (v_max**2 / (2 * g))  # simplified
    h_apogee = (v_max**2) / (2 * g) - delta_h
    apogee_reduction.append(h_apogee)

# Plot
plt.figure(figsize=(8,5))
plt.plot(airbrake_area*1e4, apogee_reduction, marker='o')
plt.xlabel("Airbrake Area (cm²)")
plt.ylabel("Estimated Apogee (m)")
plt.title("Effect of Airbrake Area on Rocket Apogee")
plt.grid(True)
plt.show()

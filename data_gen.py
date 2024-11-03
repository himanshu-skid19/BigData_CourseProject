import numpy as np
import pandas as pd
from scipy.stats import gamma, poisson, beta, uniform, truncnorm
import matplotlib.pyplot as plt
import seaborn as sns


def generate_data(start = '2024-01-01', end = '2024-12-31', n_samples= 1000):

    np.random.seed(42)

    # 1. Timestamp - Random dates in 2024
    timestamps = pd.date_range(start=start, end=end, freq='D')
    timestamps = np.random.choice(timestamps, n_samples)

    driver_ids = np.arange(1, n_samples + 1)

    driver_ages = np.clip(np.random.normal(40, 10, n_samples), 18, 70)

    genders = np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.56, 0.40, 0.04])

    credit_scores = np.clip(np.random.normal(680, 50, n_samples), 300, 850)

    scale_factor = 5  # Adjust this to control how quickly the distribution decays
    vehicle_ages = np.random.exponential(scale=scale_factor, size=n_samples)
    vehicle_ages = np.clip(vehicle_ages, 0, 25)  # Clip at 25 to match the plot's range

    vehicle_types = np.random.choice(['Sedan', 'SUV', 'Truck', 'Motorcycle'], n_samples, p=[0.4, 0.35, 0.15, 0.1])

    vehicle_use = np.random.choice(['Personal', 'Commercial'], n_samples, p=[0.85, 0.15])
    insurance_duration = np.random.uniform(1, 10, n_samples)

    years_no_claim = np.random.uniform(0, 10, n_samples)
    miles_driven = np.clip(np.random.normal(15000, 5000, n_samples), 0, 30000)

    avg_speed = np.random.uniform(0, 80, n_samples)

    max_speed = np.random.uniform(0, 120, n_samples)

    acceleration_events = poisson.rvs(100, size=n_samples)
    acceleration_events = np.clip(acceleration_events, 0, 500)

    braking_events = poisson.rvs(100, size=n_samples)
    braking_events = np.clip(braking_events, 0, 500)

    night_driving = beta.rvs(2, 5, size=n_samples) * 100

    weekend_driving = beta.rvs(2, 5, size=n_samples) * 100

    city_driving = np.random.uniform(0, 100, n_samples)

    highway_driving = 100 - city_driving

    adverse_weather = beta.rvs(2, 5, size=n_samples) * 100

    gps_speed = np.clip(np.random.normal(avg_speed, 5, n_samples), 0, 120)

    num_claims = np.clip(poisson.rvs(0.3, size=n_samples), 0, 3)

    claim_amounts = gamma.rvs(2, scale=2000, size=n_samples) * (num_claims > 0)
    claim_amounts = np.clip(claim_amounts, 0, 100000)

    claim_severity = np.where(claim_amounts < 20000, 'Minor', 
                            np.where(claim_amounts < 50000, 'Major', 'Total Loss'))

    temperatures = np.clip(np.random.normal(20, 10, n_samples), -10, 40)

    weather_conditions = np.random.choice(['Sunny', 'Rainy', 'Snowy', 'Foggy'], n_samples, p=[0.6, 0.2, 0.1, 0.1])

    fuel_efficiency = np.clip(np.random.normal(25, 5, n_samples), 10, 50)

    rpm = np.random.uniform(0, 6000, n_samples)

    engine_load = np.random.uniform(0, 100, n_samples)

    telematics_data_pd = pd.DataFrame({
        'Timestamp': timestamps,
        'DriverID': driver_ids,
        'Driver Age': driver_ages,
        'Gender': genders,
        'Credit Score': credit_scores,
        'Vehicle Age': vehicle_ages,
        'Vehicle Type': vehicle_types,
        'Vehicle Use': vehicle_use,
        'Insurance Duration': insurance_duration,
        'Years with No Claim': years_no_claim,
        'Total Miles Driven': miles_driven,
        'Average Speed': avg_speed,
        'Maximum Speed': max_speed,
        'Acceleration Events': acceleration_events,
        'Braking Events': braking_events,
        'Driving during Night (%)': night_driving,
        'Driving during Weekends (%)': weekend_driving,
        'City Driving (%)': city_driving,
        'Highway Driving (%)': highway_driving,
        'Weather Conditions (%)': adverse_weather,
        'GPS Speed': gps_speed,
        'Number of Claims': num_claims,
        'Claim Amount': claim_amounts,
        'Claim Severity': claim_severity,
        'Temperature (°C)': temperatures,
        'Weather': weather_conditions,
        'Fuel Efficiency (mpg)': fuel_efficiency,
        'RPM': rpm,
        'Engine Load (%)': engine_load
    })

    return telematics_data_pd


train_data = generate_data('2023-01-01', '2023-12-31', 10000)
train_data.to_csv("train.csv")

test_data = generate_data('2024-01-01', '2024-12-31', 1000)
test_data.to_csv('test.csv')

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import requests
# import math
# import joblib
# import os
# from datetime import datetime
# from sklearn.linear_model import LinearRegression

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# EMISSION_FACTOR = 3.114
# SPEED_KNOTS = 20
# MODEL_PATH = "fuel_model.pkl"

# # -------------------------
# # Request Models
# # -------------------------

# class Port(BaseModel):
#     lat: float
#     lon: float

# class Voyage(BaseModel):
#     start_port: Port
#     end_port: Port

# class OptimizationRequest(BaseModel):
#     voyage: Voyage

# # -------------------------
# # Haversine Distance
# # -------------------------

# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371
#     phi1 = math.radians(lat1)
#     phi2 = math.radians(lat2)
#     dphi = math.radians(lat2 - lat1)
#     dlambda = math.radians(lon2 - lon1)

#     a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

#     distance_km = R * c
#     return distance_km * 0.539957

# # -------------------------
# # Live Weather Fetch
# # -------------------------

# def fetch_live_wind(lat, lon):
#     try:
#         url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
#         response = requests.get(url, timeout=5)
#         data = response.json()
#         return data["current_weather"]["windspeed"]
#     except:
#         return 12

# def fetch_wave_height():
#     return float(np.random.uniform(1, 4))

# def fetch_ocean_current():
#     return float(np.random.uniform(0.5, 2.0))

# # -------------------------
# # ML Model
# # -------------------------

# def train_model():
#     np.random.seed(42)
#     samples = 600

#     distance = np.random.uniform(100, 5000, samples)
#     wind = np.random.uniform(0, 25, samples)
#     current = np.random.uniform(0, 3, samples)
#     wave = np.random.uniform(0, 6, samples)
#     time = distance / 20

#     fuel = (
#         0.04 * distance +
#         0.7 * wind -
#         0.5 * current +
#         0.8 * wave +
#         0.02 * time +
#         np.random.normal(0, 5, samples)
#     )

#     X = np.column_stack((distance, wind, current, wave, time))
#     y = fuel

#     model = LinearRegression()
#     model.fit(X, y)

#     joblib.dump(model, MODEL_PATH)

# def load_model():
#     if not os.path.exists(MODEL_PATH):
#         train_model()
#     return joblib.load(MODEL_PATH)

# def predict_fuel(distance_nm, wind, current, wave):
#     model = load_model()
#     voyage_hours = distance_nm / SPEED_KNOTS
#     X = np.array([[distance_nm, wind, current, wave, voyage_hours]])
#     fuel = float(model.predict(X)[0])
#     if fuel < 0:
#         fuel = abs(fuel)
#     co2 = fuel * EMISSION_FACTOR
#     return fuel, co2, voyage_hours

# # -------------------------
# # Optimization Endpoint
# # -------------------------

# @app.post("/optimize")
# def optimize_route(data: OptimizationRequest):

#     lat1 = data.voyage.start_port.lat
#     lon1 = data.voyage.start_port.lon
#     lat2 = data.voyage.end_port.lat
#     lon2 = data.voyage.end_port.lon

#     baseline_distance = haversine(lat1, lon1, lat2, lon2)

#     wind = fetch_live_wind(lat1, lon1)
#     current = fetch_ocean_current()
#     wave = fetch_wave_height()

#     safety_penalty = 0.05 if wave > 3 else 0

#     optimized_distance = baseline_distance * (
#         1 + safety_penalty + (wave * 0.02) - (current * 0.03)
#     )

#     fuel_base, co2_base, time_base = predict_fuel(baseline_distance, wind, current, wave)
#     fuel_opt, co2_opt, time_opt = predict_fuel(optimized_distance, wind, current, wave)

#     fuel_reduction = ((fuel_base - fuel_opt) / fuel_base) * 100
#     co2_reduction = co2_base - co2_opt
#     time_diff = abs(time_base - time_opt)
#     reroute_nm = abs(baseline_distance - optimized_distance)

#     return {
#         "Optimized Route Generated":
#             f"Fuel consumption reduced by {round(fuel_reduction,2)}% compared to shortest-path navigation.",
#         "Weather Impact Adjustment":
#             f"Strong headwinds avoided by rerouting {round(reroute_nm,1)} nautical miles south.",
#         "Emissions Report":
#             f"Estimated CO2 reduction — {round(co2_reduction,2)} tons for this voyage.",
#         "Operational Insight":
#             f"Arrival time maintained within ±{round(time_diff,2)} hours while improving fuel efficiency.",
#         "Data Timestamp": str(datetime.utcnow())
#     }
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import requests
# import math
# import joblib
# import os
# from datetime import datetime
# from sklearn.linear_model import LinearRegression

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# EMISSION_FACTOR = 3.114
# SPEED_KNOTS = 20
# MODEL_PATH = "fuel_model.pkl"

# class Port(BaseModel):
#     lat: float
#     lon: float

# class Voyage(BaseModel):
#     start_port: Port
#     end_port: Port

# class OptimizationRequest(BaseModel):
#     voyage: Voyage

# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371
#     phi1 = math.radians(lat1)
#     phi2 = math.radians(lat2)
#     dphi = math.radians(lat2 - lat1)
#     dlambda = math.radians(lon2 - lon1)
#     a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     distance_km = R * c
#     return distance_km * 0.539957

# def fetch_live_wind(lat, lon):
#     try:
#         url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
#         response = requests.get(url, timeout=5)
#         data = response.json()
#         return data["current_weather"]["windspeed"]
#     except:
#         return 12

# # def fetch_wave_height():
# #     return float(np.random.uniform(1, 4))

# # def fetch_ocean_current():
# #     return float(np.random.uniform(0.5, 2.0))
# def fetch_wave_height(lat, lon):
#     return round(abs(math.sin(lat)) * 2 + 1.5, 2)

# def fetch_ocean_current(lat, lon):
#     return round(abs(math.cos(lon)) * 1.5 + 0.5, 2)

# def train_model():
#     np.random.seed(42)
#     samples = 600
#     distance = np.random.uniform(100, 5000, samples)
#     wind = np.random.uniform(0, 25, samples)
#     current = np.random.uniform(0, 3, samples)
#     wave = np.random.uniform(0, 6, samples)
#     time = distance / 20

#     fuel = (
#         0.04 * distance +
#         0.7 * wind -
#         0.5 * current +
#         0.8 * wave +
#         0.02 * time +
#         np.random.normal(0, 5, samples)
#     )

#     X = np.column_stack((distance, wind, current, wave, time))
#     y = fuel

#     model = LinearRegression()
#     model.fit(X, y)
#     joblib.dump(model, MODEL_PATH)

# def load_model():
#     if not os.path.exists(MODEL_PATH):
#         train_model()
#     return joblib.load(MODEL_PATH)

# def predict_fuel(distance_nm, wind, current, wave):
#     model = load_model()
#     voyage_hours = distance_nm / SPEED_KNOTS
#     X = np.array([[distance_nm, wind, current, wave, voyage_hours]])
#     fuel = float(model.predict(X)[0])
#     fuel = abs(fuel)
#     co2 = fuel * EMISSION_FACTOR
#     return fuel, co2, voyage_hours

# @app.post("/optimize")
# def optimize_route(data: OptimizationRequest):

#     lat1 = data.voyage.start_port.lat
#     lon1 = data.voyage.start_port.lon
#     lat2 = data.voyage.end_port.lat
#     lon2 = data.voyage.end_port.lon

#     baseline_distance = haversine(lat1, lon1, lat2, lon2)

#     wind = fetch_live_wind(lat1, lon1)
#     current = fetch_ocean_current(lat1, lon1)
#     wave = fetch_wave_height(lat1, lon1)

#     safety_penalty = 0.05 if wave > 3 else 0

#     optimized_distance = baseline_distance * (
#         1 + safety_penalty + (wave * 0.02) - (current * 0.03)
#     )

#     fuel_base, co2_base, time_base = predict_fuel(baseline_distance, wind, current, wave)
#     fuel_opt, co2_opt, time_opt = predict_fuel(optimized_distance, wind, current, wave)

#     fuel_difference = fuel_base - fuel_opt
#     co2_difference = co2_base - co2_opt

#     fuel_reduction_percent = (fuel_difference / fuel_base) * 100

#     return {
#         "baseline_distance_nm": round(baseline_distance, 2),
#         "optimized_distance_nm": round(optimized_distance, 2),
#         "rerouted_distance_nm": round(abs(baseline_distance - optimized_distance), 2),
#         "fuel_reduction_percent": round(fuel_reduction_percent, 2),
#         "co2_reduction_tons": round(co2_difference, 2),
#         "time_difference_hours": round(abs(time_base - time_opt), 2),
#         "timestamp": str(datetime.utcnow())
#     }
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import requests
# import math
# import joblib
# import os
# from datetime import datetime
# from sklearn.linear_model import LinearRegression

# app = FastAPI(title="AI Marine Route Optimizer - Real Time")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------
# # Constants
# # -------------------------

# EMISSION_FACTOR = 3.114
# SPEED_KNOTS = 20
# MODEL_PATH = "fuel_model.pkl"

# # -------------------------
# # Request Models
# # -------------------------

# class Port(BaseModel):
#     lat: float
#     lon: float

# class Voyage(BaseModel):
#     start_port: Port
#     end_port: Port

# class OptimizationRequest(BaseModel):
#     voyage: Voyage

# # -------------------------
# # Distance Calculation
# # -------------------------

# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371
#     phi1 = math.radians(lat1)
#     phi2 = math.radians(lat2)
#     dphi = math.radians(lat2 - lat1)
#     dlambda = math.radians(lon2 - lon1)

#     a = math.sin(dphi / 2)**2 + \
#         math.cos(phi1) * math.cos(phi2) * \
#         math.sin(dlambda / 2)**2

#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     return R * c * 0.539957  # nautical miles

# # -------------------------
# # Route Sampling
# # -------------------------

# def interpolate_points(lat1, lon1, lat2, lon2, steps=6):
#     lats = np.linspace(lat1, lat2, steps)
#     lons = np.linspace(lon1, lon2, steps)
#     return list(zip(lats, lons))

# # -------------------------
# # Safe Float
# # -------------------------

# def safe_float(value, default=0.0):
#     try:
#         if value is None:
#             return default
#         return float(value)
#     except:
#         return default

# # -------------------------
# # Real Marine Weather (Robust)
# # -------------------------

# def fetch_marine_weather(lat, lon):
#     try:
#         # Wind API
#         weather_url = (
#             f"https://api.open-meteo.com/v1/forecast"
#             f"?latitude={lat}&longitude={lon}&current_weather=true"
#         )

#         weather_res = requests.get(weather_url, timeout=8)
#         weather_res.raise_for_status()
#         weather_data = weather_res.json()

#         wind = weather_data.get("current_weather", {}).get("windspeed", 0)

#         # Marine API
#         marine_url = (
#             f"https://marine-api.open-meteo.com/v1/marine"
#             f"?latitude={lat}&longitude={lon}"
#             f"&hourly=wave_height,ocean_current_velocity"
#         )

#         marine_res = requests.get(marine_url, timeout=8)
#         marine_res.raise_for_status()
#         marine_data = marine_res.json()

#         hourly = marine_data.get("hourly", {})

#         wave = safe_float(hourly.get("wave_height", [0])[0])
#         current = safe_float(hourly.get("ocean_current_velocity", [0])[0])
#         wind = safe_float(wind)

#         return wind, current, wave

#     except requests.exceptions.Timeout:
#         return 5.0, 0.2, 0.5  # fallback safe values

#     except Exception:
#         return 5.0, 0.2, 0.5  # fallback safe values

# # -------------------------
# # ML Model Training
# # -------------------------

# def train_model():
#     np.random.seed(42)
#     samples = 800

#     distance = np.random.uniform(100, 6000, samples)
#     wind = np.random.uniform(0, 30, samples)
#     current = np.random.uniform(0, 3, samples)
#     wave = np.random.uniform(0, 6, samples)
#     time = distance / SPEED_KNOTS

#     fuel = (
#         0.045 * distance +
#         0.9 * wind +
#         1.2 * wave -
#         0.6 * current +
#         0.03 * time +
#         np.random.normal(0, 8, samples)
#     )

#     X = np.column_stack((distance, wind, current, wave, time))
#     y = fuel

#     model = LinearRegression()
#     model.fit(X, y)
#     joblib.dump(model, MODEL_PATH)

# if not os.path.exists(MODEL_PATH):
#     train_model()

# model = joblib.load(MODEL_PATH)

# def predict_fuel(distance_nm, wind, current, wave):
#     voyage_hours = distance_nm / SPEED_KNOTS
#     X = np.array([[distance_nm, wind, current, wave, voyage_hours]])
#     fuel = abs(float(model.predict(X)[0]))
#     co2 = fuel * EMISSION_FACTOR
#     return fuel, co2, voyage_hours

# # -------------------------
# # Optimization Endpoint
# # -------------------------

# @app.post("/optimize")
# async def optimize_route(data: OptimizationRequest):

#     lat1 = data.voyage.start_port.lat
#     lon1 = data.voyage.start_port.lon
#     lat2 = data.voyage.end_port.lat
#     lon2 = data.voyage.end_port.lon

#     baseline_distance = haversine(lat1, lon1, lat2, lon2)

#     route_points = interpolate_points(lat1, lon1, lat2, lon2)

#     winds, currents, waves = [], [], []

#     for lat, lon in route_points:
#         wind, current, wave = fetch_marine_weather(lat, lon)
#         winds.append(wind)
#         currents.append(current)
#         waves.append(wave)

#     avg_wind = float(np.mean(winds))
#     avg_current = float(np.mean(currents))
#     avg_wave = float(np.mean(waves))

#     # SMART rerouting logic
#     if avg_wave > 2.5 or avg_wind > 20:
#         weather_factor = (
#             1 +
#             (avg_wave * 0.04) +
#             (avg_wind * 0.005) -
#             (avg_current * 0.03)
#         )
#     else:
#         weather_factor = 1  # calm sea → no reroute

#     optimized_distance = baseline_distance * weather_factor

#     fuel_base, co2_base, time_base = predict_fuel(
#         baseline_distance, avg_wind, avg_current, avg_wave
#     )

#     fuel_opt, co2_opt, time_opt = predict_fuel(
#         optimized_distance, avg_wind, avg_current, avg_wave
#     )

#     fuel_reduction = (
#         ((fuel_base - fuel_opt) / fuel_base) * 100
#         if fuel_base != 0 else 0
#     )
#     # Prevent worse deployment
#     # if fuel_opt > fuel_base:
#     #     optimized_distance = baseline_distance
#     #     fuel_opt = fuel_base
#     #     co2_opt = co2_base
#     #     time_opt = time_base
#     #     fuel_reduction = 0

#     return {
#         "Baseline Distance (NM)": round(baseline_distance, 2),
#         "Optimized Distance (NM)": round(optimized_distance, 2),
#         "Fuel Reduction (%)": round(fuel_reduction, 2),
#         "CO2 Reduction (tons)": round(co2_base - co2_opt, 2),
#         "Time Difference (hours)": round(abs(time_base - time_opt), 2),
#         "Route Adjustment (NM)": round(abs(optimized_distance - baseline_distance), 2),
#         "Live Weather Summary": {
#             "Average Wind (km/h)": round(avg_wind, 2),
#             "Average Wave Height (m)": round(avg_wave, 2),
#             "Average Current (m/s)": round(avg_current, 2)
#         },
#         "Weather Condition":
#             "Rerouted (Rough Sea)" if weather_factor != 1 else "Direct Route (Calm Sea)",
#         "Timestamp (UTC)": datetime.utcnow().isoformat()
#     }
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import requests
# import math
# import joblib
# import os
# from datetime import datetime
# from sklearn.linear_model import LinearRegression

# app = FastAPI(title="AI Marine Route Optimizer - Real Time")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------
# # Constants
# # -------------------------

# EMISSION_FACTOR = 3.114
# SPEED_KNOTS = 20
# MODEL_PATH = "fuel_model.pkl"

# # -------------------------
# # Request Models
# # -------------------------

# class Port(BaseModel):
#     lat: float
#     lon: float

# class Voyage(BaseModel):
#     start_port: Port
#     end_port: Port

# class OptimizationRequest(BaseModel):
#     voyage: Voyage

# # -------------------------
# # Distance Calculation
# # -------------------------

# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371
#     phi1 = math.radians(lat1)
#     phi2 = math.radians(lat2)
#     dphi = math.radians(lat2 - lat1)
#     dlambda = math.radians(lon2 - lon1)

#     a = math.sin(dphi / 2)**2 + \
#         math.cos(phi1) * math.cos(phi2) * \
#         math.sin(dlambda / 2)**2

#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     return R * c * 0.539957  # nautical miles

# # -------------------------
# # Route Sampling
# # -------------------------

# def interpolate_points(lat1, lon1, lat2, lon2, steps=6):
#     lats = np.linspace(lat1, lat2, steps)
#     lons = np.linspace(lon1, lon2, steps)
#     return list(zip(lats, lons))

# # -------------------------
# # Safe Float
# # -------------------------

# def safe_float(value, default=0.0):
#     try:
#         if value is None:
#             return default
#         return float(value)
#     except:
#         return default

# # -------------------------
# # Real Marine Weather (Robust)
# # -------------------------

# def fetch_marine_weather(lat, lon):
#     try:
#         # Wind API
#         weather_url = (
#             f"https://api.open-meteo.com/v1/forecast"
#             f"?latitude={lat}&longitude={lon}&current_weather=true"
#         )

#         weather_res = requests.get(weather_url, timeout=8)
#         weather_res.raise_for_status()
#         weather_data = weather_res.json()

#         wind = weather_data.get("current_weather", {}).get("windspeed", 0)

#         # Marine API
#         marine_url = (
#             f"https://marine-api.open-meteo.com/v1/marine"
#             f"?latitude={lat}&longitude={lon}"
#             f"&hourly=wave_height,ocean_current_velocity"
#         )

#         marine_res = requests.get(marine_url, timeout=8)
#         marine_res.raise_for_status()
#         marine_data = marine_res.json()

#         hourly = marine_data.get("hourly", {})

#         wave = safe_float(hourly.get("wave_height", [0])[0])
#         current = safe_float(hourly.get("ocean_current_velocity", [0])[0])
#         wind = safe_float(wind)

#         return wind, current, wave

#     except requests.exceptions.Timeout:
#         return 5.0, 0.2, 0.5  # fallback safe values

#     except Exception:
#         return 5.0, 0.2, 0.5  # fallback safe values

# # -------------------------
# # ML Model Training
# # -------------------------

# def train_model():
#     np.random.seed(42)
#     samples = 800

#     distance = np.random.uniform(100, 6000, samples)
#     wind = np.random.uniform(0, 30, samples)
#     current = np.random.uniform(0, 3, samples)
#     wave = np.random.uniform(0, 6, samples)
#     time = distance / SPEED_KNOTS

#     fuel = (
#         0.045 * distance +
#         0.9 * wind +
#         1.2 * wave -
#         0.6 * current +
#         0.03 * time +
#         np.random.normal(0, 8, samples)
#     )

#     X = np.column_stack((distance, wind, current, wave, time))
#     y = fuel

#     model = LinearRegression()
#     model.fit(X, y)
#     joblib.dump(model, MODEL_PATH)

# if not os.path.exists(MODEL_PATH):
#     train_model()

# model = joblib.load(MODEL_PATH)

# def predict_fuel(distance_nm, wind, current, wave):
#     voyage_hours = distance_nm / SPEED_KNOTS
#     X = np.array([[distance_nm, wind, current, wave, voyage_hours]])
#     fuel = abs(float(model.predict(X)[0]))
#     co2 = fuel * EMISSION_FACTOR
#     return fuel, co2, voyage_hours

# # -------------------------
# # Optimization Endpoint
# # -------------------------

# @app.post("/optimize")
# async def optimize_route(data: OptimizationRequest):

#     lat1 = data.voyage.start_port.lat
#     lon1 = data.voyage.start_port.lon
#     lat2 = data.voyage.end_port.lat
#     lon2 = data.voyage.end_port.lon

#     baseline_distance = haversine(lat1, lon1, lat2, lon2)

#     route_points = interpolate_points(lat1, lon1, lat2, lon2)

#     winds, currents, waves = [], [], []

#     for lat, lon in route_points:
#         wind, current, wave = fetch_marine_weather(lat, lon)
#         winds.append(wind)
#         currents.append(current)
#         waves.append(wave)

#     avg_wind = float(np.mean(winds))
#     avg_current = float(np.mean(currents))
#     avg_wave = float(np.mean(waves))

#     # SMART rerouting logic
#     if avg_wave > 2.5 or avg_wind > 20:
#         weather_factor = (
#             1 +
#             (avg_wave * 0.04) +
#             (avg_wind * 0.005) -
#             (avg_current * 0.03)
#         )
#     else:
#         weather_factor = 1  # calm sea → no reroute

#     optimized_distance = baseline_distance * weather_factor

#     fuel_base, co2_base, time_base = predict_fuel(
#         baseline_distance, avg_wind, avg_current, avg_wave
#     )

#     fuel_opt, co2_opt, time_opt = predict_fuel(
#         optimized_distance, avg_wind, avg_current, avg_wave
#     )

#     fuel_reduction = (
#         ((fuel_base - fuel_opt) / fuel_base) * 100
#         if fuel_base != 0 else 0
#     )
#     # Prevent worse deployment
#     # if fuel_opt > fuel_base:
#     #     optimized_distance = baseline_distance
#     #     fuel_opt = fuel_base
#     #     co2_opt = co2_base
#     #     time_opt = time_base
#     #     fuel_reduction = 0

#     return {
#         "Baseline Distance (NM)": round(baseline_distance, 2),
#         "Optimized Distance (NM)": round(optimized_distance, 2),
#         "Fuel Reduction (%)": round(fuel_reduction, 2),
#         "CO2 Reduction (tons)": round(co2_base - co2_opt, 2),
#         "Time Difference (hours)": round(abs(time_base - time_opt), 2),
#         "Route Adjustment (NM)": round(abs(optimized_distance - baseline_distance), 2),
#         "Live Weather Summary": {
#             "Average Wind (km/h)": round(avg_wind, 2),
#             "Average Wave Height (m)": round(avg_wave, 2),
#             "Average Current (m/s)": round(avg_current, 2)
#         },
#         "Weather Condition":
#             "Rerouted (Rough Sea)" if weather_factor != 1 else "Direct Route (Calm Sea)",
#         "Timestamp (UTC)": datetime.utcnow().isoformat()
#     }


# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import requests
# import math
# import joblib
# import os
# from datetime import datetime
# from sklearn.linear_model import LinearRegression

# app = FastAPI(title="AI Marine Route Optimizer - Real Time")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------
# # Constants
# # -------------------------

# EMISSION_FACTOR = 3.114
# SPEED_KNOTS = 20
# MODEL_PATH = "fuel_model.pkl"

# # -------------------------
# # Request Models
# # -------------------------

# class Port(BaseModel):
#     lat: float
#     lon: float

# class Voyage(BaseModel):
#     start_port: Port
#     end_port: Port

# class OptimizationRequest(BaseModel):
#     voyage: Voyage

# # -------------------------
# # Distance Calculation
# # -------------------------

# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371
#     phi1 = math.radians(lat1)
#     phi2 = math.radians(lat2)
#     dphi = math.radians(lat2 - lat1)
#     dlambda = math.radians(lon2 - lon1)

#     a = math.sin(dphi / 2)**2 + \
#         math.cos(phi1) * math.cos(phi2) * \
#         math.sin(dlambda / 2)**2

#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     return R * c * 0.539957  # nautical miles

# # -------------------------
# # Route Sampling
# # -------------------------

# def interpolate_points(lat1, lon1, lat2, lon2, steps=6):
#     lats = np.linspace(lat1, lat2, steps)
#     lons = np.linspace(lon1, lon2, steps)
#     return list(zip(lats, lons))

# # -------------------------
# # Safe Float
# # -------------------------

# def safe_float(value, default=0.0):
#     try:
#         if value is None:
#             return default
#         return float(value)
#     except:
#         return default

# # -------------------------
# # Real Marine Weather (Robust)
# # -------------------------

# def fetch_marine_weather(lat, lon):
#     try:
#         # Wind API
#         weather_url = (
#             f"https://api.open-meteo.com/v1/forecast"
#             f"?latitude={lat}&longitude={lon}&current_weather=true"
#         )

#         weather_res = requests.get(weather_url, timeout=8)
#         weather_res.raise_for_status()
#         weather_data = weather_res.json()

#         wind = weather_data.get("current_weather", {}).get("windspeed", 0)

#         # Marine API
#         marine_url = (
#             f"https://marine-api.open-meteo.com/v1/marine"
#             f"?latitude={lat}&longitude={lon}"
#             f"&hourly=wave_height,ocean_current_velocity"
#         )

#         marine_res = requests.get(marine_url, timeout=8)
#         marine_res.raise_for_status()
#         marine_data = marine_res.json()

#         hourly = marine_data.get("hourly", {})

#         wave = safe_float(hourly.get("wave_height", [0])[0])
#         current = safe_float(hourly.get("ocean_current_velocity", [0])[0])
#         wind = safe_float(wind)

#         return wind, current, wave

#     except requests.exceptions.Timeout:
#         return 5.0, 0.2, 0.5  # fallback safe values

#     except Exception:
#         return 5.0, 0.2, 0.5  # fallback safe values

# # -------------------------
# # ML Model Training
# # -------------------------

# def train_model():
#     np.random.seed(42)
#     samples = 800

#     distance = np.random.uniform(100, 6000, samples)
#     wind = np.random.uniform(0, 30, samples)
#     current = np.random.uniform(0, 3, samples)
#     wave = np.random.uniform(0, 6, samples)
#     time = distance / SPEED_KNOTS

#     fuel = (
#         0.045 * distance +
#         0.9 * wind +
#         1.2 * wave -
#         0.6 * current +
#         0.03 * time +
#         np.random.normal(0, 8, samples)
#     )

#     X = np.column_stack((distance, wind, current, wave, time))
#     y = fuel

#     model = LinearRegression()
#     model.fit(X, y)
#     joblib.dump(model, MODEL_PATH)

# if not os.path.exists(MODEL_PATH):
#     train_model()

# model = joblib.load(MODEL_PATH)

# def predict_fuel(distance_nm, wind, current, wave):
#     voyage_hours = distance_nm / SPEED_KNOTS
#     X = np.array([[distance_nm, wind, current, wave, voyage_hours]])
#     fuel = abs(float(model.predict(X)[0]))
#     co2 = fuel * EMISSION_FACTOR
#     return fuel, co2, voyage_hours

# # -------------------------
# # Optimization Endpoint
# # -------------------------

# @app.post("/optimize")
# async def optimize_route(data: OptimizationRequest):

#     lat1 = data.voyage.start_port.lat
#     lon1 = data.voyage.start_port.lon
#     lat2 = data.voyage.end_port.lat
#     lon2 = data.voyage.end_port.lon

#     baseline_distance = haversine(lat1, lon1, lat2, lon2)

#     route_points = interpolate_points(lat1, lon1, lat2, lon2)

#     winds, currents, waves = [], [], []

#     for lat, lon in route_points:
#         wind, current, wave = fetch_marine_weather(lat, lon)
#         winds.append(wind)
#         currents.append(current)
#         waves.append(wave)

#     avg_wind = float(np.mean(winds))
#     avg_current = float(np.mean(currents))
#     avg_wave = float(np.mean(waves))

#     # SMART rerouting logic
#     if avg_wave > 2.5 or avg_wind > 20:
#         weather_factor = (
#             1 +
#             (avg_wave * 0.04) +
#             (avg_wind * 0.005) -
#             (avg_current * 0.03)
#         )
#     else:
#         weather_factor = 1  # calm sea → no reroute

#     optimized_distance = baseline_distance * weather_factor

#     fuel_base, co2_base, time_base = predict_fuel(
#         baseline_distance, avg_wind, avg_current, avg_wave
#     )

#     fuel_opt, co2_opt, time_opt = predict_fuel(
#         optimized_distance, avg_wind, avg_current, avg_wave
#     )

#     fuel_reduction = (
#         ((fuel_base - fuel_opt) / fuel_base) * 100
#         if fuel_base != 0 else 0
#     )
#     # Prevent worse deployment
#     # if fuel_opt > fuel_base:
#     #     optimized_distance = baseline_distance
#     #     fuel_opt = fuel_base
#     #     co2_opt = co2_base
#     #     time_opt = time_base
#     #     fuel_reduction = 0
#     baseline_route = [
#         {"lat": lat1, "lon": lon1},
#         {"lat": lat2, "lon": lon2}
#     ]

#     optimized_route = [
#         {"lat": lat1, "lon": lon1},
#         {
#             "lat": lat2,
#             "lon": lon2 + (avg_wave * 0.3 if weather_factor != 1 else 0)
#         }
#     ]

#     return {
#     "baseline_distance_nm": round(baseline_distance, 2),
#     "optimized_distance_nm": round(optimized_distance, 2),
#     "fuel_reduction_percent": round(fuel_reduction, 2),
#     "co2_reduction_tons": round(co2_base - co2_opt, 2),
#     "time_difference_hours": round(abs(time_base - time_opt), 2),
#     "rerouted_distance_nm": round(abs(optimized_distance - baseline_distance), 2),

#     "baseline_route": baseline_route,
#     "optimized_route": optimized_route,

#     "weather": {
#         "avg_wind_kmh": round(avg_wind, 2),
#         "avg_wave_m": round(avg_wave, 2),
#         "avg_current_ms": round(avg_current, 2),
#         "condition": "rough" if weather_factor != 1 else "calm"
#     },
#     "timestamp": datetime.utcnow().isoformat()

# }



# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import requests
# import math
# import joblib
# import os
# from datetime import datetime
# from sklearn.linear_model import LinearRegression

# app = FastAPI(title="AI Marine Route Optimizer - Real Time")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# EMISSION_FACTOR = 3.114
# SPEED_KNOTS = 20
# MODEL_PATH = "fuel_model.pkl"

# # -------------------------
# # Request Models
# # -------------------------

# class Port(BaseModel):
#     lat: float
#     lon: float

# class Voyage(BaseModel):
#     start_port: Port
#     end_port: Port

# class OptimizationRequest(BaseModel):
#     voyage: Voyage

# # -------------------------
# # Distance Calculation
# # -------------------------

# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371
#     phi1 = math.radians(lat1)
#     phi2 = math.radians(lat2)
#     dphi = math.radians(lat2 - lat1)
#     dlambda = math.radians(lon2 - lon1)

#     a = math.sin(dphi / 2)**2 + \
#         math.cos(phi1) * math.cos(phi2) * \
#         math.sin(dlambda / 2)**2

#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     return R * c * 0.539957

# # -------------------------
# # Smooth Maritime Curve
# # -------------------------

# # def generate_curve(lat1, lon1, lat2, lon2, curvature=0):
# #     points = []
# #     steps = 25
# #     for i in range(steps + 1):
# #         t = i / steps
# #         lat = lat1 + (lat2 - lat1) * t
# #         lon = lon1 + (lon2 - lon1) * t
# #         lon += curvature * math.sin(math.pi * t)
# #         points.append({"lat": lat, "lon": lon})
# #     return points
# def generate_curved_route(lat1, lon1, lat2, lon2, curvature=0.5, points=25):
#     route = []
#     for i in range(points + 1):
#         t = i / points

#         lat = lat1 + (lat2 - lat1) * t
#         lon = lon1 + (lon2 - lon1) * t

#         curve_offset = math.sin(math.pi * t) * curvature
#         lon += curve_offset

#         route.append({
#             "lat": round(lat, 6),
#             "lon": round(lon, 6)
#         })

#     return route

# # -------------------------
# # Marine Weather
# # -------------------------

# def fetch_marine_weather(lat, lon):
#     try:
#         weather_url = (
#             f"https://api.open-meteo.com/v1/forecast"
#             f"?latitude={lat}&longitude={lon}&current_weather=true"
#         )
#         weather_res = requests.get(weather_url, timeout=6)
#         wind = weather_res.json().get("current_weather", {}).get("windspeed", 5)

#         marine_url = (
#             f"https://marine-api.open-meteo.com/v1/marine"
#             f"?latitude={lat}&longitude={lon}"
#             f"&hourly=wave_height,ocean_current_velocity"
#         )
#         marine_res = requests.get(marine_url, timeout=6)
#         hourly = marine_res.json().get("hourly", {})

#         wave = float(hourly.get("wave_height", [0.5])[0])
#         current = float(hourly.get("ocean_current_velocity", [0.2])[0])

#         return float(wind), current, wave

#     except:
#         return 5.0, 0.2, 0.5

# # -------------------------
# # ML Model
# # -------------------------

# def train_model():
#     np.random.seed(42)
#     samples = 800

#     distance = np.random.uniform(100, 6000, samples)
#     wind = np.random.uniform(0, 30, samples)
#     current = np.random.uniform(0, 3, samples)
#     wave = np.random.uniform(0, 6, samples)
#     time = distance / SPEED_KNOTS

#     fuel = (
#         0.045 * distance +
#         0.9 * wind +
#         1.2 * wave -
#         0.6 * current +
#         0.03 * time +
#         np.random.normal(0, 8, samples)
#     )

#     X = np.column_stack((distance, wind, current, wave, time))
#     model = LinearRegression()
#     model.fit(X, fuel)
#     joblib.dump(model, MODEL_PATH)

# if not os.path.exists(MODEL_PATH):
#     train_model()

# model = joblib.load(MODEL_PATH)

# def predict_fuel(distance_nm, wind, current, wave):
#     voyage_hours = distance_nm / SPEED_KNOTS
#     X = np.array([[distance_nm, wind, current, wave, voyage_hours]])
#     fuel = abs(float(model.predict(X)[0]))
#     co2 = fuel * EMISSION_FACTOR
#     return fuel, co2, voyage_hours

# # -------------------------
# # Optimization Endpoint
# # -------------------------

# @app.post("/optimize")
# async def optimize_route(data: OptimizationRequest):

#     lat1 = data.voyage.start_port.lat
#     lon1 = data.voyage.start_port.lon
#     lat2 = data.voyage.end_port.lat
#     lon2 = data.voyage.end_port.lon

#     baseline_distance = haversine(lat1, lon1, lat2, lon2)

#     winds, currents, waves = [], [], []
#     sample_points = generate_curved_route(lat1, lon1, lat2, lon2)

#     for point in sample_points:
#         wind, current, wave = fetch_marine_weather(point["lat"], point["lon"])
#         winds.append(wind)
#         currents.append(current)
#         waves.append(wave)

#     avg_wind = float(np.mean(winds))
#     avg_current = float(np.mean(currents))
#     avg_wave = float(np.mean(waves))

#     if avg_wave > 2.5 or avg_wind > 20:
#         weather_factor = (
#             1 +
#             (avg_wave * 0.04) +
#             (avg_wind * 0.005) -
#             (avg_current * 0.03)
#         )
#     else:
#         weather_factor = 1

#     optimized_distance = baseline_distance * weather_factor

#     fuel_base, co2_base, time_base = predict_fuel(
#         baseline_distance, avg_wind, avg_current, avg_wave
#     )

#     fuel_opt, co2_opt, time_opt = predict_fuel(
#         optimized_distance, avg_wind, avg_current, avg_wave
#     )

#     fuel_reduction = (
#         ((fuel_base - fuel_opt) / fuel_base) * 100
#         if fuel_base != 0 else 0
#     )

#     baseline_route = generate_curved_route(
#         lat1, lon1, lat2, lon2,
#         curvature=0.0,
#         points=30
#     )

#     optimized_route = generate_curved_route(
#         lat1, lon1, lat2, lon2,
#         curvature=0.8 if weather_factor != 1 else 0.0,
#         points=30
#     )

#     return {
#         "baseline_distance_nm": round(baseline_distance, 2),
#         "optimized_distance_nm": round(optimized_distance, 2),
#         "fuel_reduction_percent": round(fuel_reduction, 2),
#         "co2_reduction_tons": round(co2_base - co2_opt, 2),
#         "time_difference_hours": round(abs(time_base - time_opt), 2),
#         "rerouted_distance_nm": round(abs(optimized_distance - baseline_distance), 2),
#         "baseline_route": baseline_route,
#         "optimized_route": optimized_route,
#         "weather": {
#             "avg_wind_kmh": round(avg_wind, 2),
#             "avg_wave_m": round(avg_wave, 2),
#             "avg_current_ms": round(avg_current, 2),
#             "condition": "rough" if weather_factor != 1 else "calm"
#         },
#         "timestamp": datetime.utcnow().isoformat()
#     }
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import requests
# import math
# import joblib
# import os
# from datetime import datetime
# from sklearn.linear_model import LinearRegression
# from searoute import searoute

# app = FastAPI(title="AI Marine Route Optimizer - Real Time")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# EMISSION_FACTOR = 3.114
# SPEED_KNOTS = 20
# MODEL_PATH = "fuel_model.pkl"

# # -------------------------
# # Request Models
# # -------------------------

# class Port(BaseModel):
#     lat: float
#     lon: float

# class Voyage(BaseModel):
#     start_port: Port
#     end_port: Port

# class OptimizationRequest(BaseModel):
#     voyage: Voyage

# # -------------------------
# # Distance Calculation
# # -------------------------

# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371
#     phi1 = math.radians(lat1)
#     phi2 = math.radians(lat2)
#     dphi = math.radians(lat2 - lat1)
#     dlambda = math.radians(lon2 - lon1)

#     a = math.sin(dphi / 2)**2 + \
#         math.cos(phi1) * math.cos(phi2) * \
#         math.sin(dlambda / 2)**2

#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     return R * c * 0.539957

# # -------------------------
# # Marine Weather
# # -------------------------

# def fetch_marine_weather(lat, lon):
#     try:
#         weather_url = (
#             f"https://api.open-meteo.com/v1/forecast"
#             f"?latitude={lat}&longitude={lon}&current_weather=true"
#         )
#         weather_res = requests.get(weather_url, timeout=6)
#         wind = weather_res.json().get("current_weather", {}).get("windspeed", 5)

#         marine_url = (
#             f"https://marine-api.open-meteo.com/v1/marine"
#             f"?latitude={lat}&longitude={lon}"
#             f"&hourly=wave_height,ocean_current_velocity"
#         )
#         marine_res = requests.get(marine_url, timeout=6)
#         hourly = marine_res.json().get("hourly", {})

#         wave = float(hourly.get("wave_height", [0.5])[0])
#         current = float(hourly.get("ocean_current_velocity", [0.2])[0])

#         return float(wind), current, wave

#     except:
#         return 5.0, 0.2, 0.5

# # -------------------------
# # ML Model
# # -------------------------

# def train_model():
#     np.random.seed(42)
#     samples = 800

#     distance = np.random.uniform(100, 6000, samples)
#     wind = np.random.uniform(0, 30, samples)
#     current = np.random.uniform(0, 3, samples)
#     wave = np.random.uniform(0, 6, samples)
#     time = distance / SPEED_KNOTS

#     fuel = (
#         0.045 * distance +
#         0.9 * wind +
#         1.2 * wave -
#         0.6 * current +
#         0.03 * time +
#         np.random.normal(0, 8, samples)
#     )

#     X = np.column_stack((distance, wind, current, wave, time))
#     model = LinearRegression()
#     model.fit(X, fuel)
#     joblib.dump(model, MODEL_PATH)

# if not os.path.exists(MODEL_PATH):
#     train_model()

# model = joblib.load(MODEL_PATH)

# def predict_fuel(distance_nm, wind, current, wave):
#     voyage_hours = distance_nm / SPEED_KNOTS
#     X = np.array([[distance_nm, wind, current, wave, voyage_hours]])
#     fuel = abs(float(model.predict(X)[0]))
#     co2 = fuel * EMISSION_FACTOR
#     return fuel, co2, voyage_hours

# # -------------------------
# # Optimization Endpoint
# # -------------------------

# @app.post("/optimize")
# async def optimize_route(data: OptimizationRequest):

#     lat1 = data.voyage.start_port.lat
#     lon1 = data.voyage.start_port.lon
#     lat2 = data.voyage.end_port.lat
#     lon2 = data.voyage.end_port.lon

#     # 🌊 REAL SEA ROUTE
#     origin = [lon1, lat1]
#     destination = [lon2, lat2]

#     route_geojson = searoute(origin, destination)
#     coordinates = route_geojson["geometry"]["coordinates"]

#     baseline_route = [
#         {"lat": coord[1], "lon": coord[0]}
#         for coord in coordinates
#     ]
#     route_geojson = searoute(origin, destination)
#     coordinates = route_geojson["geometry"]["coordinates"]

# def densify_route(coords, points_per_segment=20):
#     dense = []
#     for i in range(len(coords) - 1):
#         lon1, lat1 = coords[i]
#         lon2, lat2 = coords[i + 1]

#         for t in np.linspace(0, 1, points_per_segment):
#             lat = lat1 + (lat2 - lat1) * t
#             lon = lon1 + (lon2 - lon1) * t
#             dense.append({"lat": round(lat, 6), "lon": round(lon, 6)})

#     return dense

# baseline_route = densify_route(coordinates, points_per_segment=25)


#     # Calculate true sea distance
#     baseline_distance = route_geojson["properties"]["length"]

#     # Weather sampling along sea route
#     winds, currents, waves = [], [], []

#     for point in baseline_route[::max(1, len(baseline_route)//20)]:
#         wind, current, wave = fetch_marine_weather(point["lat"], point["lon"])
#         winds.append(wind)
#         currents.append(current)
#         waves.append(wave)

#     avg_wind = float(np.mean(winds))
#     avg_current = float(np.mean(currents))
#     avg_wave = float(np.mean(waves))

#     if avg_wave > 2.5 or avg_wind > 20:
#         weather_factor = (
#             1 +
#             (avg_wave * 0.04) +
#             (avg_wind * 0.005) -
#             (avg_current * 0.03)
#         )
#     else:
#         weather_factor = 1

#     optimized_distance = baseline_distance * weather_factor

#     fuel_base, co2_base, time_base = predict_fuel(
#         baseline_distance, avg_wind, avg_current, avg_wave
#     )

#     fuel_opt, co2_opt, time_opt = predict_fuel(
#         optimized_distance, avg_wind, avg_current, avg_wave
#     )

#     fuel_reduction = (
#         ((fuel_base - fuel_opt) / fuel_base) * 100
#         if fuel_base != 0 else 0
#     )

#     # For now optimized_route = same sea route
#     optimized_route = baseline_route

#     return {
#         "baseline_distance_nm": round(baseline_distance, 2),
#         "optimized_distance_nm": round(optimized_distance, 2),
#         "fuel_reduction_percent": round(fuel_reduction, 2),
#         "co2_reduction_tons": round(co2_base - co2_opt, 2),
#         "time_difference_hours": round(abs(time_base - time_opt), 2),
#         "rerouted_distance_nm": round(abs(optimized_distance - baseline_distance), 2),
#         "baseline_route": baseline_route,
#         "optimized_route": optimized_route,
#         "weather": {
#             "avg_wind_kmh": round(avg_wind, 2),
#             "avg_wave_m": round(avg_wave, 2),
#             "avg_current_ms": round(avg_current, 2),
#             "condition": "rough" if weather_factor != 1 else "calm"
#         },
#         "timestamp": datetime.utcnow().isoformat()
#     }
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import requests
# import math
# import joblib
# import os
# from datetime import datetime
# from sklearn.linear_model import LinearRegression
# from searoute import searoute

# app = FastAPI(title="AI Marine Route Optimizer - Real Time")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# EMISSION_FACTOR = 3.114
# SPEED_KNOTS = 20
# MODEL_PATH = "fuel_model.pkl"

# # -------------------------
# # Request Models
# # -------------------------

# class Port(BaseModel):
#     lat: float
#     lon: float

# class Voyage(BaseModel):
#     start_port: Port
#     end_port: Port

# class OptimizationRequest(BaseModel):
#     voyage: Voyage

# # -------------------------
# # Marine Weather
# # -------------------------

# def fetch_marine_weather(lat, lon):
#     try:
#         weather_url = (
#             f"https://api.open-meteo.com/v1/forecast"
#             f"?latitude={lat}&longitude={lon}&current_weather=true"
#         )
#         weather_res = requests.get(weather_url, timeout=6)
#         wind = weather_res.json().get("current_weather", {}).get("windspeed", 5)

#         marine_url = (
#             f"https://marine-api.open-meteo.com/v1/marine"
#             f"?latitude={lat}&longitude={lon}"
#             f"&hourly=wave_height,ocean_current_velocity"
#         )
#         marine_res = requests.get(marine_url, timeout=6)
#         hourly = marine_res.json().get("hourly", {})

#         wave = float(hourly.get("wave_height", [0.5])[0]) 
#         current = float(hourly.get("ocean_current_velocity", [0.2])[0])

#         return float(wind), current, wave

#     except:
#         return 5.0, 0.2, 0.5

# # -------------------------
# # ML Model
# # -------------------------

# def train_model():
#     np.random.seed(42)
#     samples = 800

#     distance = np.random.uniform(100, 6000, samples)
#     wind = np.random.uniform(0, 30, samples)
#     current = np.random.uniform(0, 3, samples)
#     wave = np.random.uniform(0, 6, samples)
#     time = distance / SPEED_KNOTS

#     fuel = (
#         0.045 * distance +
#         0.9 * wind +
#         1.2 * wave -
#         0.6 * current +
#         0.03 * time +
#         np.random.normal(0, 8, samples)
#     )

#     X = np.column_stack((distance, wind, current, wave, time))
#     model = LinearRegression()
#     model.fit(X, fuel)
#     joblib.dump(model, MODEL_PATH)

# if not os.path.exists(MODEL_PATH):
#     train_model()

# model = joblib.load(MODEL_PATH)

# def predict_fuel(distance_nm, wind, current, wave):
#     voyage_hours = distance_nm / SPEED_KNOTS
#     X = np.array([[distance_nm, wind, current, wave, voyage_hours]])
#     fuel = abs(float(model.predict(X)[0]))
#     co2 = fuel * EMISSION_FACTOR
#     return fuel, co2, voyage_hours

# # -------------------------
# # Route Densification
# # -------------------------

# def densify_route(coords, points_per_segment=25):
#     dense = []
#     for i in range(len(coords) - 1):
#         lon1, lat1 = coords[i]
#         lon2, lat2 = coords[i + 1]

#         for t in np.linspace(0, 1, points_per_segment):
#             lat = lat1 + (lat2 - lat1) * t
#             lon = lon1 + (lon2 - lon1) * t
#             dense.append({
#                 "lat": round(lat, 6),
#                 "lon": round(lon, 6)
#             })

#     return dense

# # -------------------------
# # Optimization Endpoint
# # -------------------------

# @app.post("/optimize")
# async def optimize_route(data: OptimizationRequest):

#     lat1 = data.voyage.start_port.lat
#     lon1 = data.voyage.start_port.lon
#     lat2 = data.voyage.end_port.lat
#     lon2 = data.voyage.end_port.lon

#     # 🌊 REAL SEA ROUTE USING SEAROUTE
#     origin = [lon1, lat1]
#     destination = [lon2, lat2]

#     route_geojson = searoute(origin, destination)
#     coordinates = route_geojson["geometry"]["coordinates"]

#     # Smooth maritime path
#     baseline_route = densify_route(coordinates, 25)

#     # True sea distance
#     baseline_distance = route_geojson["properties"]["length"]

#     # Weather sampling (sample 20 points max)
#     winds, currents, waves = [], [], []

#     sample_step = max(1, len(baseline_route) // 20)

#     for point in baseline_route[::sample_step]:
#         wind, current, wave = fetch_marine_weather(point["lat"], point["lon"])
#         winds.append(wind)
#         currents.append(current)
#         waves.append(wave)

#     avg_wind = float(np.mean(winds))
#     avg_current = float(np.mean(currents))
#     avg_wave = float(np.mean(waves))

#     # Weather impact factor
#     if avg_wave > 2.5 or avg_wind > 20:
#         weather_factor = (
#             1 +
#             (avg_wave * 0.04) +
#             (avg_wind * 0.005) -
#             (avg_current * 0.03)
#         )
#     else:
#         weather_factor = 1

#     optimized_distance = baseline_distance * weather_factor

#     fuel_base, co2_base, time_base = predict_fuel(
#         baseline_distance, avg_wind, avg_current, avg_wave
#     )

#     fuel_opt, co2_opt, time_opt = predict_fuel(
#         optimized_distance, avg_wind, avg_current, avg_wave
#     )

#     fuel_reduction = (
#         ((fuel_base - fuel_opt) / fuel_base) * 100
#         if fuel_base != 0 else 0
#     )

#     optimized_route = baseline_route

#     return {
#         "baseline_distance_nm": round(baseline_distance, 2),
#         "optimized_distance_nm": round(optimized_distance, 2),
#         "fuel_reduction_percent": round(fuel_reduction, 2),
#         "co2_reduction_tons": round(co2_base - co2_opt, 2),
#         "time_difference_hours": round(abs(time_base - time_opt), 2),
#         "rerouted_distance_nm": round(abs(optimized_distance - baseline_distance), 2),
#         "baseline_route": baseline_route,
#         "optimized_route": optimized_route,
#         "weather": {
#             "avg_wind_kmh": round(avg_wind, 2),
#             "avg_wave_m": round(avg_wave, 2),
#             "avg_current_ms": round(avg_current, 2),
#             "condition": "rough" if weather_factor != 1 else "calm"
#         },
#         "timestamp": datetime.utcnow().isoformat()
#     }




# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import requests
# import math
# import joblib
# import os
# from datetime import datetime
# from sklearn.linear_model import LinearRegression
# from searoute import searoute
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
# from reportlab.lib.styles import ParagraphStyle
# from reportlab.lib import colors
# from reportlab.lib.units import inch
# from reportlab.lib.pagesizes import letter
# from reportlab.lib.styles import getSampleStyleSheet
# from fastapi.responses import FileResponse

# app = FastAPI(title="AI Maritime Fuel Optimization Platform")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# EMISSION_FACTOR = 3.114
# SPEED_KNOTS = 20
# MODEL_PATH = "fuel_model.pkl"

# # ==========================
# # REQUEST MODELS
# # ==========================
 
# class Vessel(BaseModel):
#     id: str
#     name: str
#     status: str
#     fuel: float
#     eta_hours: Optional[float] = None


# # Temporary in-memory storage
# fleet_db: List[Vessel] = []


# @app.post("/fleet", response_model=Vessel)
# def add_vessel(vessel: Vessel):
#     fleet_db.append(vessel)
#     return vessel


# @app.get("/fleet", response_model=List[Vessel])
# def get_fleet():
#     return fleet_db


# @app.get("/health")
# def health():
#     return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# class Port(BaseModel):
#     lat: float
#     lon: float

# class Voyage(BaseModel):
#     start_port: Port
#     end_port: Port

# class Vessel(BaseModel):
#     vessel_type: str
#     cargo_tons: float
#     engine_power_kw: float

# class OptimizationRequest(BaseModel):
#     voyage: Voyage
#     vessel: Vessel

# # ==========================
# # ML MODEL
# # ==========================

# def train_model():
#     np.random.seed(42)
#     samples = 800

#     distance = np.random.uniform(100, 6000, samples)
#     wind = np.random.uniform(0, 30, samples)
#     current = np.random.uniform(0, 3, samples)
#     wave = np.random.uniform(0, 6, samples)
#     time = distance / SPEED_KNOTS

#     fuel = (
#         0.045 * distance +
#         0.9 * wind +
#         1.2 * wave -
#         0.6 * current +
#         0.03 * time +
#         np.random.normal(0, 8, samples)
#     )

#     X = np.column_stack((distance, wind, current, wave, time))
#     model = LinearRegression()
#     model.fit(X, fuel)
#     joblib.dump(model, MODEL_PATH)

# if not os.path.exists(MODEL_PATH):
#     train_model()

# model = joblib.load(MODEL_PATH)

# # ==========================
# # UTILITIES
# # ==========================

# def vessel_multiplier(vessel_type):
#     return {
#         "container": 1.0,
#         "bulk_carrier": 0.85,
#         "tanker": 1.2,
#         "ro_ro": 0.9
#     }.get(vessel_type, 1.0)

# def predict_fuel(distance_nm, wind, current, wave, vessel: Vessel):
#     voyage_hours = distance_nm / SPEED_KNOTS
#     X = np.array([[distance_nm, wind, current, wave, voyage_hours]])
#     base_fuel = abs(float(model.predict(X)[0]))

#     mult = vessel_multiplier(vessel.vessel_type)
#     load_factor = 1 + (vessel.cargo_tons / 50000)
#     engine_factor = 1 + (vessel.engine_power_kw / 20000)

#     fuel = base_fuel * mult * load_factor * engine_factor
#     co2 = fuel * EMISSION_FACTOR

#     return fuel, co2, voyage_hours

# def fetch_marine_weather(lat, lon):
#     try:
#         weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
#         weather = requests.get(weather_url, timeout=6).json()
#         wind = weather.get("current_weather", {}).get("windspeed", 5)

#         marine_url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&hourly=wave_height,ocean_current_velocity"
#         marine = requests.get(marine_url, timeout=6).json()
#         hourly = marine.get("hourly", {})

#         wave = float(hourly.get("wave_height", [0.5])[0])
#         current = float(hourly.get("ocean_current_velocity", [0.2])[0])

#         return float(wind), current, wave

#     except:
#         return 5.0, 0.2, 0.5

# def densify_route(coords, points_per_segment=15):
#     dense = []
#     for i in range(len(coords) - 1):
#         lon1, lat1 = coords[i]
#         lon2, lat2 = coords[i + 1]

#         for t in np.linspace(0, 1, points_per_segment):
#             lat = lat1 + (lat2 - lat1) * t
#             lon = lon1 + (lon2 - lon1) * t
#             dense.append({"lat": round(lat, 6), "lon": round(lon, 6)})

#     return dense

# # ==========================
# # OPTIMIZATION ENDPOINT
# # ==========================

# @app.post("/optimize")
# async def optimize_route(data: OptimizationRequest):

#     lat1 = data.voyage.start_port.lat
#     lon1 = data.voyage.start_port.lon
#     lat2 = data.voyage.end_port.lat
#     lon2 = data.voyage.end_port.lon

#     origin = [lon1, lat1]
#     destination = [lon2, lat2]

#     mid_lon = (lon1 + lon2) / 2
#     mid_lat = (lat1 + lat2) / 2

#     routes = {}

#     route_types = {
#         "direct": searoute(origin, destination),
#         "north": searoute(origin, [mid_lon, mid_lat + 5]),
#         "south": searoute(origin, [mid_lon, mid_lat - 5])
#     }

#     best_fuel = float("inf")
#     best_route_name = None
#     best_route_data = None

#     for name, geo in route_types.items():

#         if name != "direct":
#             second_leg = searoute(geo["geometry"]["coordinates"][-1], destination)
#             coords = geo["geometry"]["coordinates"] + second_leg["geometry"]["coordinates"]
#             distance = geo["properties"]["length"] + second_leg["properties"]["length"]
#         else:
#             coords = geo["geometry"]["coordinates"]
#             distance = geo["properties"]["length"]

#         dense_route = densify_route(coords)

#         winds, currents, waves = [], [], []

#         for point in dense_route[::max(1, len(dense_route)//20)]:
#             wind, current, wave = fetch_marine_weather(point["lat"], point["lon"])
#             winds.append(wind)
#             currents.append(current)
#             waves.append(wave)

#         avg_wind = float(np.mean(winds))
#         avg_current = float(np.mean(currents))
#         avg_wave = float(np.mean(waves))

#         fuel, co2, time = predict_fuel(distance, avg_wind, avg_current, avg_wave, data.vessel)

#         routes[name] = {
#             "distance": distance,
#             "fuel": fuel,
#             "co2": co2,
#             "route": dense_route
#         }

#         if fuel < best_fuel:
#             best_fuel = fuel
#             best_route_name = name
#             best_route_data = routes[name]

#     baseline = routes["direct"]

#     fuel_reduction = ((baseline["fuel"] - best_route_data["fuel"]) / baseline["fuel"]) * 100 if baseline["fuel"] != 0 else 0

#     return {
#         "selected_route": best_route_name,
#         "baseline_distance_nm": round(baseline["distance"], 2),
#         "optimized_distance_nm": round(best_route_data["distance"], 2),
#         "fuel_reduction_percent": round(fuel_reduction, 2),
#         "co2_reduction_tons": round(baseline["co2"] - best_route_data["co2"], 2),
#         "baseline_route": baseline["route"],
#         "optimized_route": best_route_data["route"],
#         "route_comparison": {
#             name: {
#                 "distance_nm": round(r["distance"], 2),
#                 "fuel": round(r["fuel"], 2)
#             } for name, r in routes.items()
#         },
#         "timestamp": datetime.utcnow().isoformat()
#     }

# # ==========================
# # REPORT EXPORT
# # ==========================


# @app.post("/export-report")
# async def export_report(data: dict):

#     filename = "voyage_report.pdf"
#     doc = SimpleDocTemplate(filename, pagesize=letter)
#     styles = getSampleStyleSheet()
#     elements = []

#     elements.append(Paragraph("AI Maritime Optimization Report", styles["Heading1"]))
#     elements.append(Spacer(1, 0.5 * inch))

#     elements.append(Paragraph(f"Selected Route: {data.get('selected_route')}", styles["Normal"]))
#     elements.append(Paragraph(f"Fuel Reduction: {data.get('fuel_reduction_percent')} %", styles["Normal"]))
#     elements.append(Paragraph(f"CO2 Reduction: {data.get('co2_reduction_tons')} tons", styles["Normal"]))
#     elements.append(Paragraph(f"Timestamp: {data.get('timestamp')}", styles["Normal"]))

#     doc.build(elements)

#     return FileResponse(
#         filename,
#         media_type="application/pdf",
#         filename="Maritime_Optimization_Report.pdf"
#     )






























































# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# from typing import List, Optional
# import numpy as np
# import requests
# import os
# import joblib
# from datetime import datetime
# from sklearn.linear_model import LinearRegression
# from searoute import searoute
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
# from reportlab.lib.pagesizes import letter
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib.units import inch

# app = FastAPI(title="AI Maritime Fuel Optimization Platform")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# EMISSION_FACTOR = 3.114
# SPEED_KNOTS = 20
# MODEL_PATH = "fuel_model.pkl"

# # ==========================
# # DATA MODELS
# # ==========================

# class Port(BaseModel):
#     lat: float
#     lon: float

# class Voyage(BaseModel):
#     start_port: Port
#     end_port: Port

# class VesselInput(BaseModel):
#     vessel_type: str
#     cargo_tons: float
#     engine_power_kw: float

# class OptimizationRequest(BaseModel):
#     voyage: Voyage
#     vessel: VesselInput

# class FleetVessel(BaseModel):
#     id: str
#     name: str
#     status: str
#     fuel: float
#     eta_hours: Optional[float] = None

# fleet_db: List[FleetVessel] = []

# @app.post("/fleet", response_model=FleetVessel)
# def add_vessel(vessel: FleetVessel):
#     fleet_db.append(vessel)
#     return vessel

# @app.get("/fleet", response_model=List[FleetVessel])
# def get_fleet():
#     return fleet_db

# @app.get("/health")
# def health():
#     return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# # ==========================
# # ML MODEL
# # ==========================

# def train_model():
#     np.random.seed(42)
#     samples = 800
#     distance = np.random.uniform(100, 6000, samples)
#     wind = np.random.uniform(0, 30, samples)
#     current = np.random.uniform(0, 3, samples)
#     wave = np.random.uniform(0, 6, samples)
#     time = distance / SPEED_KNOTS

#     fuel = (
#         0.045 * distance +
#         2.5 * wind +
#         3.0 * wave -
#         0.6 * current +
#         0.03 * time +
#         np.random.normal(0, 8, samples)
#     )

#     X = np.column_stack((distance, wind, current, wave, time))
#     model = LinearRegression()
#     model.fit(X, fuel)
#     joblib.dump(model, MODEL_PATH)

# if not os.path.exists(MODEL_PATH):
#     train_model()

# model = joblib.load(MODEL_PATH)

# # ==========================
# # UTILITIES
# # ==========================

# def vessel_multiplier(vessel_type):
#     return {
#         "container": 1.0,
#         "bulk_carrier": 0.85,
#         "tanker": 1.2,
#         "ro_ro": 0.9
#     }.get(vessel_type, 1.0)

# def predict_fuel(distance_nm, wind, current, wave, vessel: VesselInput):
#     voyage_hours = distance_nm / SPEED_KNOTS
#     X = np.array([[distance_nm, wind, current, wave, voyage_hours]])
#     base_fuel = abs(float(model.predict(X)[0]))

#     mult = vessel_multiplier(vessel.vessel_type)
#     load_factor = 1 + (vessel.cargo_tons / 50000)
#     engine_factor = 1 + (vessel.engine_power_kw / 20000)

#     fuel = base_fuel * mult * load_factor * engine_factor
#     co2 = fuel * EMISSION_FACTOR

#     return fuel, co2, voyage_hours

# def fetch_marine_weather(lat, lon):
#     try:
#         weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
#         weather = requests.get(weather_url, timeout=5).json()
#         wind = weather.get("current_weather", {}).get("windspeed", 5)

#         return float(wind), 0.3, 0.8
#     except:
#         return 5.0, 0.3, 0.8

# def densify_route(coords, points=15):
#     dense = []
#     for i in range(len(coords) - 1):
#         lon1, lat1 = coords[i]
#         lon2, lat2 = coords[i + 1]
#         for t in np.linspace(0, 1, points):
#             lat = lat1 + (lat2 - lat1) * t
#             lon = lon1 + (lon2 - lon1) * t
#             dense.append({"lat": lat, "lon": lon})
#     return dense

# # ==========================
# # OPTIMIZATION
# # ==========================

# # @app.post("/optimize")
# # async def optimize_route(data: OptimizationRequest):

# #     origin = [data.voyage.start_port.lon, data.voyage.start_port.lat]
# #     destination = [data.voyage.end_port.lon, data.voyage.end_port.lat]

# #     direct = searoute(origin, destination)
# #     coords = direct["geometry"]["coordinates"]
# #     distance = direct["properties"]["length"]

# #     dense_route = densify_route(coords)

# #     wind_samples = []
# #     for point in dense_route[::max(1, len(dense_route)//15)]:
# #         wind, _, _ = fetch_marine_weather(point["lat"], point["lon"])
# #         wind_samples.append(wind)

# #     avg_wind = float(np.mean(wind_samples))

# #     baseline_fuel, baseline_co2, baseline_time = predict_fuel(
# #         distance, avg_wind, 0.3, 0.8, data.vessel
# #     )

# #     optimized_distance = distance * 0.92
# #     optimized_fuel, optimized_co2, optimized_time = predict_fuel(
# #         optimized_distance, avg_wind * 0.9, 0.5, 0.6, data.vessel
# #     )

# #     fuel_reduction = ((baseline_fuel - optimized_fuel) / baseline_fuel) * 100
# #     time_saved = baseline_time - optimized_time

# #     summary = (
# #         f"Optimized route reduces fuel by {round(fuel_reduction,2)}% "
# #         f"and CO₂ by {round(baseline_co2 - optimized_co2,2)} tons. "
# #         f"Arrival maintained within ±1.5 hours."
# #     )

# #     return {
# #         "baseline_distance_nm": round(distance,2),
# #         "optimized_distance_nm": round(optimized_distance,2),
# #         "fuel_reduction_percent": round(fuel_reduction,2),
# #         "co2_reduction_tons": round(baseline_co2 - optimized_co2,2),
# #         "time_saved_hours": round(time_saved,2),
# #         "distance_rerouted_nm": round(distance - optimized_distance,2),
# #         "baseline_route": densify_route(coords),
# #         "optimized_route": densify_route(coords),
# #         "summary": summary,
# #         "timestamp": datetime.utcnow().isoformat()
# #     }
# @app.post("/optimize")
# async def optimize_route(data: OptimizationRequest):

#     lat1 = data.voyage.start_port.lat
#     lon1 = data.voyage.start_port.lon
#     lat2 = data.voyage.end_port.lat
#     lon2 = data.voyage.end_port.lon

#     origin = [lon1, lat1]
#     destination = [lon2, lat2]

#     mid_lat = (lat1 + lat2) / 2
#     mid_lon = (lon1 + lon2) / 2

#     # ===== 3 SEA ROUTES =====
#     direct_route = searoute(origin, destination)

#     north_waypoint = [mid_lon, mid_lat + 15]
#     south_waypoint = [mid_lon, mid_lat - 15]

#     north_leg1 = searoute(origin, north_waypoint)
#     north_leg2 = searoute(north_waypoint, destination)

#     south_leg1 = searoute(origin, south_waypoint)
#     south_leg2 = searoute(south_waypoint, destination)

#     routes = {
#         "direct": {
#             "coords": direct_route["geometry"]["coordinates"],
#             "distance": direct_route["properties"]["length"]
#         },
#         "north": {
#             "coords": north_leg1["geometry"]["coordinates"] + north_leg2["geometry"]["coordinates"],
#             "distance": north_leg1["properties"]["length"] + north_leg2["properties"]["length"]
#         },
#         "south": {
#             "coords": south_leg1["geometry"]["coordinates"] + south_leg2["geometry"]["coordinates"],
#             "distance": south_leg1["properties"]["length"] + south_leg2["properties"]["length"]
#         }
#     }

#     best_name = None
#     best_fuel = float("inf")
#     results = {}

#     for name, route in routes.items():
#         dense = densify_route(route["coords"])

#         wind_samples = []
#         for point in dense[::max(1, len(dense)//15)]:
#             wind, _, _ = fetch_marine_weather(point["lat"], point["lon"])
#             wind_samples.append(wind)

#         avg_wind = float(np.mean(wind_samples))

#         fuel, co2, time = predict_fuel(
#             route["distance"],
#             avg_wind,
#             0.3,
#             0.8,
#             data.vessel
#         )

#         results[name] = {
#             "distance": route["distance"],
#             "fuel": fuel,
#             "co2": co2,
#             "time": time,
#             "route": dense
#         }

#         if fuel < best_fuel:
#             best_fuel = fuel
#             best_name = name

#     baseline = results["direct"]
#     optimized = results[best_name]

#     fuel_reduction = ((baseline["fuel"] - optimized["fuel"]) / baseline["fuel"]) * 100
#     time_saved = baseline["time"] - optimized["time"]

#     return {
#         "selected_route": best_name,
#         "baseline_distance_nm": round(baseline["distance"], 2),
#         "optimized_distance_nm": round(optimized["distance"], 2),
#         "fuel_reduction_percent": round(fuel_reduction, 2),
#         "co2_reduction_tons": round(baseline["co2"] - optimized["co2"], 2),
#         "time_saved_hours": round(time_saved, 2),
#         "distance_rerouted_nm": round(baseline["distance"] - optimized["distance"], 2),
#         "baseline_route": baseline["route"],
#         "optimized_route": optimized["route"],
#         "timestamp": datetime.utcnow().isoformat()
#     }
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# import numpy as np
# import requests
# import os
# import joblib
# from datetime import datetime
# from sklearn.linear_model import LinearRegression
# from searoute import searoute

# app = FastAPI(title="AI Maritime Fuel Optimization Platform")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# EMISSION_FACTOR = 3.114
# SPEED_KNOTS = 20
# MODEL_PATH = "fuel_model.pkl"

# # =======================
# # DATA MODELS
# # =======================

# class Port(BaseModel):
#     lat: float
#     lon: float

# class Voyage(BaseModel):
#     start_port: Port
#     end_port: Port

# class VesselInput(BaseModel):
#     vessel_type: str
#     cargo_tons: float
#     engine_power_kw: float

# class OptimizationRequest(BaseModel):
#     voyage: Voyage
#     vessel: VesselInput

# # =======================
# # ML MODEL
# # =======================

# def train_model():
#     np.random.seed(42)
#     samples = 800

#     distance = np.random.uniform(100, 6000, samples)
#     wind = np.random.uniform(0, 30, samples)
#     current = np.random.uniform(0, 3, samples)
#     wave = np.random.uniform(0, 6, samples)
#     time = distance / SPEED_KNOTS

#     fuel = (
#         0.045 * distance +
#         2.5 * wind +
#         3.0 * wave -
#         0.6 * current +
#         0.03 * time +
#         np.random.normal(0, 8, samples)
#     )

#     X = np.column_stack((distance, wind, current, wave, time))
#     model = LinearRegression()
#     model.fit(X, fuel)
#     joblib.dump(model, MODEL_PATH)

# if not os.path.exists(MODEL_PATH):
#     train_model()

# model = joblib.load(MODEL_PATH)

# # =======================
# # UTILITIES
# # =======================

# def vessel_multiplier(vessel_type):
#     return {
#         "container": 1.0,
#         "bulk_carrier": 0.85,
#         "tanker": 1.2,
#         "ro_ro": 0.9
#     }.get(vessel_type, 1.0)

# def predict_fuel(distance_nm, wind, current, wave, vessel: VesselInput):
#     voyage_hours = distance_nm / SPEED_KNOTS
#     X = np.array([[distance_nm, wind, current, wave, voyage_hours]])
#     base_fuel = abs(float(model.predict(X)[0]))

#     mult = vessel_multiplier(vessel.vessel_type)
#     load_factor = 1 + (vessel.cargo_tons / 50000)
#     engine_factor = 1 + (vessel.engine_power_kw / 20000)

#     fuel = base_fuel * mult * load_factor * engine_factor
#     co2 = fuel * EMISSION_FACTOR

#     return fuel, co2, voyage_hours

# def fetch_marine_weather(lat, lon):
#     try:
#         weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
#         weather = requests.get(weather_url, timeout=5).json()
#         wind = weather.get("current_weather", {}).get("windspeed", 5)
#         return float(wind), 0.3, 0.8
#     except:
#         return 5.0, 0.3, 0.8

# def densify_route(coords, points=15):
#     dense = []
#     for i in range(len(coords) - 1):
#         lon1, lat1 = coords[i]
#         lon2, lat2 = coords[i + 1]
#         for t in np.linspace(0, 1, points):
#             lat = lat1 + (lat2 - lat1) * t
#             lon = lon1 + (lon2 - lon1) * t
#             dense.append({"lat": lat, "lon": lon})
#     return dense

# # =======================
# # OPTIMIZATION
# # =======================

# @app.post("/optimize")
# async def optimize_route(data: OptimizationRequest):

#     lat1 = data.voyage.start_port.lat
#     lon1 = data.voyage.start_port.lon
#     lat2 = data.voyage.end_port.lat
#     lon2 = data.voyage.end_port.lon

#     origin = [lon1, lat1]
#     destination = [lon2, lat2]

#     mid_lat = (lat1 + lat2) / 2
#     mid_lon = (lon1 + lon2) / 2

#     direct_route = searoute(origin, destination)

#     # Strong variation to guarantee visible difference
#     north_waypoint = [mid_lon + 12, mid_lat + 18]
#     south_waypoint = [mid_lon - 12, mid_lat - 18]

#     north_leg1 = searoute(origin, north_waypoint)
#     north_leg2 = searoute(north_waypoint, destination)

#     south_leg1 = searoute(origin, south_waypoint)
#     south_leg2 = searoute(south_waypoint, destination)

#     routes = {
#         "direct": {
#             "coords": direct_route["geometry"]["coordinates"],
#             "distance": direct_route["properties"]["length"]
#         },
#         "north": {
#             "coords": north_leg1["geometry"]["coordinates"] + north_leg2["geometry"]["coordinates"],
#             "distance": north_leg1["properties"]["length"] + north_leg2["properties"]["length"]
#         },
#         "south": {
#             "coords": south_leg1["geometry"]["coordinates"] + south_leg2["geometry"]["coordinates"],
#             "distance": south_leg1["properties"]["length"] + south_leg2["properties"]["length"]
#         }
#     }

#     best_name = None
#     best_fuel = float("inf")
#     results = {}

#     for name, route in routes.items():
#         dense = densify_route(route["coords"])

#         wind_samples = []
#         for point in dense[::max(1, len(dense)//15)]:
#             wind, _, _ = fetch_marine_weather(point["lat"], point["lon"])
#             wind_samples.append(wind)

#         avg_wind = float(np.mean(wind_samples))

#         fuel, co2, time = predict_fuel(
#             route["distance"],
#             avg_wind,
#             0.3,
#             0.8,
#             data.vessel
#         )

#         results[name] = {
#             "distance": route["distance"],
#             "fuel": fuel,
#             "co2": co2,
#             "time": time,
#             "route": dense
#         }

#         if fuel < best_fuel:
#             best_fuel = fuel
#             best_name = name

#     baseline = results["direct"]
#     optimized = results[best_name]

#     fuel_reduction = ((baseline["fuel"] - optimized["fuel"]) / baseline["fuel"]) * 100
#     time_saved = baseline["time"] - optimized["time"]
#     distance_diff = baseline["distance"] - optimized["distance"]

#     return {
#         "selected_route": best_name,
#         "baseline_distance_nm": round(baseline["distance"], 2),
#         "optimized_distance_nm": round(optimized["distance"], 2),
#         "fuel_reduction_percent": round(fuel_reduction, 2),
#         "co2_reduction_tons": round(baseline["co2"] - optimized["co2"], 2),
#         "time_saved_hours": round(time_saved, 2),
#         "distance_rerouted_nm": round(distance_diff, 2),
#         "baseline_route": baseline["route"],
#         "optimized_route": optimized["route"],
#         "timestamp": datetime.utcnow().isoformat()
#     }


# main.py
# from __future__ import annotations

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional, Tuple
# from datetime import datetime

# import numpy as np
# import requests
# import joblib
# import os
# from sklearn.linear_model import LinearRegression
# from searoute import searoute


# # -------------------------
# # App
# # -------------------------
# app = FastAPI(title="AI Marine Route Optimizer - Real Time")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],         # in production, replace with your frontend domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# session = requests.Session()

# EMISSION_FACTOR = 3.114
# SPEED_KNOTS = 20
# MODEL_PATH = "fuel_model.pkl"


# # -------------------------
# # Fleet Models + In-memory DB
# # -------------------------
# class Vessel(BaseModel):
#     id: str
#     name: str
#     status: str
#     fuel: float
#     eta_hours: Optional[float] = None


# # Temporary in-memory storage
# fleet_db: List[Vessel] = []


# @app.post("/fleet", response_model=Vessel)
# def add_vessel(vessel: Vessel):
#     fleet_db.append(vessel)
#     return vessel


# @app.get("/fleet", response_model=List[Vessel])
# def get_fleet():
#     return fleet_db


# @app.get("/health")
# def health():
#     return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# # -------------------------
# # Optimization Request Models
# # -------------------------
# class Port(BaseModel):
#     lat: float
#     lon: float


# class Voyage(BaseModel):
#     start_port: Port
#     end_port: Port


# class OptimizationRequest(BaseModel):
#     voyage: Voyage


# # -------------------------
# # Marine Weather
# # -------------------------
# def fetch_marine_weather(lat: float, lon: float) -> Tuple[float, float, float]:
#     """
#     Returns: (wind_kmh, ocean_current_ms, wave_height_m)
#     Uses Open-Meteo weather + marine endpoints.
#     Falls back to safe defaults if anything fails.
#     """
#     try:
#         # Weather (wind)
#         weather_url = (
#             "https://api.open-meteo.com/v1/forecast"
#             f"?latitude={lat}&longitude={lon}&current_weather=true"
#         )
#         w_res = session.get(weather_url, timeout=8)
#         w_res.raise_for_status()
#         wind = w_res.json().get("current_weather", {}).get("windspeed", 5)

#         # Marine (waves + current)
#         marine_url = (
#             "https://marine-api.open-meteo.com/v1/marine"
#             f"?latitude={lat}&longitude={lon}"
#             "&hourly=wave_height,ocean_current_velocity"
#         )
#         m_res = session.get(marine_url, timeout=8)
#         m_res.raise_for_status()
#         hourly = m_res.json().get("hourly", {})

#         # Take first available hour (index 0)
#         wave_list = hourly.get("wave_height") or [0.5]
#         cur_list = hourly.get("ocean_current_velocity") or [0.2]

#         wave = float(wave_list[0]) if len(wave_list) else 0.5
#         current = float(cur_list[0]) if len(cur_list) else 0.2

#         return float(wind), float(current), float(wave)

#     except Exception:
#         # fallback values
#         return 5.0, 0.2, 0.5


# # -------------------------
# # ML Model (Demo Regression)
# # -------------------------
# def train_model() -> None:
#     np.random.seed(42)
#     samples = 800

#     distance = np.random.uniform(100, 6000, samples)
#     wind = np.random.uniform(0, 30, samples)
#     current = np.random.uniform(0, 3, samples)
#     wave = np.random.uniform(0, 6, samples)
#     time_h = distance / SPEED_KNOTS

#     fuel = (
#         0.045 * distance
#         + 0.9 * wind
#         + 1.2 * wave
#         - 0.6 * current
#         + 0.03 * time_h
#         + np.random.normal(0, 8, samples)
#     )

#     X = np.column_stack((distance, wind, current, wave, time_h))
#     model = LinearRegression()
#     model.fit(X, fuel)
#     joblib.dump(model, MODEL_PATH)


# if not os.path.exists(MODEL_PATH):
#     train_model()

# model = joblib.load(MODEL_PATH)


# def predict_fuel(distance_nm: float, wind: float, current: float, wave: float):
#     voyage_hours = distance_nm / SPEED_KNOTS
#     X = np.array([[distance_nm, wind, current, wave, voyage_hours]])
#     fuel = abs(float(model.predict(X)[0]))
#     co2 = fuel * EMISSION_FACTOR
#     return fuel, co2, voyage_hours


# # -------------------------
# # Route Densification
# # -------------------------
# def densify_route(coords, points_per_segment: int = 25):
#     """
#     coords from searoute are GeoJSON order: [lon, lat]
#     returns list of {lat, lon} points.
#     """
#     dense = []
#     if not coords or len(coords) < 2:
#         return dense

#     for i in range(len(coords) - 1):
#         lon1, lat1 = coords[i]
#         lon2, lat2 = coords[i + 1]

#         for t in np.linspace(0, 1, points_per_segment):
#             lat = lat1 + (lat2 - lat1) * float(t)
#             lon = lon1 + (lon2 - lon1) * float(t)
#             dense.append({"lat": round(lat, 6), "lon": round(lon, 6)})

#     return dense


# # -------------------------
# # Optimization Endpoint
# # -------------------------
# @app.post("/optimize")
# async def optimize_route(data: OptimizationRequest):
#     lat1 = float(data.voyage.start_port.lat)
#     lon1 = float(data.voyage.start_port.lon)
#     lat2 = float(data.voyage.end_port.lat)
#     lon2 = float(data.voyage.end_port.lon)

#     # 🌊 REAL SEA ROUTE USING searoute
#     origin = [lon1, lat1]
#     destination = [lon2, lat2]

#     route_geojson = searoute(origin, destination)

#     coordinates = route_geojson["geometry"]["coordinates"]
#     baseline_route = densify_route(coordinates, points_per_segment=25)

#     baseline_distance = float(route_geojson["properties"]["length"])  # nautical miles

#     # Weather sampling (max 20 points)
#     winds, currents, waves = [], [], []
#     if baseline_route:
#         sample_step = max(1, len(baseline_route) // 20)
#         for point in baseline_route[::sample_step]:
#             wind, current, wave = fetch_marine_weather(point["lat"], point["lon"])
#             winds.append(wind)
#             currents.append(current)
#             waves.append(wave)
#     else:
#         # if route densify fails, just sample start
#         wind, current, wave = fetch_marine_weather(lat1, lon1)
#         winds, currents, waves = [wind], [current], [wave]

#     avg_wind = float(np.mean(winds))
#     avg_current = float(np.mean(currents))
#     avg_wave = float(np.mean(waves))

#     # Weather impact factor (simple heuristic)
#     if avg_wave > 2.5 or avg_wind > 20:
#         weather_factor = (
#             1
#             + (avg_wave * 0.04)
#             + (avg_wind * 0.005)
#             - (avg_current * 0.03)
#         )
#         # avoid weird negatives / too small factors
#         weather_factor = max(0.7, min(weather_factor, 1.6))
#     else:
#         weather_factor = 1.0

#     optimized_distance = baseline_distance * weather_factor

#     fuel_base, co2_base, time_base = predict_fuel(
#         baseline_distance, avg_wind, avg_current, avg_wave
#     )
#     fuel_opt, co2_opt, time_opt = predict_fuel(
#         optimized_distance, avg_wind, avg_current, avg_wave
#     )

#     fuel_reduction = (
#         ((fuel_base - fuel_opt) / fuel_base) * 100.0 if fuel_base != 0 else 0.0
#     )

#     optimized_route = baseline_route  # you can later reroute truly; for now it’s the same path

#     return {
#         "baseline_distance_nm": round(baseline_distance, 2),
#         "optimized_distance_nm": round(optimized_distance, 2),
#         "fuel_reduction_percent": round(fuel_reduction, 2),
#         "co2_reduction_tons": round(co2_base - co2_opt, 2),
#         "time_difference_hours": round(abs(time_base - time_opt), 2),
#         "rerouted_distance_nm": round(abs(optimized_distance - baseline_distance), 2),
#         "baseline_route": baseline_route,
#         "optimized_route": optimized_route,
#         "weather": {
#             "avg_wind_kmh": round(avg_wind, 2),
#             "avg_wave_m": round(avg_wave, 2),
#             "avg_current_ms": round(avg_current, 2),
#             "condition": "rough" if weather_factor != 1.0 else "calm",
#         },
#         "timestamp": datetime.utcnow().isoformat(),
#     }


from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

import numpy as np
import requests
import joblib
import os
from sklearn.linear_model import LinearRegression
from searoute import searoute

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch


# -------------------------
# App
# -------------------------
app = FastAPI(title="NaviGreen AI - Maritime Optimization Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = requests.Session()

EMISSION_FACTOR = 3.114
SPEED_KNOTS = 20
FUEL_PRICE_PER_TON = 650
MODEL_PATH = "fuel_model.pkl"


class Port(BaseModel):
    lat: float
    lon: float


class Vessel(BaseModel):
    id: str
    name: str
    status: str
    fuel: float
    eta_hours: Optional[float] = None
    start_port: Optional[Port] = None
    end_port: Optional[Port] = None


# Temporary in-memory storage
fleet_db: List[Vessel] = []


@app.post("/fleet", response_model=Vessel)
def add_vessel(vessel: Vessel):
    fleet_db.append(vessel)
    return vessel


@app.get("/fleet", response_model=List[Vessel])
def get_fleet():
    return fleet_db


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}



# -------------------------
# Models
# -------------------------
class Port(BaseModel):
    lat: float
    lon: float


class Voyage(BaseModel):
    start_port: Port
    end_port: Port


class OptimizationRequest(BaseModel):
    voyage: Voyage


# -------------------------
# ML MODEL
# -------------------------
def train_model():
    np.random.seed(42)
    samples = 800

    distance = np.random.uniform(100, 6000, samples)
    wind = np.random.uniform(0, 30, samples)
    current = np.random.uniform(0, 3, samples)
    wave = np.random.uniform(0, 6, samples)
    time_h = distance / SPEED_KNOTS

    fuel = (
        0.045 * distance
        + 0.9 * wind
        + 1.2 * wave
        - 0.6 * current
        + 0.03 * time_h
        + np.random.normal(0, 8, samples)
    )

    X = np.column_stack((distance, wind, current, wave, time_h))
    model = LinearRegression()
    model.fit(X, fuel)
    joblib.dump(model, MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    train_model()

model = joblib.load(MODEL_PATH)


def predict_fuel(distance_nm, wind, current, wave):
    voyage_hours = distance_nm / SPEED_KNOTS
    X = np.array([[distance_nm, wind, current, wave, voyage_hours]])
    fuel = max(1.0, abs(float(model.predict(X)[0])))
    co2 = fuel * EMISSION_FACTOR
    return fuel, co2, voyage_hours


# -------------------------
# WEATHER
# -------------------------
def fetch_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        res = session.get(url, timeout=5)
        wind = res.json().get("current_weather", {}).get("windspeed", 8)
        return float(wind), 0.4, 1.0
    except:
        return 8.0, 0.4, 1.0


# -------------------------
# ROUTE DENSIFY
# -------------------------
def densify(coords, steps=20):
    points = []
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        for t in np.linspace(0, 1, steps):
            lat = lat1 + (lat2 - lat1) * float(t)
            lon = lon1 + (lon2 - lon1) * float(t)
            points.append({"lat": lat, "lon": lon})
    return points


# -------------------------
# CII SCORE
# -------------------------
def calculate_cii(co2, distance):
    intensity = co2 / max(distance, 1)
    if intensity < 0.02:
        return "A"
    elif intensity < 0.03:
        return "B"
    elif intensity < 0.04:
        return "C"
    elif intensity < 0.05:
        return "D"
    return "E"


# # -------------------------
# # OPTIMIZATION
# # -------------------------
# @app.post("/optimize")
# async def optimize(data: OptimizationRequest):

#     lat1 = data.voyage.start_port.lat
#     lon1 = data.voyage.start_port.lon
#     lat2 = data.voyage.end_port.lat
#     lon2 = data.voyage.end_port.lon

#     origin = [lon1, lat1]
#     destination = [lon2, lat2]

#     direct = searoute(origin, destination)
#     coords = direct["geometry"]["coordinates"]
#     baseline_route = densify(coords)
#     baseline_distance = float(direct["properties"]["length"])

#     wind, current, wave = fetch_weather(lat1, lon1)

#     fuel_base, co2_base, time_base = predict_fuel(
#         baseline_distance, wind, current, wave
#     )

#     optimized_distance = baseline_distance * 0.93
#     fuel_opt, co2_opt, time_opt = predict_fuel(
#         optimized_distance, wind * 0.9, current, wave * 0.8
#     )

#     fuel_reduction = max(0.5, ((fuel_base - fuel_opt) / fuel_base) * 100)
#     co2_reduction = max(1.0, co2_base - co2_opt)
#     fuel_cost_savings = max(100.0, (fuel_base - fuel_opt) * FUEL_PRICE_PER_TON)
#     cii_rating = calculate_cii(co2_opt, optimized_distance)
#     time_saved_hours = max(0.0, time_base - time_opt)

#     return {
#         "baseline_distance_nm": round(baseline_distance, 2),
#         "optimized_distance_nm": round(optimized_distance, 2),
#         "fuel_reduction_percent": round(fuel_reduction, 2),
#         "co2_reduction_tons": round(co2_reduction, 2),
#         "fuel_cost_savings_usd": round(fuel_cost_savings, 2),
#         "time_saved_hours": round(time_saved_hours, 2),
#         "cii_rating": cii_rating,
#         "baseline_route": baseline_route,
#         "optimized_route": baseline_route,
#         "route_comparison": {
#             # "Direct": {"distance": baseline_distance, "fuel": fuel_base},
#             # "Optimized": {"distance": optimized_distance, "fuel": fuel_opt},
#             "Direct": {
#                 "distance": round(baseline_distance, 2),
#                 "fuel": round(fuel_base, 2)
#             },
#             "Optimized": {
#                 "distance": round(optimized_distance, 2),
#                 "fuel": round(fuel_opt, 2)
#             },
#         },
#         "timestamp": datetime.utcnow().isoformat(),
#     }


# # -------------------------
# # PDF REPORT
# # -------------------------
# @app.post("/generate-report")
# async def generate_report(data: dict):
#     filename = "Maritime_Optimization_Report.pdf"
#     doc = SimpleDocTemplate(filename, pagesize=letter)
#     styles = getSampleStyleSheet()
#     elements = []

#     elements.append(Paragraph("NaviGreen AI - Optimization Report", styles["Heading1"]))
#     elements.append(Spacer(1, 0.5 * inch))

#     for key, value in data.items():
#         elements.append(Paragraph(f"{key}: {value}", styles["Normal"]))
#         elements.append(Spacer(1, 0.2 * inch))

#     doc.build(elements)

#     return FileResponse(
#         filename,
#         media_type="application/pdf",
#         filename="NaviGreen_AI_Report.pdf"
#     )
# @app.post("/optimize")
# async def optimize(data: OptimizationRequest):

#     lat1 = data.voyage.start_port.lat
#     lon1 = data.voyage.start_port.lon
#     lat2 = data.voyage.end_port.lat
#     lon2 = data.voyage.end_port.lon

#     origin = [lon1, lat1]
#     destination = [lon2, lat2]

#     # 🌍 --- GENERATE 3 ROUTES ---
#     direct = searoute(origin, destination)

#     mid_lat = (lat1 + lat2) / 2
#     mid_lon = (lon1 + lon2) / 2

#     north_wp = [mid_lon, mid_lat + 5]
#     south_wp = [mid_lon, mid_lat - 5]

#     north1 = searoute(origin, north_wp)
#     north2 = searoute(north_wp, destination)

#     south1 = searoute(origin, south_wp)
#     south2 = searoute(south_wp, destination)

#     routes = {
#         "Direct": direct,
#         "North": {
#             "geometry": {
#                 "coordinates": north1["geometry"]["coordinates"] +
#                                north2["geometry"]["coordinates"]
#             },
#             "properties": {
#                 "length": north1["properties"]["length"] +
#                           north2["properties"]["length"]
#             }
#         },
#         "South": {
#             "geometry": {
#                 "coordinates": south1["geometry"]["coordinates"] +
#                                south2["geometry"]["coordinates"]
#             },
#             "properties": {
#                 "length": south1["properties"]["length"] +
#                           south2["properties"]["length"]
#             }
#         }
#     }

#     # 🌦 Weather sampling (use start point for now)
#     wind, current, wave = fetch_weather(lat1, lon1)

#     route_results = {}

#     for name, route in routes.items():
#         distance = float(route["properties"]["length"])
#         fuel, co2, time = predict_fuel(distance, wind, current, wave)

#         route_results[name] = {
#             "distance": round(distance, 2),
#             "fuel": round(fuel, 2),
#             "co2": round(co2, 2),
#             "time": round(time, 2),
#             "coords": route["geometry"]["coordinates"]
#         }

#     # 🏆 Select best route (minimum fuel)
#     best_route_name = min(route_results, key=lambda x: route_results[x]["fuel"])

#     baseline = route_results["Direct"]
#     optimized = route_results[best_route_name]

#     # 📊 Calculations
#     fuel_reduction = max(
#         0.5,
#         ((baseline["fuel"] - optimized["fuel"]) / baseline["fuel"]) * 100
#     )

#     co2_reduction = max(
#         1.0,
#         baseline["co2"] - optimized["co2"]
#     )

#     fuel_cost_savings = max(
#         100.0,
#         (baseline["fuel"] - optimized["fuel"]) * FUEL_PRICE_PER_TON
#     )

#     time_saved_hours = max(
#         0.0,
#         baseline["time"] - optimized["time"]
#     )

#     cii_rating = calculate_cii(
#         optimized["co2"],
#         optimized["distance"]
#     )

#     baseline_route = densify(baseline["coords"])
#     optimized_route = densify(optimized["coords"])

#     return {
#         "selected_route": best_route_name,

#         "baseline_distance_nm": baseline["distance"],
#         "optimized_distance_nm": optimized["distance"],

#         "fuel_reduction_percent": round(fuel_reduction, 2),
#         "co2_reduction_tons": round(co2_reduction, 2),
#         "fuel_cost_savings_usd": round(fuel_cost_savings, 2),
#         "time_saved_hours": round(time_saved_hours, 2),

#         "cii_rating": cii_rating,

#         "baseline_route": baseline_route,
#         "optimized_route": optimized_route,

#         # 🔥 For Chart
#         "route_comparison": {
#             "Direct": {
#                 "distance": baseline["distance"],
#                 "fuel": baseline["fuel"]
#             },
#             "North": {
#                 "distance": route_results["North"]["distance"],
#                 "fuel": route_results["North"]["fuel"]
#             },
#             "South": {
#                 "distance": route_results["South"]["distance"],
#                 "fuel": route_results["South"]["fuel"]
#             }
#         },

#         "timestamp": datetime.utcnow().isoformat(),
#     }

# @app.post("/generate-report")
# async def generate_report(data: dict):
#     filename = "Maritime_Optimization_Report.pdf"
#     doc = SimpleDocTemplate(filename, pagesize=letter)
#     styles = getSampleStyleSheet()
#     elements = []

#     elements.append(Paragraph("NaviGreen AI - Optimization Report", styles["Heading1"]))
#     elements.append(Spacer(1, 0.5 * inch))

#     for key, value in data.items():
#         elements.append(Paragraph(f"{key}: {value}", styles["Normal"]))
#         elements.append(Spacer(1, 0.2 * inch))

#     doc.build(elements)

#     return FileResponse(
#         filename,
#         media_type="application/pdf",
#         filename="NaviGreen_AI_Report.pdf"
#     )
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import enums


@app.post("/generate-report")
async def generate_report(data: dict):

    filename = "NaviGreen_AI_Report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # ===============================
    # TITLE
    # ===============================
    title_style = ParagraphStyle(
        name="TitleStyle",
        parent=styles["Heading1"],
        alignment=enums.TA_CENTER,
        spaceAfter=20
    )

    elements.append(
        Paragraph("NaviGreen AI - Maritime Optimization Report", title_style)
    )
    elements.append(Spacer(1, 0.3 * inch))

    # ===============================
    # EXECUTIVE SUMMARY
    # ===============================
    elements.append(Paragraph("Executive Summary", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    summary_text = f"""
    Selected Optimal Route: <b>{data.get('selected_route', 'N/A')}</b><br/>
    Fuel Reduction: <b>{data.get('fuel_reduction_percent', 0)}%</b><br/>
    CO₂ Reduction: <b>{data.get('co2_reduction_tons', 0)} tons</b><br/>
    Fuel Cost Savings: <b>${data.get('fuel_cost_savings_usd', 0)}</b><br/>
    Time Saved: <b>{data.get('time_saved_hours', 0)} hours</b><br/>
    IMO CII Rating: <b>{data.get('cii_rating', 'N/A')}</b>
    """

    elements.append(Paragraph(summary_text, styles["Normal"]))
    elements.append(Spacer(1, 0.4 * inch))

    # ===============================
    # ROUTE DISTANCE SECTION
    # ===============================
    elements.append(Paragraph("Distance Comparison", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    distance_text = f"""
    Baseline Distance: <b>{data.get('baseline_distance_nm', 0)} nm</b><br/>
    Optimized Distance: <b>{data.get('optimized_distance_nm', 0)} nm</b>
    """

    elements.append(Paragraph(distance_text, styles["Normal"]))
    elements.append(Spacer(1, 0.4 * inch))

    # ===============================
    # ROUTE COMPARISON TABLE
    # ===============================
    elements.append(Paragraph("Route Comparison Analysis", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    route_comparison = data.get("route_comparison", {})

    table_data = [["Route", "Distance (nm)", "Fuel (tons)"]]

    for route, values in route_comparison.items():
        table_data.append([
            route,
            values.get("distance", 0),
            values.get("fuel", 0)
        ])

    table = Table(table_data, hAlign="LEFT")

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("ROWHEIGHT", (0, 0), (-1, -1), 18),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.5 * inch))

    # ===============================
    # FOOTER
    # ===============================
    elements.append(
        Paragraph(
            f"Report Generated On: {data.get('timestamp', '')}",
            styles["Italic"]
        )
    )

    # BUILD PDF
    doc.build(elements)

    return FileResponse(
        filename,
        media_type="application/pdf",
        filename="NaviGreen_AI_Report.pdf"
    )
@app.post("/optimize")
async def optimize(data: OptimizationRequest):

    lat1 = data.voyage.start_port.lat
    lon1 = data.voyage.start_port.lon
    lat2 = data.voyage.end_port.lat
    lon2 = data.voyage.end_port.lon

    origin = [lon1, lat1]
    destination = [lon2, lat2]

    # 🌍 --- GENERATE 3 ROUTES ---
    direct = searoute(origin, destination)

    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2

    north_wp = [mid_lon, mid_lat + 5]
    south_wp = [mid_lon, mid_lat - 5]

    north1 = searoute(origin, north_wp)
    north2 = searoute(north_wp, destination)

    south1 = searoute(origin, south_wp)
    south2 = searoute(south_wp, destination)

    routes = {
        "Direct": direct,
        "North": {
            "geometry": {
                "coordinates": north1["geometry"]["coordinates"] +
                               north2["geometry"]["coordinates"]
            },
            "properties": {
                "length": north1["properties"]["length"] +
                          north2["properties"]["length"]
            }
        },
        "South": {
            "geometry": {
                "coordinates": south1["geometry"]["coordinates"] +
                               south2["geometry"]["coordinates"]
            },
            "properties": {
                "length": south1["properties"]["length"] +
                          south2["properties"]["length"]
            }
        }
    }

    route_results = {}

    # 🔥 WEATHER + FUEL CALCULATION FOR EACH ROUTE
    for name, route in routes.items():

        coords = route["geometry"]["coordinates"]
        distance = float(route["properties"]["length"])

        dense_points = densify(coords)

        weather_samples = []
        winds = []

        sample_step = max(1, len(dense_points) // 15)

        for point in dense_points[::sample_step]:
            wind, current, wave = fetch_weather(point["lat"], point["lon"])
            winds.append(wind)

            weather_samples.append({
                "lat": round(point["lat"], 6),
                "lon": round(point["lon"], 6),
                "wind": round(wind, 2)
            })

        avg_wind = max(5.0, float(np.mean(winds)))  # avoid zero

        fuel, co2, time = predict_fuel(distance, avg_wind, 0.4, 1.0)

        route_results[name] = {
            "distance": round(distance, 2),
            "fuel": round(max(1.0, fuel), 2),
            "co2": round(max(1.0, co2), 2),
            "time": round(max(0.1, time), 2),
            "coords": coords,
            "weather_samples": weather_samples
        }

    # 🏆 Select best route (minimum fuel)
    best_route_name = min(route_results, key=lambda x: route_results[x]["fuel"])

    baseline = route_results["Direct"]
    optimized = route_results[best_route_name]

    # 📊 SAFE CALCULATIONS (NO ZERO VALUES)
    fuel_reduction = max(
        0.5,
        ((baseline["fuel"] - optimized["fuel"]) / baseline["fuel"]) * 100
    )

    co2_reduction = max(
        1.0,
        baseline["co2"] - optimized["co2"]
    )

    fuel_cost_savings = max(
        100.0,
        (baseline["fuel"] - optimized["fuel"]) * FUEL_PRICE_PER_TON
    )

    time_saved_hours = max(
        0.1,
        baseline["time"] - optimized["time"]
    )

    cii_rating = calculate_cii(
        optimized["co2"],
        optimized["distance"]
    )

    baseline_route = densify(baseline["coords"])
    optimized_route = densify(optimized["coords"])

    return {
        "selected_route": best_route_name,

        "baseline_distance_nm": baseline["distance"],
        "optimized_distance_nm": optimized["distance"],

        "fuel_reduction_percent": round(fuel_reduction, 2),
        "co2_reduction_tons": round(co2_reduction, 2),
        "fuel_cost_savings_usd": round(fuel_cost_savings, 2),
        "time_saved_hours": round(time_saved_hours, 2),

        "cii_rating": cii_rating,

        "baseline_route": baseline_route,
        "optimized_route": optimized_route,

        # 🌡 REAL WEATHER DATA FOR HEAT MAP
        "weather_samples": optimized["weather_samples"],

        # 📊 ROUTE COMPARISON
        "route_comparison": {
            "Direct": {
                "distance": baseline["distance"],
                "fuel": baseline["fuel"]
            },
            "North": {
                "distance": route_results["North"]["distance"],
                "fuel": route_results["North"]["fuel"]
            },
            "South": {
                "distance": route_results["South"]["distance"],
                "fuel": route_results["South"]["fuel"]
            }
        },

        "timestamp": datetime.utcnow().isoformat(),
    }

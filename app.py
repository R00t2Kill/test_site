"""
Campus Microgrid Orchestration System (CMOS)
FastAPI Backend with ML Forecasting and Optimization Engine
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pytz

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ==================== UTILITY FUNCTIONS ====================

def get_ist_now():
    """Get current IST time"""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

def get_latest_data():
    """Fetch most recent energy data from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT * FROM energy_data 
        ORDER BY timestamp DESC 
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    
    if row:
        # Use database data but apply simulation state for battery
        now = get_ist_now()
        dt = (now - sim_state.last_update).total_seconds()
        sim_state.update(dt)
        sim_state.last_update = now
        
        return {
            'timestamp': now.isoformat(),
            'solar_kw': row[2] or 0,
            'wind_kw': row[3] or 0,
            'demand_kw': row[4] or 0,
            'battery_soc': sim_state.battery_soc,
            'grid_import_kw': row[6] or 0,
            'grid_export_kw': row[7] or 0
        }
    
    # Return simulated current state if no data
    return simulate_current_state()

def simulate_current_state():
    """Generate realistic current state based on time of day"""
    now = get_ist_now()
    hour = now.hour
    
    # Update simulation state
    dt = (now - sim_state.last_update).total_seconds()
    sim_state.update(dt)
    sim_state.last_update = now
    
    # Solar generation (0 at night, peak at noon)
    if 6 <= hour <= 18:
        solar_factor = np.sin((hour - 6) * np.pi / 12)
        solar_kw = CAMPUS_CONFIG['solar_capacity_kw'] * solar_factor * np.random.uniform(0.85, 0.98)
    else:
        solar_kw = 0
    
    # Wind (higher at night)
    wind_base = 18 if 18 <= hour or hour <= 6 else 10
    wind_kw = wind_base * np.random.uniform(0.85, 1.15)
    
    # Base demand (academic hours peak)
    if 8 <= hour <= 17:
        base_demand_kw = np.random.uniform(220, 280)
    elif 18 <= hour <= 22:
        base_demand_kw = np.random.uniform(160, 190)
    else:
        base_demand_kw = np.random.uniform(90, 120)
    
    # Apply load shedding
    demand_kw = base_demand_kw
    if sim_state.shed_tier4:
        demand_kw -= base_demand_kw * 0.15  # HVAC/lighting ~15%
    if sim_state.shed_tier3:
        demand_kw -= base_demand_kw * 0.10  # Workshops ~10%
    
    # Add scheduled loads
    for load in sim_state.scheduled_loads:
        if load.get('active', False):
            demand_kw += load['power_kw']
    
    # Calculate generation and grid balance
    total_generation = solar_kw + wind_kw
    net = total_generation - demand_kw
    
    # Battery auto-management (if no manual control active)
    if not sim_state.manual_battery_action:
        if net > 5 and sim_state.battery_soc < CAMPUS_CONFIG['battery_max_soc']:
            # Charge battery with surplus
            charge_rate = min(net * 0.7, CAMPUS_CONFIG['battery_max_charge_kw'])
            charge_kwh = (charge_rate * dt / 3600) * CAMPUS_CONFIG['battery_efficiency']
            sim_state.battery_kwh_stored = min(
                sim_state.battery_kwh_stored + charge_kwh,
                CAMPUS_CONFIG['battery_capacity_kwh'] * CAMPUS_CONFIG['battery_max_soc'] / 100
            )
            net -= charge_rate
        elif net < -5 and sim_state.battery_soc > CAMPUS_CONFIG['battery_min_soc']:
            # Discharge battery for deficit
            discharge_rate = min(abs(net) * 0.7, CAMPUS_CONFIG['battery_max_discharge_kw'])
            discharge_kwh = discharge_rate * dt / 3600
            sim_state.battery_kwh_stored = max(
                sim_state.battery_kwh_stored - discharge_kwh,
                CAMPUS_CONFIG['battery_capacity_kwh'] * CAMPUS_CONFIG['battery_min_soc'] / 100
            )
            net += discharge_rate
    
    sim_state.battery_soc = (sim_state.battery_kwh_stored / CAMPUS_CONFIG['battery_capacity_kwh']) * 100
    
    # Grid import/export with limiter
    if sim_state.grid_import_limited:
        grid_import_kw = max(0, min(-net, 100))  # Limited to 100 kW
    else:
        grid_import_kw = max(0, -net)
    
    grid_export_kw = max(0, net)
    
    return {
        'timestamp': now.isoformat(),
        'solar_kw': round(solar_kw, 2),
        'wind_kw': round(wind_kw, 2),
        'demand_kw': round(demand_kw, 2),
        'battery_soc': round(sim_state.battery_soc, 2),
        'battery_kwh_stored': round(sim_state.battery_kwh_stored, 2),
        'grid_import_kw': round(grid_import_kw, 2),
        'grid_export_kw': round(grid_export_kw, 2)
    }

def generate_recommendations(current_state, forecast_df=None):
    """Generate actionable recommendations for operators"""
    recommendations = []
    
    # Battery action
    battery_action = optimizer.calculate_battery_action(current_state, forecast_df)
    if battery_action['action'] != 'idle':
        priority = 'high' if 'critically' in battery_action['reason'].lower() else 'medium'
        recommendations.append({
            'type': 'battery',
            'priority': priority,
            'message': f"{battery_action['action'].title()} battery at {battery_action['power_kw']:.1f} kW - {battery_action['reason']}",
            'timestamp': get_ist_now().isoformat()
        })
    
    # Surplus/deficit alerts
    solar = current_state['solar_kw']
    wind = current_state['wind_kw']
    demand = current_state['demand_kw']
    surplus = solar + wind - demand
    
    if surplus > 30:
        recommendations.append({
            'type': 'scheduling',
            'priority': 'medium',
            'message': f"Large surplus ({surplus:.1f} kW) available - ideal time to schedule workshops or heavy machinery",
            'timestamp': get_ist_now().isoformat()
        })
    
    if current_state['grid_import_kw'] > 100:
        recommendations.append({
            'type': 'load_management',
            'priority': 'high',
            'message': f"High grid import ({current_state['grid_import_kw']:.1f} kW) - consider deferring non-critical loads",
            'timestamp': get_ist_now().isoformat()
        })
    
    # Battery warnings
    if current_state['battery_soc'] < 25:
        recommendations.append({
            'type': 'alert',
            'priority': 'high',
            'message': f"Battery SOC critically low ({current_state['battery_soc']:.1f}%) - ensure backup power available",
            'timestamp': get_ist_now().isoformat()
        })
    
    return recommendations

# ==================== CONFIGURATION ====================

# Campus Configuration
CAMPUS_CONFIG = {
    "name": "Rajasthan Technical University - Main Campus",
    "timezone": "Asia/Kolkata",
    "location": {"lat": 26.9124, "lon": 75.7873},  # Jaipur
    
    # Installed Capacity
    "solar_capacity_kw": 250,
    "wind_capacity_kw": 50,
    "battery_capacity_kwh": 200,
    "battery_max_charge_kw": 50,
    "battery_max_discharge_kw": 50,
    "battery_efficiency": 0.92,
    "battery_min_soc": 20,  # %
    "battery_max_soc": 90,  # %
    
    # Grid Configuration
    "grid_import_limit_kw": 300,
    "grid_export_limit_kw": 100,
    "grid_import_cost": 6.50,  # ₹/kWh
    "grid_export_price": 2.80,  # ₹/kWh
    "demand_charge": 350,  # ₹/kW/month
    
    # Carbon
    "grid_carbon_intensity": 0.82,  # kg CO₂/kWh
    
    # Load Tiers
    "load_tiers": {
        1: {"name": "Critical", "description": "Labs, servers, security", "sheddable": False},
        2: {"name": "Essential", "description": "Hostels, libraries, admin", "sheddable": False},
        3: {"name": "Scheduled", "description": "Workshops, heavy machinery", "sheddable": True},
        4: {"name": "Discretionary", "description": "HVAC, outdoor lighting", "sheddable": True}
    }
}

# ==================== APP INITIALIZATION ====================

app = FastAPI(
    title="Campus Microgrid Orchestration System",
    description="Vendor-neutral energy intelligence platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================== DATABASE SETUP ====================

DB_PATH = "campus_energy.db"

def init_database():
    """Initialize SQLite database with time-series tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Historical energy data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS energy_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            solar_kw REAL,
            wind_kw REAL,
            demand_kw REAL,
            battery_soc REAL,
            grid_import_kw REAL,
            grid_export_kw REAL,
            UNIQUE(timestamp)
        )
    """)
    
    # Predictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            forecast_timestamp TEXT NOT NULL,
            solar_kw_pred REAL,
            wind_kw_pred REAL,
            demand_kw_pred REAL
        )
    """)
    
    # Recommendations log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            type TEXT,
            message TEXT,
            priority TEXT,
            acknowledged INTEGER DEFAULT 0
        )
    """)
    
    # Alerts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level TEXT,
            message TEXT,
            resolved INTEGER DEFAULT 0
        )
    """)
    
    conn.commit()
    conn.close()

init_database()

# ==================== ML MODELS ====================

class EnergyPredictor:
    """ML-based forecasting for solar, wind, and demand"""
    
    def __init__(self):
        self.solar_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.wind_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.demand_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
    
    def prepare_features(self, df):
        """Extract time-based features from datetime"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        features = ['hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        return df[features]
    
    def train(self, df):
        """Train models on historical data"""
        if len(df) < 168:  # Need at least 1 week
            return False
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        self.solar_model.fit(X_scaled, df['solar_kw'])
        self.wind_model.fit(X_scaled, df['wind_kw'])
        self.demand_model.fit(X_scaled, df['demand_kw'])
        
        self.trained = True
        return True
    
    def predict(self, future_timestamps):
        """Predict for future timestamps"""
        if not self.trained:
            return None
        
        df_future = pd.DataFrame({'timestamp': future_timestamps})
        X_future = self.prepare_features(df_future)
        X_future_scaled = self.scaler.transform(X_future)
        
        predictions = {
            'timestamp': future_timestamps,
            'solar_kw': self.solar_model.predict(X_future_scaled).tolist(),
            'wind_kw': self.wind_model.predict(X_future_scaled).tolist(),
            'demand_kw': self.demand_model.predict(X_future_scaled).tolist()
        }
        
        return predictions

predictor = EnergyPredictor()

# ==================== OPTIMIZATION ENGINE ====================

class MicrogridOptimizer:
    """Battery scheduling and load management optimizer"""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_battery_action(self, current_state, forecast_df):
        """
        Determine optimal battery charge/discharge action
        
        Returns: {
            'action': 'charge' | 'discharge' | 'idle',
            'power_kw': float,
            'reason': str
        }
        """
        solar = current_state['solar_kw']
        wind = current_state['wind_kw']
        demand = current_state['demand_kw']
        battery_soc = current_state['battery_soc']
        
        generation = solar + wind
        net = generation - demand
        
        # Check SOC limits
        if battery_soc >= self.config['battery_max_soc']:
            return {'action': 'idle', 'power_kw': 0, 'reason': 'Battery full'}
        
        if battery_soc <= self.config['battery_min_soc']:
            return {'action': 'charge', 'power_kw': min(30, self.config['battery_max_charge_kw']), 
                    'reason': 'Battery critically low'}
        
        # Surplus available - charge battery
        if net > 5:  # More than 5 kW surplus
            charge_power = min(net * 0.8, self.config['battery_max_charge_kw'], 
                             self.config['battery_capacity_kwh'] * (self.config['battery_max_soc'] - battery_soc) / 100)
            return {'action': 'charge', 'power_kw': charge_power, 
                    'reason': f'Surplus {net:.1f} kW available'}
        
        # Deficit - discharge battery
        if net < -5:  # More than 5 kW deficit
            discharge_power = min(abs(net) * 0.8, self.config['battery_max_discharge_kw'],
                                self.config['battery_capacity_kwh'] * (battery_soc - self.config['battery_min_soc']) / 100)
            return {'action': 'discharge', 'power_kw': discharge_power, 
                    'reason': f'Deficit {abs(net):.1f} kW'}
        
        # Look ahead - charge if peak solar coming
        if forecast_df is not None and len(forecast_df) > 0:
            next_3h = forecast_df.head(3)
            avg_surplus = (next_3h['solar_kw'] + next_3h['wind_kw'] - next_3h['demand_kw']).mean()
            if avg_surplus > 20 and battery_soc < 70:
                return {'action': 'charge', 'power_kw': 20, 
                        'reason': 'Preparing for predicted surplus'}
        
        return {'action': 'idle', 'power_kw': 0, 'reason': 'Balanced operation'}
    
    def recommend_load_schedule(self, load_kw, duration_hours, tier, forecast_df):
        """
        Find optimal time window for scheduling a load
        
        Returns: {
            'start_time': datetime,
            'reason': str,
            'estimated_renewable_percent': float
        }
        """
        if forecast_df is None or len(forecast_df) == 0:
            return None
        
        # Calculate renewable availability for each hour
        forecast_df['renewable_kw'] = forecast_df['solar_kw'] + forecast_df['wind_kw']
        forecast_df['surplus'] = forecast_df['renewable_kw'] - forecast_df['demand_kw']
        
        # Find best window
        best_score = -999999
        best_idx = 0
        
        for i in range(len(forecast_df) - duration_hours):
            window = forecast_df.iloc[i:i+duration_hours]
            avg_surplus = window['surplus'].mean()
            min_surplus = window['surplus'].min()
            
            # Score: prioritize average surplus, penalize negative minimums
            score = avg_surplus * 2 + min_surplus
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        start_time = forecast_df.iloc[best_idx]['timestamp']
        window = forecast_df.iloc[best_idx:best_idx+duration_hours]
        renewable_pct = (window['renewable_kw'].sum() / (window['demand_kw'].sum() + load_kw * duration_hours)) * 100
        
        return {
            'start_time': start_time,
            'reason': f'Peak renewable availability ({window["renewable_kw"].mean():.1f} kW avg)',
            'estimated_renewable_percent': min(renewable_pct, 100)
        }
    
    def calculate_metrics(self, current_state):
        """Calculate real-time performance metrics"""
        solar = current_state['solar_kw']
        wind = current_state['wind_kw']
        demand = current_state['demand_kw']
        grid_import = current_state.get('grid_import_kw', 0)
        
        renewable_gen = solar + wind
        renewable_util = (renewable_gen / demand * 100) if demand > 0 else 0
        
        # Carbon savings
        grid_displaced = renewable_gen - grid_import
        carbon_avoided = max(0, grid_displaced) * self.config['grid_carbon_intensity']
        
        # Cost
        import_cost = grid_import * self.config['grid_import_cost']
        
        return {
            'renewable_utilization_percent': min(renewable_util, 100),
            'grid_dependency_percent': (grid_import / demand * 100) if demand > 0 else 0,
            'carbon_avoided_kg': carbon_avoided,
            'hourly_cost_inr': import_cost
        }

optimizer = MicrogridOptimizer(CAMPUS_CONFIG)

# ==================== SIMULATION STATE (IN-MEMORY) ====================

class SimulationState:
    """Maintains live simulation state between requests"""
    def __init__(self):
        self.battery_soc = 50.0  # Starting at 50%
        self.battery_kwh_stored = CAMPUS_CONFIG['battery_capacity_kwh'] * 0.5
        self.manual_battery_action = None  # {'action': 'charge'/'discharge', 'power_kw': float, 'until': timestamp}
        self.scheduled_loads = []  # List of active scheduled loads
        self.last_update = get_ist_now()
        self.shed_tier3 = False
        self.shed_tier4 = False
        self.grid_import_limited = False
    
    def update(self, dt_seconds):
        """Update state based on time elapsed"""
        # Apply manual battery action if active
        if self.manual_battery_action:
            action = self.manual_battery_action
            if datetime.now(pytz.timezone('Asia/Kolkata')) < action['until']:
                if action['action'] == 'charge':
                    # Charge battery
                    charge_kwh = (action['power_kw'] * dt_seconds / 3600) * CAMPUS_CONFIG['battery_efficiency']
                    self.battery_kwh_stored = min(
                        self.battery_kwh_stored + charge_kwh,
                        CAMPUS_CONFIG['battery_capacity_kwh'] * CAMPUS_CONFIG['battery_max_soc'] / 100
                    )
                elif action['action'] == 'discharge':
                    # Discharge battery
                    discharge_kwh = action['power_kw'] * dt_seconds / 3600
                    self.battery_kwh_stored = max(
                        self.battery_kwh_stored - discharge_kwh,
                        CAMPUS_CONFIG['battery_capacity_kwh'] * CAMPUS_CONFIG['battery_min_soc'] / 100
                    )
            else:
                # Action expired
                self.manual_battery_action = None
        
        # Update SOC percentage
        self.battery_soc = (self.battery_kwh_stored / CAMPUS_CONFIG['battery_capacity_kwh']) * 100

# Global simulation state
sim_state = SimulationState()

# ==================== API ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main dashboard"""
    try:
        with open("static/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <body>
                <h1>CMOS Dashboard</h1>
                <p>Static files not found. Please ensure static/index.html exists.</p>
                <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
            </body>
        </html>
        """

@app.get("/api/status")
async def get_status():
    """Get current system status with recommendations"""
    current_state = get_latest_data()
    
    # Get forecast
    forecast_df = None
    if predictor.trained:
        future_times = [get_ist_now() + timedelta(hours=i) for i in range(1, 25)]
        predictions = predictor.predict(future_times)
        if predictions:
            forecast_df = pd.DataFrame(predictions)
    
    # Calculate metrics
    metrics = optimizer.calculate_metrics(current_state)
    
    # Generate recommendations
    recommendations = generate_recommendations(current_state, forecast_df)
    
    # Determine alert level
    if current_state['battery_soc'] < 25 or current_state['grid_import_kw'] > 150:
        alert_level = 'red'
    elif current_state['grid_import_kw'] > 80 or current_state['battery_soc'] < 40:
        alert_level = 'yellow'
    else:
        alert_level = 'green'
    
    # Add manual control status
    manual_status = None
    if sim_state.manual_battery_action:
        remaining = (sim_state.manual_battery_action['until'] - get_ist_now()).total_seconds() / 60
        if remaining > 0:
            manual_status = f"{sim_state.manual_battery_action['action'].title()} active: {sim_state.manual_battery_action['power_kw']:.1f} kW ({remaining:.0f} min remaining)"
    
    return {
        **current_state,
        **metrics,
        'alert_level': alert_level,
        'recommendations': recommendations[:3],  # Top 3
        'battery_capacity_kwh': CAMPUS_CONFIG['battery_capacity_kwh'],
        'manual_control_active': manual_status,
        'load_shedding_active': sim_state.shed_tier3 or sim_state.shed_tier4,
        'active_scheduled_loads': len([l for l in sim_state.scheduled_loads if l.get('active', False)])
    }

@app.get("/api/history")
async def get_history(hours: int = Query(24, description="Hours of historical data")):
    """Get historical energy data"""
    conn = sqlite3.connect(DB_PATH)
    
    cutoff = (get_ist_now() - timedelta(hours=hours)).isoformat()
    
    df = pd.read_sql_query("""
        SELECT timestamp, solar_kw, wind_kw, demand_kw, battery_soc, grid_import_kw, grid_export_kw
        FROM energy_data
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
    """, conn, params=(cutoff,))
    
    conn.close()
    
    if len(df) == 0:
        return {'data': []}
    
    return {'data': df.to_dict(orient='records')}

@app.get("/api/forecast")
async def get_forecast(hours: int = Query(24, description="Hours to forecast")):
    """Get energy generation and demand forecast"""
    if not predictor.trained:
        # Train model if we have data
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM energy_data ORDER BY timestamp DESC LIMIT 720", conn)
        conn.close()
        
        if len(df) >= 168:
            predictor.train(df)
    
    if not predictor.trained:
        return {'error': 'Insufficient historical data for forecasting. Need at least 7 days.'}
    
    future_times = [get_ist_now() + timedelta(hours=i) for i in range(1, hours + 1)]
    predictions = predictor.predict(future_times)
    
    return {'forecast': predictions}

@app.get("/api/recommendations")
async def get_recommendations():
    """Get all active recommendations"""
    current_state = get_latest_data()
    
    # Get forecast for better recommendations
    forecast_df = None
    if predictor.trained:
        future_times = [get_ist_now() + timedelta(hours=i) for i in range(1, 25)]
        predictions = predictor.predict(future_times)
        if predictions:
            forecast_df = pd.DataFrame(predictions)
    
    recommendations = generate_recommendations(current_state, forecast_df)
    
    return {'recommendations': recommendations}

@app.post("/api/schedule_load")
async def schedule_load(request: dict):
    """
    Find optimal time to schedule a load - now actually adds it to simulation
    Request: {
        "load_name": str,
        "power_kw": float,
        "duration_hours": float,
        "tier": int,
        "preferred_time": "morning" | "afternoon" | "evening" | "any"
    }
    """
    if not predictor.trained:
        raise HTTPException(400, "Forecasting not available - insufficient historical data")
    
    # Get forecast
    future_times = [get_ist_now() + timedelta(hours=i) for i in range(1, 49)]
    predictions = predictor.predict(future_times)
    forecast_df = pd.DataFrame(predictions)
    forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])
    
    # Filter by preferred time if specified
    preferred = request.get('preferred_time', 'any')
    if preferred != 'any':
        if preferred == 'morning':
            forecast_df = forecast_df[forecast_df['timestamp'].dt.hour.between(6, 12)]
        elif preferred == 'afternoon':
            forecast_df = forecast_df[forecast_df['timestamp'].dt.hour.between(12, 17)]
        elif preferred == 'evening':
            forecast_df = forecast_df[forecast_df['timestamp'].dt.hour.between(17, 22)]
    
    schedule = optimizer.recommend_load_schedule(
        request['power_kw'],
        request['duration_hours'],
        request['tier'],
        forecast_df
    )
    
    if schedule:
        # Add to simulation state
        load_id = len(sim_state.scheduled_loads) + 1
        start_time = pd.to_datetime(schedule['start_time'])
        end_time = start_time + timedelta(hours=request['duration_hours'])
        
        new_load = {
            'id': load_id,
            'load_name': request['load_name'],
            'power_kw': request['power_kw'],
            'duration_hours': request['duration_hours'],
            'optimal_start_time': schedule['start_time'],
            'end_time': end_time.isoformat(),
            'estimated_renewable_percent': schedule['estimated_renewable_percent'],
            'reason': schedule['reason'],
            'active': False,  # Will activate when time comes
            'tier': request['tier']
        }
        
        sim_state.scheduled_loads.append(new_load)
        
        return {
            **new_load,
            'message': 'Load scheduled successfully and added to simulation'
        }
    else:
        raise HTTPException(400, "Could not determine optimal schedule")

@app.get("/api/reports/carbon")
async def get_carbon_report(period: str = Query("monthly", description="daily, weekly, monthly")):
    """Generate carbon savings report"""
    if period == "daily":
        hours = 24
    elif period == "weekly":
        hours = 168
    else:  # monthly
        hours = 720
    
    conn = sqlite3.connect(DB_PATH)
    cutoff = (get_ist_now() - timedelta(hours=hours)).isoformat()
    
    df = pd.read_sql_query("""
        SELECT timestamp, solar_kw, wind_kw, demand_kw, grid_import_kw
        FROM energy_data
        WHERE timestamp >= ?
    """, conn, params=(cutoff,))
    conn.close()
    
    if len(df) == 0:
        return {'error': 'No data available for period'}
    
    df['renewable_kw'] = df['solar_kw'] + df['wind_kw']
    
    total_generation = df['renewable_kw'].sum()
    total_demand = df['demand_kw'].sum()
    total_grid_import = df['grid_import_kw'].sum()
    
    grid_displaced = total_generation - total_grid_import
    carbon_avoided = grid_displaced * CAMPUS_CONFIG['grid_carbon_intensity']
    renewable_utilization = (total_generation / total_demand * 100) if total_demand > 0 else 0
    
    return {
        'period': period,
        'start_date': df['timestamp'].min(),
        'end_date': df['timestamp'].max(),
        'total_renewable_generation_kwh': round(total_generation, 2),
        'total_demand_kwh': round(total_demand, 2),
        'grid_import_kwh': round(total_grid_import, 2),
        'grid_displaced_kwh': round(grid_displaced, 2),
        'carbon_avoided_kg': round(carbon_avoided, 2),
        'carbon_avoided_tonnes': round(carbon_avoided / 1000, 3),
        'renewable_utilization_percent': round(renewable_utilization, 2)
    }

@app.get("/api/reports/financial")
async def get_financial_report(period: str = Query("monthly", description="daily, weekly, monthly")):
    """Generate financial savings report"""
    if period == "daily":
        hours = 24
    elif period == "weekly":
        hours = 168
    else:  # monthly
        hours = 720
    
    conn = sqlite3.connect(DB_PATH)
    cutoff = (get_ist_now() - timedelta(hours=hours)).isoformat()
    
    df = pd.read_sql_query("""
        SELECT timestamp, solar_kw, wind_kw, grid_import_kw, grid_export_kw
        FROM energy_data
        WHERE timestamp >= ?
    """, conn, params=(cutoff,))
    conn.close()
    
    if len(df) == 0:
        return {'error': 'No data available for period'}
    
    total_grid_import = df['grid_import_kw'].sum()
    total_grid_export = df['grid_export_kw'].sum()
    
    import_cost = total_grid_import * CAMPUS_CONFIG['grid_import_cost']
    export_revenue = total_grid_export * CAMPUS_CONFIG['grid_export_price']
    net_cost = import_cost - export_revenue
    
    # Calculate baseline cost (if 100% grid powered)
    df['renewable_kw'] = df['solar_kw'] + df['wind_kw']
    baseline_cost = (total_grid_import + df['renewable_kw'].sum()) * CAMPUS_CONFIG['grid_import_cost']
    savings = baseline_cost - net_cost
    
    return {
        'period': period,
        'grid_import_kwh': round(total_grid_import, 2),
        'grid_export_kwh': round(total_grid_export, 2),
        'import_cost_inr': round(import_cost, 2),
        'export_revenue_inr': round(export_revenue, 2),
        'net_cost_inr': round(net_cost, 2),
        'baseline_cost_inr': round(baseline_cost, 2),
        'savings_inr': round(savings, 2),
        'savings_percent': round((savings / baseline_cost * 100) if baseline_cost > 0 else 0, 2)
    }

@app.post("/api/import_data")
async def import_data(file: UploadFile = File(...)):
    """Import historical data from CSV"""
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        required_cols = ['timestamp', 'solar_kw', 'wind_kw', 'demand_kw']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(400, f"CSV must contain columns: {required_cols}")
        
        # Insert into database
        conn = sqlite3.connect(DB_PATH)
        
        for _, row in df.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO energy_data 
                (timestamp, solar_kw, wind_kw, demand_kw, battery_soc, grid_import_kw, grid_export_kw)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                row['timestamp'],
                row['solar_kw'],
                row['wind_kw'],
                row['demand_kw'],
                row.get('battery_soc', 50),
                row.get('grid_import_kw', 0),
                row.get('grid_export_kw', 0)
            ))
        
        conn.commit()
        conn.close()
        
        # Retrain model
        conn = sqlite3.connect(DB_PATH)
        df_train = pd.read_sql_query("SELECT * FROM energy_data ORDER BY timestamp DESC LIMIT 1000", conn)
        conn.close()
        
        if len(df_train) >= 168:
            predictor.train(df_train)
        
        return {
            'success': True,
            'rows_imported': len(df),
            'model_trained': predictor.trained
        }
    
    except Exception as e:
        raise HTTPException(400, f"Error importing data: {str(e)}")

@app.get("/api/config")
async def get_config():
    """Get system configuration"""
    return CAMPUS_CONFIG

@app.post("/api/control")
async def manual_control(request: dict):
    """
    Manual control commands - now actually affects simulation
    Request: {
        "action": "charge_battery" | "discharge_battery" | "shed_load" | "auto_mode",
        "value": float | str,
        "duration": int (seconds)
    }
    """
    action = request.get('action', '')
    value = request.get('value', 0)
    duration = request.get('duration', 3600)  # Default 1 hour
    
    now = get_ist_now()
    until = now + timedelta(seconds=duration)
    
    message = ""
    
    if action == 'charge_battery':
        power_kw = float(value)
        sim_state.manual_battery_action = {
            'action': 'charge',
            'power_kw': power_kw,
            'until': until
        }
        message = f"Battery charging at {power_kw} kW for {duration/60:.0f} minutes"
        
    elif action == 'discharge_battery':
        power_kw = float(value)
        sim_state.manual_battery_action = {
            'action': 'discharge',
            'power_kw': power_kw,
            'until': until
        }
        message = f"Battery discharging at {power_kw} kW for {duration/60:.0f} minutes"
        
    elif action == 'auto_mode':
        sim_state.manual_battery_action = None
        message = "Returned to automatic battery management"
        
    elif action == 'shed_load':
        # Parse load shedding settings
        try:
            settings = json.loads(value) if isinstance(value, str) else value
            sim_state.shed_tier4 = settings.get('tier4', False)
            sim_state.shed_tier3 = settings.get('tier3', False)
            sim_state.grid_import_limited = settings.get('gridLimit', False)
            
            shed_items = []
            if sim_state.shed_tier4:
                shed_items.append("Tier 4 (HVAC/Lighting)")
            if sim_state.shed_tier3:
                shed_items.append("Tier 3 (Workshops)")
            if sim_state.grid_import_limited:
                shed_items.append("Grid Import Limited to 100kW")
            
            message = f"Load shedding applied: {', '.join(shed_items)}" if shed_items else "Load shedding cleared"
        except:
            message = "Load shedding configuration updated"
    
    # Log the control action
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO recommendations (timestamp, type, message, priority)
        VALUES (?, ?, ?, ?)
    """, (
        now.isoformat(),
        'manual_control',
        message,
        'high'
    ))
    conn.commit()
    conn.close()
    
    return {
        'success': True,
        'message': message,
        'current_battery_soc': sim_state.battery_soc,
        'current_battery_kwh': sim_state.battery_kwh_stored
    }

@app.get("/api/scheduled_loads")
async def get_scheduled_loads():
    """Get all scheduled loads"""
    # Check and activate loads whose time has come
    now = get_ist_now()
    for load in sim_state.scheduled_loads:
        start = pd.to_datetime(load['optimal_start_time'])
        end = pd.to_datetime(load['end_time'])
        
        if start <= now <= end:
            load['active'] = True
        elif now > end:
            load['active'] = False
            load['completed'] = True
    
    return {
        'scheduled_loads': sim_state.scheduled_loads,
        'active_count': len([l for l in sim_state.scheduled_loads if l.get('active', False)])
    }

@app.delete("/api/scheduled_loads/{load_id}")
async def delete_scheduled_load(load_id: int):
    """Remove a scheduled load"""
    sim_state.scheduled_loads = [l for l in sim_state.scheduled_loads if l['id'] != load_id]
    return {'success': True, 'message': f'Load {load_id} removed'}

@app.get("/api/simulation_state")
async def get_simulation_state():
    """Get detailed simulation state for debugging"""
    return {
        'battery_soc': sim_state.battery_soc,
        'battery_kwh_stored': sim_state.battery_kwh_stored,
        'manual_action_active': sim_state.manual_battery_action is not None,
        'manual_action_details': sim_state.manual_battery_action,
        'shed_tier3': sim_state.shed_tier3,
        'shed_tier4': sim_state.shed_tier4,
        'grid_limited': sim_state.grid_import_limited,
        'scheduled_loads_count': len(sim_state.scheduled_loads),
        'active_loads_count': len([l for l in sim_state.scheduled_loads if l.get('active', False)])
    }

# ==================== MAIN ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Campus Microgrid Orchestration System (CMOS)")
    print("=" * 60)
    print(f"Campus: {CAMPUS_CONFIG['name']}")
    print(f"Solar Capacity: {CAMPUS_CONFIG['solar_capacity_kw']} kW")
    print(f"Wind Capacity: {CAMPUS_CONFIG['wind_capacity_kw']} kW")
    print(f"Battery Storage: {CAMPUS_CONFIG['battery_capacity_kwh']} kWh")
    print("=" * 60)
    print("\nStarting server...")
    print("Dashboard: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
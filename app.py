"""
Campus Microgrid Orchestration System (CMOS) - Solar Only
Enhanced FastAPI Backend with Advanced ML & Optimization
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
        now = get_ist_now()
        dt = (now - sim_state.last_update).total_seconds()
        sim_state.update(dt)
        sim_state.last_update = now
        
        return {
            'timestamp': now.isoformat(),
            'solar_kw': row[2] or 0,
            'demand_kw': row[3] or 0,
            'battery_soc': sim_state.battery_soc,
            'battery_kwh_stored': sim_state.battery_kwh_stored,
            'grid_import_kw': row[5] or 0,
            'grid_export_kw': row[6] or 0
        }
    
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
        demand_kw -= base_demand_kw * 0.15
    if sim_state.shed_tier3:
        demand_kw -= base_demand_kw * 0.10
    
    # Add scheduled loads
    for load in sim_state.scheduled_loads:
        if load.get('active', False):
            demand_kw += load['power_kw']
    
    # Calculate net
    net = solar_kw - demand_kw
    
    # Battery auto-management
    if not sim_state.manual_battery_action:
        if net > 5 and sim_state.battery_soc < CAMPUS_CONFIG['battery_max_soc']:
            charge_rate = min(net * 0.7, CAMPUS_CONFIG['battery_max_charge_kw'])
            charge_kwh = (charge_rate * dt / 3600) * CAMPUS_CONFIG['battery_efficiency']
            sim_state.battery_kwh_stored = min(
                sim_state.battery_kwh_stored + charge_kwh,
                CAMPUS_CONFIG['battery_capacity_kwh'] * CAMPUS_CONFIG['battery_max_soc'] / 100
            )
            net -= charge_rate
        elif net < -5 and sim_state.battery_soc > CAMPUS_CONFIG['battery_min_soc']:
            discharge_rate = min(abs(net) * 0.7, CAMPUS_CONFIG['battery_max_discharge_kw'])
            discharge_kwh = discharge_rate * dt / 3600
            sim_state.battery_kwh_stored = max(
                sim_state.battery_kwh_stored - discharge_kwh,
                CAMPUS_CONFIG['battery_capacity_kwh'] * CAMPUS_CONFIG['battery_min_soc'] / 100
            )
            net += discharge_rate
    
    sim_state.battery_soc = (sim_state.battery_kwh_stored / CAMPUS_CONFIG['battery_capacity_kwh']) * 100
    
    # Grid balance
    grid_import_kw = max(0, -net)
    grid_export_kw = max(0, net)
    
    return {
        'timestamp': now.isoformat(),
        'solar_kw': round(solar_kw, 2),
        'demand_kw': round(demand_kw, 2),
        'battery_soc': round(sim_state.battery_soc, 2),
        'battery_kwh_stored': round(sim_state.battery_kwh_stored, 2),
        'grid_import_kw': round(grid_import_kw, 2),
        'grid_export_kw': round(grid_export_kw, 2)
    }

def generate_recommendations(current_state, forecast_df=None):
    """Generate actionable recommendations"""
    recommendations = []
    
    battery_action = optimizer.calculate_battery_action(current_state, forecast_df)
    if battery_action['action'] != 'idle':
        priority = 'high' if 'critically' in battery_action['reason'].lower() else 'medium'
        recommendations.append({
            'type': 'battery',
            'priority': priority,
            'message': f"{battery_action['action'].title()} battery at {battery_action['power_kw']:.1f} kW - {battery_action['reason']}",
            'timestamp': get_ist_now().isoformat()
        })
    
    solar = current_state['solar_kw']
    demand = current_state['demand_kw']
    surplus = solar - demand
    
    if surplus > 30:
        recommendations.append({
            'type': 'scheduling',
            'priority': 'medium',
            'message': f"Large solar surplus ({surplus:.1f} kW) - Ideal time for heavy loads",
            'timestamp': get_ist_now().isoformat()
        })
    
    if current_state['grid_import_kw'] > 100:
        recommendations.append({
            'type': 'load_management',
            'priority': 'high',
            'message': f"High grid import ({current_state['grid_import_kw']:.1f} kW) - Consider load shedding",
            'timestamp': get_ist_now().isoformat()
        })
    
    if current_state['battery_soc'] < 25:
        recommendations.append({
            'type': 'alert',
            'priority': 'high',
            'message': f"Battery critically low ({current_state['battery_soc']:.1f}%)",
            'timestamp': get_ist_now().isoformat()
        })
    
    return recommendations

# ==================== CONFIGURATION ====================

CAMPUS_CONFIG = {
    "name": "Rajasthan Technical University - Main Campus",
    "timezone": "Asia/Kolkata",
    "location": {"lat": 26.9124, "lon": 75.7873},
    
    # Solar Configuration
    "solar_capacity_kw": 250,
    "solar_panels": 625,  # 400W panels
    "panel_efficiency": 0.21,
    "panel_degradation_yearly": 0.005,
    
    # Battery Configuration
    "battery_capacity_kwh": 200,
    "battery_max_charge_kw": 50,
    "battery_max_discharge_kw": 50,
    "battery_efficiency": 0.92,
    "battery_min_soc": 20,
    "battery_max_soc": 90,
    "battery_cycles": 0,
    "battery_max_cycles": 6000,
    
    # Grid Configuration
    "grid_import_limit_kw": 300,
    "grid_export_limit_kw": 100,
    "grid_import_cost": 6.50,
    "grid_export_price": 2.80,
    "demand_charge": 350,
    
    # Carbon
    "grid_carbon_intensity": 0.82,
    
    # Load Tiers
    "load_tiers": {
        1: {"name": "Critical", "description": "Labs, servers, security", "sheddable": False, "avg_kw": 80},
        2: {"name": "Essential", "description": "Hostels, libraries, admin", "sheddable": False, "avg_kw": 120},
        3: {"name": "Scheduled", "description": "Workshops, heavy machinery", "sheddable": True, "avg_kw": 40},
        4: {"name": "Discretionary", "description": "HVAC, outdoor lighting", "sheddable": True, "avg_kw": 60}
    }
}

# ==================== APP INITIALIZATION ====================

app = FastAPI(
    title="Campus Microgrid Orchestration System",
    description="Solar-powered energy intelligence platform",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================== DATABASE SETUP ====================

DB_PATH = "campus_energy.db"

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS energy_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            solar_kw REAL,
            demand_kw REAL,
            battery_soc REAL,
            grid_import_kw REAL,
            grid_export_kw REAL,
            UNIQUE(timestamp)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            forecast_timestamp TEXT NOT NULL,
            solar_kw_pred REAL,
            demand_kw_pred REAL
        )
    """)
    
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
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level TEXT,
            message TEXT,
            resolved INTEGER DEFAULT 0
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS control_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT,
            details TEXT,
            user TEXT
        )
    """)
    
    conn.commit()
    conn.close()

init_database()

# ==================== ML MODELS ====================

class EnergyPredictor:
    """ML forecasting for solar and demand"""
    
    def __init__(self):
        self.solar_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        self.demand_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
    
    def prepare_features(self, df):
        """Extract time-based features"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        features = ['hour', 'day_of_week', 'month', 'is_weekend', 
                   'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        return df[features]
    
    def train(self, df):
        """Train models on historical data"""
        if len(df) < 168:
            return False
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        self.solar_model.fit(X_scaled, df['solar_kw'])
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
            'demand_kw': self.demand_model.predict(X_future_scaled).tolist()
        }
        
        return predictions

predictor = EnergyPredictor()

# ==================== OPTIMIZATION ENGINE ====================

class MicrogridOptimizer:
    """Battery scheduling and load management"""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_battery_action(self, current_state, forecast_df):
        """Determine optimal battery action"""
        solar = current_state['solar_kw']
        demand = current_state['demand_kw']
        battery_soc = current_state['battery_soc']
        
        net = solar - demand
        
        if battery_soc >= self.config['battery_max_soc']:
            return {'action': 'idle', 'power_kw': 0, 'reason': 'Battery full'}
        
        if battery_soc <= self.config['battery_min_soc']:
            return {'action': 'charge', 'power_kw': min(30, self.config['battery_max_charge_kw']), 
                    'reason': 'Battery critically low'}
        
        if net > 5:
            charge_power = min(net * 0.8, self.config['battery_max_charge_kw'])
            return {'action': 'charge', 'power_kw': charge_power, 
                    'reason': f'Solar surplus {net:.1f} kW available'}
        
        if net < -5:
            discharge_power = min(abs(net) * 0.8, self.config['battery_max_discharge_kw'])
            return {'action': 'discharge', 'power_kw': discharge_power, 
                    'reason': f'Solar deficit {abs(net):.1f} kW'}
        
        if forecast_df is not None and len(forecast_df) > 0:
            next_3h = forecast_df.head(3)
            avg_surplus = (next_3h['solar_kw'] - next_3h['demand_kw']).mean()
            if avg_surplus > 20 and battery_soc < 70:
                return {'action': 'charge', 'power_kw': 20, 
                        'reason': 'Preparing for predicted surplus'}
        
        return {'action': 'idle', 'power_kw': 0, 'reason': 'Balanced operation'}
    
    def recommend_load_schedule(self, load_kw, duration_hours, forecast_df):
        """Find optimal time for load scheduling"""
        if forecast_df is None or len(forecast_df) == 0:
            return None
        
        forecast_df['surplus'] = forecast_df['solar_kw'] - forecast_df['demand_kw']
        
        best_score = -999999
        best_idx = 0
        
        for i in range(len(forecast_df) - duration_hours):
            window = forecast_df.iloc[i:i+duration_hours]
            avg_surplus = window['surplus'].mean()
            min_surplus = window['surplus'].min()
            score = avg_surplus * 2 + min_surplus
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        start_time = forecast_df.iloc[best_idx]['timestamp']
        window = forecast_df.iloc[best_idx:best_idx+duration_hours]
        solar_coverage = (window['solar_kw'].sum() / (window['demand_kw'].sum() + load_kw * duration_hours)) * 100
        
        return {
            'start_time': start_time,
            'reason': f'Peak solar availability ({window["solar_kw"].mean():.1f} kW avg)',
            'solar_coverage_percent': min(solar_coverage, 100)
        }
    
    def calculate_metrics(self, current_state):
        """Calculate performance metrics"""
        solar = current_state['solar_kw']
        demand = current_state['demand_kw']
        grid_import = current_state.get('grid_import_kw', 0)
        
        solar_util = (solar / demand * 100) if demand > 0 else 0
        self_consumption = ((solar - current_state.get('grid_export_kw', 0)) / solar * 100) if solar > 0 else 0
        
        grid_displaced = solar - grid_import
        carbon_avoided = max(0, grid_displaced) * self.config['grid_carbon_intensity']
        
        import_cost = grid_import * self.config['grid_import_cost']
        export_revenue = current_state.get('grid_export_kw', 0) * self.config['grid_export_price']
        
        return {
            'solar_utilization_percent': min(solar_util, 100),
            'self_consumption_percent': min(self_consumption, 100),
            'grid_dependency_percent': (grid_import / demand * 100) if demand > 0 else 0,
            'carbon_avoided_kg': carbon_avoided,
            'hourly_cost_inr': import_cost - export_revenue
        }

optimizer = MicrogridOptimizer(CAMPUS_CONFIG)

# ==================== SIMULATION STATE ====================

class SimulationState:
    """Maintains live simulation state"""
    def __init__(self):
        self.battery_soc = 50.0
        self.battery_kwh_stored = CAMPUS_CONFIG['battery_capacity_kwh'] * 0.5
        self.manual_battery_action = None
        self.scheduled_loads = []
        self.last_update = get_ist_now()
        self.shed_tier3 = False
        self.shed_tier4 = False
    
    def update(self, dt_seconds):
        """Update state based on elapsed time"""
        if self.manual_battery_action:
            action = self.manual_battery_action
            if datetime.now(pytz.timezone('Asia/Kolkata')) < action['until']:
                if action['action'] == 'charge':
                    charge_kwh = (action['power_kw'] * dt_seconds / 3600) * CAMPUS_CONFIG['battery_efficiency']
                    self.battery_kwh_stored = min(
                        self.battery_kwh_stored + charge_kwh,
                        CAMPUS_CONFIG['battery_capacity_kwh'] * CAMPUS_CONFIG['battery_max_soc'] / 100
                    )
                elif action['action'] == 'discharge':
                    discharge_kwh = action['power_kw'] * dt_seconds / 3600
                    self.battery_kwh_stored = max(
                        self.battery_kwh_stored - discharge_kwh,
                        CAMPUS_CONFIG['battery_capacity_kwh'] * CAMPUS_CONFIG['battery_min_soc'] / 100
                    )
            else:
                self.manual_battery_action = None
        
        self.battery_soc = (self.battery_kwh_stored / CAMPUS_CONFIG['battery_capacity_kwh']) * 100

sim_state = SimulationState()

# ==================== API ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main dashboard"""
    try:
        with open("static/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<html><body><h1>CMOS Dashboard</h1><p>Visit <a href='/docs'>/docs</a> for API</p></body></html>"

@app.get("/api/status")
async def get_status():
    """Get current system status"""
    current_state = get_latest_data()
    
    forecast_df = None
    if predictor.trained:
        future_times = [get_ist_now() + timedelta(hours=i) for i in range(1, 25)]
        predictions = predictor.predict(future_times)
        if predictions:
            forecast_df = pd.DataFrame(predictions)
    
    metrics = optimizer.calculate_metrics(current_state)
    recommendations = generate_recommendations(current_state, forecast_df)
    
    if current_state['battery_soc'] < 25 or current_state['grid_import_kw'] > 150:
        alert_level = 'red'
    elif current_state['grid_import_kw'] > 80 or current_state['battery_soc'] < 40:
        alert_level = 'yellow'
    else:
        alert_level = 'green'
    
    manual_status = None
    if sim_state.manual_battery_action:
        remaining = (sim_state.manual_battery_action['until'] - get_ist_now()).total_seconds() / 60
        if remaining > 0:
            manual_status = f"{sim_state.manual_battery_action['action'].title()}: {sim_state.manual_battery_action['power_kw']:.1f} kW ({remaining:.0f}min)"
    
    return {
        **current_state,
        **metrics,
        'alert_level': alert_level,
        'recommendations': recommendations[:3],
        'battery_capacity_kwh': CAMPUS_CONFIG['battery_capacity_kwh'],
        'manual_control_active': manual_status,
        'load_shedding_active': sim_state.shed_tier3 or sim_state.shed_tier4,
        'active_scheduled_loads': len([l for l in sim_state.scheduled_loads if l.get('active', False)])
    }

@app.get("/api/history")
async def get_history(hours: int = Query(24)):
    """Get historical data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Get data from the last X hours
        cutoff = (get_ist_now() - timedelta(hours=hours))
        
        # Query for data
        df = pd.read_sql_query("""
            SELECT timestamp, solar_kw, demand_kw, battery_soc, grid_import_kw, grid_export_kw
            FROM energy_data 
            ORDER BY timestamp ASC
            LIMIT 100
        """, conn)
        
        conn.close()
        
        print(f"History query found {len(df)} records")  # Debug print
        
        # Ensure numeric columns are properly formatted
        numeric_columns = ['solar_kw', 'demand_kw', 'battery_soc', 'grid_import_kw', 'grid_export_kw']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Convert timestamps to ISO format for frontend
        if len(df) > 0 and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Remove any invalid timestamps
            df = df[df['timestamp'].notna()]
            # Convert to ISO format strings
            df['timestamp'] = df['timestamp'].apply(lambda x: x.isoformat() if pd.notna(x) else '')
        
        result = {'data': df.to_dict(orient='records') if len(df) > 0 else []}
        print(f"Returning {len(result['data'])} data points to frontend")  # Debug print
        return result
        
    except Exception as e:
        print(f"Error in history endpoint: {e}")  # Debug print
        return {'data': [], 'error': str(e)}

@app.get("/api/forecast")
async def get_forecast(hours: int = Query(24)):
    """Get forecast"""
    if not predictor.trained:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM energy_data ORDER BY timestamp DESC LIMIT 720", conn)
        conn.close()
        if len(df) >= 168:
            predictor.train(df)
    
    if not predictor.trained:
        return {'error': 'Insufficient data for forecasting'}
    
    future_times = [get_ist_now() + timedelta(hours=i) for i in range(1, hours + 1)]
    predictions = predictor.predict(future_times)
    
    return {'forecast': predictions}

@app.post("/api/schedule_load")
async def schedule_load(request: dict):
    """Schedule load for optimal time"""
    if not predictor.trained:
        raise HTTPException(400, "Forecasting not available")
    
    future_times = [get_ist_now() + timedelta(hours=i) for i in range(1, 49)]
    predictions = predictor.predict(future_times)
    forecast_df = pd.DataFrame(predictions)
    forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])
    
    schedule = optimizer.recommend_load_schedule(
        request['power_kw'],
        request['duration_hours'],
        forecast_df
    )
    
    if schedule:
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
            'solar_coverage_percent': schedule['solar_coverage_percent'],
            'reason': schedule['reason'],
            'active': False,
            'tier': request.get('tier', 3)
        }
        
        sim_state.scheduled_loads.append(new_load)
        return {**new_load, 'message': 'Load scheduled successfully'}
    else:
        raise HTTPException(400, "Could not determine optimal schedule")

@app.get("/api/reports/carbon")
async def get_carbon_report(period: str = Query("monthly")):
    """Generate carbon savings report"""
    hours = {'daily': 24, 'weekly': 168, 'monthly': 720}.get(period, 720)
    
    conn = sqlite3.connect(DB_PATH)
    cutoff = (get_ist_now() - timedelta(hours=hours)).isoformat()
    
    df = pd.read_sql_query("""
        SELECT timestamp, solar_kw, demand_kw, grid_import_kw
        FROM energy_data WHERE timestamp >= ?
    """, conn, params=(cutoff,))
    conn.close()
    
    if len(df) == 0:
        return {'error': 'No data available'}
    
    total_solar = df['solar_kw'].sum()
    total_demand = df['demand_kw'].sum()
    total_grid_import = df['grid_import_kw'].sum()
    
    grid_displaced = total_solar - total_grid_import
    carbon_avoided = grid_displaced * CAMPUS_CONFIG['grid_carbon_intensity']
    solar_utilization = (total_solar / total_demand * 100) if total_demand > 0 else 0
    
    return {
        'period': period,
        'start_date': df['timestamp'].min(),
        'end_date': df['timestamp'].max(),
        'total_solar_generation_kwh': round(total_solar, 2),
        'total_demand_kwh': round(total_demand, 2),
        'grid_import_kwh': round(total_grid_import, 2),
        'grid_displaced_kwh': round(grid_displaced, 2),
        'carbon_avoided_kg': round(carbon_avoided, 2),
        'carbon_avoided_tonnes': round(carbon_avoided / 1000, 3),
        'solar_utilization_percent': round(solar_utilization, 2)
    }

@app.get("/api/reports/financial")
async def get_financial_report(period: str = Query("monthly")):
    """Generate financial report"""
    hours = {'daily': 24, 'weekly': 168, 'monthly': 720}.get(period, 720)
    
    conn = sqlite3.connect(DB_PATH)
    cutoff = (get_ist_now() - timedelta(hours=hours)).isoformat()
    
    df = pd.read_sql_query("""
        SELECT timestamp, solar_kw, grid_import_kw, grid_export_kw
        FROM energy_data WHERE timestamp >= ?
    """, conn, params=(cutoff,))
    conn.close()
    
    if len(df) == 0:
        return {'error': 'No data available'}
    
    total_grid_import = df['grid_import_kw'].sum()
    total_grid_export = df['grid_export_kw'].sum()
    
    import_cost = total_grid_import * CAMPUS_CONFIG['grid_import_cost']
    export_revenue = total_grid_export * CAMPUS_CONFIG['grid_export_price']
    net_cost = import_cost - export_revenue
    
    baseline_cost = (total_grid_import + df['solar_kw'].sum()) * CAMPUS_CONFIG['grid_import_cost']
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
        
        required_cols = ['timestamp', 'solar_kw', 'demand_kw']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(400, f"CSV must contain: {required_cols}")
        
        conn = sqlite3.connect(DB_PATH)
        
        for _, row in df.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO energy_data 
                (timestamp, solar_kw, demand_kw, battery_soc, grid_import_kw, grid_export_kw)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row['timestamp'],
                row['solar_kw'],
                row['demand_kw'],
                row.get('battery_soc', 50),
                row.get('grid_import_kw', 0),
                row.get('grid_export_kw', 0)
            ))
        
        conn.commit()
        conn.close()
        
        conn = sqlite3.connect(DB_PATH)
        df_train = pd.read_sql_query("SELECT * FROM energy_data ORDER BY timestamp DESC LIMIT 1000", conn)
        conn.close()
        
        if len(df_train) >= 168:
            predictor.train(df_train)
        
        return {'success': True, 'rows_imported': len(df), 'model_trained': predictor.trained}
    
    except Exception as e:
        raise HTTPException(400, f"Error: {str(e)}")

@app.post("/api/import_sample_data")
async def import_sample_data():
    """Import sample data from CSV file"""
    try:
        print("Starting sample data import...")  # Debug
        
        # Read the CSV file
        df = pd.read_csv('./data/sample_data.csv')
        print(f"Read {len(df)} rows from CSV")  # Debug
        
        conn = sqlite3.connect(DB_PATH)
        
        # Clear existing data
        conn.execute("DELETE FROM energy_data")
        print("Cleared existing data")  # Debug
        
        # Insert new data - handle timestamp conversion carefully
        inserted_count = 0
        for index, row in df.iterrows():
            try:
                timestamp = row['timestamp']
                
                # Convert timestamp to proper format
                if isinstance(timestamp, str):
                    # If it's already a string, try to parse it
                    timestamp_dt = pd.to_datetime(timestamp)
                    timestamp_iso = timestamp_dt.isoformat()
                else:
                    # If it's already a datetime, convert to ISO
                    timestamp_iso = timestamp.isoformat()
                
                # Insert the record
                conn.execute("""
                    INSERT INTO energy_data 
                    (timestamp, solar_kw, demand_kw, battery_soc, grid_import_kw, grid_export_kw)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    timestamp_iso,
                    float(row.get('solar_kw', 0)),
                    float(row.get('demand_kw', 0)),
                    float(row.get('battery_soc', 50)),
                    float(row.get('grid_import_kw', 0)),
                    float(row.get('grid_export_kw', 0))
                ))
                inserted_count += 1
                
            except Exception as row_error:
                print(f"Error processing row {index}: {row_error}")  # Debug
                continue
        
        conn.commit()
        conn.close()
        print(f"Successfully inserted {inserted_count} records")  # Debug
        
        # Verify the data was inserted
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM energy_data")
        final_count = cursor.fetchone()[0]
        conn.close()
        
        print(f"Final record count in database: {final_count}")  # Debug
        
        return {
            'success': True, 
            'rows_imported': inserted_count,
            'final_count': final_count,
            'message': f'Imported {inserted_count} records successfully'
        }
    
    except Exception as e:
        print(f"Error in import_sample_data: {e}")  # Debug
        raise HTTPException(500, f"Error importing sample data: {str(e)}")

@app.get("/api/check_sample_data")
async def check_sample_data():
    """Check if sample data file exists and can be read"""
    try:
        df = pd.read_csv('./data/sample_data.csv')
        return {
            'file_exists': True,
            'rows': len(df),
            'columns': list(df.columns),
            'first_timestamp': df['timestamp'].iloc[0] if 'timestamp' in df.columns else None,
            'last_timestamp': df['timestamp'].iloc[-1] if 'timestamp' in df.columns else None
        }
    except Exception as e:
        return {'file_exists': False, 'error': str(e)}

@app.get("/api/debug/database")
async def debug_database():
    """Debug endpoint to check database contents"""
    conn = sqlite3.connect(DB_PATH)
    
    # Check if energy_data table exists and has data
    cursor = conn.cursor()
    
    # Check table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='energy_data'")
    table_exists = cursor.fetchone() is not None
    
    if table_exists:
        # Count records
        cursor.execute("SELECT COUNT(*) FROM energy_data")
        count = cursor.fetchone()[0]
        
        # Get sample records
        cursor.execute("SELECT * FROM energy_data LIMIT 5")
        sample_records = cursor.fetchall()
        
        # Get column names
        cursor.execute("PRAGMA table_info(energy_data)")
        columns = [col[1] for col in cursor.fetchall()]
    else:
        count = 0
        sample_records = []
        columns = []
    
    conn.close()
    
    return {
        'table_exists': table_exists,
        'record_count': count,
        'columns': columns,
        'sample_records': sample_records
    }

@app.get("/api/debug/sample_file")
async def debug_sample_file():
    """Check if sample data file exists and can be read"""
    try:
        import os
        file_path = './data/sample_data.csv'
        file_exists = os.path.exists(file_path)
        
        if file_exists:
            df = pd.read_csv(file_path)
            return {
                'file_exists': True,
                'file_path': os.path.abspath(file_path),
                'rows': len(df),
                'columns': list(df.columns),
                'first_row': df.iloc[0].to_dict() if len(df) > 0 else None
            }
        else:
            return {
                'file_exists': False,
                'file_path': os.path.abspath(file_path),
                'error': 'File not found'
            }
    except Exception as e:
        return {
            'file_exists': False,
            'error': str(e)
        }

@app.post("/api/manual_import_test")
async def manual_import_test():
    """Manual test import with hardcoded data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Clear existing data
        conn.execute("DELETE FROM energy_data")
        
        # Insert some test data manually
        test_data = [
            ("2025-01-01T00:00:00", 0.0, 95.3, 45.2, 76.8, 0.0),
            ("2025-01-01T01:00:00", 0.0, 88.7, 43.8, 67.4, 0.0),
            ("2025-01-01T02:00:00", 0.0, 82.1, 42.5, 62.3, 0.0),
            ("2025-01-01T03:00:00", 0.0, 79.4, 41.1, 62.2, 0.0),
            ("2025-01-01T04:00:00", 0.0, 81.8, 40.3, 65.9, 0.0),
            ("2025-01-01T05:00:00", 5.2, 85.1, 39.8, 74.1, 0.0),
            ("2025-01-01T06:00:00", 25.7, 120.4, 42.1, 68.6, 0.0),
            ("2025-01-01T07:00:00", 68.3, 185.2, 45.3, 51.6, 0.0),
            ("2025-01-01T08:00:00", 125.8, 245.7, 52.8, 24.9, 0.0),
            ("2025-01-01T09:00:00", 178.4, 268.9, 61.2, 0.0, 15.3),
        ]
        
        for data in test_data:
            conn.execute("""
                INSERT INTO energy_data 
                (timestamp, solar_kw, demand_kw, battery_soc, grid_import_kw, grid_export_kw)
                VALUES (?, ?, ?, ?, ?, ?)
            """, data)
        
        conn.commit()
        
        # Verify insertion
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM energy_data")
        count = cursor.fetchone()[0]
        
        conn.close()
        
        return {'success': True, 'message': f'Manual test data inserted - {count} records total'}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.get("/api/config")
async def get_config():
    """Get system configuration"""
    return CAMPUS_CONFIG

@app.post("/api/control")
async def manual_control(request: dict):
    """Manual control commands"""
    action = request.get('action', '')
    value = request.get('value', 0)
    duration = request.get('duration', 3600)
    
    now = get_ist_now()
    until = now + timedelta(seconds=duration)
    message = ""
    
    if action == 'charge_battery':
        power_kw = float(value)
        sim_state.manual_battery_action = {'action': 'charge', 'power_kw': power_kw, 'until': until}
        message = f"Battery charging at {power_kw} kW for {duration/60:.0f} minutes"
        
    elif action == 'discharge_battery':
        power_kw = float(value)
        sim_state.manual_battery_action = {'action': 'discharge', 'power_kw': power_kw, 'until': until}
        message = f"Battery discharging at {power_kw} kW for {duration/60:.0f} minutes"
        
    elif action == 'auto_mode':
        sim_state.manual_battery_action = None
        message = "Returned to automatic battery management"
        
    elif action == 'shed_load':
        try:
            settings = json.loads(value) if isinstance(value, str) else value
            sim_state.shed_tier4 = settings.get('tier4', False)
            sim_state.shed_tier3 = settings.get('tier3', False)
            
            shed_items = []
            if sim_state.shed_tier4:
                shed_items.append("Tier 4 (HVAC/Lighting)")
            if sim_state.shed_tier3:
                shed_items.append("Tier 3 (Workshops)")
            
            message = f"Load shedding: {', '.join(shed_items)}" if shed_items else "Load shedding cleared"
        except:
            message = "Load shedding updated"
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO control_log (timestamp, action, details, user)
        VALUES (?, ?, ?, ?)
    """, (now.isoformat(), action, message, 'operator'))
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
    """Remove scheduled load"""
    sim_state.scheduled_loads = [l for l in sim_state.scheduled_loads if l['id'] != load_id]
    return {'success': True, 'message': f'Load {load_id} removed'}

@app.get("/api/analytics/performance")
async def get_performance_analytics():
    """Get detailed performance analytics"""
    conn = sqlite3.connect(DB_PATH)
    
    # Last 24 hours
    df = pd.read_sql_query("""
        SELECT * FROM energy_data 
        ORDER BY timestamp DESC
        LIMIT 24
    """, conn)
    conn.close()
    
    if len(df) == 0:
        return {'error': 'No data available'}
    
    # Calculate metrics
    total_solar = df['solar_kw'].sum()
    total_demand = df['demand_kw'].sum()
    peak_solar = df['solar_kw'].max()
    peak_demand = df['demand_kw'].max()
    avg_battery_soc = df['battery_soc'].mean()
    
    # Solar generation hours
    solar_hours = len(df[df['solar_kw'] > 5])
    
    # Self-sufficiency
    grid_free_hours = len(df[df['grid_import_kw'] < 1])
    
    return {
        'period': '24 hours',
        'total_solar_kwh': round(total_solar, 2),
        'total_demand_kwh': round(total_demand, 2),
        'peak_solar_kw': round(peak_solar, 2),
        'peak_demand_kw': round(peak_demand, 2),
        'avg_battery_soc': round(avg_battery_soc, 2),
        'solar_generation_hours': solar_hours,
        'grid_free_hours': grid_free_hours,
        'self_sufficiency_percent': round((grid_free_hours / len(df) * 100), 2)
    }

@app.get("/api/system/health")
async def get_system_health():
    """Get system health status"""
    current = get_latest_data()
    
    health_issues = []
    
    # Check battery health
    if current['battery_soc'] < 20:
        health_issues.append({'component': 'Battery', 'severity': 'critical', 'message': 'SOC critically low'})
    elif current['battery_soc'] < 30:
        health_issues.append({'component': 'Battery', 'severity': 'warning', 'message': 'SOC low'})
    
    # Check solar performance
    now = get_ist_now()
    if 10 <= now.hour <= 16:  # Peak hours
        expected_solar = CAMPUS_CONFIG['solar_capacity_kw'] * 0.6
        if current['solar_kw'] < expected_solar * 0.5:
            health_issues.append({'component': 'Solar', 'severity': 'warning', 'message': 'Below expected output'})
    
    # Check grid dependency
    if current['grid_import_kw'] > 150:
        health_issues.append({'component': 'Grid', 'severity': 'warning', 'message': 'High grid dependency'})
    
    overall_health = 'healthy' if len(health_issues) == 0 else 'degraded' if all(i['severity'] == 'warning' for i in health_issues) else 'critical'
    
    return {
        'overall_health': overall_health,
        'issues': health_issues,
        'battery_cycles': CAMPUS_CONFIG['battery_cycles'],
        'battery_health_percent': round((1 - CAMPUS_CONFIG['battery_cycles'] / CAMPUS_CONFIG['battery_max_cycles']) * 100, 2),
        'solar_panel_efficiency': round(CAMPUS_CONFIG['panel_efficiency'] * (1 - CAMPUS_CONFIG['panel_degradation_yearly'] * 2), 4)
    }

@app.on_event("startup")
async def auto_import_sample_data():
    """Automatically import sample data if database is empty"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM energy_data")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            print("No data found, attempting to auto-import sample data...")
            # Try to import sample data
            try:
                df = pd.read_csv('./data/sample_data.csv')
                
                conn = sqlite3.connect(DB_PATH)
                for _, row in df.iterrows():
                    timestamp = row['timestamp']
                    # Convert to ISO format if needed
                    if isinstance(timestamp, str) and ' ' in timestamp:
                        timestamp = pd.to_datetime(timestamp).isoformat()
                    
                    conn.execute("""
                        INSERT INTO energy_data 
                        (timestamp, solar_kw, demand_kw, battery_soc, grid_import_kw, grid_export_kw)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp,
                        float(row.get('solar_kw', 0)),
                        float(row.get('demand_kw', 0)),
                        float(row.get('battery_soc', 50)),
                        float(row.get('grid_import_kw', 0)),
                        float(row.get('grid_export_kw', 0))
                    ))
                
                conn.commit()
                conn.close()
                print(f"Auto-imported {len(df)} sample data points")
                
            except Exception as e:
                print(f"Auto-import failed: {e}. You can manually import data later.")
            
    except Exception as e:
        print(f"Auto-import check failed: {e}")

# ==================== MAIN ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Campus Microgrid Orchestration System (CMOS)")
    print("Solar-Only Configuration")
    print("=" * 60)
    print(f"Campus: {CAMPUS_CONFIG['name']}")
    print(f"Solar Capacity: {CAMPUS_CONFIG['solar_capacity_kw']} kW")
    print(f"Solar Panels: {CAMPUS_CONFIG['solar_panels']} x 400W")
    print(f"Battery Storage: {CAMPUS_CONFIG['battery_capacity_kwh']} kWh")
    print("=" * 60)
    print("\nStarting server...")
    print("Dashboard: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
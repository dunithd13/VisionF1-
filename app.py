import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from PIL import Image
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score

# Set page config
st.set_page_config(
    page_title="VisionF1 - F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for F1 theme
st.markdown("""
<style>
    .main {
        background-color: #0a0a0a;
        color: white;
    }
    
    .stApp {
        background-color: #0a0a0a;
    }
    
    .title-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 2px solid #ff0000;
    }
    
    .main-title {
        font-size: 3.8rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .vision-text {
        color: #ffffff;
    }
    
    .f1-text {
        color: #ff0000;
    }
    
    .subtitle {
        font-size: 1.8rem;
        color: #cccccc;
        margin-top: 0.5rem;
    }
    
    .selection-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #444;
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        border: 2px solid #ffffff;
    }
    
    .prediction-title {
        font-size: 2rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #ffffff;
        margin: 0.5rem 0;
    }
    
    .prediction-label {
        font-size: 1.2rem;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .stSelectbox > div > div > div {
        background-color: #2a2a2a;
        color: white;
        border: 1px solid #ff0000;
    }
    
    .stSelectbox > div > div > div > div {
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
        color: white;
        border: 2px solid #ffffff;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #cc0000 0%, #aa0000 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255,0,0,0.3);
    }
    
    .driver-info {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #ff0000;
    }
    
    .race-info {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #ffffff;
    }
    
    .image-container {
        text-align: center;
        margin: 2rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border-radius: 10px;
        border: 1px solid #444;
    }
    
    .warning-text {
        color: #ff6b6b;
        font-style: italic;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data folder path
BASE_PATH = r"C:\Users\dunit\Desktop\VisionF1"
DATA_PATH = os.path.join(BASE_PATH, "data")

# Current 2025 drivers and teams
current_drivers_info = {
    'Pierre Gasly': {'constructor': 'Alpine', 'driverId': -1},
    'Franco Colapinto': {'constructor': 'Alpine', 'driverId': -2},
    'Fernando Alonso': {'constructor': 'Aston Martin', 'driverId': -3},
    'Lance Stroll': {'constructor': 'Aston Martin', 'driverId': -4},
    'Charles Leclerc': {'constructor': 'Ferrari', 'driverId': -5},
    'Lewis Hamilton': {'constructor': 'Ferrari', 'driverId': -6},
    'Oliver Bearman': {'constructor': 'Haas', 'driverId': -7},
    'Esteban Ocon': {'constructor': 'Haas', 'driverId': -8},
    'Oscar Piastri': {'constructor': 'McLaren', 'driverId': -9},
    'Lando Norris': {'constructor': 'McLaren', 'driverId': -10},
    'George Russell': {'constructor': 'Mercedes', 'driverId': -11},
    'Kimi Antonelli': {'constructor': 'Mercedes', 'driverId': -12},
    'Liam Lawson': {'constructor': 'Racing Bulls', 'driverId': -13},
    'Isack Hadjar': {'constructor': 'Racing Bulls', 'driverId': -14},
    'Max Verstappen': {'constructor': 'Red Bull', 'driverId': -15},
    'Yuki Tsunoda': {'constructor': 'Red Bull', 'driverId': -16},
    'Nico Hulkenberg': {'constructor': 'Sauber', 'driverId': -17},
    'Gabriel Bortoleto': {'constructor': 'Sauber', 'driverId': -18},
    'Alex Albon': {'constructor': 'Williams', 'driverId': -19},
    'Carlos Sainz Jr': {'constructor': 'Williams', 'driverId': -20}
}

# 2025 races
races_2025 = {
    'Belgian GP': {'circuitId': 1, 'round': 13, 'date': '2025-07-27'},
    'Hungarian GP': {'circuitId': 2, 'round': 14, 'date': '2025-08-03'},
    'Dutch GP': {'circuitId': 3, 'round': 15, 'date': '2025-08-31'},
    'Italian GP': {'circuitId': 4, 'round': 16, 'date': '2025-09-07'},
    'Azerbaijan GP': {'circuitId': 5, 'round': 17, 'date': '2025-09-21'},
    'Singapore GP': {'circuitId': 6, 'round': 18, 'date': '2025-10-05'},
    'United States GP': {'circuitId': 7, 'round': 19, 'date': '2025-10-19'},
    'Mexican GP': {'circuitId': 8, 'round': 20, 'date': '2025-10-26'},
    'São Paulo GP': {'circuitId': 9, 'round': 21, 'date': '2025-11-09'},
    'Las Vegas GP': {'circuitId': 10, 'round': 22, 'date': '2025-11-22'},
    'Qatar GP': {'circuitId': 11, 'round': 23, 'date': '2025-11-30'},
    'Abu Dhabi GP': {'circuitId': 12, 'round': 24, 'date': '2025-12-07'}
}

@st.cache_data
def load_dataset(filename):
    """Load dataset from data folder"""
    try:
        path = os.path.join(DATA_PATH, filename)
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Error: {filename} not found in {DATA_PATH}")
        return None
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None

@st.cache_data
def load_and_prepare_data():
    """Load and prepare all F1 data"""
    # Load datasets
    results = load_dataset('results.csv')
    races = load_dataset('races.csv')
    drivers = load_dataset('drivers.csv')
    qualifying = load_dataset('qualifying.csv')
    driver_standings = load_dataset('driver_standings.csv')
    constructors = load_dataset('constructors.csv')
    circuits = load_dataset('circuits.csv')
    constructor_standings = load_dataset('constructor_standings.csv')
    
    if any(df is None for df in [results, races, drivers, qualifying, driver_standings, constructors, circuits, constructor_standings]):
        return None, None, None
    
    # Data preprocessing
    races['date'] = pd.to_datetime(races['date'])
    results['position'] = pd.to_numeric(results['position'], errors='coerce')
    results = results.dropna(subset=['position'])
    
    drivers['driver_name'] = drivers['forename'] + ' ' + drivers['surname']
    circuits = circuits.rename(columns={'name': 'circuit_name'})
    constructors = constructors.rename(columns={'name': 'constructor_name'})
    
    # Merge data
    df = merge_f1_data(results, races, drivers, constructors, circuits, qualifying, driver_standings)
    df = df[df['year'] >= 2015]
    
    # Create target variables
    df['podium'] = (df['positionOrder'] <= 3).astype(int)
    df['points_finish'] = (df['positionOrder'] <= 10).astype(int)
    
    # Create features
    df = create_rolling_features(df, 'driverId')
    df = create_rolling_features(df, 'constructorId')
    
    # Circuit features
    circuit_features = df.groupby(['driverId', 'circuitId']).agg(
        circuit_avg_position=('positionOrder', 'mean'),
        circuit_podium_pct=('podium', 'mean'),
        circuit_points_pct=('points_finish', 'mean')
    ).reset_index()
    df = df.merge(circuit_features, on=['driverId', 'circuitId'], how='left')
    
    for col in ['circuit_avg_position', 'circuit_podium_pct', 'circuit_points_pct']:
        df[col] = df.groupby('driverId')[col].transform(lambda x: x.fillna(x.mean()))
    
    # Season features
    df['season_position_avg'] = df.groupby(['year', 'driverId'])['positionOrder'].transform(
        lambda x: x.expanding().mean())
    df['season_podium_pct'] = df.groupby(['year', 'driverId'])['podium'].transform(
        lambda x: x.expanding().mean())
    df['season_points_pct'] = df.groupby(['year', 'driverId'])['points_finish'].transform(
        lambda x: x.expanding().mean())
    
    return df, constructors, circuits

def merge_f1_data(results, races, drivers, constructors, circuits, qualifying, driver_standings):
    """Merge F1 datasets"""
    df = results.merge(races, on='raceId')
    df = df.merge(drivers[['driverId', 'driver_name']], on='driverId')
    df = df.merge(constructors[['constructorId', 'constructor_name']], on='constructorId')
    df = df.merge(circuits[['circuitId', 'circuit_name', 'country']], on='circuitId')
    
    qualifying_latest = qualifying.sort_values(['raceId', 'driverId', 'qualifyId']).drop_duplicates(
        ['raceId', 'driverId'], keep='last')
    df = df.merge(qualifying_latest[['raceId', 'driverId', 'position']].rename(
        columns={'position': 'qualifying_position'}), on=['raceId', 'driverId'], how='left')
    
    df = df.merge(driver_standings.rename(columns={
        'points': 'points_before_race',
        'position': 'standing_position_before_race',
        'wins': 'wins_before_race'
    }), on=['raceId', 'driverId'], how='left')
    
    return df

def create_rolling_features(df, group_col, window=5):
    """Create rolling performance features"""
    df = df.sort_values(['year', 'round'])
    
    df[f'{group_col}_rolling_position'] = df.groupby(group_col)['positionOrder'].transform(
        lambda x: x.rolling(window, min_periods=1).mean())
    
    df[f'{group_col}_rolling_podium_pct'] = df.groupby(group_col)['podium'].transform(
        lambda x: x.rolling(window, min_periods=1).mean())
    
    df[f'{group_col}_rolling_points_pct'] = df.groupby(group_col)['points_finish'].transform(
        lambda x: x.rolling(window, min_periods=1).mean())
    
    return df

def prepare_prediction_data(driver_name, race_name, df, current_drivers_info, races_2025, constructor_name_to_id, X_train_columns):
    """Prepare prediction data for selected driver and race"""
    driver_info = current_drivers_info[driver_name]
    constructor_name = driver_info['constructor']
    constructor_id = constructor_name_to_id.get(constructor_name, -1)
    circuit_id = races_2025[race_name]['circuitId']
    
    # Get driver's historical data
    driver_history = df[df['driver_name'] == driver_name].sort_values(['year', 'round'])
    
    # Default features for new/unknown drivers
    if len(driver_history) == 0:
        features = {
            'grid': 10,
            'qualifying_position': 10,
            'driverId_rolling_position': df['positionOrder'].mean(),
            'driverId_rolling_podium_pct': df['podium'].mean(),
            'driverId_rolling_points_pct': df['points_finish'].mean(),
            'constructorId_rolling_position': df[df['constructorId'] == constructor_id]['positionOrder'].mean() if constructor_id != -1 else df['positionOrder'].mean(),
            'constructorId_rolling_podium_pct': df[df['constructorId'] == constructor_id]['podium'].mean() if constructor_id != -1 else df['podium'].mean(),
            'constructorId_rolling_points_pct': df[df['constructorId'] == constructor_id]['points_finish'].mean() if constructor_id != -1 else df['points_finish'].mean(),
            'circuit_avg_position': df['positionOrder'].mean(),
            'circuit_podium_pct': df['podium'].mean(),
            'circuit_points_pct': df['points_finish'].mean(),
            'season_position_avg': df['positionOrder'].mean(),
            'season_podium_pct': df['podium'].mean(),
            'season_points_pct': df['points_finish'].mean(),
            'points_before_race': 0,
            'standing_position_before_race': 20,
            'wins_before_race': 0,
            'constructorId': constructor_id
        }
    else:
        # Use historical data
        last_race = driver_history.iloc[-1]
        features = {
            'grid': last_race['grid'],
            'qualifying_position': last_race.get('qualifying_position', 10),
            'driverId_rolling_position': last_race['driverId_rolling_position'],
            'driverId_rolling_podium_pct': last_race['driverId_rolling_podium_pct'],
            'driverId_rolling_points_pct': last_race['driverId_rolling_points_pct'],
            'constructorId_rolling_position': last_race['constructorId_rolling_position'],
            'constructorId_rolling_podium_pct': last_race['constructorId_rolling_podium_pct'],
            'constructorId_rolling_points_pct': last_race['constructorId_rolling_points_pct'],
            'season_position_avg': last_race['season_position_avg'],
            'season_podium_pct': last_race['season_podium_pct'],
            'season_points_pct': last_race['season_points_pct'],
            'points_before_race': last_race.get('points_before_race', 0),
            'standing_position_before_race': last_race.get('standing_position_before_race', 20),
            'wins_before_race': last_race.get('wins_before_race', 0),
            'constructorId': constructor_id
        }
        
        # Circuit-specific features
        circuit_history = driver_history[driver_history['circuitId'] == circuit_id]
        if len(circuit_history) > 0:
            features.update({
                'circuit_avg_position': circuit_history['positionOrder'].mean(),
                'circuit_podium_pct': circuit_history['podium'].mean(),
                'circuit_points_pct': circuit_history['points_finish'].mean()
            })
    
    # Create DataFrame
    features_df = pd.DataFrame([features])
    
    # FIXED: Only extract actual constructor ID columns (not rolling features)
    constructor_columns = [col for col in X_train_columns if col.startswith('constructorId_') 
                          and not any(suffix in col for suffix in ['rolling', 'avg', 'pct'])]
    
    # One-hot encode constructorId
    for col in constructor_columns:
        constructor_id_from_col = int(col.split('constructorId_')[1])
        features_df[col] = (features_df['constructorId'] == constructor_id_from_col).astype(int)
    
    features_df = features_df.drop('constructorId', axis=1)
    
    # Ensure all expected columns are present
    for col in X_train_columns:
        if col not in features_df.columns:
            features_df[col] = 0
    
    return features_df[X_train_columns]

@st.cache_resource
def train_models(df, constructors):
    """Train ML models"""
    features = [
        'grid', 'qualifying_position',
        'driverId_rolling_position', 'driverId_rolling_podium_pct', 'driverId_rolling_points_pct',
        'constructorId_rolling_position', 'constructorId_rolling_podium_pct', 'constructorId_rolling_points_pct',
        'circuit_avg_position', 'circuit_podium_pct', 'circuit_points_pct',
        'season_position_avg', 'season_podium_pct', 'season_points_pct',
        'points_before_race', 'standing_position_before_race', 'wins_before_race',
        'constructorId'
    ]
    
    # Prepare data
    X = df[features].copy()
    y_position = df['positionOrder']
    y_points = df['points_finish']
    
    # Encode constructorId
    X = pd.get_dummies(X, columns=['constructorId'], drop_first=True)
    
    # Split data
    train_mask = df['year'] < 2023
    X_train, X_test = X[train_mask], X[~train_mask]
    y_position_train, y_position_test = y_position[train_mask], y_position[~train_mask]
    y_points_train, y_points_test = y_points[train_mask], y_points[~train_mask]
    
    # Train models
    position_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    position_model.fit(X_train, y_position_train)
    
    points_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
    points_model.fit(X_train, y_points_train)
    
    return position_model, points_model, X_train.columns

# Main App
def main():
    # Title
    st.markdown("""
    <div class="title-container">
        <h1 class="main-title">
            <span class="vision-text">Vision</span><span class="f1-text">F1</span>
        </h1>
        <p class="subtitle">Formula 1 Race Predictor ML Project</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and train models
    with st.spinner("Loading F1 data and training models..."):
        df, constructors, circuits = load_and_prepare_data()
        
        if df is None:
            st.error("Failed to load data. Please check your data files.")
            return
        
        position_model, points_model, X_train_columns = train_models(df, constructors)
        constructor_name_to_id = {row['constructor_name']: row['constructorId'] for _, row in constructors.iterrows()}
    
    # Create columns for selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="selection-container">', unsafe_allow_html=True)
        st.markdown("### 🏎️ Select Driver")
        
        # Group drivers by team for better UX
        drivers_by_team = {}
        for driver, info in current_drivers_info.items():
            team = info['constructor']
            if team not in drivers_by_team:
                drivers_by_team[team] = []
            drivers_by_team[team].append(driver)
        
        # Create driver options with team info
        driver_options = []
        for team in sorted(drivers_by_team.keys()):
            for driver in sorted(drivers_by_team[team]):
                driver_options.append(f"{driver} ({team})")
        
        selected_driver_with_team = st.selectbox(
            "Choose a driver:",
            driver_options,
            index=0
        )
        
        selected_driver = selected_driver_with_team.split(' (')[0]
        
        # Display driver info
        driver_info = current_drivers_info[selected_driver]
        st.markdown(f"""
        <div class="driver-info">
            <strong>Team:</strong> {driver_info['constructor']}<br>
            <strong>Driver:</strong> {selected_driver}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="selection-container">', unsafe_allow_html=True)
        st.markdown("### 🏁 Select Race")
        
        race_options = list(races_2025.keys())
        selected_race = st.selectbox(
            "Choose a race:",
            race_options,
            index=0
        )
        
        # Display race info
        race_info = races_2025[selected_race]
        st.markdown(f"""
        <div class="race-info">
            <strong>Race:</strong> {selected_race}<br>
            <strong>Date:</strong> {race_info['date']}<br>
            <strong>Round:</strong> {race_info['round']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔮 Give Predicted Position and Analysis", use_container_width=True):
            with st.spinner("Analyzing data and making predictions..."):
                try:
                    # Prepare prediction data
                    sample_input = prepare_prediction_data(
                        selected_driver, selected_race, df, 
                        current_drivers_info, races_2025, 
                        constructor_name_to_id, X_train_columns
                    )
                    
                    # Make predictions
                    position_pred = position_model.predict(sample_input)[0]
                    points_prob = points_model.predict_proba(sample_input)[0][1]
                    
                    # Display results
                    st.markdown(f"""
                    <div class="prediction-container">
                        <h2 class="prediction-title">🏆 Prediction Results</h2>
                        <div class="prediction-label">Predicted Finishing Position for {selected_driver}</div>
                        <div class="prediction-value">P{round(position_pred)}</div>
                        <div class="prediction-label">Points Probability (Top 10 Finish)</div>
                        <div class="prediction-value">{points_prob:.1%}</div>
                        <div class="prediction-label">📊 Race: {selected_race}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Analysis
                    st.markdown("### 📈 Analysis")
                    
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        st.markdown(f"""
                        **Position Analysis:**
                        - Predicted finish: **P{round(position_pred)}**
                        - Expected performance: {'🔥 Excellent' if position_pred <= 3 else '✅ Strong' if position_pred <= 8 else '⚠️ Challenging' if position_pred <= 15 else '🔴 Difficult'}
                        """)
                    
                    with analysis_col2:
                        st.markdown(f"""
                        **Points Analysis:**
                        - Points probability: **{points_prob:.1%}**
                        - Championship impact: {'🏆 High' if points_prob > 0.7 else '📈 Moderate' if points_prob > 0.4 else '📉 Low'}
                        """)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.error("Please check the debug information below:")
                    st.write(f"Selected driver: {selected_driver}")
                    st.write(f"Selected race: {selected_race}")
                    st.write(f"Constructor mapping: {constructor_name_to_id}")
    
    # Display F1 drivers image
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.markdown("### 🏎️ 2025 Formula 1 Drivers")
    
    try:
        image_path = os.path.join(BASE_PATH, "F1image.jpg")
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption="Formula 1 2025 Season Drivers", use_column_width=True)
        else:
            st.info("F1 drivers image not found. Please ensure F1image.jpg is in the VisionF1 folder.")
    except Exception as e:
        st.warning(f"Could not load driver image: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🏎️ VisionF1 - Powered by Machine Learning | Formula 1 Predictions 2025 by Dunith Desitha Ranawansha</p>
        <p class="warning-text">⚠️ Predictions are based on historical data </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def get_alert_data(engine, lookback_hours=2):
    """Fetch alert data from PostgreSQL with a specified lookback window"""
    query = """
    WITH recent_alerts AS (
        SELECT 
            incident_id,
            app_name,
            policy_name,
            condition_name,
            category,
            entity_type,
            entity_name,
            alert_description,
            is_root_incident,
            alert_start_time,
            priority,
            COALESCE(alert_end_time, CURRENT_TIMESTAMP) as alert_end_time,
            EXTRACT(EPOCH FROM 
                COALESCE(alert_end_time, CURRENT_TIMESTAMP) - alert_start_time
            )/3600.0 as alert_duration_hours,
            CASE 
                WHEN alert_end_time IS NULL 
                THEN EXTRACT(EPOCH FROM CURRENT_TIMESTAMP - alert_start_time)/3600.0 
                ELSE 0 
            END as alert_age_hours,
            datasource,
            environment
        FROM alerts
        WHERE alert_start_time >= NOW() - INTERVAL '%s hours'
        AND alert_start_time <= NOW()
    )
    SELECT * FROM recent_alerts
    ORDER BY alert_start_time DESC;
    """
    
    return pd.read_sql_query(query % lookback_hours, engine)

def calculate_alert_temperature(alert_df, entity_group_cols):
    """Enhanced temperature calculation including duration metrics"""
    def get_temperature_score(group):
        total_alerts = len(group)
        priority_weights = {'critical': 1.0, 'high': 0.7, 'medium': 0.4, 'low': 0.2}
        
        priority_score = group['priority'].map(priority_weights).mean()
        root_incident_ratio = group['is_root_incident'].mean()
        
        # Duration metrics
        avg_duration = group['alert_duration_hours'].mean()
        max_age = group['alert_age_hours'].max()
        duration_score = min((avg_duration / 2.0), 1.0)  # Normalize to 0-1
        age_score = min((max_age / 2.0), 1.0)  # Normalize to 0-1
        
        # Calculate frequency (alerts per hour)
        time_range = (group['alert_start_time'].max() - group['alert_start_time'].min()).total_seconds() / 3600
        frequency = total_alerts / (time_range if time_range > 0 else 1)
        
        # Enhanced temperature calculation
        temperature = (
            (frequency * 20) +  # 20% weight for frequency
            (priority_score * 25) +  # 25% weight for priority
            (root_incident_ratio * 15) +  # 15% weight for root incidents
            (duration_score * 20) +  # 20% weight for duration
            (age_score * 20)  # 20% weight for age
        )
        
        return min(temperature * 100, 100)
    
    temperature_df = alert_df.groupby(entity_group_cols).apply(
        get_temperature_score
    ).reset_index(name='temperature')
    
    temperature_df['temp_category'] = pd.cut(
        temperature_df['temperature'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    return temperature_df

def create_feature_timeseries(alert_df, outage_df, sample_minutes=15):
    """Create time series features leading up to each outage"""
    features_over_time = []
    
    for _, outage in outage_df.iterrows():
        outage_start = pd.to_datetime(outage['start'])
        lookback_start = outage_start - timedelta(hours=2)
        
        # Create time windows
        time_windows = pd.date_range(
            start=lookback_start,
            end=outage_start,
            freq=f'{sample_minutes}T'
        )
        
        for window_end in time_windows:
            window_start = window_end - timedelta(minutes=sample_minutes)
            
            # Filter alerts for this window
            window_alerts = alert_df[
                (alert_df['alert_start_time'] >= window_start) &
                (alert_df['alert_start_time'] < window_end) &
                (alert_df['app_name'] == outage['applid'])
            ]
            
            # Calculate features for this window
            window_features = {
                'outage_id': outage['incident_number'],
                'app_name': outage['applid'],
                'time_to_outage': (outage_start - window_end).total_seconds() / 3600,
                'timestamp': window_end,
                'alert_count': len(window_alerts),
                'unique_conditions': window_alerts['condition_name'].nunique(),
                'avg_duration': window_alerts['alert_duration_hours'].mean(),
                'max_age': window_alerts['alert_age_hours'].max(),
                'critical_alerts': len(window_alerts[window_alerts['priority'] == 'critical']),
                'root_incidents': window_alerts['is_root_incident'].sum()
            }
            
            features_over_time.append(window_features)
    
    return pd.DataFrame(features_over_time)

def visualize_outage_patterns(feature_timeseries, outage_df):
    """Create visualizations for feature patterns leading to outages"""
    def create_outage_plot(outage_id):
        outage_data = feature_timeseries[feature_timeseries['outage_id'] == outage_id]
        
        if outage_data.empty:
            return None
            
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Alert Count', 'Unique Conditions',
                'Average Duration', 'Maximum Age',
                'Critical Alerts', 'Root Incidents'
            )
        )
        
        # Alert Count
        fig.add_trace(
            go.Scatter(x=outage_data['time_to_outage'], y=outage_data['alert_count'],
                      name='Alert Count', mode='lines+markers'),
            row=1, col=1
        )
        
        # Unique Conditions
        fig.add_trace(
            go.Scatter(x=outage_data['time_to_outage'], y=outage_data['unique_conditions'],
                      name='Unique Conditions', mode='lines+markers'),
            row=1, col=2
        )
        
        # Average Duration
        fig.add_trace(
            go.Scatter(x=outage_data['time_to_outage'], y=outage_data['avg_duration'],
                      name='Avg Duration (hours)', mode='lines+markers'),
            row=2, col=1
        )
        
        # Maximum Age
        fig.add_trace(
            go.Scatter(x=outage_data['time_to_outage'], y=outage_data['max_age'],
                      name='Max Age (hours)', mode='lines+markers'),
            row=2, col=2
        )
        
        # Critical Alerts
        fig.add_trace(
            go.Scatter(x=outage_data['time_to_outage'], y=outage_data['critical_alerts'],
                      name='Critical Alerts', mode='lines+markers'),
            row=3, col=1
        )
        
        # Root Incidents
        fig.add_trace(
            go.Scatter(x=outage_data['time_to_outage'], y=outage_data['root_incidents'],
                      name='Root Incidents', mode='lines+markers'),
            row=3, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text=f"Alert Patterns Leading to Outage {outage_id}",
            showlegend=False
        )
        
        # Update x-axes
        for i in range(1, 7):
            fig.update_xaxes(title_text='Hours Before Outage', row=(i-1)//2 + 1, col=(i-1)%2 + 1)
        
        return fig

    return {
        outage_id: create_outage_plot(outage_id)
        for outage_id in outage_df['incident_number'].unique()
    }

def main():
    # Initialize database connection
    engine = create_db_connection()
    
    # Load data
    alert_df = get_alert_data(engine, lookback_hours=2)
    outage_df = load_outage_data('outages.xlsx')
    
    # Calculate temperatures
    temperature_results = analyze_alerts(alert_df, outage_df)
    
    # Create time series features
    feature_timeseries = create_feature_timeseries(alert_df, outage_df, sample_minutes=15)
    
    # Generate visualizations
    outage_visualizations = visualize_outage_patterns(feature_timeseries, outage_df)
    
    return temperature_results, feature_timeseries, outage_visualizations

if __name__ == "__main__":
    temperature_results, feature_timeseries, outage_visualizations = main()

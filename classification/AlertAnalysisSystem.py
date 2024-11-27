import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os
import yaml
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alert_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AlertAnalysisSystem:
    def __init__(self, config_path='config.yaml'):
        """Initialize the analysis system with configuration"""
        self.config = self._load_config(config_path)
        self.engine = self._create_db_connection()
        self.output_dir = Path(self.config.get('output_dir', 'output'))
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise

    def _create_db_connection(self):
        """Create database connection using SQLAlchemy"""
        try:
            db_config = self.config['database']
            connection_string = (
                f"postgresql://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            engine = create_engine(connection_string)
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
            return engine
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise

    def get_alert_data_for_window(self, window_start, window_end):
        """Fetch alert data for a specific time window"""
        try:
            query = """
            WITH window_alerts AS (
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
                WHERE alert_start_time >= :start_time
                AND alert_start_time <= :end_time
            )
            SELECT * FROM window_alerts
            ORDER BY alert_start_time DESC;
            """
            
            df = pd.read_sql_query(
                text(query),
                self.engine,
                params={'start_time': window_start, 'end_time': window_end}
            )
            logger.debug(f"Retrieved {len(df)} alerts for window {window_start} to {window_end}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch alert data: {str(e)}")
            raise

    def calculate_alert_temperature(self, alert_df, entity_group_cols):
        """Calculate temperature scores for alert groups"""
        if alert_df.empty:
            logger.warning("No alerts provided for temperature calculation")
            return pd.DataFrame(columns=entity_group_cols + ['temperature', 'temp_category'])

        try:
            def get_temperature_score(group):
                total_alerts = len(group)
                priority_weights = {'critical': 1.0, 'high': 0.7, 'medium': 0.4, 'low': 0.2}
                
                priority_score = group['priority'].map(
                    lambda x: priority_weights.get(x.lower(), 0.1)
                ).mean()
                
                root_incident_ratio = group['is_root_incident'].mean()
                avg_duration = group['alert_duration_hours'].mean()
                max_age = group['alert_age_hours'].max()
                
                duration_score = min((avg_duration / 2.0), 1.0)
                age_score = min((max_age / 2.0), 1.0)
                
                time_range = (
                    (group['alert_start_time'].max() - group['alert_start_time'].min())
                    .total_seconds() / 3600
                )
                frequency = total_alerts / (time_range if time_range > 0 else 1)
                
                temperature = (
                    (frequency * 20) +
                    (priority_score * 25) +
                    (root_incident_ratio * 15) +
                    (duration_score * 20) +
                    (age_score * 20)
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
        except Exception as e:
            logger.error(f"Temperature calculation failed: {str(e)}")
            raise

    def create_outage_visualization(self, feature_df, outage_row):
        """Create visualization for a single outage"""
        if feature_df.empty:
            logger.warning(f"No features to visualize for outage {outage_row['incident_number']}")
            return None

        try:
            fig = make_subplots(
                rows=4, cols=2,
                subplot_titles=(
                    'Alert Count', 'Unique Conditions',
                    'Average Duration', 'Maximum Age',
                    'Critical Alerts', 'Root Incidents',
                    'Datasource Diversity', 'Category Diversity'
                )
            )

            metrics = [
                ('alert_count', 'Alert Count'),
                ('unique_conditions', 'Unique Conditions'),
                ('avg_duration', 'Avg Duration (hours)'),
                ('max_age', 'Max Age (hours)'),
                ('critical_alerts', 'Critical Alerts'),
                ('root_incidents', 'Root Incidents'),
                ('datasource_diversity', 'Datasource Count'),
                ('category_diversity', 'Category Count')
            ]

            for idx, (metric, name) in enumerate(metrics):
                row = (idx // 2) + 1
                col = (idx % 2) + 1

                fig.add_trace(
                    go.Scatter(
                        x=feature_df['time_to_outage'],
                        y=feature_df[metric],
                        name=name,
                        mode='lines+markers'
                    ),
                    row=row, col=col
                )

            fig.update_layout(
                height=1000,
                title_text=f"Alert Patterns Leading to Outage {outage_row['incident_number']} ({outage_row['applid']})",
                showlegend=False
            )

            for i in range(1, 9):
                fig.update_xaxes(
                    title_text='Hours Before Outage',
                    row=(i-1)//2 + 1,
                    col=(i-1)%2 + 1
                )

            # Save plot
            plot_path = self.output_dir / f"outage_{outage_row['incident_number']}_analysis.html"
            fig.write_html(str(plot_path))
            logger.info(f"Visualization saved to {plot_path}")

            return fig
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            raise

    def analyze_outage(self, outage_row, sample_minutes=15):
        """Analyze alerts for a single outage"""
        logger.info(f"Analyzing outage {outage_row['incident_number']}")
        
        try:
            outage_start = pd.to_datetime(outage_row['start'])
            window_start = outage_start - timedelta(hours=2)
            
            # Get alerts for this outage's time window
            alert_df = self.get_alert_data_for_window(window_start, outage_start)
            
            # Filter for relevant application
            app_alerts = alert_df[alert_df['app_name'] == outage_row['applid']]
            
            if app_alerts.empty:
                logger.warning(f"No alerts found for outage {outage_row['incident_number']}")
                return None, None, None
            
            # Create time windows
            time_windows = pd.date_range(
                start=window_start,
                end=outage_start,
                freq=f'{sample_minutes}T'
            )
            
            # Calculate features for each window
            window_features = []
            for window_end in time_windows:
                window_start_local = window_end - timedelta(minutes=sample_minutes)
                
                window_alerts = app_alerts[
                    (app_alerts['alert_start_time'] >= window_start_local) &
                    (app_alerts['alert_start_time'] < window_end)
                ]
                
                features = {
                    'outage_id': outage_row['incident_number'],
                    'app_name': outage_row['applid'],
                    'time_to_outage': (outage_start - window_end).total_seconds() / 3600,
                    'timestamp': window_end,
                    'alert_count': len(window_alerts),
                    'unique_conditions': window_alerts['condition_name'].nunique(),
                    'avg_duration': window_alerts['alert_duration_hours'].mean(),
                    'max_age': window_alerts['alert_age_hours'].max(),
                    'critical_alerts': len(window_alerts[window_alerts['priority'] == 'critical']),
                    'root_incidents': window_alerts['is_root_incident'].sum(),
                    'datasource_diversity': window_alerts['datasource'].nunique(),
                    'category_diversity': window_alerts['category'].nunique()
                }
                
                window_features.append(features)
            
            feature_df = pd.DataFrame(window_features)
            
            # Calculate temperatures
            temperatures = self.calculate_alert_temperature(
                app_alerts, ['datasource', 'category']
            )
            
            # Create visualization
            fig = self.create_outage_visualization(feature_df, outage_row)
            
            # Save intermediate results
            feature_df.to_csv(
                self.output_dir / f"outage_{outage_row['incident_number']}_features.csv",
                index=False
            )
            temperatures.to_csv(
                self.output_dir / f"outage_{outage_row['incident_number']}_temperatures.csv",
                index=False
            )
            
            return feature_df, temperatures, fig
            
        except Exception as e:
            logger.error(f"Outage analysis failed: {str(e)}")
            raise

    def run_analysis(self, outage_file):
        """Run analysis for all outages"""
        try:
            # Load outage data
            outage_df = pd.read_excel(outage_file)
            logger.info(f"Loaded {len(outage_df)} outages from {outage_file}")
            
            # Validate outage data
            required_columns = ['incident_number', 'applid', 'start']
            missing_columns = [col for col in required_columns if col not in outage_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            results = []
            temperatures = {}
            visualizations = {}
            
            for idx, outage_row in outage_df.iterrows():
                features, temp_scores, viz = self.analyze_outage(outage_row)
                
                if features is not None:
                    results.append(features)
                    temperatures[outage_row['incident_number']] = temp_scores
                    visualizations[outage_row['incident_number']] = viz
            
            # Combine all features
            if results:
                all_features = pd.concat(results, ignore_index=True)
                
                # Save final results
                all_features.to_csv(self.output_dir / 'all_features.csv', index=False)
                
                logger.info("Analysis completed successfully")
                return all_features, temperatures, visualizations
            else:
                logger.warning("No results generated from analysis")
                return None, {}, {}
                
        except Exception as e:
            logger.error(f"Analysis run failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        # Initialize the analysis system
        system = AlertAnalysisSystem()
        
        # Run analysis
        features, temperatures, visualizations = system.run_analysis('outages.xlsx')
        
        logger.info("Analysis completed successfully")
        return features, temperatures, visualizations
        
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    features, temperatures, visualizations = main()

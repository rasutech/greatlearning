import yaml
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import google.generativeai as genai  # For Gemini Pro
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                           roc_curve, auc, precision_recall_curve)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OutagePredictionSystem:
    def __init__(self, config_path: str):
        """Initialize the Outage Prediction System with configuration."""
        self.config = self._load_config(config_path)
        self.db_engine = self._create_db_connection()
        self._setup_gemini()
        self.alert_scores_cache = {}  # Cache for LLM-generated alert scores
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _create_db_connection(self) -> create_engine:
        """Create database connection using SQLAlchemy."""
        db_params = self.config['database']
        connection_string = (
            f"postgresql://{db_params['user']}:{db_params['password']}@"
            f"{db_params['host']}:{db_params['port']}/{db_params['database']}"
        )
        return create_engine(connection_string)

    def _setup_gemini(self):
        """Initialize Gemini Pro API."""
        genai.configure(api_key=self.config['gemini']['api_key'])
        self.model = genai.GenerativeModel('gemini-pro')

    def _get_alert_score(self, condition: str, description: str) -> float:
        """Get alert severity score from Gemini Pro."""
        cache_key = f"{condition}_{description}"
        if cache_key in self.alert_scores_cache:
            return self.alert_scores_cache[cache_key]

        prompt = f"""
        Analyze this alert and assign a severity score from 1-10 (1 being least severe, 10 being most severe):
        Condition: {condition}
        Description: {description}
        
        Consider factors like:
        - System impact
        - Urgency
        - Potential business impact
        - Recovery complexity
        
        Return only the numeric score.
        """
        
        try:
            response = self.model.generate_content(prompt)
            score = float(response.text.strip())
            self.alert_scores_cache[cache_key] = score
            return score
        except Exception as e:
            logger.error(f"Error getting alert score: {e}")
            return 5.0  # Default middle score

    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess alerts and outages data."""
        # Load alerts data
        alerts_query = """
        SELECT * FROM alerts_table
        WHERE alert_start_time >= NOW() - INTERVAL '1 year'
        """
        alerts_df = pd.read_sql(alerts_query, self.db_engine)
        
        # Load outages data
        outages_df = pd.read_excel('outages.xlsx')
        
        # Preprocess alerts
        alerts_df = alerts_df.fillna({
            col: 'NA' for col in alerts_df.select_dtypes(include=['object']).columns
        })
        
        # Normalize datetime fields
        alerts_df['alert_start_time'] = pd.to_datetime(alerts_df['alert_start_time'])
        alerts_df['alert_end_time'] = pd.to_datetime(alerts_df['alert_end_time'])
        outages_df['Start'] = pd.to_datetime(outages_df['Start'])
        outages_df['End'] = pd.to_datetime(outages_df['End'])
        
        return alerts_df, outages_df

    def generate_features(self, alerts_df: pd.DataFrame, outages_df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for the prediction model."""
        features_list = []
        
        # Generate 10-minute intervals for the entire time range
        start_time = min(alerts_df['alert_start_time'].min(), outages_df['Start'].min())
        end_time = max(alerts_df['alert_end_time'].max(), outages_df['End'].max())
        intervals = pd.date_range(start_time, end_time, freq='10T')
        
        for interval_start in intervals:
            interval_end = interval_start + pd.Timedelta(minutes=10)
            
            # Get alerts in this interval
            interval_alerts = alerts_df[
                (alerts_df['alert_start_time'] >= interval_start) &
                (alerts_df['alert_start_time'] < interval_end)
            ]
            
            if len(interval_alerts) == 0:
                continue
                
            # Calculate features
            features = {
                'interval_start': interval_start,
                'alert_temperature': self._calculate_alert_temperature(interval_alerts),
                'alert_density': self._calculate_alert_density(interval_alerts),
                'peak_hour': 1 if self._is_peak_hour(interval_start) else 0,
                'day_of_week': interval_start.dayofweek,
                'alert_duration': self._calculate_weighted_duration(interval_alerts),
                'is_outage': self._check_outage(interval_start, outages_df)
            }
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)

    def _calculate_alert_temperature(self, alerts: pd.DataFrame) -> float:
        """Calculate weighted average alert score."""
        if len(alerts) == 0:
            return 0.0
            
        scores = alerts.apply(
            lambda x: self._get_alert_score(x['condition_name'], x['alert_description']),
            axis=1
        )
        return scores.mean()

    def _calculate_alert_density(self, alerts: pd.DataFrame) -> Dict[str, int]:
        """Calculate alert density by source and type."""
        return len(alerts.groupby(['datasource', 'condition_name', 'policy_name']))

    def _is_peak_hour(self, timestamp: pd.Timestamp) -> bool:
        """Check if time is during peak hours (8 AM - 5 PM EST)."""
        est_time = timestamp.tz_localize('UTC').tz_convert('US/Eastern')
        return 8 <= est_time.hour < 17

    def _calculate_weighted_duration(self, alerts: pd.DataFrame) -> float:
        """Calculate weighted average duration of alerts."""
        durations = []
        for _, alert in alerts.iterrows():
            end_time = alert['alert_end_time'] if pd.notna(alert['alert_end_time']) else pd.Timestamp.now()
            duration = (end_time - alert['alert_start_time']).total_seconds() / 3600  # hours
            durations.append(duration)
        return np.mean(durations) if durations else 0.0

    def _check_outage(self, timestamp: pd.Timestamp, outages_df: pd.DataFrame) -> int:
        """Check if timestamp falls within any outage period."""
        for _, outage in outages_df.iterrows():
            if outage['Start'] <= timestamp <= outage['End']:
                return 1
        return 0

    def train_models(self, features_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Train both versions of the Random Forest model."""
        X = features_df.drop(['interval_start', 'is_outage'], axis=1)
        y = features_df['is_outage']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Version 1: Without SMOTE
        model_v1 = RandomForestClassifier(n_estimators=100, random_state=42)
        model_v1.fit(X_train, y_train)
        metrics_v1 = self._calculate_metrics(model_v1, X_test, y_test, "Version 1")
        
        # Version 2: With SMOTE
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        model_v2 = RandomForestClassifier(n_estimators=100, random_state=42)
        model_v2.fit(X_train_smote, y_train_smote)
        metrics_v2 = self._calculate_metrics(model_v2, X_test, y_test, "Version 2")
        
        # Save models
        self._save_models(model_v1, model_v2)
        
        return metrics_v1, metrics_v2

    def _calculate_metrics(self, model, X_test, y_test, version: str) -> Dict:
        """Calculate and return all model metrics."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate various metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        return {
            'version': version,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'roc_curve': (fpr, tpr, roc_auc)
        }

    def _save_models(self, model_v1, model_v2):
        """Save trained models to disk."""
        with open('model_v1.pkl', 'wb') as f:
            pickle.dump(model_v1, f)
        with open('model_v2.pkl', 'wb') as f:
            pickle.dump(model_v2, f)

    def generate_report(self, metrics_v1: Dict, metrics_v2: Dict):
        """Generate HTML report with all metrics and visualizations."""
        plt.style.use('seaborn')
        
        # Create HTML file
        html_content = """
        <html>
        <head>
            <title>Outage Prediction Model Analysis</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
        <h1>Outage Prediction Model Analysis</h1>
        """
        
        # Add confusion matrices
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Version 1", "Version 2"))
        
        for i, metrics in enumerate([metrics_v1, metrics_v2], 1):
            conf_matrix = metrics['confusion_matrix']
            
            heatmap = go.Heatmap(
                z=conf_matrix,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                colorscale='Viridis'
            )
            
            fig.add_trace(heatmap, row=1, col=i)
            
        fig.update_layout(height=500, title_text="Confusion Matrices")
        html_content += fig.to_html(full_html=False)
        
        # Add ROC curves
        fig = go.Figure()
        for metrics, name in [(metrics_v1, "Version 1"), (metrics_v2, "Version 2")]:
            fpr, tpr, roc_auc = metrics['roc_curve']
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC = {roc_auc:.2f})'))
            
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500
        )
        html_content += fig.to_html(full_html=False)
        
        # Add classification reports
        html_content += """
        <h2>Classification Reports</h2>
        <div style="display: flex; justify-content: space-around;">
        """
        
        for metrics in [metrics_v1, metrics_v2]:
            report = metrics['classification_report']
            html_content += f"""
            <div style="margin: 20px;">
                <h3>{metrics['version']}</h3>
                <table border="1">
                    <tr><th>Metric</th><th>Precision</th><th>Recall</th><th>F1-score</th></tr>
            """
            
            for label in ['0', '1']:
                html_content += f"""
                    <tr>
                        <td>Class {label}</td>
                        <td>{report[label]['precision']:.3f}</td>
                        <td>{report[label]['recall']:.3f}</td>
                        <td>{report[label]['f1-score']:.3f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        html_content += """
        </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open('model_analysis_report.html', 'w') as f:
            f.write(html_content)

    def run_pipeline(self):
        """Run the complete prediction pipeline."""
        logger.info("Starting prediction pipeline...")
        
        # Load and preprocess data
        alerts_df, outages_df = self.load_and_preprocess_data()
        logger.info("Data loaded and preprocessed")
        
        # Generate features
        features_df = self.generate_features(alerts_df, outages_df)
        logger.info("Features generated")
        
        # Train models and get metrics
        metrics_v1, metrics_v2 = self.train_models(features_df)
        logger.info("Models trained")
        
        # Generate report
        self.generate_report(metrics_v1, metrics_v2)
        logger.info("Report generated")
        
        return "Pipeline completed successfully"

if __name__ == "__main__":
    predictor = OutagePredictionSystem('config.yaml')
    predictor.run_pipeline()

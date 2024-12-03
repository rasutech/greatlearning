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

def _check_outage(self, interval_timestamp: pd.Timestamp, outages_df: pd.DataFrame) -> int:
    """
    Determine if a given timestamp falls within any outage period, with robust error handling.
    
    This function implements careful validation and conversion of timestamps before comparing
    them, ensuring that we don't encounter NoneType comparison errors. Each timestamp is 
    verified and standardized before use.
    
    Args:
        interval_timestamp: The timestamp to check for outages
        outages_df: DataFrame containing outage records with 'Start' and 'End' columns
        
    Returns:
        1 if the timestamp falls within an outage period, 0 otherwise
    """
    try:
        # First, validate the input timestamp
        if not isinstance(interval_timestamp, pd.Timestamp):
            logger.error(f"Invalid interval timestamp type: {type(interval_timestamp)}")
            return 0
            
        # Ensure the interval timestamp is timezone-aware
        if interval_timestamp.tz is None:
            interval_timestamp = interval_timestamp.tz_localize('UTC')
            
        # Log the timestamp we're checking
        logger.debug(f"Checking for outages at: {interval_timestamp}")
        
        # Create a standardized timestamp converter
        def standardize_timestamp(ts) -> Optional[pd.Timestamp]:
            """Convert and validate a timestamp value."""
            try:
                if pd.isna(ts):
                    return None
                    
                # Convert to pandas timestamp if it isn't already
                if not isinstance(ts, pd.Timestamp):
                    ts = pd.to_datetime(ts)
                    
                # Add timezone if needed
                if ts.tz is None:
                    ts = ts.tz_localize('UTC')
                else:
                    ts = ts.tz_convert('UTC')
                    
                return ts
                
            except Exception as e:
                logger.error(f"Error converting timestamp {ts}: {str(e)}")
                return None
        
        # Check each outage period with careful validation
        for idx, outage in outages_df.iterrows():
            # Convert and validate both timestamps
            start_time = standardize_timestamp(outage['Start'])
            end_time = standardize_timestamp(outage['End'])
            
            # Skip invalid outage records
            if start_time is None or end_time is None:
                logger.warning(
                    f"Skipping outage record {idx} due to invalid timestamps: "
                    f"Start={outage['Start']}, End={outage['End']}"
                )
                continue
                
            # Verify the outage period is valid (start before end)
            if start_time > end_time:
                logger.warning(
                    f"Skipping outage record {idx} with invalid period: "
                    f"start ({start_time}) is after end ({end_time})"
                )
                continue
                
            # Now we can safely compare timestamps
            if start_time <= interval_timestamp <= end_time:
                logger.debug(
                    f"Found outage at {interval_timestamp} "
                    f"(between {start_time} and {end_time})"
                )
                return 1
        
        # If we get here, no outage was found
        logger.debug(f"No outage found at {interval_timestamp}")
        return 0
        
    except Exception as e:
        logger.error(f"Error in outage checking: {str(e)}")
        return 0

def generate_features(self, alerts_df: pd.DataFrame, outages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features for the prediction model with enhanced error handling and validation.
    
    This function creates time-based features from alerts and outages data, including
    proper handling of the target variable (is_outage).
    
    Args:
        alerts_df: DataFrame containing alert records
        outages_df: DataFrame containing outage records
        
    Returns:
        DataFrame containing generated features and target variable
    """
    try:
        features_list = []
        
        # Get valid time range
        start_time = self._get_valid_start_time(alerts_df, outages_df)
        end_time = self._get_valid_end_time(alerts_df, outages_df)
        
        logger.info(f"Generating features for time range: {start_time} to {end_time}")
        
        # Generate 10-minute intervals
        intervals = pd.date_range(start=start_time, end=end_time, freq='10T', tz='UTC')
        
        total_intervals = len(intervals)
        logger.info(f"Processing {total_intervals} intervals")
        
        for i, interval_start in enumerate(intervals):
            if i % 100 == 0:  # Log progress periodically
                logger.info(f"Processing interval {i+1}/{total_intervals}")
                
            interval_end = interval_start + pd.Timedelta(minutes=10)
            
            # Get alerts in this interval
            interval_alerts = alerts_df[
                (alerts_df['alert_start_time'] >= interval_start) &
                (alerts_df['alert_start_time'] < interval_end)
            ].copy()
            
            # Only create features for intervals with alerts or outages
            is_outage = self._check_outage(interval_start, outages_df)
            
            if len(interval_alerts) > 0 or is_outage:
                try:
                    features = {
                        'interval_start': interval_start,
                        'alert_temperature': self._calculate_alert_temperature(interval_alerts) if len(interval_alerts) > 0 else 0,
                        'alert_density': self._calculate_alert_density(interval_alerts) if len(interval_alerts) > 0 else 0,
                        'peak_hour': 1 if self._is_peak_hour(interval_start) else 0,
                        'day_of_week': interval_start.dayofweek,
                        'alert_duration': self._calculate_weighted_duration(interval_alerts) if len(interval_alerts) > 0 else 0,
                        'is_outage': is_outage  # Explicitly include the target variable
                    }
                    features_list.append(features)
                except Exception as e:
                    logger.error(f"Error calculating features for interval {interval_start}: {str(e)}")
                    continue
        
        if not features_list:
            logger.warning("No features were generated. Check if your data contains valid intervals.")
            return pd.DataFrame()
        
        # Create the features DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Validate the presence of required columns
        required_columns = ['interval_start', 'alert_temperature', 'alert_density', 
                          'peak_hour', 'day_of_week', 'alert_duration', 'is_outage']
        missing_columns = [col for col in required_columns if col not in features_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in features DataFrame: {missing_columns}")
            
        logger.info(f"Successfully generated features for {len(features_df)} intervals")
        logger.info(f"Outage distribution: {features_df['is_outage'].value_counts().to_dict()}")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error in feature generation: {str(e)}")
        raise

def _get_valid_start_time(self, alerts_df: pd.DataFrame, outages_df: pd.DataFrame) -> pd.Timestamp:
    """Helper method to determine valid start time for feature generation."""
    alert_start = alerts_df['alert_start_time'].min()
    outage_start = outages_df['Start'].min()
    current_time = pd.Timestamp.now(tz='UTC')
    
    valid_starts = [ts for ts in [alert_start, outage_start] 
                   if not pd.isna(ts)]
    
    if not valid_starts:
        return current_time - pd.Timedelta(days=1)
    
    start_time = min(valid_starts)
    return start_time if start_time.tz else start_time.tz_localize('UTC')

def _get_valid_end_time(self, alerts_df: pd.DataFrame, outages_df: pd.DataFrame) -> pd.Timestamp:
    """Helper method to determine valid end time for feature generation."""
    alert_end = alerts_df['alert_end_time'].max()
    outage_end = outages_df['End'].max()
    current_time = pd.Timestamp.now(tz='UTC')
    
    valid_ends = [ts for ts in [alert_end, outage_end, current_time] 
                 if not pd.isna(ts)]
    
    end_time = max(valid_ends)
    return end_time if end_time.tz else end_time.tz_localize('UTC')
  
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
    """
    Train both versions of the Random Forest model with proper feature handling.
    
    Args:
        features_df: DataFrame containing the generated features and target variable
        
    Returns:
        Tuple of dictionaries containing metrics for both model versions
    """
    logger.info("Starting model training process")
    
    try:
        # First, let's identify our feature columns and target variable
        # We'll exclude metadata columns and the target variable from our features
        metadata_columns = ['interval_start']  # Columns not used for prediction
        target_column = 'is_outage'
        
        # Get all feature columns by excluding metadata and target
        feature_columns = [col for col in features_df.columns 
                         if col not in metadata_columns + [target_column]]
        
        logger.info(f"Using the following features for training: {feature_columns}")
        
        # Prepare feature matrix X and target vector y
        X = features_df[feature_columns]
        y = features_df[target_column]
        
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Version 1: Without SMOTE
        logger.info("Training Version 1 (without SMOTE)...")
        model_v1 = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        model_v1.fit(X_train, y_train)
        metrics_v1 = self._calculate_metrics(model_v1, X_test, y_test, "Version 1")
        
        # Version 2: With SMOTE
        logger.info("Training Version 2 (with SMOTE)...")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        logger.info(f"SMOTE-resampled training set size: {len(X_train_smote)}")
        
        model_v2 = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model_v2.fit(X_train_smote, y_train_smote)
        metrics_v2 = self._calculate_metrics(model_v2, X_test, y_test, "Version 2")
        
        # Save feature importance information
        self._save_feature_importance(model_v1, feature_columns, 'v1')
        self._save_feature_importance(model_v2, feature_columns, 'v2')
        
        # Save models
        self._save_models(model_v1, model_v2)
        
        logger.info("Model training completed successfully")
        return metrics_v1, metrics_v2
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

def _save_feature_importance(self, model: RandomForestClassifier, 
                           feature_names: List[str], version: str):
    """
    Save feature importance information for analysis.
    
    Args:
        model: Trained RandomForestClassifier model
        feature_names: List of feature names
        version: Model version identifier
    """
    try:
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create a DataFrame with feature importances
        importance_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices]
        })
        
        # Save to CSV
        importance_df.to_csv(f'feature_importance_v{version}.csv', index=False)
        logger.info(f"Saved feature importance information for version {version}")
        
    except Exception as e:
        logger.error(f"Error saving feature importance: {str(e)}")

def _save_models(self, model_v1: RandomForestClassifier, 
                model_v2: RandomForestClassifier):
    """
    Save trained models to disk with proper error handling.
    
    Args:
        model_v1: First version of the trained model
        model_v2: Second version of the trained model (with SMOTE)
    """
    try:
        # Save models with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(f'model_v1_{timestamp}.pkl', 'wb') as f:
            pickle.dump(model_v1, f)
        
        with open(f'model_v2_{timestamp}.pkl', 'wb') as f:
            pickle.dump(model_v2, f)
            
        logger.info(f"Models saved successfully with timestamp {timestamp}")
        
    except Exception as e:
        logger.error(f"Error saving models: {str(e)}")
        raise

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

# outage_prediction.py

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import configparser
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from alert_intelligence import AlertIntelligence
import warnings
warnings.filterwarnings('ignore')

class OutagePredictionSystem:
    """
    Main class for predicting system outages based on alert patterns
    """
    def __init__(self, config_file='out_pred.ini'):
        """
        Initialize the prediction system
        
        Parameters:
        config_file (str): Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.setup_logging()
        self.engine = self._create_db_connection()
        
    def _load_config(self, config_file):
        """Load configuration from INI file"""
        config = configparser.ConfigParser()
        config.read(config_file)
        return config
        
    def setup_logging(self):
        """Configure logging based on settings"""
        logging.basicConfig(
            filename=self.config['Logging']['log_file'],
            level=getattr(logging, self.config['Logging']['level']),
            format=self.config['Logging']['format']
        )
        
    def _create_db_connection(self):
        """Create database connection using SQLAlchemy"""
        db_params = self.config['Database']
        connection_string = (
            f"postgresql://{db_params['user']}:{db_params['password']}"
            f"@{db_params['host']}:{db_params['port']}/{db_params['database']}"
        )
        return create_engine(connection_string)
    
    def load_data(self, app_names):
        """
        Load alert and outage data for specified applications
        
        Parameters:
        app_names (str): Comma-separated list of application names
        """
        logging.info(f"Loading data for applications: {app_names}")
        
        # Convert comma-separated string to list and clean
        app_list = [app.strip() for app in app_names.split(',')]
        
        # Load alerts from database
        query = """
            SELECT *
            FROM alerts
            WHERE app_name = ANY(:app_list)
            AND alert_start_time >= NOW() - INTERVAL '90 days'
        """
        self.alerts_df = pd.read_sql(
            query, 
            self.engine,
            params={'app_list': app_list}
        )
        
        # Load outages from Excel
        self.outages_df = pd.read_excel(
            self.config['Files']['outage_file'],
            parse_dates=['Start', 'End']
        )
        self.outages_df = self.outages_df[
            self.outages_df['app_name'].isin(app_list)
        ]
        
        logging.info(f"Loaded {len(self.alerts_df)} alerts and {len(self.outages_df)} outages")
    
    def preprocess_data(self):
        """Prepare data for feature engineering"""
        # Initialize alert intelligence
        self.alert_intelligence = AlertIntelligence(
            self.alerts_df,
            self.outages_df
        )
        
        # Calculate alert weights
        self.alert_weights = self.alert_intelligence.calculate_alert_weights()
        
        # Analyze duration patterns
        self.duration_patterns = self.alert_intelligence.analyze_duration_patterns()
        
        logging.info("Data preprocessing completed")
    
    def create_window_features(self, window_alerts, is_outage=False):
        """
        Create features for a specific time window
        
        Parameters:
        window_alerts (pd.DataFrame): Alerts within the time window
        is_outage (bool): Whether this window precedes an outage
        
        Returns:
        dict: Feature dictionary for the window
        """
        features = {
            'total_alerts': len(window_alerts),
            'unique_policies': window_alerts['policy_name'].nunique(),
            'unique_conditions': window_alerts['condition_name'].nunique(),
            'critical_alerts': len(window_alerts[window_alerts['priority'] == 'critical']),
            'is_outage': int(is_outage)
        }
        
        # Add weighted features
        weighted_features = {
            'cluster_weighted_sum': 0,
            'source_weighted_sum': 0,
            'category_weighted_sum': 0,
            'policy_weighted_sum': 0
        }
        
        for _, alert in window_alerts.iterrows():
            # Add cluster weights
            weighted_features['cluster_weighted_sum'] += self.alert_weights['cluster'].get(
                alert['alert_cluster'], 0
            )
            
            # Add source weights
            weighted_features['source_weighted_sum'] += self.alert_weights['source'].get(
                alert['datasource'], 0
            )
            
            # Add category weights
            weighted_features['category_weighted_sum'] += self.alert_weights['category'].get(
                alert['category'], 0
            )
            
            # Add policy weights
            weighted_features['policy_weighted_sum'] += self.alert_weights['policy'].get(
                alert['policy_name'], 0
            )
        
        features.update(weighted_features)
        
        # Add duration-based features
        duration_features = {
            'mean_alert_duration': window_alerts['duration'].mean(),
            'max_alert_duration': window_alerts['duration'].max(),
            'std_alert_duration': window_alerts['duration'].std(),
            'total_alert_duration': window_alerts['duration'].sum()
        }
        
        # Add weighted duration sum using significance scores
        weighted_duration_sum = sum(
            alert['duration'] * self.duration_patterns.get(
                alert['alert_cluster'], {}
            ).get('significance_score', 0)
            for _, alert in window_alerts.iterrows()
        )
        duration_features['weighted_duration_sum'] = weighted_duration_sum
        
        features.update(duration_features)
        
        # Add category-specific features
        for category in self.alerts_df['category'].unique():
            cat_alerts = window_alerts[window_alerts['category'] == category]
            features[f'{category.lower()}_alerts'] = len(cat_alerts)
            features[f'{category.lower()}_weighted_sum'] = sum(
                self.alert_weights['category'].get(category, 0)
                for _ in range(len(cat_alerts))
            )
        
        return features
    
    def feature_engineering(self):
        """Generate features for model training"""
        logging.info("Starting feature engineering")
        features_list = []
        
        # Create features for outage windows
        for _, outage in self.outages_df.iterrows():
            window_start = outage['Start'] - timedelta(hours=8)
            window_alerts = self.alerts_df[
                (self.alerts_df['alert_start_time'] >= window_start) &
                (self.alerts_df['alert_start_time'] <= outage['Start']) &
                (self.alerts_df['app_name'] == outage['app_name'])
            ]
            
            features = self.create_window_features(window_alerts, is_outage=True)
            features['app_name'] = outage['app_name']
            features['window_end'] = outage['Start']
            features_list.append(features)
        
        # Create features for non-outage windows
        time_windows = pd.date_range(
            start=self.alerts_df['alert_start_time'].min(),
            end=self.alerts_df['alert_start_time'].max(),
            freq='8H'
        )
        
        for app_name in self.outages_df['app_name'].unique():
            for window_end in time_windows:
                # Skip if window overlaps with any outage
                if not any(
                    (window_end >= outage['Start']) & 
                    (window_end <= outage['End'])
                    for _, outage in self.outages_df.iterrows()
                ):
                    window_start = window_end - timedelta(hours=8)
                    window_alerts = self.alerts_df[
                        (self.alerts_df['alert_start_time'] >= window_start) &
                        (self.alerts_df['alert_start_time'] <= window_end) &
                        (self.alerts_df['app_name'] == app_name)
                    ]
                    
                    features = self.create_window_features(window_alerts, is_outage=False)
                    features['app_name'] = app_name
                    features['window_end'] = window_end
                    features_list.append(features)
        
        self.features_df = pd.DataFrame(features_list)
        logging.info(f"Generated {len(self.features_df)} feature records")
        return self.features_df
    
    def train_and_evaluate(self):
        """Train models and evaluate their performance"""
        logging.info("Starting model training and evaluation")
        
        # Prepare data for training
        X = self.features_df.drop(['is_outage', 'app_name', 'window_end'], axis=1)
        y = self.features_df['is_outage']
        
        # Handle missing values
        X = X.fillna(0)
        
        # Handle imbalanced dataset
        smote = SMOTE(random_state=int(self.config['Model']['random_state']))
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled,
            y_resampled,
            test_size=float(self.config['Model']['test_size']),
            random_state=int(self.config['Model']['random_state'])
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=int(self.config['Model']['n_estimators']),
                random_state=int(self.config['Model']['random_state'])
            ),
            'SVM': SVC(
                probability=True,
                random_state=int(self.config['Model']['random_state'])
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=int(self.config['Model']['random_state'])
            )
        }
        
        # Train and evaluate each model
        results = []
        feature_importance_data = {}
        
        for name, model in models.items():
            logging.info(f"Training {name} model")
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'ROC AUC': roc_auc_score(y_test, y_prob)
            }
            results.append(metrics)
            
            # Store feature importance for applicable models
            if hasattr(model, 'feature_importances_'):
                feature_importance_data[name] = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        self.generate_report(results, feature_importance_data)
        logging.info("Model training and evaluation completed")
    
    def generate_report(self, results, feature_importance_data):
        """Generate HTML report with visualizations"""
        logging.info("Generating evaluation report")
        
        html_content = """
        <html>
        <head>
            <title>Outage Prediction Model Evaluation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-table { border-collapse: collapse; margin: 20px 0; }
                .metric-table td, .metric-table th { 
                    border: 1px solid #ddd; padding: 8px; 
                }
                .metric-table th { background-color: #f2f2f2; }
                .plot { margin: 20px 0; }
            </style>
        </head>
        <body>
        """
        
        # Add metrics table
        metrics_df = pd.DataFrame(results)
        html_content += "<h2>Model Performance Metrics</h2>"
        html_content += metrics_df.to_html(classes='metric-table', index=False)
        
        # Generate and save plots
        self._generate_performance_plots(metrics_df, feature_importance_data)
        
        # Add plots to HTML
        html_content += """
        <div class='plot'>
            <h2>Model Performance Comparison</h2>
            <img src='model_comparison.png' />
        </div>
        """
        
        for model_name in feature_importance_data.keys():
            html_content += f"""
            <div class='plot'>
                <h2>Feature Importance - {model_name}</h2>
                <img src='feature_importance_{model_name.lower().replace(" ", "_")}.png' />
            </div>
            """
        
        html_content += "</body></html>"
        
        # Save HTML report
        with open(self.config['Files']['output_file'], 'w') as f:
            f.write(html_content)
        
        logging.info(f"Report generated: {self.config['Files']['output_file']}")
    
    def _generate_performance_plots(self, metrics_df, feature_importance_data):
        """Generate performance visualization plots"""
        # Model comparison plot
        plt.figure(figsize=(12, 6))
        metrics_df.plot(
            x='Model',
            y=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            kind='bar',
            width=0.8
        )
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        
        # Feature importance plots
        for model_name, importance_df in feature_importance_data.items():
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=importance_df.head(15),
                x='importance',
                y='feature'
            )
            plt.title(f'Top 15 Important Features - {model_name}')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png')

# Example usage
if __name__ == "__main__":
    # Initialize prediction system
    predictor = OutagePredictionSystem()
    
    # Load and process data
    app_names = "

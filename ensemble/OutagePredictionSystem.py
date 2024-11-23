# outage_prediction.py

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import configparser
import logging
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, auc, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from alert_intelligence import AlertIntelligence
import warnings
warnings.filterwarnings('ignore')

class OutagePredictionSystem:
    """Main class for predicting system outages based on alert patterns"""
    
    def __init__(self, config_file='out_pred.ini'):
        """Initialize the prediction system"""
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
        """Load alert and outage data for specified applications"""
        logging.info(f"Loading data for applications: {app_names}")
        
        # Convert comma-separated string to list and clean
        app_list = [app.strip() for app in app_names.split(',')]
        
        try:
            # Load alerts using direct string formatting
            app_names_quoted = "','".join(app_list)
            query = f"""
                SELECT *
                FROM alerts
                WHERE app_name IN ('{app_names_quoted}')
                AND alert_start_time >= CURRENT_TIMESTAMP - INTERVAL '90 days'
            """
            
            self.alerts_df = pd.read_sql(query, self.engine)
            logging.info(f"Successfully loaded {len(self.alerts_df)} alerts from database")
            
            # Load outages from Excel
            self.outages_df = pd.read_excel(
                self.config['Files']['outage_file'],
                parse_dates=['Start', 'End']
            )
            self.outages_df = self.outages_df[
                self.outages_df['app_name'].isin(app_list)
            ]
            
            logging.info(f"Successfully loaded {len(self.outages_df)} outages from Excel")
            
            if len(self.alerts_df) == 0:
                logging.warning("No alerts found for the specified applications")
            if len(self.outages_df) == 0:
                logging.warning("No outages found for the specified applications")
                
        except Exception as e:
            logging.error(f"Error in data loading process: {str(e)}")
            raise
    
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
        
        # Get processed alerts DataFrame
        self.alerts_df = self.alert_intelligence.get_processed_alerts()
        
        logging.info("Data preprocessing completed")
    
    def create_window_features(self, window_alerts, is_outage=False):
        """Create features for a specific time window"""
        try:
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
                    alert.get('alert_cluster', -1), 0
                )
                
                # Add source weights
                weighted_features['source_weighted_sum'] += self.alert_weights['source'].get(
                    alert.get('datasource', 'unknown'), 0
                )
                
                # Add category weights
                weighted_features['category_weighted_sum'] += self.alert_weights['category'].get(
                    alert.get('category', 'unknown'), 0
                )
                
                # Add policy weights
                weighted_features['policy_weighted_sum'] += self.alert_weights['policy'].get(
                    alert.get('policy_name', 'unknown'), 0
                )
            
            features.update(weighted_features)
            
            # Add duration-based features
            duration_features = {
                'mean_alert_duration': window_alerts['duration'].mean() if 'duration' in window_alerts else 0,
                'max_alert_duration': window_alerts['duration'].max() if 'duration' in window_alerts else 0,
                'std_alert_duration': window_alerts['duration'].std() if 'duration' in window_alerts else 0,
                'total_alert_duration': window_alerts['duration'].sum() if 'duration' in window_alerts else 0
            }
            
            # Add weighted duration sum
            weighted_duration_sum = 0
            if 'duration' in window_alerts.columns:
                for _, alert in window_alerts.iterrows():
                    pattern = self.duration_patterns.get(alert.get('alert_cluster', -1), {})
                    significance_score = pattern.get('significance_score', 0)
                    weighted_duration_sum += alert['duration'] * significance_score
                    
            duration_features['weighted_duration_sum'] = weighted_duration_sum
            features.update(duration_features)
            
            # Add category-specific features
            unique_categories = set(
                cat.lower() for cat in self.alerts_df['category'].unique() 
                if cat is not None and isinstance(cat, str)
            )
            
            for category in unique_categories:
                cat_alerts = window_alerts[
                    window_alerts['category'].fillna('unknown').str.lower() == category
                ]
                safe_category = category.replace(' ', '_').replace('-', '_')
                features[f'{safe_category}_alerts'] = len(cat_alerts)
                features[f'{safe_category}_weighted_sum'] = sum(
                    self.alert_weights['category'].get(category, 0)
                    for _ in range(len(cat_alerts))
                )
            
            return features
            
        except Exception as e:
            logging.error(f"Error in create_window_features: {str(e)}")
            raise
    
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
            ].copy()
            
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
                if not any((window_end >= outage['Start']) & 
                          (window_end <= outage['End'])
                          for _, outage in self.outages_df.iterrows()):
                    window_start = window_end - timedelta(hours=8)
                    window_alerts = self.alerts_df[
                        (self.alerts_df['alert_start_time'] >= window_start) &
                        (self.alerts_df['alert_start_time'] <= window_end) &
                        (self.alerts_df['app_name'] == app_name)
                    ].copy()
                    
                    features = self.create_window_features(window_alerts, is_outage=False)
                    features['app_name'] = app_name
                    features['window_end'] = window_end
                    features_list.append(features)
        
        self.features_df = pd.DataFrame(features_list)
        logging.info(f"Generated {len(self.features_df)} feature records")
        return self.features_df

    def _generate_confusion_matrix(self, y_true, y_pred, model_name):
        """Generate confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            cm, 
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Outage', 'Outage'],
            yticklabels=['No Outage', 'Outage']
        )
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add percentages
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm[i, j] / total * 100
                plt.text(
                    j + 0.5, 
                    i + 0.7, 
                    f'({percentage:.1f}%)',
                    ha='center',
                    va='center'
                )
        
        plt.tight_layout()
        filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename

    def _generate_roc_curve(self, y_true, y_prob, model_name):
        """Generate ROC curve visualization"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        filename = f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename)
        plt.close()
        return filename

    def _generate_precision_recall_curve(self, y_true, y_prob, model_name):
        """Generate Precision-Recall curve visualization"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        filename = f'pr_curve_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename)
        plt.close()
        return filename

    def _generate_learning_curves(self, model, X, y, model_name):
        """Generate learning curves to analyze overfitting"""
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y,
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='f1'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score',
                color='darkorange', lw=2)
        plt.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1,
                        color='darkorange')
        plt.plot(train_sizes, test_mean, label='Cross-validation score',
                color='navy', lw=2)
        plt.fill_between(train_sizes, test_mean - test_std,
                        test_mean + test_std, alpha=0.1,
                        color='navy')
        plt.xlabel('Training Examples')
        plt.ylabel('F1 Score')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        filename = f'learning_curve_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename)
        plt.close()
        return filename

    def _generate_feature_importance_plot(self, importances, feature_names, model_name

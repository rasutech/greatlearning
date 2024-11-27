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
    """Generate features for model training with 10-minute frequency"""
    logging.info("Starting feature engineering")
    features_list = []
    
    # Create a complete time series range with 10-minute frequency
    full_range = pd.date_range(
        start=self.alerts_df['alert_start_time'].min(),
        end=self.alerts_df['alert_start_time'].max(),
        freq='10T'  # 10-minute frequency
    )
    
    # Track processed windows
    processed_windows = set()
    
    # Configure window parameters
    WINDOW_SIZE = timedelta(hours=8)  # Keep 8-hour look-back
    MIN_ALERTS_THRESHOLD = 1  # Minimum alerts needed to create a feature window
    
    # First, process outage windows
    for _, outage in self.outages_df.iterrows():
        window_start = outage['Start'] - WINDOW_SIZE
        window_end = outage['Start']
        
        window_alerts = self.alerts_df[
            (self.alerts_df['alert_start_time'] >= window_start) &
            (self.alerts_df['alert_start_time'] <= window_end) &
            (self.alerts_df['app_name'] == outage['app_name'])
        ].copy()
        
        features = self.create_window_features(window_alerts, is_outage=True)
        features.update({
            'app_name': outage['app_name'],
            'window_start': window_start,
            'window_end': window_end,
            'window_type': 'outage',
            'alert_count': len(window_alerts),
            'unique_alert_types': window_alerts['condition_name'].nunique(),
            'alert_frequency': len(window_alerts) / (WINDOW_SIZE.total_seconds() / 3600),  # alerts per hour
            'time_to_outage': 0  # For outage windows
        })
        features_list.append(features)
        
        # Add window to processed set
        processed_windows.add((outage['app_name'], window_start, window_end))
    
    # Process non-outage windows
    for app_name in self.alerts_df['app_name'].unique():
        # Get next outage time for this app (if any)
        app_outages = self.outages_df[self.outages_df['app_name'] == app_name]
        
        # Create sliding windows
        for window_end in full_range:
            window_start = window_end - WINDOW_SIZE
            
            # Skip if this window overlaps with any outage window
            if any((window_end >= outage['Start'] - timedelta(minutes=10)) & 
                   (window_end <= outage['End'])
                   for _, outage in self.outages_df.iterrows()):
                continue
            
            # Skip if window already processed
            if (app_name, window_start, window_end) in processed_windows:
                continue
            
            window_alerts = self.alerts_df[
                (self.alerts_df['alert_start_time'] >= window_start) &
                (self.alerts_df['alert_start_time'] <= window_end) &
                (self.alerts_df['app_name'] == app_name)
            ].copy()
            
            # Only create features if there are sufficient alerts
            if len(window_alerts) >= MIN_ALERTS_THRESHOLD:
                # Calculate time to next outage
                next_outage = app_outages[app_outages['Start'] > window_end]
                time_to_outage = float('inf')
                if not next_outage.empty:
                    time_to_outage = (next_outage.iloc[0]['Start'] - window_end).total_seconds() / 3600  # in hours
                
                features = self.create_window_features(window_alerts, is_outage=False)
                features.update({
                    'app_name': app_name,
                    'window_start': window_start,
                    'window_end': window_end,
                    'window_type': 'normal',
                    'alert_count': len(window_alerts),
                    'unique_alert_types': window_alerts['condition_name'].nunique(),
                    'alert_frequency': len(window_alerts) / (WINDOW_SIZE.total_seconds() / 3600),  # alerts per hour
                    'time_to_outage': time_to_outage
                })
                features_list.append(features)
    
    self.features_df = pd.DataFrame(features_list)
    
    # Add time-based features
    self.features_df['hour_of_day'] = self.features_df['window_end'].dt.hour
    self.features_df['day_of_week'] = self.features_df['window_end'].dt.dayofweek
    self.features_df['is_business_hours'] = (
        (self.features_df['hour_of_day'] >= 9) & 
        (self.features_df['hour_of_day'] < 17) & 
        (self.features_df['day_of_week'] < 5)
    ).astype(int)
    
    # Sort by time
    self.features_df.sort_values(['app_name', 'window_end'], inplace=True)
    
    # Generate and export enhanced summaries
    self._export_enhanced_summaries()
    
    logging.info(f"Generated {len(self.features_df)} feature records")
    return self.features_df

def _export_enhanced_summaries(self):
    """Export detailed summaries with 10-minute granularity analysis"""
    
    # 1. Feature Summary with enhanced time analysis
    with pd.ExcelWriter('feature_summary.xlsx') as writer:
        # Overall summary
        feature_summary = pd.DataFrame({
            'total_windows': len(self.features_df),
            'outage_windows': len(self.features_df[self.features_df['window_type'] == 'outage']),
            'normal_windows': len(self.features_df[self.features_df['window_type'] == 'normal']),
            'total_alerts_processed': self.features_df['alert_count'].sum(),
            'avg_alerts_per_window': self.features_df['alert_count'].mean(),
            'max_alerts_in_window': self.features_df['alert_count'].max(),
            'avg_alert_frequency': self.features_df['alert_frequency'].mean(),
            'unique_alert_types': self.features_df['unique_alert_types'].max(),
        }, index=[0])
        feature_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Time-based statistics
        time_stats = self.features_df.groupby([
            self.features_df['window_end'].dt.date,
            self.features_df['window_end'].dt.hour,
            pd.cut(self.features_df['window_end'].dt.minute, bins=range(0, 61, 10))
        ]).agg({
            'alert_count': ['count', 'sum', 'mean'],
            'is_outage': 'sum',
            'alert_frequency': 'mean'
        }).round(2)
        time_stats.to_excel(writer, sheet_name='10-Min Intervals')
        
        # Hourly patterns
        hourly_patterns = self.features_df.groupby('hour_of_day').agg({
            'alert_count': ['mean', 'max'],
            'is_outage': 'sum',
            'alert_frequency': 'mean'
        }).round(2)
        hourly_patterns.to_excel(writer, sheet_name='Hourly Patterns')
        
        # Business hours analysis
        business_hours = self.features_df.groupby('is_business_hours').agg({
            'alert_count': ['mean', 'sum'],
            'is_outage': 'sum',
            'alert_frequency': 'mean'
        }).round(2)
        business_hours.to_excel(writer, sheet_name='Business Hours')
    
    # 2. Enhanced Alert Summary
    with pd.ExcelWriter('alert_summary.xlsx') as writer:
        # Basic summary
        alert_summary = pd.DataFrame({
            'total_alerts': len(self.alerts_df),
            'unique_apps': self.alerts_df['app_name'].nunique(),
            'unique_conditions': self.alerts_df['condition_name'].nunique(),
            'unique_categories': self.alerts_df['category'].nunique(),
            'date_range': f"{self.alerts_df['alert_start_time'].min()} to {self.alerts_df['alert_start_time'].max()}",
            'avg_duration': self.alerts_df['duration'].mean(),
        }, index=[0])
        alert_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # 10-minute interval analysis
        alert_intervals = self.alerts_df.set_index('alert_start_time').resample('10T').agg({
            'app_name': 'count',
            'category': 'nunique',
            'condition_name': 'nunique',
            'duration': 'mean'
        })
        alert_intervals.columns = ['alert_count', 'unique_categories', 'unique_conditions', 'avg_duration']
        alert_intervals.to_excel(writer, sheet_name='10-Min Intervals')
        
        # Alert patterns by time
        time_patterns = self.alerts_df.groupby([
            self.alerts_df['alert_start_time'].dt.hour,
            self.alerts_df['alert_start_time'].dt.dayofweek
        ]).size().unstack()
        time_patterns.to_excel(writer, sheet_name='Time Patterns')

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

def _generate_feature_importance_plot(self, importances, feature_names, model_name):
        """Generate feature importance visualization"""
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        sns.barplot(
            data=importance_df.head(15),
            x='importance',
            y='feature'
        )
        plt.title(f'Top 15 Important Features - {model_name}')
        plt.tight_layout()
        
        filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename)
        plt.close()
        return filename

    def train_and_evaluate(self):
        # Prepare data for training
        X = self.features_df.drop(['is_outage', 'app_name', 'window_start', 'window_end', 'window_type'], axis=1)
        y = self.features_df['is_outage']
        
        # Handle missing values
        X = X.fillna(0)
        
        # Log class distribution
        class_dist = y.value_counts()
        logging.info(f"Class distribution before split: {dict(class_dist)}")
        
        # Use time-based split
        train_date = self.features_df['window_end'].max() - timedelta(days=30)
        
        train_idx = self.features_df['window_end'] <= train_date
        test_idx = self.features_df['window_end'] > train_date
        
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to DataFrame to keep feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
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
        confusion_matrices = {}
        detailed_metrics = {}
        
        for name, model in models.items():
            logging.info(f"Training {name} model")
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices[name] = cm
            
            # Calculate detailed metrics
            tn, fp, fn, tp = cm.ravel()
            
            detailed_metrics[name] = {
                'True Positives': tp,
                'True Negatives': tn,
                'False Positives': fp,
                'False Negatives': fn,
                'Sensitivity (True Positive Rate)': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'Specificity (True Negative Rate)': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'ROC AUC': roc_auc_score(y_test, y_prob),
                'Accuracy': accuracy_score(y_test, y_pred),
                'False Positive Rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'False Negative Rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
                'Positive Predictive Value': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'Negative Predictive Value': tn / (tn + fn) if (tn + fn) > 0 else 0
            }
            
            # Store basic metrics for summary
            results.append({
                'Model': name,
                'Accuracy': detailed_metrics[name]['Accuracy'],
                'Precision': detailed_metrics[name]['Precision'],
                'Recall': detailed_metrics[name]['Recall'],
                'F1 Score': detailed_metrics[name]['F1 Score'],
                'ROC AUC': detailed_metrics[name]['ROC AUC']
            })
            
            # Store feature importance for applicable models
            if hasattr(model, 'feature_importances_'):
                feature_importance_data[name] = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        self.generate_report(results, feature_importance_data, confusion_matrices, detailed_metrics)
        logging.info("Model training and evaluation completed")

    def generate_report(self, results, visualization_data,detailed_metrics):
        """Generate HTML report with all visualizations"""
        html_content = """
        <html>
        <head>
            <title>Outage Prediction Model Evaluation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-table { border-collapse: collapse; margin: 20px 0; width: 100%; }
                .metric-table td, .metric-table th { 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: left;
                }
                .metric-table th { background-color: #f2f2f2; }
                .plot { margin: 20px 0; }
                .model-section { margin: 40px 0; padding: 20px; border: 1px solid #ddd; }
                .visualization-section { display: flex; flex-wrap: wrap; justify-content: space-between; }
                .visualization-card { width: 45%; margin: 10px; padding: 10px; border: 1px solid #eee; }
                .detailed-metrics { margin: 20px 0; }
            </style>
        </head>
        <body>
        """
        
        # Add summary metrics table
        metrics_df = pd.DataFrame(results)
        html_content += "<h2>Model Performance Summary</h2>"
        html_content += metrics_df.to_html(classes='metric-table', index=False)
        
        # Add sections for each model
        for model_name, visuals in visualization_data.items():
            html_content += f"""
            <div class='model-section'>
                <h2>Model Analysis - {model_name}</h2>
                
                <div class='visualization-section'>
                    <div class='visualization-card'>
                        <h3>Confusion Matrix</h3>
                        <img src='{visuals["confusion_matrix"]}' />
                    </div>
                    
                    <div class='visualization-card'>
                        <h3>ROC Curve</h3>
                        <img src='{visuals["roc_curve"]}' />
                    </div>
                    
                    <div class='visualization-card'>
                        <h3>Precision-Recall Curve</h3>
                        <img src='{visuals["pr_curve"]}' />
                    </div>
                    
                    <div class='visualization-card'>
                        <h3>Learning Curves</h3>
                        <img src='{visuals["learning_curve"]}' />
                    </div>
            """
            
            if 'feature_importance' in visuals:
                html_content += f"""
                    <div class='visualization-card'>
                        <h3>Feature Importance</h3>
                        <img src='{visuals["feature_importance"]}' />
                    </div>
                """
            # Add detailed metrics
            html_content += "<div class='detailed-metrics'>"
            html_content += "<h3>Detailed Metrics</h3>"
            metrics_table = pd.DataFrame(
                detailed_metrics[model_name].items(),
                columns=['Metric', 'Value']
            )
            metrics_table['Value'] = metrics_table['Value'].apply(
                lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
            )
            html_content += metrics_table.to_html(
                classes='metric-table',
                index=False
            )
            html_content += "</div>"
            html_content += "</div></div>"

        
            
        html_content += "</body></html>"
        
        # Save report
        with open(self.config['Files']['output_file'], 'w') as f:
            f.write(html_content)
        
        logging.info(f"Report generated: {self.config['Files']['output_file']}")

if __name__ == "__main__":
    # Example usage
    predictor = OutagePredictionSystem()
    predictor.load_data("app1,app2,app3")  # Replace with actual app names
    predictor.preprocess_data()
    predictor.feature_engineering()
    predictor.train_and_evaluate()

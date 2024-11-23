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
from sklearn.model_selection import learning_curve

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
            
            # Generate all visualizations
            visualization_data[name] = {
                'confusion_matrix': self._generate_confusion_matrix(y_test, y_pred, name),
                'roc_curve': self._generate_roc_curve(y_test, y_prob, name),
                'pr_curve': self._generate_precision_recall_curve(y_test, y_prob, name),
                'learning_curve': self._generate_learning_curves(model, X_resampled, y_resampled, name)
            }
            
            if hasattr(model, 'feature_importances_'):
                visualization_data[name]['feature_importance'] = self._generate_feature_importance_plot(
                    model.feature_importances_, X.columns, name
                )
         # Generate feature correlation matrix once for all features
        correlation_matrix_file = self._generate_feature_correlation_matrix(X)
        
        self.generate_enhanced_report(results, visualization_data, detailed_metrics, correlation_matrix_file)
        logging.info("Model training and evaluation completed")

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

    def _generate_feature_correlation_matrix(self, X):
        """Generate feature correlation matrix visualization"""
        plt.figure(figsize=(12, 10))
        correlation_matrix = X.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', center=0,
                   annot=False, fmt='.2f', square=True)
        plt.title('Feature Correlation Matrix')
        
        filename = 'feature_correlation_matrix.png'
        plt.savefig(filename)
        plt.close()
        return filename

    def train_and_evaluate(self):
        """Enhanced train and evaluate method with additional visualizations"""
        # ... (previous code remains the same until model evaluation) ...
        
        visualization_data = {}
        
        for name, model in models.items():
            logging.info(f"Training and evaluating {name}")
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            # Generate all visualizations
            visualization_data[name] = {
                'confusion_matrix': self._generate_confusion_matrix(y_test, y_pred, name),
                'roc_curve': self._generate_roc_curve(y_test, y_prob, name),
                'pr_curve': self._generate_precision_recall_curve(y_test, y_prob, name),
                'learning_curve': self._generate_learning_curves(model, X_resampled, y_resampled, name)
            }
            
            if hasattr(model, 'feature_importances_'):
                visualization_data[name]['feature_importance'] = self._generate_feature_importance_plot(
                    model.feature_importances_, X.columns, name
                )
        
        # Generate feature correlation matrix once for all features
        correlation_matrix_file = self._generate_feature_correlation_matrix(X)
        
        self.generate_enhanced_report(results, visualization_data, detailed_metrics, correlation_matrix_file)

    def generate_enhanced_report(self, results, visualization_data, detailed_metrics, correlation_matrix_file):
        """Generate enhanced HTML report with all visualizations"""
        html_content = """
        <html>
        <head>
            <title>Enhanced Outage Prediction Model Evaluation</title>
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
                .visualization-card { 
                    width: 45%; 
                    margin: 10px; 
                    padding: 10px; 
                    border: 1px solid #eee;
                }
                .interpretation { margin: 10px 0; padding: 10px; background-color: #f9f9f9; }
            </style>
        </head>
        <body>
        """
        
        # Add feature correlation matrix section
        html_content += """
        <h2>Feature Correlation Analysis</h2>
        <div class='visualization-card'>
            <img src='{}' />
            <div class='interpretation'>
                <p><strong>Interpretation:</strong></p>
                <ul>
                    <li>Strong positive correlations are shown in red</li>
                    <li>Strong negative correlations are shown in blue</li>
                    <li>This helps identify redundant features and potential feature interactions</li>
                </ul>
            </div>
        </div>
        """.format(correlation_matrix_file)
        
        # Add sections for each model
        for model_name, visuals in visualization_data.items():
            html_content += f"""
            <div class='model-section'>
                <h2>Model Analysis - {model_name}</h2>
                
                <div class='visualization-section'>
                    <div class='visualization-card'>
                        <h3>Confusion Matrix</h3>
                        <img src='{visuals["confusion_matrix"]}' />
                        <div class='interpretation'>
                            <p><strong>Key Insights:</strong></p>
                            <ul>
                                <li>True Positives: Correctly predicted outages</li>
                                <li>False Positives: False alarms</li>
                                <li>False Negatives: Missed outages</li>
                                <li>True Negatives: Correctly predicted non-outages</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class='visualization-card'>
                        <h3>ROC Curve</h3>
                        <img src='{visuals["roc_curve"]}' />
                        <div class='interpretation'>
                            <p><strong>Key Insights:</strong></p>
                            <ul>
                                <li>Shows trade-off between sensitivity and specificity</li>
                                <li>Higher AUC indicates better model performance</li>
                                <li>Closer to top-left corner is better</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class='visualization-card'>
                        <h3>Precision-Recall Curve</h3>
                        <img src='{visuals["pr_curve"]}' />
                        <div class='interpretation'>
                            <p><strong>Key Insights:</strong></p>
                            <ul>
                                <li>Shows trade-off between precision and recall</li>
                                <li>Particularly important for imbalanced datasets</li>
                                <li>Higher curve indicates better performance</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class='visualization-card'>
                        <h3>Learning Curves</h3>
                        <img src='{visuals["learning_curve"]}' />
                        <div class='interpretation'>
                            <p><strong>Key Insights:</strong></p>
                            <ul>
                                <li>Shows how model learns with more data</li>
                                <li>Gap between training and CV indicates overfitting</li>
                                <li>Converging curves indicate good fit</li>
                            </ul>
                        </div>
                    </div>
                    """
            
            if 'feature_importance' in visuals:
                html_content += f"""
                <div class='visualization-card'>
                    <h3>Feature Importance</h3>
                    <img src='{visuals["feature_importance"]}' />
                    <div class='interpretation'>
                        <p><strong>Key Insights:</strong></p>
                        <ul>
                            <li>Shows most influential features</li>
                            <li>Helps in feature selection</li>
                            <li>Guides feature engineering efforts</li>
                        </ul>
                    </div>
                </div>
                """
            
            html_content += "</div></div>"
        
        html_content += "</body></html>"
        
        # Save enhanced report
        with open(self.config['Files']['output_file'], 'w') as f:
            f.write(html_content)
        
        logging.info(f"Enhanced report generated: {self.config['Files']['output_file']}")

# Example usage
if __name__ == "__main__":
    # Initialize prediction system
    predictor = OutagePredictionSystem()
    
    # Load and process data
    app_names = "

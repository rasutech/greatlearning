import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import psycopg2
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class OutagePredictionSystem:
    def __init__(self, db_params, app_name, excel_path):
        """Initialize the Outage Prediction System"""
        self.db_params = db_params
        self.app_name = app_name
        self.excel_path = excel_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None

    def connect_to_db(self):
        """Establish database connection"""
        try:
            conn = psycopg2.connect(**self.db_params)
            return conn
        except Exception as e:
            raise Exception(f"Error connecting to database: {str(e)}")

    def load_outage_data(self):
        """Load outage data from Excel file"""
        try:
            outage_df = pd.read_excel(self.excel_path)
            
            # Convert date columns to datetime
            outage_df['date'] = pd.to_datetime(outage_df['date'])
            outage_df['opened'] = pd.to_datetime(outage_df['opened'])
            outage_df['outage_end_time'] = outage_df['opened'] + pd.to_timedelta(outage_df['duration'], unit='minutes')
            
            return outage_df[outage_df['app_name'] == self.app_name]
        except Exception as e:
            raise Exception(f"Error loading outage data: {str(e)}")

    def load_alerts_data(self, conn, start_date, end_date):
        """Load alerts data from database"""
        try:
            # Add buffer before and after the date range
            buffer_hours = 24
            query_start = start_date - pd.Timedelta(hours=buffer_hours)
            query_end = end_date + pd.Timedelta(hours=buffer_hours)
            
            query = """
            SELECT 
                incident_id,
                app_name,
                policy_name,
                condition_name,
                category,
                entity_type,
                entity_name,
                alert_description,
                alert_start_time,
                priority,
                datasource,
                environment
            FROM alerts_archive
            WHERE app_name = %s
            AND priority = 'critical'
            AND alert_start_time BETWEEN %s AND %s
            ORDER BY alert_start_time
            """
            
            print(f"Fetching alerts from {query_start} to {query_end}")
            df = pd.read_sql_query(query, conn, params=(self.app_name, query_start, query_end))
            print(f"Fetched {len(df)} critical alerts")
            
            return df
        except Exception as e:
            raise Exception(f"Error loading alerts data: {str(e)}")

    def create_minute_features(self, timestamp, current_alerts, all_alerts, time_window, outage_details):
        """Create features for a specific minute"""
        features = {}
        
        # Create combination features
        if len(current_alerts) > 0:
            # Create the combination string for each alert
            current_alerts['alert_combination'] = (current_alerts['datasource'] + '_' + 
                                                 current_alerts['policy_name'] + '_' + 
                                                 current_alerts['condition_name']).str.replace(' ', '_')
            
            # Count occurrences of each combination
            combinations = current_alerts['alert_combination'].value_counts()
            for combo, count in combinations.items():
                features[f"alert_{combo}"] = count
        
        # Add total critical alerts count
        features['total_critical_alerts'] = len(current_alerts)
        
        # Time-based lookback features
        lookback_start = timestamp - pd.Timedelta(minutes=time_window)
        lookback_alerts = all_alerts[
            (all_alerts['alert_start_time'] >= lookback_start) &
            (all_alerts['alert_start_time'] < timestamp)
        ]
        
        if len(lookback_alerts) > 0:
            lookback_alerts['alert_combination'] = (lookback_alerts['datasource'] + '_' + 
                                                  lookback_alerts['policy_name'] + '_' + 
                                                  lookback_alerts['condition_name']).str.replace(' ', '_')
            
            lookback_combinations = lookback_alerts['alert_combination'].value_counts()
            for combo, count in lookback_combinations.items():
                features[f"lookback_{combo}"] = count
        
        # Temporal features
        features['hour_of_day'] = timestamp.hour
        features['day_of_week'] = timestamp.dayofweek
        features['is_business_hours'] = 1 if (timestamp.hour >= 9 and timestamp.hour < 17) else 0
        features['is_weekend'] = 1 if timestamp.dayofweek >= 5 else 0
        
        return features

    def create_feature_matrix(self, alerts_df, outage_df, time_window=120):
        """Create feature matrix"""
        try:
            if len(alerts_df) == 0:
                raise ValueError("No critical alerts found in the data")
            
            print(f"Creating features from {len(alerts_df)} critical alerts")
            
            # Create a complete time range
            start_time = min(alerts_df['alert_start_time'].min(), outage_df['opened'].min())
            end_time = max(alerts_df['alert_start_time'].max(), outage_df['opened'].max())
            
            all_timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
            feature_records = []
            
            # Create all possible combinations
            alerts_df['alert_combination'] = (alerts_df['datasource'] + '_' + 
                                            alerts_df['policy_name'] + '_' + 
                                            alerts_df['condition_name']).str.replace(' ', '_')
            all_combinations = set(alerts_df['alert_combination'].unique())
            
            print(f"Processing {len(all_timestamps)} time points with {len(all_combinations)} unique alert combinations")
            
            # Process each timestamp
            for timestamp in all_timestamps:
                # Get alerts in current minute
                current_alerts = alerts_df[
                    (alerts_df['alert_start_time'] >= timestamp) &
                    (alerts_df['alert_start_time'] < timestamp + pd.Timedelta(minutes=1))
                ]
                
                # Check for outage
                is_outage = 0
                outage_details = None
                for _, outage in outage_df.iterrows():
                    if timestamp >= outage['opened'] and timestamp <= outage['outage_end_time']:
                        is_outage = 1
                        outage_details = outage
                        break
                
                # Create features
                features = self.create_minute_features(
                    timestamp, current_alerts, alerts_df, time_window, outage_details
                )
                
                # Ensure all combinations are represented
                for combo in all_combinations:
                    if f"alert_{combo}" not in features:
                        features[f"alert_{combo}"] = 0
                    if f"lookback_{combo}" not in features:
                        features[f"lookback_{combo}"] = 0
                
                features['is_outage'] = is_outage
                feature_records.append(features)
            
            # Create DataFrame
            feature_matrix = pd.DataFrame(feature_records)
            
            # Split features and target
            y = feature_matrix.pop('is_outage')
            X = feature_matrix
            
            print(f"Created feature matrix with shape: {X.shape}")
            print(f"Number of outage points: {y.sum()}")
            
            return X, y
            
        except Exception as e:
            raise Exception(f"Error creating feature matrix: {str(e)}")

    def train_model(self, X, y):
        """Train the prediction model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create and fit preprocessor
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            self.preprocessor = ColumnTransformer(
                transformers=[('num', numeric_transformer, X.columns)],
                remainder='drop'
            )
            
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)
            
            # Apply PCA
            pca = PCA(n_components=0.95)
            X_train_pca = pca.fit_transform(X_train_processed)
            X_test_pca = pca.transform(X_test_processed)
            
            # Train model
            self.model = LogisticRegression(class_weight='balanced', random_state=42)
            self.model.fit(X_train_pca, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_pca)
            y_pred_proba = self.model.predict_proba(X_test_pca)[:, 1]
            
            # Calculate metrics
            results = {
                'pca': pca,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'feature_names': X.columns.tolist()
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")

    def generate_report(self, results, outage_ratio):
        """Generate analysis report"""
        report_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Outage Prediction Analysis - {app_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Outage Prediction Analysis - {app_name}</h1>
            
            <div class="section">
                <h2>Model Performance</h2>
                <div class="metric">
                    <strong>ROC-AUC Score:</strong> {roc_auc:.4f}
                </div>
                <div class="metric">
                    <strong>Outage Ratio:</strong> {outage_ratio:.2%}
                </div>
                <h3>Classification Report</h3>
                <pre>{classification_report}</pre>
                
                <h3>Confusion Matrix</h3>
                <pre>{confusion_matrix}</pre>
            </div>
        </body>
        </html>
        """
        
        report_content = report_template.format(
            app_name=self.app_name,
            roc_auc=results['roc_auc'],
            outage_ratio=outage_ratio,
            classification_report=results['classification_report'],
            confusion_matrix=results['confusion_matrix']
        )
        
        filename = f"outage_prediction_{self.app_name}_{datetime.now().strftime('%Y%m%d')}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\nAnalysis report generated: {filename}")

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            # Load outage data
            print("Loading outage data...")
            outage_df = self.load_outage_data()
            
            if len(outage_df) == 0:
                raise ValueError(f"No outage data found for app: {self.app_name}")
            
            # Calculate date range
            start_date = outage_df['opened'].min()
            end_date = outage_df['opened'].max()
            print(f"Analyzing period from {start_date} to {end_date}")
            
            # Load alerts data
            with self.connect_to_db() as conn:
                alerts_df = self.load_alerts_data(conn, start_date, end_date)
            
            # Create feature matrix
            print("\nCreating feature matrix...")
            X, y = self.create_feature_matrix(alerts_df, outage_df)
            
            # Train model
            print("\nTraining model...")
            results = self.train_model(X, y)
            
            # Generate report
            print("\nGenerating report...")
            self.generate_report(results, y.mean())
            
            return results
            
        except Exception as e:
            print(f"Error in analysis pipeline: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

def main():
    """Main execution function"""
    # Configuration
    db_params = {
        "dbname": "your_database",
        "user": "your_username",
        "password": "your_password",
        "host": "your_host",
        "port": "your_port"
    }
    
    app_name = "YOUR_APP_NAME"
    excel_path = "path_to_your_outage_data.xlsx"
    
    # Create and run the prediction system
    predictor = OutagePredictionSystem(db_params, app_name, excel_path)
    
    try:
        results = predictor.run_analysis()
        print("\nAnalysis completed successfully!")
        print(f"ROC-AUC Score: {results['roc_auc']:.4f}")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()

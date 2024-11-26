# alert_intelligence.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import timedelta
import logging

class AlertIntelligence:
    """Handles the intelligent processing of alerts including clustering and weight calculations"""
    
    def __init__(self, alerts_df, outages_df):
        """
        Initialize AlertIntelligence with alert and outage data
        
        Parameters:
        alerts_df (pd.DataFrame): DataFrame containing alert data
        outages_df (pd.DataFrame): DataFrame containing outage data
        """
        self.alerts_df = alerts_df.copy()
        self.outages_df = outages_df.copy()
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Prepare data for analysis with enhanced time handling"""
        # Calculate alert durations
        self.alerts_df['duration'] = (
            self.alerts_df['alert_end_time'] - 
            self.alerts_df['alert_start_time']
        ).dt.total_seconds() / 3600  # Convert to hours
        
        # Handle missing or invalid durations
        self.alerts_df['duration'] = self.alerts_df['duration'].clip(lower=0)
        self.alerts_df.loc[self.alerts_df['duration'].isna(), 'duration'] = 0
        
        # Add time-based features
        self.alerts_df['hour_of_day'] = self.alerts_df['alert_start_time'].dt.hour
        self.alerts_df['minute_of_hour'] = self.alerts_df['alert_start_time'].dt.minute
        self.alerts_df['day_of_week'] = self.alerts_df['alert_start_time'].dt.dayofweek
        
        # Clean and standardize categorical columns
        for col in ['category', 'datasource', 'policy_name', 'condition_name']:
            self.alerts_df[col] = (
                self.alerts_df[col]
                .fillna('Unknown')
                .astype(str)
                .str.strip()
                .str.lower()
            )
        
        # Create alert text for clustering
        self.alerts_df['alert_text'] = (
            self.alerts_df['alert_description'].fillna('') + 
            ' ' + 
            self.alerts_df['condition_name'].fillna('')
        ).str.lower()
        
        # Calculate 10-minute interval index
        self.alerts_df['interval_index'] = (
            self.alerts_df['alert_start_time']
            .dt.floor('10T')
            .view(np.int64) // 10**9 // 600
        )

    def _cluster_alerts(self):
        """Cluster similar alerts with enhanced handling"""
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            text_features = vectorizer.fit_transform(self.alerts_df['alert_text'])
            
            # Determine optimal number of clusters based on data size
            n_clusters = min(
                50,  # Maximum clusters
                max(
                    10,  # Minimum clusters
                    len(self.alerts_df) // 100  # Or 1 cluster per 100 alerts
                )
            )
            
            # Perform clustering
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            self.alerts_df['alert_cluster'] = kmeans.fit_predict(text_features)
            
            # Calculate cluster densities
            cluster_counts = self.alerts_df['alert_cluster'].value_counts()
            self.cluster_densities = cluster_counts / len(self.alerts_df)
            
            return vectorizer.get_feature_names_out()
            
        except Exception as e:
            logging.error(f"Error in alert clustering: {str(e)}")
            self.alerts_df['alert_cluster'] = 0
            return []

    def calculate_alert_weights(self):
        """Calculate weights with 10-minute interval consideration"""
        # First cluster the alerts
        self._cluster_alerts()
        
        # Initialize weight dictionaries
        weights = {
            'cluster': {},
            'source': {},
            'category': {},
            'policy': {}
        }
        
        # Calculate weights based on proximity to outages
        for _, outage in self.outages_df.iterrows():
            # Look at alerts within 8 hours of outage
            window_start = outage['Start'] - timedelta(hours=8)
            window_alerts = self.alerts_df[
                (self.alerts_df['alert_start_time'] >= window_start) &
                (self.alerts_df['alert_start_time'] <= outage['Start']) &
                (self.alerts_df['app_name'] == outage['app_name'])
            ]
            
            for _, alert in window_alerts.iterrows():
                # Calculate time-based weight (more weight to recent alerts)
                time_to_outage = (
                    (outage['Start'] - alert['alert_start_time']).total_seconds() / 600  # In 10-minute intervals
                )
                proximity_weight = 1 / (1 + time_to_outage/6)  # Normalize by hour
                
                # Add duration factor
                duration_factor = min(1, alert['duration'] / 24)  # Cap at 24 hours
                combined_weight = proximity_weight * (1 + duration_factor)
                
                # Update weights
                self._update_weight(weights['cluster'], alert['alert_cluster'], combined_weight)
                self._update_weight(weights['source'], alert['datasource'], combined_weight)
                self._update_weight(weights['category'], alert['category'], combined_weight)
                self._update_weight(weights['policy'], alert['policy_name'], combined_weight)
        
        # Normalize weights
        self.weights = {
            dim: self._normalize_weights(weights[dim])
            for dim in weights.keys()
        }
        
        return self.weights

    def analyze_duration_patterns(self):
        """Analyze alert duration patterns with 10-minute granularity"""
        patterns = {}
        
        # Calculate statistics for each alert cluster
        cluster_stats = self.alerts_df.groupby('alert_cluster').agg({
            'duration': ['mean', 'std', 'count'],
            'interval_index': lambda x: x.nunique()
        })
        
        # Calculate patterns for each cluster
        for cluster in cluster_stats.index:
            mean_duration = cluster_stats.loc[cluster, ('duration', 'mean')]
            std_duration = cluster_stats.loc[cluster, ('duration', 'std')]
            count = cluster_stats.loc[cluster, ('duration', 'count')]
            interval_count = cluster_stats.loc[cluster, ('interval_index', 'lambda')]
            
            # Calculate density (alerts per interval)
            density = count / interval_count if interval_count > 0 else 0
            
            # Calculate significance score considering:
            # - Consistency (lower std dev is better)
            # - Frequency (more occurrences is better)
            # - Density (more alerts per interval is better)
            significance_score = (
                mean_duration * count * density / (1 + std_duration)
            ) if std_duration > 0 else mean_duration * count * density
            
            patterns[cluster] = {
                'mean_duration': mean_duration,
                'std_duration': std_duration,
                'count': count,
                'density': density,
                'significance_score': significance_score
            }
        
        self.duration_patterns = patterns
        return patterns

    @staticmethod
    def _update_weight(weight_dict, key, value):
        """Helper method to update weight dictionaries"""
        if key not in weight_dict:
            weight_dict[key] = []
        weight_dict[key].append(value)

    @staticmethod
    def _normalize_weights(weight_dict):
        """Normalize weights to [0, 1] range"""
        final_weights = {k: np.mean(v) for k, v in weight_dict.items()}
        if not final_weights:
            return {}
            
        max_weight = max(final_weights.values())
        min_weight = min(final_weights.values())
        weight_range = max_weight - min_weight
        
        if weight_range > 0:
            return {
                k: (v - min_weight) / weight_range
                for k, v in final_weights.items()
            }
        return final_weights

    def get_processed_alerts(self):
        """Return the processed alerts DataFrame"""
        return self.alerts_df.copy()

    def get_alert_statistics(self):
        """Get detailed alert statistics with 10-minute granularity"""
        stats = {
            'total_alerts': len(self.alerts_df),
            'unique_clusters': self.alerts_df['alert_cluster'].nunique(),
            'avg_duration': self.alerts_df['duration'].mean(),
            'total_intervals': self.alerts_df['interval_index'].nunique(),
            'alerts_per_interval': len(self.alerts_df) / self.alerts_df['interval_index'].nunique()
            if self.alerts_df['interval_index'].nunique() > 0 else 0,
            'cluster_stats': self.alerts_df.groupby('alert_cluster').agg({
                'duration': ['mean', 'std', 'count'],
                'interval_index': 'nunique'
            }).to_dict(),
            'temporal_patterns': self.alerts_df.groupby([
                'hour_of_day',
                pd.cut(self.alerts_df['minute_of_hour'], bins=range(0, 61, 10))
            ]).size().to_dict()
        }
        return stats

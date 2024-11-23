import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import timedelta

class AlertIntelligence:
    """
    Handles the intelligent processing of alerts including:
    - Alert clustering
    - Weight calculation
    - Duration pattern analysis
    """
    
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
        """Prepare data for analysis"""
        # Calculate alert durations
        self.alerts_df['duration'] = (
            self.alerts_df['alert_end_time'] - 
            self.alerts_df['alert_start_time']
        ).dt.total_seconds() / 3600  # Convert to hours
        
        # Clean text data
        self.alerts_df['alert_text'] = (
            self.alerts_df['alert_description'].fillna('') + 
            ' ' + 
            self.alerts_df['condition_name'].fillna('')
        ).str.lower()
    
    def _cluster_alerts(self):
        """
        Cluster similar alerts based on their descriptions and conditions
        using TF-IDF and K-means clustering
        """
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        text_features = vectorizer.fit_transform(self.alerts_df['alert_text'])
        
        # Determine optimal number of clusters
        n_clusters = min(50, len(self.alerts_df) // 100)
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        self.alerts_df['alert_cluster'] = kmeans.fit_predict(text_features)
        
        return vectorizer.get_feature_names_out()
    
    def calculate_alert_weights(self):
        """
        Calculate weights for different alert attributes based on their
        historical correlation with outages
        """
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
            # Get alerts within 8 hours of outage
            window_start = outage['Start'] - timedelta(hours=8)
            window_alerts = self.alerts_df[
                (self.alerts_df['alert_start_time'] >= window_start) &
                (self.alerts_df['alert_start_time'] <= outage['Start']) &
                (self.alerts_df['app_name'] == outage['app_name'])
            ]
            
            for _, alert in window_alerts.iterrows():
                # Calculate time-based weight
                time_to_outage = (
                    outage['Start'] - alert['alert_start_time']
                ).total_seconds() / 3600
                proximity_weight = 1 / (1 + time_to_outage)
                
                # Update weights for each dimension
                self._update_weight(weights['cluster'], alert['alert_cluster'], proximity_weight)
                self._update_weight(weights['source'], alert['datasource'], proximity_weight)
                self._update_weight(weights['category'], alert['category'], proximity_weight)
                self._update_weight(weights['policy'], alert['policy_name'], proximity_weight)
        
        # Normalize weights
        self.weights = {
            dim: self._normalize_weights(weights[dim])
            for dim in weights.keys()
        }
        
        return self.weights
    
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
    
    def analyze_duration_patterns(self):
        """
        Analyze patterns in alert durations to identify significant
        duration-based features
        """
        patterns = {}
        
        # Calculate statistics for each alert cluster
        cluster_stats = self.alerts_df.groupby('alert_cluster')['duration'].agg([
            'mean',
            'std',
            'count'
        ])
        
        # Calculate significance scores
        for cluster in cluster_stats.index:
            mean_duration = cluster_stats.loc[cluster, 'mean']
            std_duration = cluster_stats.loc[cluster, 'std']
            count = cluster_stats.loc[cluster, 'count']
            
            # Calculate significance score
            # Higher score for:
            # - More consistent durations (lower std)
            # - More frequent occurrences (higher count)
            significance_score = (
                mean_duration * count / (1 + std_duration)
            ) if std_duration > 0 else mean_duration * count
            
            patterns[cluster] = {
                'mean_duration': mean_duration,
                'std_duration': std_duration,
                'count': count,
                'significance_score': significance_score
            }
        
        self.duration_patterns = patterns
        return patterns

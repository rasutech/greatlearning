-- Key metrics and anomaly detection queries for each table

-- NetworkSample: Focus on network throughput and errors
WITH network_stats AS (
  SELECT 
    timestamp,
    entity_name,
    receive_bytes_per_second,
    transmit_bytes_per_second,
    receive_errors_per_second,
    transmit_errors_per_second,
    AVG(receive_bytes_per_second) OVER w AS avg_receive,
    STDDEV(receive_bytes_per_second) OVER w AS stddev_receive,
    AVG(transmit_bytes_per_second) OVER w AS avg_transmit,
    STDDEV(transmit_bytes_per_second) OVER w AS stddev_transmit
  FROM NetworkSample
  WHERE timestamp >= DATEADD(day, -6, CURRENT_TIMESTAMP)
  WINDOW w AS (PARTITION BY entity_name ORDER BY timestamp 
               ROWS BETWEEN 576 PRECEDING AND 1 PRECEDING) -- 6 days of 15-min samples
)
SELECT 
  timestamp,
  entity_name,
  receive_bytes_per_second,
  transmit_bytes_per_second,
  CASE 
    WHEN ABS(receive_bytes_per_second - avg_receive) > 3 * stddev_receive THEN 'Anomaly'
    ELSE 'Normal'
  END AS receive_status,
  CASE 
    WHEN ABS(transmit_bytes_per_second - avg_transmit) > 3 * stddev_transmit THEN 'Anomaly'
    ELSE 'Normal'
  END AS transmit_status
FROM network_stats
WHERE receive_status = 'Anomaly' OR transmit_status = 'Anomaly';

-- StorageSample: Focus on disk usage and I/O
WITH storage_stats AS (
  SELECT 
    timestamp,
    entity_name,
    device,
    disk_used_percent,
    read_bytes_per_second,
    write_bytes_per_second,
    AVG(disk_used_percent) OVER w AS avg_disk_used,
    STDDEV(disk_used_percent) OVER w AS stddev_disk_used,
    AVG(read_bytes_per_second) OVER w AS avg_read,
    STDDEV(read_bytes_per_second) OVER w AS stddev_read,
    AVG(write_bytes_per_second) OVER w AS avg_write,
    STDDEV(write_bytes_per_second) OVER w AS stddev_write
  FROM StorageSample
  WHERE timestamp >= DATEADD(day, -6, CURRENT_TIMESTAMP)
  WINDOW w AS (PARTITION BY entity_name, device ORDER BY timestamp 
               ROWS BETWEEN 576 PRECEDING AND 1 PRECEDING)
)
SELECT 
  timestamp,
  entity_name,
  device,
  disk_used_percent,
  CASE 
    WHEN disk_used_percent > 90 THEN 'Critical'
    WHEN ABS(disk_used_percent - avg_disk_used) > 3 * stddev_disk_used THEN 'Anomaly'
    ELSE 'Normal'
  END AS disk_status,
  CASE 
    WHEN ABS(read_bytes_per_second - avg_read) > 3 * stddev_read OR
         ABS(write_bytes_per_second - avg_write) > 3 * stddev_write THEN 'Anomaly'
    ELSE 'Normal'
  END AS io_status
FROM storage_stats
WHERE disk_status IN ('Critical', 'Anomaly') OR io_status = 'Anomaly';

-- ProcessSample: Focus on CPU and memory usage
WITH process_stats AS (
  SELECT 
    timestamp,
    entity_name,
    command_name,
    cpu_percent,
    memory_resident_size_bytes,
    AVG(cpu_percent) OVER w AS avg_cpu,
    STDDEV(cpu_percent) OVER w AS stddev_cpu,
    AVG(memory_resident_size_bytes) OVER w AS avg_memory,
    STDDEV(memory_resident_size_bytes) OVER w AS stddev_memory
  FROM ProcessSample
  WHERE timestamp >= DATEADD(day, -6, CURRENT_TIMESTAMP)
  WINDOW w AS (PARTITION BY entity_name, command_name ORDER BY timestamp 
               ROWS BETWEEN 576 PRECEDING AND 1 PRECEDING)
)
SELECT 
  timestamp,
  entity_name,
  command_name,
  cpu_percent,
  memory_resident_size_bytes,
  CASE 
    WHEN cpu_percent > 90 THEN 'Critical'
    WHEN ABS(cpu_percent - avg_cpu) > 3 * stddev_cpu THEN 'Anomaly'
    ELSE 'Normal'
  END AS cpu_status,
  CASE 
    WHEN ABS(memory_resident_size_bytes - avg_memory) > 3 * stddev_memory THEN 'Anomaly'
    ELSE 'Normal'
  END AS memory_status
FROM process_stats
WHERE cpu_status IN ('Critical', 'Anomaly') OR memory_status = 'Anomaly';

-- SystemSample: Focus on system-wide metrics
WITH system_stats AS (
  SELECT 
    timestamp,
    entity_name,
    cpu_percent,
    memory_used_percent,
    load_average_five_minute,
    swap_used_bytes,
    AVG(cpu_percent) OVER w AS avg_cpu,
    STDDEV(cpu_percent) OVER w AS stddev_cpu,
    AVG(memory_used_percent) OVER w AS avg_memory,
    STDDEV(memory_used_percent) OVER w AS stddev_memory,
    AVG(load_average_five_minute) OVER w AS avg_load,
    STDDEV(load_average_five_minute) OVER w AS stddev_load
  FROM SystemSample
  WHERE timestamp >= DATEADD(day, -6, CURRENT_TIMESTAMP)
  WINDOW w AS (PARTITION BY entity_name ORDER BY timestamp 
               ROWS BETWEEN 576 PRECEDING AND 1 PRECEDING)
)
SELECT 
  timestamp,
  entity_name,
  cpu_percent,
  memory_used_percent,
  load_average_five_minute,
  CASE 
    WHEN cpu_percent > 90 OR memory_used_percent > 90 THEN 'Critical'
    WHEN ABS(cpu_percent - avg_cpu) > 3 * stddev_cpu OR
         ABS(memory_used_percent - avg_memory) > 3 * stddev_memory THEN 'Anomaly'
    ELSE 'Normal'
  END AS system_status,
  CASE 
    WHEN ABS(load_average_five_minute - avg_load) > 3 * stddev_load THEN 'Anomaly'
    ELSE 'Normal'
  END AS load_status
FROM system_stats
WHERE system_status IN ('Critical', 'Anomaly') OR load_status = 'Anomaly';

-- Python code for anomaly detection system
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class MetricsAnomalyDetector:
    def __init__(self, lookback_days=6, sample_interval_minutes=15):
        self.lookback_days = lookback_days
        self.sample_interval = sample_interval_minutes
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AnomalyDetector')
        
    def detect_anomalies(self, df, metrics, entity_column='entity_name', 
                        timestamp_column='timestamp'):
        """
        Detect anomalies in specified metrics using z-score method
        
        Parameters:
        df: DataFrame containing the metrics
        metrics: List of metric columns to analyze
        entity_column: Column name for entity identification
        timestamp_column: Column name for timestamp
        """
        results = []
        
        for entity in df[entity_column].unique():
            entity_data = df[df[entity_column] == entity].copy()
            
            for metric in metrics:
                try:
                    # Calculate rolling statistics
                    window_size = int((self.lookback_days * 24 * 60) / self.sample_interval)
                    rolling_mean = entity_data[metric].rolling(window=window_size).mean()
                    rolling_std = entity_data[metric].rolling(window=window_size).std()
                    
                    # Calculate z-scores
                    z_scores = np.abs((entity_data[metric] - rolling_mean) / rolling_std)
                    
                    # Detect anomalies (z-score > 3)
                    anomalies = z_scores > 3
                    
                    if anomalies.any():
                        anomaly_times = entity_data[anomalies][timestamp_column]
                        anomaly_values = entity_data[anomalies][metric]
                        
                        for time, value in zip(anomaly_times, anomaly_values):
                            results.append({
                                'timestamp': time,
                                'entity': entity,
                                'metric': metric,
                                'value': value,
                                'z_score': z_scores[anomalies][time],
                                'severity': 'Critical' if z_scores[anomalies][time] > 5 else 'Warning'
                            })
                            
                except Exception as e:
                    self.logger.error(f"Error processing {metric} for {entity}: {str(e)}")
                    
        return pd.DataFrame(results)

    def detect_network_anomalies(self, df):
        """Detect anomalies in NetworkSample metrics"""
        metrics = [
            'receive_bytes_per_second',
            'transmit_bytes_per_second',
            'receive_errors_per_second',
            'transmit_errors_per_second'
        ]
        return self.detect_anomalies(df, metrics)
    
    def detect_storage_anomalies(self, df):
        """Detect anomalies in StorageSample metrics"""
        metrics = [
            'disk_used_percent',
            'read_bytes_per_second',
            'write_bytes_per_second',
            'total_utilization_percent'
        ]
        return self.detect_anomalies(df, metrics)
    
    def detect_process_anomalies(self, df):
        """Detect anomalies in ProcessSample metrics"""
        metrics = [
            'cpu_percent',
            'memory_resident_size_bytes',
            'io_read_bytes_per_second',
            'io_write_bytes_per_second'
        ]
        return self.detect_anomalies(df, metrics)
    
    def detect_system_anomalies(self, df):
        """Detect anomalies in SystemSample metrics"""
        metrics = [
            'cpu_percent',
            'memory_used_percent',
            'load_average_five_minute',
            'swap_used_bytes'
        ]
        return self.detect_anomalies(df, metrics)

# Example usage
if __name__ == "__main__":
    detector = MetricsAnomalyDetector()
    
    # Example for NetworkSample
    network_df = pd.read_sql("""
        SELECT timestamp, entity_name, receive_bytes_per_second, 
               transmit_bytes_per_second, receive_errors_per_second, 
               transmit_errors_per_second
        FROM NetworkSample
        WHERE timestamp >= DATEADD(minute, -15, CURRENT_TIMESTAMP)
    """, connection)
    
    network_anomalies = detector.detect_network_anomalies(network_df)
    
    if not network_anomalies.empty:
        logging.info(f"Found {len(network_anomalies)} network anomalies")
        # Send alerts or take action based on anomalies

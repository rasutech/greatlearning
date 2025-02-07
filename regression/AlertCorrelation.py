-- NetworkSample Anomalies
-- Detects network throughput and error anomalies
"""SELECT 
    timestamp,
    entity_name,
    interface_name,
    receive_bytes_per_second,
    transmit_bytes_per_second,
    receive_errors_per_second,
    transmit_errors_per_second,
    -- Compare current values to historical averages
    CASE 
        WHEN receive_bytes_per_second > (SELECT percentile(receive_bytes_per_second, 95) 
                                       FROM NetworkSample 
                                       WHERE timestamp >= NOW() - 7 DAYS)
        THEN 'High Receive Traffic'
        WHEN receive_errors_per_second > (SELECT percentile(receive_errors_per_second, 95) 
                                        FROM NetworkSample 
                                        WHERE timestamp >= NOW() - 7 DAYS)
        THEN 'High Receive Errors'
        ELSE 'Normal'
    END as network_status
FROM NetworkSample
WHERE timestamp >= NOW() - 30 MINUTES;

-- StorageSample Anomalies
-- Monitors disk usage and I/O patterns
SELECT 
    timestamp,
    entity_name,
    device,
    disk_used_percent,
    disk_free_percent,
    read_bytes_per_second,
    write_bytes_per_second,
    CASE
        WHEN disk_used_percent > 90 THEN 'Critical Space'
        WHEN disk_used_percent > (SELECT AVG(disk_used_percent) + 2 * STDDEV(disk_used_percent)
                                FROM StorageSample
                                WHERE timestamp >= NOW() - 7 DAYS)
        THEN 'Unusual Usage'
        ELSE 'Normal'
    END as storage_status,
    CASE
        WHEN total_utilization_percent > (SELECT percentile(total_utilization_percent, 95)
                                        FROM StorageSample
                                        WHERE timestamp >= NOW() - 7 DAYS)
        THEN 'High IO'
        ELSE 'Normal'
    END as io_status
FROM StorageSample
WHERE timestamp >= NOW() - 30 MINUTES;

-- ProcessSample Anomalies
-- Identifies process-level resource usage anomalies
WITH process_baselines AS (
    SELECT 
        command_name,
        AVG(cpu_percent) as avg_cpu,
        STDDEV(cpu_percent) as stddev_cpu,
        AVG(memory_resident_size_bytes) as avg_memory,
        STDDEV(memory_resident_size_bytes) as stddev_memory
    FROM ProcessSample
    WHERE timestamp >= NOW() - 7 DAYS
    GROUP BY command_name
)
SELECT 
    p.timestamp,
    p.entity_name,
    p.command_name,
    p.cpu_percent,
    p.memory_resident_size_bytes,
    p.io_read_bytes_per_second,
    p.io_write_bytes_per_second,
    CASE
        WHEN p.cpu_percent > pb.avg_cpu + 3 * pb.stddev_cpu THEN 'CPU Spike'
        WHEN p.memory_resident_size_bytes > pb.avg_memory + 3 * pb.stddev_memory THEN 'Memory Spike'
        ELSE 'Normal'
    END as process_status
FROM ProcessSample p
JOIN process_baselines pb ON p.command_name = pb.command_name
WHERE p.timestamp >= NOW() - 30 MINUTES;

-- SystemSample Anomalies
-- Monitors system-wide performance metrics
WITH system_baselines AS (
    SELECT 
        entity_name,
        AVG(cpu_percent) as avg_cpu,
        STDDEV(cpu_percent) as stddev_cpu,
        AVG(memory_used_percent) as avg_memory,
        STDDEV(memory_used_percent) as stddev_memory,
        AVG(load_average_five_minute) as avg_load,
        STDDEV(load_average_five_minute) as stddev_load
    FROM SystemSample
    WHERE timestamp >= NOW() - 7 DAYS
    GROUP BY entity_name
)
SELECT 
    s.timestamp,
    s.entity_name,
    s.cpu_percent,
    s.memory_used_percent,
    s.load_average_five_minute,
    s.swap_used_bytes,
    CASE
        WHEN s.cpu_percent > 90 THEN 'Critical CPU'
        WHEN s.cpu_percent > sb.avg_cpu + 2 * sb.stddev_cpu THEN 'High CPU'
        WHEN s.memory_used_percent > 90 THEN 'Critical Memory'
        WHEN s.memory_used_percent > sb.avg_memory + 2 * sb.stddev_memory THEN 'High Memory'
        WHEN s.load_average_five_minute > sb.avg_load + 2 * sb.stddev_load THEN 'High Load'
        ELSE 'Normal'
    END as system_status
FROM SystemSample s
JOIN system_baselines sb ON s.entity_name = sb.entity_name
WHERE s.timestamp >= NOW() - 30 MINUTES;

-- Metric Table Anomalies
-- Focuses on service errors and performance
SELECT 
    timestamp,
    entity.name,
    service.name,
    metric_name,
    error.group.name,
    error.group.message,
    COUNT(*) as occurrence_count,
    SUM(CASE WHEN success = false THEN 1 ELSE 0 END) as error_count,
    ROUND(SUM(CASE WHEN success = false THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as error_rate
FROM Metric
WHERE timestamp >= NOW() - 30 MINUTES
GROUP BY 
    timestamp,
    entity.name,
    service.name,
    metric_name,
    error.group.name,
    error.group.message
HAVING 
    error_rate > (
        SELECT AVG(error_rate) + 2 * STDDEV(error_rate)
        FROM (
            SELECT 
                service.name,
                timestamp,
                SUM(CASE WHEN success = false THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as error_rate
            FROM Metric
            WHERE timestamp >= NOW() - 7 DAYS
            GROUP BY service.name, timestamp
        ) historical_errors
    )
    OR error_count > 5; """

-- Usage Instructions:
-- 1. Each query looks at the last 30 minutes of data and compares it against 7-day baselines
-- 2. Anomaly detection uses various statistical methods:
--    - Standard deviation based thresholds (2-3 sigma)
--    - 95th percentile comparisons
--    - Absolute thresholds for critical conditions
-- 3. Results include both the raw metrics and interpreted status
-- 4. Queries can be modified to adjust:
--    - Time windows (currently 30 min for current data, 7 days for baseline)
--    - Threshold sensitivity
--    - Metrics being monitored

-- Best Practices for New Relic:
-- 1. Use FACET for grouping when creating alerts
-- 2. Add TIMESERIES clauses when setting up continuous monitoring
-- 3. Consider adding LIMIT clauses for large result sets
-- 4. Add appropriate WHERE clauses to filter by environment or other tags

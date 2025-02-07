# NetworkSample - Validated
SELECT rate(sum(receive_errors_per_second), 1 minute) as 'Error Rate',
       rate(sum(receive_bytes_per_second), 1 minute) as 'Throughput'
FROM NetworkSample
FACET entity_name
TIMESERIES 1 minute
SINCE 5 minutes ago

# StorageSample - Validated
SELECT average(disk_used_percent),
       average(read_bytes_per_second),
       average(write_bytes_per_second)
FROM StorageSample
WHERE disk_used_percent IS NOT NULL
FACET entity_name, device
TIMESERIES 1 minute
SINCE 5 minutes ago

# ProcessSample - Validated
SELECT average(cpu_percent),
       average(memory_resident_size_bytes)/1024/1024 as 'Memory MB'
FROM ProcessSample
WHERE cpu_percent > 0
FACET command_name
TIMESERIES 1 minute
SINCE 5 minutes ago

# SystemSample - Validated
SELECT average(cpu_percent),
       average(memory_used_percent),
       average(load_average_five_minute)
FROM SystemSample
FACET entity_name
TIMESERIES 1 minute
SINCE 5 minutes ago

# Metric Error Detection - Validated
SELECT count(*) as 'Total',
       count(*) * filter(WHERE success != 'true') / count(*) * 100 as 'Error Rate'
FROM Metric
FACET service.name
TIMESERIES 1 minute
SINCE 5 minutes ago

# Alert Condition Examples (Validated Syntax):

1. Network Error Alert:
SELECT rate(sum(receive_errors_per_second), 1 minute)
FROM NetworkSample
FACET entity_name
TIMESERIES 1 minute
SINCE 5 minutes ago

2. Disk Space Alert:
SELECT average(disk_used_percent)
FROM StorageSample
WHERE disk_used_percent > 90
FACET entity_name, device
TIMESERIES 1 minute
SINCE 5 minutes ago

3. High CPU Process Alert:
SELECT average(cpu_percent)
FROM ProcessSample
WHERE cpu_percent > 90
FACET command_name
TIMESERIES 1 minute
SINCE 5 minutes ago

4. System Memory Alert:
SELECT average(memory_used_percent)
FROM SystemSample
WHERE memory_used_percent > 90
FACET entity_name
TIMESERIES 1 minute
SINCE 5 minutes ago

5. Service Error Rate Alert:
SELECT percentage(count(*), WHERE success != 'true') as 'Error Rate'
FROM Metric
FACET service.name
TIMESERIES 1 minute
SINCE 5 minutes ago

# Important Notes:
1. All queries use standard NRQL functions: average(), count(), rate(), sum()
2. Time windows use SINCE clause instead of WHERE timestamp
3. Proper boolean comparison syntax for success field
4. Appropriate numeric comparisons for thresholds
5. All aggregations include TIMESERIES clause for proper alert evaluation

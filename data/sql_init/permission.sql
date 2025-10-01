-- permission.sql  (runs only on first init when /var/lib/mysql is empty)

-- Ensure DB exists (harmless if classicmodels.sql also creates it)
CREATE DATABASE IF NOT EXISTS classicmodels;

-- Create the ingestion user
CREATE USER IF NOT EXISTS 'datahub'@'%' IDENTIFIED BY 'datahub';

-- Minimal grants required for DataHub + profiling (GE uses TEMP TABLEs)
GRANT SELECT, SHOW VIEW, CREATE TEMPORARY TABLES
ON classicmodels.* TO 'datahub'@'%';

-- (Optional) let the user read metadata schemas; harmless and useful
GRANT SELECT ON information_schema.* TO 'datahub'@'%';
GRANT SELECT ON performance_schema.* TO 'datahub'@'%';

FLUSH PRIVILEGES;
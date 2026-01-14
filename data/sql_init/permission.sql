-- permission.sql

-- Ensure DB exists
CREATE DATABASE IF NOT EXISTS classicmodels;

-- Create the ingestion user
-- Menggunakan 'IF NOT EXISTS' agar tidak error jika script dijalankan ulang
CREATE USER IF NOT EXISTS 'datahub'@'%' IDENTIFIED BY 'datahub';

-- Grant akses ke database target (classicmodels)
-- Ini SUDAH CUKUP agar datahub bisa membaca metadata tabel ini di information_schema
GRANT SELECT, SHOW VIEW, CREATE TEMPORARY TABLES
ON classicmodels.* TO 'datahub'@'%';

-- performance_schema Boleh di-grant (opsional, bagus untuk profiling query)
GRANT SELECT ON performance_schema.* TO 'datahub'@'%';

FLUSH PRIVILEGES;
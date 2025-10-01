CREATE USER IF NOT EXISTS 'datahub'@'%' IDENTIFIED BY 'datahub';
GRANT SELECT, SHOW VIEW ON classicmodels.* TO 'datahub'@'%';
FLUSH PRIVILEGES;
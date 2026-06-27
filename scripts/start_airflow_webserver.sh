#!/usr/bin/env bash
set -e

echo "Airflow UI: http://localhost:8080  (login: admin / admin)"
exec airflow webserver

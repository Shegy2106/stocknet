#!/bin/bash

# Start influxd in the background
influxd &

# Wait for a moment to ensure influxd starts properly
sleep 5

influx setup --username $INFLUXDB_USER --password $INFLUXDB_PASSWORD --org $INFLUXDB_ORG --bucket $INFLUXDB_BUCKET --token $INFLUXDB_TOKEN --force

tail -f /dev/null

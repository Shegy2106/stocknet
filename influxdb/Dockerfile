FROM influxdb:latest

# Copy the script into the container
COPY ../influxdb/start.sh /start.sh

# Give execution rights on the script
RUN chmod +x /start.sh


EXPOSE 8086


# Set the script to run when the container starts
CMD ["/start.sh"]

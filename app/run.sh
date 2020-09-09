#!/bin/sh
sudo gunicorn -c config.py main:app --workers 8 --worker-class uvicorn.workers.UvicornWorker --bind=localhost:80 --timeout=300
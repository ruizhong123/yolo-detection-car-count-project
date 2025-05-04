#!/bin/sh
echo "Starting entrypoint script..."
python manage.py collectstatic --noinput
python manage.py migrate --noinput
exec gunicorn --bind 0.0.0.0:8000 --workers 1 --worker-class gthread  APP_SETTING.wsgi:application


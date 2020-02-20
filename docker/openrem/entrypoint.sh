#!/bin/sh

if [ "$DATABASE" = "postgres" ]
then
    echo "Waiting for postgres..."

    while ! nc -z $SQL_HOST $SQL_PORT; do
      sleep 0.1
    done

    echo "PostgreSQL started"
fi
OPENREM_PATH="/opt/venv/lib/python3.8/site-packages/openrem"
python $OPENREM_PATH/manage.py flush --no-input
python $OPENREM_PATH/manage.py migrate
#python $OPENREM_PATH/manage.py collectstatic --no-input --clear

exec "$@"
# pull official base image
FROM python:3.10-slim

RUN useradd -ms /bin/bash app \
 && adduser www-data app
ENV HOME=/home/app
ENV APP_HOME=/home/app/openrem
ENV APP_VENV=/home/app/venv

RUN mkdir $APP_HOME \
 && mkdir $APP_HOME/mediafiles \
 && mkdir $APP_HOME/staticfiles \
 && mkdir $HOME/pixelmed
ADD http://www.dclunie.com/pixelmed/software/webstart/pixelmed.jar $HOME/pixelmed/
WORKDIR $HOME

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN mkdir -p /usr/share/man/man1
RUN apt-get update && apt-get -y dist-upgrade && apt install -y netcat dcmtk default-jre gettext supervisor \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mv $HOME/stuff/v1initial.py $APP_HOME/remapp/migrations/0001_initial.py.1-0-upgrade \
 && mv $APP_HOME/openremproject/wsgi.py.example $APP_HOME/openremproject/wsgi.py

RUN chmod -R 775 $APP_HOME/mediafiles \
 && chmod -R 775 $APP_HOME/staticfiles \
 && chmod -R g+s $APP_HOME/mediafiles \
 && chmod -R g+s $APP_HOME/staticfiles \
 && mkdir /logs \
 && mkdir /imports \
 && chmod 555 $HOME/pixelmed/pixelmed.jar

RUN pip install -e .

WORKDIR $APP_HOME

# run entrypoint.sh
ENTRYPOINT ["/home/app/openrem/docker/entrypoint.prod.sh"]
CMD ["/usr/bin/supervisord", "-c", "/home/app/openrem/docker/supervisord.conf"]
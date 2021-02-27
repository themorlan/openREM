# pull official base image
FROM python:3.8-slim

RUN useradd -ms /bin/bash app \
 && adduser www-data app
ENV HOME=/home/app
ENV APP_HOME=/home/app/openrem
ENV APP_VENV=/home/app/venv
USER app
RUN mkdir $APP_HOME \
 && mkdir $APP_HOME/mediafiles \
 && mkdir $APP_HOME/staticfiles \
 && mkdir $HOME/pixelmed
ADD http://www.dclunie.com/pixelmed/software/webstart/pixelmed.jar $HOME/pixelmed/
USER root
WORKDIR $HOME

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN mkdir -p /usr/share/man/man1
RUN apt-get update && apt-get -y dist-upgrade && apt install -y netcat dcmtk default-jre gettext \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

USER app
RUN python -m venv $APP_VENV
# Make sure we use the virtualenv:
ENV PATH="$APP_VENV/bin:$PATH"
# install dependencies
# hadolint ignore=DL3013
RUN pip install --upgrade pip
COPY --chown=app:app ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .
RUN mv $HOME/stuff/0002_0_7_fresh_install_add_median.py.inactive $APP_HOME/remapp/migrations/ \
 && mv $HOME/stuff/v1initial.py $APP_HOME/remapp/migrations/0001_initial.py.1-0-upgrade \
 && mv $APP_HOME/openremproject/wsgi.py.example $APP_HOME/openremproject/wsgi.py

USER root
RUN chmod -R 775 $APP_HOME/mediafiles \
 && chmod -R 775 $APP_HOME/staticfiles \
 && chmod -R g+s $APP_HOME/mediafiles \
 && chmod -R g+s $APP_HOME/staticfiles \
 && mkdir /logs && chown app:app /logs \
 && mkdir /imports && chown app:app /imports \
 && chown app:app $HOME/pixelmed/pixelmed.jar \
 && chmod 555 $HOME/pixelmed/pixelmed.jar

USER app
RUN pip install -e .

WORKDIR $APP_HOME

# run entrypoint.sh
ENTRYPOINT ["/home/app/openrem/entrypoint.prod.sh"]
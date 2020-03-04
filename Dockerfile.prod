# pull official base image
FROM python:slim

RUN useradd -ms /bin/bash app
RUN adduser www-data app
ENV HOME=/home/app
ENV APP_HOME=/home/app/openrem
ENV APP_VENV=/home/app/venv
RUN mkdir $APP_HOME
RUN mkdir $APP_HOME/mediafiles
RUN mkdir $APP_HOME/staticfiles
WORKDIR $HOME

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get -y dist-upgrade
# enable nc command in entrypoint
RUN apt install -y netcat

USER app
RUN python -m venv $APP_VENV
# Make sure we use the virtualenv:
ENV PATH="$APP_VENV/bin:$PATH"
# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY . .

USER root
RUN chown -R app:app $HOME
RUN chown -R app:app $HOME/.[^.]*
RUN chmod -R 775 $APP_HOME/mediafiles
RUN chmod -R g+s $APP_HOME/mediafiles

USER app
RUN pip install -e .

WORKDIR $APP_HOME

# run entrypoint.sh
ENTRYPOINT ["/home/app/openrem/entrypoint.prod.sh"]
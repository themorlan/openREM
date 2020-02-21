# pull official base image
FROM python:slim

# set work directory
WORKDIR /usr/src

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get -y dist-upgrade
# enable nc command in entrypoint
RUN apt install -y netcat

RUN python -m venv /opt/venv
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"
# install dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY code/ .
RUN pip install -e .

# Move into the /usr/src/openrem folder
WORKDIR openrem

COPY ./entrypoint.sh .

# run entrypoint.sh
ENTRYPOINT ["/usr/src/openrem/entrypoint.sh"]
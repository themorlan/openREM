# Django settings for OpenREM project.

import os
import multiprocessing

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Choose your database and fill in the details below. If testing, you
# can use the sqlite3 database as it doesn't require any further configuration
DATABASES = {
    "default": {
        "ENGINE": os.environ.get("SQL_ENGINE", "django.db.backends.sqlite3"),
        "NAME": os.environ.get("POSTGRES_DB", os.path.join(BASE_DIR, "db.sqlite3")),
        "USER": os.environ.get("POSTGRES_USER", "user"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", "password"),
        "HOST": os.environ.get("SQL_HOST", "localhost"),
        "PORT": os.environ.get("SQL_PORT", "5432"),
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.AutoField"

SECRET_KEY = os.environ.get("SECRET_KEY", default="shouldn'tbethisone")

DEBUG = int(os.environ.get("DEBUG", default=0))

# 'DJANGO_ALLOWED_HOSTS' should be a single string of hosts with a space between each.
# For example: 'DJANGO_ALLOWED_HOSTS=localhost 127.0.0.1 [::1]'
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", default="localhost").split(" ")

MEDIA_URL = os.environ.get("MEDIA_URL", default="/media/")
MEDIA_ROOT = os.environ.get("MEDIA_ROOT", default=os.path.join(BASE_DIR, "mediafiles"))
STATIC_URL = os.environ.get("STATIC_URL", default="/static/")
STATIC_ROOT = os.environ.get(
    "STATIC_ROOT", default=os.path.join(BASE_DIR, "staticfiles")
)
JS_REVERSE_OUTPUT_PATH = os.path.join(STATIC_ROOT, "js", "django_reverse")
VIRTUAL_DIRECTORY = os.environ.get("VIRTUAL_DIRECTORY", default="")
FIXTURES_DIRS = os.path.join(BASE_DIR, "/remapp/fixtures")

ROOT_PROJECT = os.path.join(os.path.split(__file__)[0], "..")

# Local time zone for this installation. Choices can be found here:
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# although not all choices may be available on all operating systems.
# In a Windows environment this must be set to your system time zone.
TIME_ZONE = os.environ.get("TIME_ZONE", default="Europe/London")

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = os.environ.get("LANGUAGE_CODE", default="en-us")

SITE_ID = 1

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = os.environ.get("USE_I18N", default=True)

# If you set this to False, Django will not format dates, numbers and
# calendars according to the current locale.
USE_L10N = os.environ.get("USE_L10N", default=True)

# If you set this to False, Django will not use timezone-aware datetimes.
USE_TZ = os.environ.get("USE_TZ", default=False)

XLSX_DATE = os.environ.get("XLSX_DATE", default="dd/mm/yyyy")
XLSX_TIME = os.environ.get("XLSX_TIME", default="hh:mm:ss")

# URL name of the login page (as defined in urls.py)
LOGIN_URL = "login"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [
            # insert your TEMPLATE_DIRS here
        ],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                # Insert your TEMPLATE_CONTEXT_PROCESSORS here or use this
                # list if you haven't customized them:
                "django.contrib.auth.context_processors.auth",
                "django.template.context_processors.debug",
                "django.template.context_processors.i18n",
                "django.template.context_processors.media",
                "django.template.context_processors.static",
                "django.template.context_processors.tz",
                "django.template.context_processors.request",  # Added by ETM
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "openremproject.urls"

# Python dotted path to the WSGI application used by Django's runserver.
WSGI_APPLICATION = "openremproject.wsgi.application"

INSTALLED_APPS = (
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.sites",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.admin",
    "remapp",
    "django_filters",
    "django.contrib.humanize",
    "solo",
    "crispy_forms",
    "django_js_reverse",
    "huey.contrib.djhuey",
)

CRISPY_TEMPLATE_PACK = "bootstrap3"

TASK_QUEUE_ROOT = BASE_DIR

HUEY_CLASS = "huey.PriorityRedisHuey"
HUEY_WORKER_TYPE = "process"
HUEY_NUMBER_OF_WORKERS = int(
    os.environ.get("HUEY_NUMBER_OF_WORKERS", multiprocessing.cpu_count())
)

HUEY = {
    "huey_class": HUEY_CLASS,
    "immediate": False,
    "connection": {
        "host": os.environ.get("REDIS_HOST", "localhost"),
        "port": int(os.environ.get("REDIS_PORT", 6379)),
    },
    "consumer": {
        "workers": HUEY_NUMBER_OF_WORKERS,
        "worker_type": HUEY_WORKER_TYPE,
    },
}

on_rtd = os.environ.get("READTHEDOCS") == "True"
if not on_rtd:
    LOG_ROOT = os.environ.get("LOG_ROOT", default=MEDIA_ROOT)
else:
    LOG_ROOT = BASE_DIR

# Ensure log directory exists
if not os.path.exists(LOG_ROOT):
    os.makedirs(LOG_ROOT, exist_ok=True)

LOG_FILENAME = os.path.join(LOG_ROOT, "openrem.log")
QR_FILENAME = os.path.join(LOG_ROOT, "openrem_qr.log")
EXTRACTOR_FILENAME = os.path.join(LOG_ROOT, "openrem_extractor.log")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {message}",
            "style": "{",
        },
    },
    "filters": {"require_debug_false": {"()": "django.utils.log.RequireDebugFalse"}},
    "handlers": {
        "mail_admins": {
            "level": "ERROR",
            "filters": ["require_debug_false"],
            "class": "django.utils.log.AdminEmailHandler",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": "openrem_rdsr.log",
            "formatter": "verbose",
        },
        "qr_file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": "openrem_qrscu.log",
            "formatter": "verbose",
        },
        "extractor_file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": "openrem_extractor.log",
            "formatter": "verbose",
        },
    },
    "loggers": {
        "django.request": {
            "handlers": ["mail_admins"],
            "level": "ERROR",
            "propagate": True,
        },
        "remapp": {"handlers": ["file"], "level": "INFO"},
        "remapp.netdicom.qrscu": {
            "handlers": ["qr_file"],
            "level": "INFO",
            "propagate": False,
        },
        "remapp.extractors": {
            "handlers": ["extractor_file"],
            "level": "INFO",
            "propagate": False,
        },
        "remapp.extractors.rdsr": {
            "handlers": ["file"],
            "level": "DEBUG",
            "propagate": True,
        },
    },
}
LOGGING["handlers"]["file"]["filename"] = LOG_FILENAME  # General logs
LOGGING["handlers"]["qr_file"]["filename"] = QR_FILENAME  # Query Retrieve SCU logs
LOGGING["handlers"]["extractor_file"]["filename"] = EXTRACTOR_FILENAME  # Extractor logs

# Set log message format. Options are 'verbose' or 'simple'. Recommend leaving as 'verbose'.
LOGGING["handlers"]["file"]["formatter"] = "verbose"  # General logs
LOGGING["handlers"]["qr_file"]["formatter"] = "verbose"  # Query Retrieve SCU logs
LOGGING["handlers"]["extractor_file"]["formatter"] = "verbose"  # Extractor logs

# Set the log level. Options are 'DEBUG', 'INFO', 'WARNING', 'ERROR', and 'CRITICAL', with progressively less logging.
LOGGING["loggers"]["remapp"]["level"] = os.environ.get(
    "LOG_LEVEL", default="INFO"
)  # General logs
# Query Retrieve SCU logs
LOGGING["loggers"]["remapp.netdicom.qrscu"]["level"] = os.environ.get(
    "LOG_LEVEL_QRSCU", default="INFO"
)
# Extractor logs
LOGGING["loggers"]["remapp.extractors"]["level"] = os.environ.get(
    "LOG_LEVEL_EXTRACTOR", default="INFO"
)

# Dummy locations of various tools for DICOM RDSR creation from CT images. Don't set value here - copy variables into
# # local_settings.py and configure there.
DCMTK_PATH = "/usr/bin"
DCMCONV = os.path.join(DCMTK_PATH, "dcmconv.exe")
DCMMKDIR = os.path.join(DCMTK_PATH, "dcmmkdir.exe")
JAVA_EXE = "/usr/bin/java"
JAVA_OPTIONS = "-Xms256m -Xmx512m -Xss1m -cp"
PIXELMED_JAR = "/home/app/pixelmed/pixelmed.jar"
PIXELMED_JAR_OPTIONS = "-Djava.awt.headless=true com.pixelmed.doseocr.OCR -"

# E-mail server settings - see https://docs.djangoproject.com/en/1.8/topics/email/
EMAIL_HOST = os.environ.get("EMAIL_HOST", default="localhost")
EMAIL_PORT = os.environ.get("EMAIL_PORT", default="25")
EMAIL_HOST_USER = os.environ.get("EMAIL_HOST_USER", default="")
EMAIL_HOST_PASSWORD = os.environ.get("EMAIL_HOST_PASSWORD", default="")  # nosec
EMAIL_USE_TLS = int(os.environ.get("EMAIL_USE_TLS", default=0))
EMAIL_USE_SSL = int(os.environ.get("EMAIL_USE_SSL", default=0))
EMAIL_DOSE_ALERT_SENDER = os.environ.get(
    "EMAIL_DOSE_ALERT_SENDER", default="your.alert@email.address"
)
EMAIL_OPENREM_URL = os.environ.get(
    "EMAIL_OPENREM_URL", default="http://your.openrem.server"
)

# Save charts as HTML files in a charts sub-folder of the MEDIA_ROOT folder on the OpenREM server.
SAVE_CHARTS_AS_HTML = False

# Ignore the Device Observer UID of these equipment models when creating display name entries during import of DICOM
# RDSR data using the rdsr.py extractor.
# See https://docs.openrem.org/en/latest/env_variables.html#device-observer-uid-settings.
IGNORE_DEVICE_OBSERVER_UID_FOR_THESE_MODELS = []

DOCKER_INSTALL = int(os.environ.get("DOCKER_INSTALL", default=False))

LOCALE_PATHS = (os.path.join(BASE_DIR, "locale"),)

try:
    from .local_settings import *  # NOQA: F401
except ImportError:
    # For Docker builds, there will not be a local_settings.py, 'local settings' are passed via environment variables
    pass

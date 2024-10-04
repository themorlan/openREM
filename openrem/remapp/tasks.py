import os
import sys
import logging
from datetime import datetime, timedelta
import django
from hl7.hl7settings import *

# Setup django. This is required on windows, because process is created via spawn and
# django will not be initialized anymore then (On Linux this will only be executed once)
basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"
django.setup()

from .models import Hl7Message
from huey import crontab
from huey.contrib.djhuey import db_periodic_task

logger = logging.getLogger(__name__)

@db_periodic_task(crontab(hour='5'))
def del_old_hl7_messages_from_db():
    logger.info('removing old HL7-Messages')

    if Hl7Message.objects.exclude(study_date__gte=(datetime.today()-timedelta(days=HL7_STUDY_DATE_OFFSET))
                                 .strftime('%Y%m%d')).exists():
        logger.info('HL7-Messages found to delete.')
        Hl7Message.objects.exclude(study_date__gte=(datetime.today()-timedelta(days=HL7_STUDY_DATE_OFFSET))
                                  .strftime('%Y%m%d')).delete()
    else:
        logger.info('No old HL7-Messages found.')
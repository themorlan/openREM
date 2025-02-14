from remapp.tools.send_high_dose_alert_emails import send_ct_high_dose_alert_email
from django.core.exceptions import ObjectDoesNotExist
from django.utils.log import logger

def ct_siemens(dataset, g):
    """Extrahiert Daten aus Siemens CT DICOM Header"""
    # ... existing code ...
    
    try:
        g.number_of_events = (
            g.ctradiationdose_set.get().ctirradiationeventdata_set.count()
        )
        g.save()
    except ObjectDoesNotExist:
        logger.warning(
            "Study UID {0} of modality {1}. Unable to get event count!".format(
                g.study_instance_uid, get_value_kw("ManufacturerModelName", dataset)
            )
        )
    
    # Am Ende der Funktion nach erfolgreicher Speicherung
    send_ct_high_dose_alert_email(study_pk=g.pk) 
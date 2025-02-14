from remapp.tools.send_high_dose_alert_emails import send_ct_high_dose_alert_email

def ct_ge(dataset, g):
    """Extrahiert Daten aus GE CT DICOM Header"""
    # ... existing code ...
    
    # Am Ende der Funktion nach erfolgreicher Speicherung
    send_ct_high_dose_alert_email(study_pk=g.pk) 
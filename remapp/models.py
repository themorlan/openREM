# Die relevanten Felder sind:
alert_ctdi = models.FloatField(
    blank=True,
    null=True, 
    default=5.0,
    verbose_name="Alert level for CTDIvol from CT examination (mGy)"
)
send_high_dose_metric_alert_emails_ct = models.BooleanField(
    default=False,
    verbose_name="Send notification e-mails when alert levels for CTDIvol are exceeded?"
) 
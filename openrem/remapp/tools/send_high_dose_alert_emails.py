# This Python file uses the following encoding: utf-8
#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2012,2013  The Royal Marsden NHS Foundation Trust
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    Additional permission under section 7 of GPLv3:
#    You shall not make any use of the name of The Royal Marsden NHS
#    Foundation trust in connection with this Program in any press or
#    other public announcement without the prior written consent of
#    The Royal Marsden NHS Foundation Trust.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
..  module:: send_high_dose_alert_emails.
    :synopsis: Module to send high dose alert e-mails.

..  moduleauthor:: David Platten

"""

import os
import sys
import django
from django.db.models import Sum
from django.core.mail import EmailMultiAlternatives, send_mail
from django.conf import settings
from remapp.models import GeneralStudyModuleAttr, HighDoseMetricAlertSettings, SkinDoseMapResults, UserProfile
from datetime import timedelta
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist
from openremproject import settings
from socket import error as socket_error
from socket import gaierror as gai_error
from smtplib import SMTPException
from ssl import SSLError
import logging
from django.template.loader import render_to_string

# setup django/OpenREM
basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"
django.setup()


def send_rf_high_dose_alert_email(study_pk=None, test_message=None, test_user=None):
    """
    Function to send users a fluoroscopy high dose alert e-mail
    """
    logger = logging.getLogger(__name__)
    
    # Send a test message to the e-mail address contained in test_user
    if test_message:
        if test_user:
            try:
                # Versuche Templates zu laden mit expliziter Fehlerbehandlung
                try:
                    text_msg_content = render_to_string("remapp/email_test_template.txt")
                except Exception as e:
                    logger.error(f"Fehler beim Laden des Text-Templates: {str(e)}")
                    return f"Fehler beim Laden des Text-Templates: {str(e)}"

                try:
                    html_msg_content = render_to_string("remapp/email_test_template.html")
                except Exception as e:
                    logger.error(f"Fehler beim Laden des HTML-Templates: {str(e)}")
                    return f"Fehler beim Laden des HTML-Templates: {str(e)}"

                if not settings.EMAIL_DOSE_ALERT_SENDER:
                    msg = "EMAIL_DOSE_ALERT_SENDER nicht konfiguriert"
                    logger.error(msg)
                    return msg

                recipients = [test_user]
                msg_subject = "OpenREM e-mail test message"
                
                msg = EmailMultiAlternatives(
                    msg_subject,
                    text_msg_content, 
                    settings.EMAIL_DOSE_ALERT_SENDER,
                    recipients
                )
                msg.attach_alternative(html_msg_content, "text/html")
                msg.send()
                logger.info(f"Test-Email erfolgreich gesendet an {test_user}")
                
            except (SSLError, SMTPException, ValueError, gai_error, socket_error) as the_error:
                error_msg = f"Email-Versand fehlgeschlagen: {str(the_error)}"
                logger.error(error_msg)
                return error_msg
            except Exception as e:
                error_msg = f"Unerwarteter Fehler: {str(e)}"
                logger.error(error_msg)
                return error_msg
        return

    if study_pk:
        study = GeneralStudyModuleAttr.objects.get(pk=study_pk)
    else:
        return
    try:
        skinresult = SkinDoseMapResults.objects.get(
            general_study_module_attributes=study
        )
        peak_skin_dose = skinresult.peak_skin_dose
    except ObjectDoesNotExist:
        peak_skin_dose = None

    this_study_dap = (
        study.projectionxrayradiationdose_set.get()
        .accumxraydose_set.last()
        .accumintegratedprojradiogdose_set.get()
        .convert_gym2_to_cgycm2()
    )
    this_study_rp_dose = (
        study.projectionxrayradiationdose_set.get()
        .accumxraydose_set.last()
        .accumintegratedprojradiogdose_set.get()
        .dose_rp_total
    )
    accum_dap = (
        study.projectionxrayradiationdose_set.get()
        .accumxraydose_set.last()
        .accumintegratedprojradiogdose_set.get()
        .total_dap_delta_gym2_to_cgycm2()
    )
    accum_rp_dose = (
        study.projectionxrayradiationdose_set.get()
        .accumxraydose_set.last()
        .accumintegratedprojradiogdose_set.get()
        .dose_rp_total_over_delta_weeks
    )

    alert_levels = HighDoseMetricAlertSettings.objects.values(
        "show_accum_dose_over_delta_weeks",
        "alert_total_dap_rf",
        "alert_total_rp_dose_rf",
        "accum_dose_delta_weeks",
        "alert_skindose",
        "send_high_dose_metric_alert_emails_ref",
        "send_high_dose_metric_alert_emails_skin",
    )[0]

    if alert_levels["show_accum_dose_over_delta_weeks"]:
        patient_id = study.patientmoduleattr_set.values_list("patient_id", flat=True)[0]
        if patient_id:
            study_date = study.study_date
            week_delta = HighDoseMetricAlertSettings.objects.values_list(
                "accum_dose_delta_weeks", flat=True
            )[0]
            oldest_date = study_date - timedelta(weeks=week_delta)
            included_studies = GeneralStudyModuleAttr.objects.filter(
                modality_type__exact="RF",
                patientmoduleattr__patient_id__exact=patient_id,
                study_date__range=[oldest_date, study_date],
            )
        else:
            included_studies = None
            week_delta = None
    else:
        included_studies = None
        week_delta = None

    if this_study_dap is None:
        this_study_dap = 0
    if this_study_rp_dose is None:
        this_study_rp_dose = 0
    if accum_dap is None:
        accum_dap = 0
    if accum_rp_dose is None:
        accum_rp_dose = 0
    if peak_skin_dose is None:
        peak_skin_dose = 0
    accum_peak_skin_dose = peak_skin_dose
    if included_studies:
        accum_peak_skin_dose = list(
            included_studies.aggregate(
                Sum("skindosemapresults__peak_skin_dose")
            ).values()
        )[0]
        if accum_peak_skin_dose is None:
            accum_peak_skin_dose = 0

    alert_for_dap_or_rp_dose = alert_levels[
        "send_high_dose_metric_alert_emails_ref"
    ] and (
        this_study_dap >= alert_levels["alert_total_dap_rf"]
        or this_study_rp_dose >= alert_levels["alert_total_rp_dose_rf"]
        or accum_dap >= alert_levels["alert_total_dap_rf"]
        or accum_rp_dose >= alert_levels["alert_total_rp_dose_rf"]
    )

    alert_for_skin_dose = alert_levels["send_high_dose_metric_alert_emails_skin"] and (
        peak_skin_dose >= alert_levels["alert_skindose"]
        or accum_peak_skin_dose >= alert_levels["alert_skindose"]
    )

    if alert_for_dap_or_rp_dose or alert_for_skin_dose:

        projection_xray_dose_set = study.projectionxrayradiationdose_set.get()
        accumxraydose_set_all_planes = (
            projection_xray_dose_set.accumxraydose_set.select_related(
                "acquisition_plane"
            ).all()
        )

        content_dict = {
            "study": study,
            "accumxraydose_set_all_planes": accumxraydose_set_all_planes,
            "all_studies": included_studies,
            "week_delta": week_delta,
            "alert_levels": alert_levels,
            "studies_in_week_delta": included_studies,
            "server_url": settings.EMAIL_OPENREM_URL,
            "accum_peak_skin_dose": accum_peak_skin_dose,
        }

        text_msg_content = render_to_string(
            "remapp/rf_dose_alert_email_template.txt", content_dict
        )
        html_msg_content = render_to_string(
            "remapp/rf_dose_alert_email_template.html", content_dict
        )
        recipients = User.objects.filter(
            highdosemetricalertrecipients__receive_high_dose_metric_alerts__exact=True
        ).values_list("email", flat=True)
        if included_studies:
            oldest_accession = included_studies.order_by("study_date", "study_time")[
                0
            ].accession_number
        else:
            oldest_accession = study.accession_number
        msg_subject = f"OpenREM high dose alert {oldest_accession}"
        msg = EmailMultiAlternatives(
            msg_subject, text_msg_content, settings.EMAIL_DOSE_ALERT_SENDER, recipients
        )
        msg.attach_alternative(html_msg_content, "text/html")
        msg.send()

def send_import_success_email(study_pk, study_uid):
    """
    Sendet eine E-Mail-Benachrichtigung nach erfolgreichem Import von Dosisdaten.
    
    :param study_pk: Primary Key des importierten Studies
    :param study_uid: Study Instance UID des importierten Studies
    """
    try:
        study = GeneralStudyModuleAttr.objects.get(pk=study_pk)
        alert_settings = HighDoseMetricAlertSettings.objects.get()
        
        if not alert_settings.send_import_success_emails:
            return
            
        subject = f'OpenREM: Dosisdaten erfolgreich importiert'
        
        message = f"""
        Neue Dosisdaten wurden erfolgreich in OpenREM importiert:
        
        Study Instance UID: {study_uid}
        Modalität: {study.modality_type}
        Studien Datum: {study.study_date}
        """
        
        if study.patientmoduleattr_set.exists():
            patient = study.patientmoduleattr_set.get()
            message += f"""
            Patient ID: {patient.patient_id}
            Name: {patient.patient_name}
            """
            
        from_email = settings.EMAIL_FROM
        recipient_list = alert_settings.alert_recipient_emails()
        
        if recipient_list:
            send_mail(
                subject,
                message, 
                from_email,
                recipient_list,
                fail_silently=True
            )
            
    except Exception as e:
        logger.error(f"Fehler beim Senden der Import-Erfolgs-Email: {e}")

def send_ct_high_dose_alert_email(study_pk, max_ctdi, limit_ctdi):
    """Sendet eine Alarm-Email wenn der CTDIvol über dem Limit liegt"""
    logger = logging.getLogger(__name__)
    
    try:
        study = GeneralStudyModuleAttr.objects.get(pk=study_pk)
        alert_settings = HighDoseMetricAlertSettings.objects.get()
        
        logger.info(f"Prüfe CT Dosis-Alarm: max_ctdi={max_ctdi}, limit_ctdi={limit_ctdi}")
        
        if alert_settings.send_high_dose_metric_alert_emails_ct:
            equipment = study.generalequipmentmoduleattr_set.get()
            
            # Hole alle User Profile mit aktivierten Warnungen - KORRIGIERTE ABFRAGE
            user_profiles = UserProfile.objects.filter(
                user__highdosemetricalertrecipients__receive_high_dose_metric_alerts=True
            )
            logger.info(f"Gefundene User Profile mit aktivierten Warnungen: {user_profiles.count()}")
            
            for user_profile in user_profiles:
                try:
                    # Individuellen Schwellenwert berechnen
                    multiplier = user_profile.ct_dose_alert_multiplier
                    adjusted_ctdi_threshold = limit_ctdi * multiplier

                    logger.info(f"User {user_profile.user.email}: multiplier={multiplier}, "
                              f"adjusted_threshold={adjusted_ctdi_threshold}, max_ctdi={max_ctdi}")

                    # Prüfe ob Schwellenwert überschritten wurde
                    if max_ctdi > adjusted_ctdi_threshold:
                        logger.info(f"Schwellenwert überschritten für {user_profile.user.email}")
                        subject = f'CT Hohe Dosis Warnung - {equipment.station_name}'
                        
                        message = f"""CT Untersuchung mit erhöhter Dosis:

Studien UID: {study.study_instance_uid}
Untersuchungsdatum: {study.study_date}
Station: {equipment.station_name}

CTDIvol max: {max_ctdi:.1f} mGy (Schwellenwert: {adjusted_ctdi_threshold:.1f} mGy)

Dies ist eine automatische Benachrichtigung basierend auf Ihren persönlichen Schwellenwerten 
(Multiplikator: {multiplier:.1f})."""

                        # Füge Patient ID hinzu falls vorhanden
                        if study.patientmoduleattr_set.exists():
                            patient = study.patientmoduleattr_set.get()
                            message += f"\nPatient ID: {patient.patient_id}"

                        # Link zu den Details
                        message += f"\n\nDetails unter: {settings.EMAIL_OPENREM_URL}/openrem/ct/{study_pk}/"

                        try:
                            # Email senden
                            send_mail(
                                subject,
                                message,
                                settings.EMAIL_DOSE_ALERT_SENDER,
                                [user_profile.user.email],
                                fail_silently=False
                            )
                            logger.info(f"CT Dosis-Alarm Email erfolgreich gesendet an {user_profile.user.email}")
                        except Exception as mail_error:
                            logger.error(f"Fehler beim Email-Versand an {user_profile.user.email}: {str(mail_error)}")
                            
                except Exception as e:
                    logger.error(f"Fehler beim Verarbeiten des User Profiles {user_profile.user.email}: {str(e)}")
                    continue
        else:
            logger.info("CT Dosis-Alarm Emails sind deaktiviert in den Einstellungen")
                    
    except Exception as e:
        logger.error(f"Fehler beim Versenden der CT Dosis-Alarm Email: {str(e)}", exc_info=True)

def send_high_dose_alert_emails(study):
    """Sendet CT Dosis-Alarm Emails an berechtigte Empfänger mit individuellen Schwellenwerten"""
    logger = logging.getLogger(__name__)
    
    try:
        # Hole die Basis-Schwellenwerte aus den Einstellungen
        base_dlp_threshold = settings.ALERT_DLP_THRESHOLD
        base_ctdi_threshold = settings.ALERT_CTDI_THRESHOLD
        
        # Hole die Studien-Details
        total_dlp = study.total_dlp
        max_ctdi = study.ctdi_vol
        
        # Hole alle User Profile mit aktivierten Warnungen
        for user_profile in UserProfile.objects.filter(receive_high_dose_alert_emails=True):
            try:
                # Individuelle Schwellenwerte berechnen
                multiplier = user_profile.ct_dose_alert_multiplier
                adjusted_dlp_threshold = base_dlp_threshold * multiplier
                adjusted_ctdi_threshold = base_ctdi_threshold * multiplier
                
                # Prüfe ob Schwellenwerte überschritten wurden
                if total_dlp > adjusted_dlp_threshold or max_ctdi > adjusted_ctdi_threshold:
                    # Email-Nachricht vorbereiten
                    subject = 'CT Hohe Dosis Warnung'
                    
                    message = f"""CT Untersuchung mit erhöhter Dosis:

Studien UID: {study.study_instance_uid}
Untersuchungsdatum: {study.study_date}

DLP: {total_dlp:.1f} mGy·cm (Schwellenwert: {adjusted_dlp_threshold:.1f} mGy·cm)
CTDIvol max: {max_ctdi:.1f} mGy (Schwellenwert: {adjusted_ctdi_threshold:.1f} mGy)

Dies ist eine automatische Benachrichtigung basierend auf Ihren persönlichen Schwellenwerten 
(Multiplikator: {multiplier:.1f}).
"""
                    
                    # Füge Patienten-Info hinzu falls vorhanden
                    if study.patientmoduleattr_set.exists():
                        patient = study.patientmoduleattr_set.get()
                        message += f"\nPatient ID: {patient.patient_id}"
                    
                    # Link zu den Details
                    message += f"\n\nDetails unter: {settings.EMAIL_OPENREM_URL}/openrem/ct/{study.pk}/"
                    
                    # Email senden
                    send_mail(
                        subject,
                        message,
                        settings.EMAIL_DOSE_ALERT_SENDER,
                        [user_profile.user.email],
                        fail_silently=False
                    )
                    
                    logger.info(f"CT Dosis-Alarm Email gesendet an {user_profile.user.email}")
                    
            except Exception as e:
                logger.error(f"Fehler beim Verarbeiten des User Profiles {user_profile.user.email}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Fehler beim Versenden der CT Dosis-Alarm Emails: {str(e)}", exc_info=True)

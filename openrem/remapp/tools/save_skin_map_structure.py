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
..  module:: save_skin_map_structure.
    :synopsis: Module to save openskin structure

..  moduleauthor:: Ed McDonagh, David Platten, Wens Kong

"""
import gzip
import os
import pickle

from django.conf import settings

from remapp.models import HighDoseMetricAlertSettings
from .send_high_dose_alert_emails import send_rf_high_dose_alert_email


def save_openskin_structure(study, return_struct):
    # Save the return_structure as a pickle in a skin_maps sub-folder of the MEDIA_ROOT folder
    if study:
        study_date = study.study_date
        if study_date:
            skin_map_path = os.path.join(
                settings.MEDIA_ROOT,
                "skin_maps",
                f"{study_date.year:0>4}",
                f"{study_date.month:0>2}",
                f"{study_date.day:0>2}",
            )
        else:
            skin_map_path = os.path.join(settings.MEDIA_ROOT, "skin_maps")

        if not os.path.exists(skin_map_path):
            os.makedirs(skin_map_path)

        with gzip.open(
            os.path.join(skin_map_path, "skin_map_" + str(study.pk) + ".p"), "wb"
        ) as pickle_file:
            pickle.dump(return_struct, pickle_file)

        # send alert email if option toggled on
        HighDoseMetricAlertSettings.objects.get()
        send_alert_emails_skin = HighDoseMetricAlertSettings.objects.values_list(
            "send_high_dose_metric_alert_emails_skin", flat=True
        )[0]
        send_alert_emails_ref = HighDoseMetricAlertSettings.objects.values_list(
            "send_high_dose_metric_alert_emails_ref", flat=True
        )[0]
        if send_alert_emails_skin or send_alert_emails_ref:
            send_rf_high_dose_alert_email(study.pk)

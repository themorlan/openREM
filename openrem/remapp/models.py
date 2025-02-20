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
..  module:: models.
    :synopsis: Models to create the database tables and relationships.

..  moduleauthor:: Ed McDonagh

"""

# Following two lines added so that sphinx autodocumentation works.
from builtins import object  # pylint: disable=redefined-builtin
import json
from django.db import models
from django.urls import reverse
from solo.models import SingletonModel
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

# hoping to remove the next two lines
# import os
# os.environ['DJANGO_SETTINGS_MODULE'] = 'openremproject.settings'
# pylint: disable=unused-variable


class AdminTaskQuestions(SingletonModel):
    """
    Record if admin tasks have been dealt with
    """

    ask_revert_to_074_question = models.BooleanField(default=True)


class NotPatientIndicatorsID(models.Model):
    """
    Table to record strings that indicate a patient ID is really a test or QA ID
    """

    not_patient_id = models.CharField(max_length=64)

    def get_absolute_url(self):
        return reverse("not_patient_indicators")

    def __unicode__(self):
        return self.not_patient_id

    class Meta(object):
        """Meta class to define verbose names for not patient indicators for IDs"""

        verbose_name = "Not-patient indicator ID"
        verbose_name_plural = "Not-patient indicator IDs"


class NotPatientIndicatorsName(models.Model):
    """
    Table to record strings that indicate a patient name is really a test or QA name
    """

    not_patient_name = models.CharField(max_length=64)

    def get_absolute_url(self):
        return reverse("not_patient_indicators")

    def __unicode__(self):
        return self.not_patient_name

    class Meta(object):
        """Meta class to define verbose names for not-patient indicator names table"""

        verbose_name = "Not-patient indicator name"
        verbose_name_plural = "Not-patient indicator names"


class SkinDoseMapCalcSettings(SingletonModel):
    """
    Table to store skin dose map calculation settings
    """

    enable_skin_dose_maps = models.BooleanField(
        default=False, verbose_name="Enable skin dose maps?"
    )
    calc_on_import = models.BooleanField(
        default=True, verbose_name="Calculate skin dose map on import?"
    )
    allow_safelist_modify = models.BooleanField(
        default=False, verbose_name="Allow safelist to be updated?"
    )

    def get_absolute_url(self):
        return reverse("skin_dose_map_settings_update", kwargs={"pk": 1})


class OpenSkinSafeList(models.Model):
    """
    Table to store systems names and software versions that are suitable for OpenSkin
    """

    manufacturer = models.TextField(blank=True, null=True)
    manufacturer_model_name = models.TextField(blank=True, null=True)
    software_version = models.TextField(blank=True, default="")

    def get_absolute_url(self):
        return reverse("display_names_view")


class HighDoseMetricAlertSettings(SingletonModel):
    """
    Table to store high dose fluoroscopy alert settings
    """

    alert_total_dap_rf = models.IntegerField(
        blank=True,
        null=True,
        default=20000,
        verbose_name="Alert level for total DAP from fluoroscopy examination (cGy.cm<sup>2</sup>)",
    )
    alert_total_rp_dose_rf = models.FloatField(
        blank=True,
        null=True,
        default=2.0,
        verbose_name="Alert level for total dose at reference point from fluoroscopy examination (Gy)",
    )
    alert_skindose = models.FloatField(
        blank=True,
        null=True,
        default=2.0,
        verbose_name="Alert level for the peak skin dose from fluoroscopy examination (Gy)",
    )
    accum_dose_delta_weeks = models.IntegerField(
        blank=True,
        null=True,
        default=12,
        verbose_name="Number of previous weeks over which to sum DAP and RP dose for each patient",
    )
    changed_accum_dose_delta_weeks = models.BooleanField(default=True)
    show_accum_dose_over_delta_weeks = models.BooleanField(
        default=True,
        verbose_name="Enable display of summed DAP and RP dose in e-mail alerts and on summary and detail pages?",
    )
    calc_accum_dose_over_delta_weeks_on_import = models.BooleanField(
        default=True,
        verbose_name="Calculate summed DAP and RP dose for incoming fluoroscopy studies?",
    )
    send_high_dose_metric_alert_emails_ref = models.BooleanField(
        default=False,
        verbose_name="Send notification e-mails when alert levels for total DAP or total dose at reference point are"
        " exceeded?",
    )
    send_high_dose_metric_alert_emails_skin = models.BooleanField(
        default=False,
        verbose_name="Send notification e-mails when alert levels for peak skin dose are exceeded?",
    )

    def get_absolute_url(self):
        return reverse("rf_alert_settings_update", kwargs={"pk": 1})


class HighDoseMetricAlertRecipients(models.Model):
    """
    Table to store whether users should receive high dose fluoroscopy alerts
    """

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    receive_high_dose_metric_alerts = models.BooleanField(
        default=False, verbose_name="Receive high dose e-mail alerts?"
    )


@receiver(post_save, sender=User)
def create_or_save_high_dose_metric_alert_recipient_setting(sender, instance, **kwargs):
    """
    Function to create or save fluoroscopy high dose alert recipient settings
    """
    if not hasattr(instance, "highdosemetricalertrecipients"):
        new_objects = HighDoseMetricAlertRecipients.objects.create(user=instance)
        new_objects.save()
    else:
        instance.highdosemetricalertrecipients.save()


class HomePageAdminSettings(SingletonModel):
    """
    Table to store home page settings
    """

    enable_workload_stats = models.BooleanField(
        default=False,
        verbose_name="Enable calculation and display of workload stats on "
        "home page?",
    )
    #
    # def get_absolute_url(self):
    #     return '/admin/homepagesettings/1/'


class MergeOnDeviceObserverUIDSettings(SingletonModel):
    """
    Table to store setting(s) for autmoatic setting of Display Name and Modality type based on same Device observer UID
    """

    match_on_device_observer_uid = models.BooleanField(
        default=False,
        verbose_name="Set Display Name and Modality type if Device Observer UID is matching.",
    )


class DicomDeleteSettings(SingletonModel):
    """
    Table to store DICOM deletion settings
    """

    del_no_match = models.BooleanField(
        default=False,
        verbose_name="delete objects that don't match any import functions?",
    )
    del_rdsr = models.BooleanField(
        default=False,
        verbose_name="delete radiation dose structured reports after processing?",
    )
    del_mg_im = models.BooleanField(
        default=False, verbose_name="delete mammography images after processing?"
    )
    del_dx_im = models.BooleanField(
        default=False, verbose_name="delete radiography images after processing?"
    )
    del_ct_phil = models.BooleanField(
        default=False,
        verbose_name="delete Philips CT dose info images after processing?",
    )
    del_nm_im = models.BooleanField(
        default=False, verbose_name="delete nuclear medicine images after processing?"
    )

    def __unicode__(self):
        return "Delete DICOM objects settings"

    class Meta(object):
        """Meta class to define verbose name for DICOM delete settings"""

        verbose_name = "Delete DICOM objects settings"

    def get_absolute_url(self):
        return reverse("dicom_summary")


class PatientIDSettings(SingletonModel):
    """
    Table to store patient ID settings
    """

    name_stored = models.BooleanField(default=False)
    name_hashed = models.BooleanField(default=True)
    id_stored = models.BooleanField(default=False)
    id_hashed = models.BooleanField(default=True)
    accession_hashed = models.BooleanField(default=False)
    dob_stored = models.BooleanField(default=False)

    def __unicode__(self):
        return "Patient ID Settings"

    class Meta(object):
        """
        Verbose name for PatientIDSettings
        """

        verbose_name = "Patient ID Settings"

    def get_absolute_url(self):
        return reverse("home")


class DicomStoreSCP(models.Model):
    """
    Table to store DICOM store settings
    """

    name = models.CharField(
        max_length=64,
        unique=True,
        verbose_name="Name of local store node - fewer than 64 characters, spaces allowed",
    )
    aetitle = models.CharField(
        max_length=16,
        blank=True,
        null=True,
        verbose_name="AE Title of this node - 16 or fewer letters and numbers, no spaces",
    )
    peer = models.CharField(max_length=32, blank=True)
    port = models.IntegerField(default=104)
    task_id = models.CharField(max_length=64, blank=True, null=True)
    status = models.CharField(max_length=64, blank=True, null=True)

    def get_absolute_url(self):
        return reverse("dicom_summary")


class DicomRemoteQR(models.Model):
    """
    Table to store DICOM remote QR settings
    """

    name = models.CharField(
        max_length=64,
        unique=True,
        verbose_name="Name of QR node - fewer than 64 characters, spaces allowed",
    )
    aetitle = models.CharField(
        max_length=16,
        blank=True,
        null=True,
        verbose_name="AE Title of the remote node - 16 or fewer letters and numbers, no spaces",
    )
    port = models.IntegerField(blank=True, null=True, verbose_name="Remote port")
    ip = models.GenericIPAddressField(
        blank=True, null=True, verbose_name="Remote IP address"
    )
    hostname = models.CharField(
        max_length=32, blank=True, null=True, verbose_name="Or remote hostname"
    )
    callingaet = models.CharField(
        max_length=16,
        blank=True,
        null=True,
        verbose_name="AE Title of this OpenREM server - 16 or fewer letters and numbers, no spaces",
    )
    use_modality_tag = models.BooleanField(
        default=False, verbose_name="Use modality tag in study query"
    )
    enabled = models.BooleanField(default=False)

    def get_absolute_url(self):
        return reverse("dicom_summary")

    def __unicode__(self):
        return self.name


class BackgroundTaskMaximumRows(SingletonModel):
    """
    Table to store the maximum number of rows allowed in the BackgroundTask table
    """

    max_background_task_rows = models.IntegerField(
        default=2000,
        verbose_name="The maximum number of historic background task records to keep",
    )

    def get_absolute_url(self):
        return reverse("background_task_settings", kwargs={"pk": 1})


def limit_background_task_table_rows(  # pylint: disable=unused-argument
    sender, instance, **kwargs
):
    """
    Method to limit the number of rows in the BackgroundTask table. This method is triggered by a post_save
    signal associated with the BackgroundTask table.
    """

    all_tasks_qs = BackgroundTask.objects.order_by("id")
    if (
        all_tasks_qs.count()
        > BackgroundTaskMaximumRows.get_solo().max_background_task_rows
    ):
        all_tasks_qs[0].delete()


class BackgroundTask(models.Model):
    uuid = models.TextField()
    proc_id = models.IntegerField()
    task_type = models.TextField()
    info = models.TextField(blank=True, null=True)
    error = models.TextField(blank=True, null=True)
    completed_successfully = models.BooleanField(default=False)
    complete = models.BooleanField(default=False)
    started_at = models.DateTimeField(blank=True, null=True)


post_save.connect(limit_background_task_table_rows, sender=BackgroundTask)


class DicomQuery(models.Model):
    """
    Table to store DICOM query settings
    """

    started_at = models.DateTimeField(blank=True, null=True)
    complete = models.BooleanField(default=False)
    query_id = models.CharField(max_length=64)
    query_summary = models.TextField(blank=True, null=True)
    failed = models.BooleanField(default=False)
    message = models.TextField(blank=True, null=True)
    stage = models.TextField(blank=True, null=True)
    errors = models.TextField(blank=True, null=True)
    qr_scp_fk = models.ForeignKey(
        DicomRemoteQR, blank=True, null=True, on_delete=models.CASCADE
    )
    store_scp_fk = models.ForeignKey(
        DicomStoreSCP, blank=True, null=True, on_delete=models.CASCADE
    )
    move_completed_sub_ops = models.IntegerField(default=0)
    move_failed_sub_ops = models.IntegerField(default=0)
    move_warning_sub_ops = models.IntegerField(default=0)
    move_complete = models.BooleanField(default=False)
    move_summary = models.TextField(blank=True)
    move_uuid = models.UUIDField(null=True)
    query_task = models.ForeignKey(
        BackgroundTask,
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        related_name="query_part",
    )
    move_task = models.ForeignKey(
        BackgroundTask,
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        related_name="move_part",
    )


class DicomQRRspStudy(models.Model):
    dicom_query = models.ForeignKey(DicomQuery, on_delete=models.CASCADE)
    query_id = models.CharField(max_length=64)
    study_instance_uid = models.TextField(blank=True, null=True)
    modality = models.CharField(max_length=16, blank=True, null=True)
    modalities_in_study = models.CharField(max_length=100, blank=True, null=True)
    study_description = models.TextField(blank=True, null=True)
    number_of_study_related_series = models.IntegerField(blank=True, null=True)
    sop_classes_in_study = models.TextField(blank=True, null=True)
    station_name = models.CharField(max_length=32, blank=True, null=True)
    deleted_flag = models.BooleanField(default=False)
    deleted_reason = models.TextField(default="Downloaded")
    related_imports = models.ManyToManyField(BackgroundTask)

    def set_modalities_in_study(self, x):
        self.modalities_in_study = json.dumps(list(x or []))

    def get_modalities_in_study(self):
        return json.loads(self.modalities_in_study)

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "dicom_query",
                ]
            ),
        ]


class DicomQRRspSeries(models.Model):
    dicom_qr_rsp_study = models.ForeignKey(DicomQRRspStudy, on_delete=models.CASCADE)
    query_id = models.CharField(max_length=64)
    series_instance_uid = models.TextField(blank=True, null=True)
    series_number = models.IntegerField(blank=True, null=True)
    series_time = models.TimeField(blank=True, null=True)
    modality = models.CharField(max_length=16, blank=True, null=True)
    series_description = models.TextField(blank=True, null=True)
    number_of_series_related_instances = models.IntegerField(blank=True, null=True)
    station_name = models.CharField(max_length=32, blank=True, null=True)
    sop_class_in_series = models.TextField(blank=True, null=True)
    image_level_move = models.BooleanField(default=False)
    deleted_flag = models.BooleanField(default=False)
    deleted_reason = models.TextField(default="Downloaded")

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "dicom_qr_rsp_study",
                ]
            ),
        ]


class DicomQRRspImage(models.Model):
    dicom_qr_rsp_series = models.ForeignKey(DicomQRRspSeries, on_delete=models.CASCADE)
    query_id = models.CharField(max_length=64)
    sop_instance_uid = models.TextField(blank=True, null=True)
    instance_number = models.IntegerField(blank=True, null=True)
    sop_class_uid = models.TextField(blank=True, null=True)
    deleted_flag = models.BooleanField(default=False)
    deleted_reason = models.TextField(default="Downloaded")

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "dicom_qr_rsp_series",
                ]
            ),
        ]


class CommonVariables:
    DEFAULT_COLOUR_MAP = "RdYlBu"
    CHART_COLOUR_MAPS = (
        (DEFAULT_COLOUR_MAP, "Red-yellow-blue (default)"),
        ("Spectral", "Spectral"),
        ("RdYlGn", "Red-yellow-green"),
        ("PiYG", "Pink-green"),
        ("PRGn", "Purple-green"),
        ("BrBG", "Brown-blue-green"),
        ("PuOr", "Purple-orange"),
        ("RdBu", "Red-blue"),
        ("RdGy", "Red-grey"),
        ("YlGnBu", "Yellow-green-blue"),
        ("YlOrBr", "Yellow-orange-brown"),
        ("hot", "Hot"),
        ("inferno", "Inferno"),
        ("magma", "Magma"),
        ("plasma", "Plasma"),
        ("viridis", "Viridis"),
        ("cividis", "Cividis"),
    )
    PLOTLY_THEME = "plotly"
    CHART_THEMES = (
        (PLOTLY_THEME, "Plotly (default)"),
        ("plotly_white", "Plotly white"),
        ("plotly_dark", "Plotly dark"),
        ("presentation", "Presentation"),
        ("ggplot2", "ggplot2"),
        ("seaborn", "Seaborn"),
        ("simple_white", "Simple white"),
    )
    SERIES = "series"
    SYSTEM = "system"
    CHART_GROUPING = ((SYSTEM, "System names"), (SERIES, "Series item names"))
    CHART_GROUPING_RF = ((SYSTEM, "System or physician"), (SERIES, "Series item names"))
    ITEMS_PER_PAGE = (
        (10, "10"),
        (25, "25"),
        (50, "50"),
        (100, "100"),
        (200, "200"),
        (400, "400"),
    )
    DESCENDING = 0
    ASCENDING = 1
    SORTING_DIRECTION = ((ASCENDING, "Ascending"), (DESCENDING, "Descending"))
    VALUE = "value"
    FREQ = "frequency"
    NAME = "name"
    SORTING_CHOICES = ((NAME, "Name"), (FREQ, "Frequency"), (VALUE, "Value"))
    YEARS = "A"
    QUARTERS = "Q"
    MONTHS = "M"
    WEEKS = "W"
    DAYS = "D"
    TIME_PERIOD = (
        (DAYS, "Days"),
        (WEEKS, "Weeks"),
        (MONTHS, "Months"),
        (QUARTERS, "Quarters"),
        (YEARS, "Years"),
    )

    MEAN = "mean"
    MEDIAN = "median"
    BOXPLOT = "boxplot"
    AVERAGES = ((MEAN, "Mean"), (MEDIAN, "Median"), (BOXPLOT, "Boxplot"))

    # Using DICOM code meanings from http://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_10013.html
    CT_SEQUENCED_ACQUISITION_TYPE = "Sequenced Acquisition"
    CT_SPIRAL_ACQUISITION_TYPE = "Spiral Acquisition"
    CT_CONSTANT_ANGLE_ACQUISITION_TYPE = "Constant Angle Acquisition"
    CT_STATIONARY_ACQUISITION_TYPE = "Stationary Acquisition"
    CT_FREE_ACQUISITION_TYPE = "Free Acquisition"
    CT_CONE_BEAM_ACQUISITION = "Cone Beam Acquisition"

    CT_ACQUISITION_TYPES = (
        (CT_SEQUENCED_ACQUISITION_TYPE, "Sequenced"),
        (CT_SPIRAL_ACQUISITION_TYPE, "Spiral"),
        (CT_CONSTANT_ANGLE_ACQUISITION_TYPE, "Constant angle"),
        (CT_STATIONARY_ACQUISITION_TYPE, "Stationary"),
        (CT_FREE_ACQUISITION_TYPE, "Free"),
        (CT_CONE_BEAM_ACQUISITION, "Cone beam"),
    )

    CT_ACQUISITION_TYPE_CODES = {
        CT_SEQUENCED_ACQUISITION_TYPE: ["113804"],
        CT_SPIRAL_ACQUISITION_TYPE: ["116152004", "P5-08001", "C0860888"],
        CT_CONSTANT_ANGLE_ACQUISITION_TYPE: ["113805"],
        CT_STATIONARY_ACQUISITION_TYPE: ["113806"],
        CT_FREE_ACQUISITION_TYPE: ["113807"],
        CT_CONE_BEAM_ACQUISITION: ["702569007", "R-FB8F1", "C3839509"],
    }


class UserProfile(models.Model, CommonVariables):
    """
    Table to store user profile settings
    """

    itemsPerPage = models.IntegerField(
        null=True, choices=CommonVariables.ITEMS_PER_PAGE, default=25
    )

    # This field is required.
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    plotGroupingChoice = models.CharField(
        max_length=6,
        choices=CommonVariables.CHART_GROUPING,
        default=CommonVariables.SYSTEM,
    )

    plotThemeChoice = models.CharField(
        max_length=12,
        choices=CommonVariables.CHART_THEMES,
        default=CommonVariables.PLOTLY_THEME,
    )

    plotColourMapChoice = models.CharField(
        max_length=8,
        choices=CommonVariables.CHART_COLOUR_MAPS,
        default=CommonVariables.DEFAULT_COLOUR_MAP,
    )

    plotFacetColWrapVal = models.PositiveSmallIntegerField(default=3)

    plotInitialSortingDirection = models.IntegerField(
        null=True,
        choices=CommonVariables.SORTING_DIRECTION,
        default=CommonVariables.DESCENDING,
    )

    plotBoxplots = models.BooleanField(default=False, editable=False)
    plotMean = models.BooleanField(default=True, editable=False)
    plotMedian = models.BooleanField(default=False, editable=False)

    # Plotting controls
    plotCharts = models.BooleanField(default=False)

    plotDXAcquisitionMeanDAP = models.BooleanField(default=True)
    plotDXAcquisitionMeankVp = models.BooleanField(default=False)
    plotDXAcquisitionMeanmAs = models.BooleanField(default=False)
    plotDXAcquisitionFreq = models.BooleanField(default=False)
    plotDXAcquisitionDAPvsMass = models.BooleanField(default=False)
    plotDXAcquisitionMeanDAPOverTime = models.BooleanField(default=False)
    plotDXAcquisitionMeankVpOverTime = models.BooleanField(default=False)
    plotDXAcquisitionMeanmAsOverTime = models.BooleanField(default=False)

    plotDXStandardAcquisitionMeanDAP = models.BooleanField(default=False)
    plotDXStandardAcquisitionMeankVp = models.BooleanField(default=False)
    plotDXStandardAcquisitionMeanmAs = models.BooleanField(default=False)
    plotDXStandardAcquisitionFreq = models.BooleanField(default=False)
    plotDXStandardAcquisitionDAPvsMass = models.BooleanField(default=False)
    plotDXStandardAcquisitionMeanDAPOverTime = models.BooleanField(default=False)
    plotDXStandardAcquisitionMeankVpOverTime = models.BooleanField(default=False)
    plotDXStandardAcquisitionMeanmAsOverTime = models.BooleanField(default=False)

    plotDXStudyMeanDAP = models.BooleanField(default=True)
    plotDXStudyFreq = models.BooleanField(default=True)
    plotDXStudyDAPvsMass = models.BooleanField(default=False)
    plotDXStudyPerDayAndHour = models.BooleanField(default=False)
    plotDXRequestMeanDAP = models.BooleanField(default=True)
    plotDXRequestFreq = models.BooleanField(default=True)
    plotDXRequestDAPvsMass = models.BooleanField(default=False)

    plotDXStandardStudyMeanDAP = models.BooleanField(default=False)
    plotDXStandardStudyFreq = models.BooleanField(default=False)
    plotDXStandardStudyDAPvsMass = models.BooleanField(default=False)
    plotDXStandardStudyPerDayAndHour = models.BooleanField(default=False)

    plotDXAcquisitionMeanDAPOverTimePeriod = models.CharField(
        max_length=13,
        choices=CommonVariables.TIME_PERIOD,
        default=CommonVariables.MONTHS,
    )
    plotDXInitialSortingChoice = models.CharField(
        max_length=9,
        choices=CommonVariables.SORTING_CHOICES,
        default=CommonVariables.FREQ,
    )

    plotNMStudyFreq = models.BooleanField(default=False)
    plotNMStudyPerDayAndHour = models.BooleanField(default=False)
    plotNMInjectedDosePerStudy = models.BooleanField(default=False)
    plotNMInjectedDoseOverTime = models.BooleanField(default=False)
    plotNMInjectedDoseOverWeight = models.BooleanField(default=False)
    plotNMOverTimePeriod = models.CharField(
        max_length=13,
        choices=CommonVariables.TIME_PERIOD,
        default=CommonVariables.MONTHS,
    )
    plotNMInitialSortingChoice = models.CharField(
        max_length=9,
        choices=CommonVariables.SORTING_CHOICES,
        default=CommonVariables.FREQ,
    )

    plotCTAcquisitionMeanDLP = models.BooleanField(default=True)
    plotCTAcquisitionMeanCTDI = models.BooleanField(default=True)
    plotCTAcquisitionFreq = models.BooleanField(default=False)
    plotCTAcquisitionCTDIvsMass = models.BooleanField(default=False)
    plotCTAcquisitionDLPvsMass = models.BooleanField(default=False)
    plotCTAcquisitionCTDIOverTime = models.BooleanField(default=False)
    plotCTAcquisitionDLPOverTime = models.BooleanField(default=False)
    plotCTStandardAcquisitionFreq = models.BooleanField(default=False)
    plotCTStandardAcquisitionMeanDLP = models.BooleanField(default=False)
    plotCTStandardAcquisitionMeanCTDI = models.BooleanField(default=False)
    plotCTStandardAcquisitionDLPOverTime = models.BooleanField(default=False)
    plotCTStandardAcquisitionCTDIOverTime = models.BooleanField(default=False)
    plotCTStandardAcquisitionCTDIvsMass = models.BooleanField(default=False)
    plotCTStandardAcquisitionDLPvsMass = models.BooleanField(default=False)
    plotCTStudyMeanDLP = models.BooleanField(default=True)
    plotCTStudyMeanCTDI = models.BooleanField(default=True)
    plotCTStudyFreq = models.BooleanField(default=False)
    plotCTStudyNumEvents = models.BooleanField(default=False)
    plotCTRequestMeanDLP = models.BooleanField(default=False)
    plotCTRequestFreq = models.BooleanField(default=False)
    plotCTRequestNumEvents = models.BooleanField(default=False)
    plotCTRequestDLPOverTime = models.BooleanField(default=False)
    plotCTStudyPerDayAndHour = models.BooleanField(default=False)
    plotCTStudyMeanDLPOverTime = models.BooleanField(default=False)
    plotCTStandardStudyMeanDLP = models.BooleanField(default=False)
    plotCTStandardStudyNumEvents = models.BooleanField(default=False)
    plotCTStandardStudyFreq = models.BooleanField(default=False)
    plotCTStandardStudyPerDayAndHour = models.BooleanField(default=False)
    plotCTStandardStudyMeanDLPOverTime = models.BooleanField(default=False)
    plotCTOverTimePeriod = models.CharField(
        max_length=13,
        choices=CommonVariables.TIME_PERIOD,
        default=CommonVariables.MONTHS,
    )
    plotCTInitialSortingChoice = models.CharField(
        max_length=9,
        choices=CommonVariables.SORTING_CHOICES,
        default=CommonVariables.FREQ,
    )
    plotCTSequencedAcquisition = models.BooleanField(default=True)
    plotCTSpiralAcquisition = models.BooleanField(default=True)
    plotCTConstantAngleAcquisition = models.BooleanField(default=False)
    plotCTStationaryAcquisition = models.BooleanField(default=False)
    plotCTFreeAcquisition = models.BooleanField(default=False)
    plotCTConeBeamAcquisition = models.BooleanField(default=False)

    plotRFStudyPerDayAndHour = models.BooleanField(default=False)
    plotRFStudyFreq = models.BooleanField(default=False)
    plotRFStudyDAP = models.BooleanField(default=True)
    plotRFStudyDAPOverTime = models.BooleanField(default=False)
    plotRFRequestDAP = models.BooleanField(default=True)
    plotRFRequestFreq = models.BooleanField(default=True)
    plotRFRequestDAPOverTime = models.BooleanField(default=False)

    plotRFStandardStudyFreq = models.BooleanField(default=False)
    plotRFStandardStudyDAP = models.BooleanField(default=False)
    plotRFStandardStudyDAPOverTime = models.BooleanField(default=False)
    plotRFStandardStudyPerDayAndHour = models.BooleanField(default=False)

    plotRFOverTimePeriod = models.CharField(
        max_length=13,
        choices=CommonVariables.TIME_PERIOD,
        default=CommonVariables.MONTHS,
    )
    plotRFInitialSortingChoice = models.CharField(
        max_length=9,
        choices=CommonVariables.SORTING_CHOICES,
        default=CommonVariables.FREQ,
    )
    plotRFSplitByPhysician = models.BooleanField(default=False)

    plotMGStudyPerDayAndHour = models.BooleanField(default=False)
    plotMGAGDvsThickness = models.BooleanField(default=False)
    plotMGkVpvsThickness = models.BooleanField(default=False)
    plotMGmAsvsThickness = models.BooleanField(default=False)
    plotMGaverageAGDvsThickness = models.BooleanField(default=False)
    plotMGaverageAGD = models.BooleanField(default=False)
    plotMGacquisitionFreq = models.BooleanField(default=False)
    plotMGAcquisitionAGDOverTime = models.BooleanField(default=False)

    plotMGStandardStudyPerDayAndHour = models.BooleanField(default=False)
    plotMGStandardAGDvsThickness = models.BooleanField(default=False)
    plotMGStandardkVpvsThickness = models.BooleanField(default=False)
    plotMGStandardmAsvsThickness = models.BooleanField(default=False)
    plotMGStandardAverageAGDvsThickness = models.BooleanField(default=False)
    plotMGStandardAverageAGD = models.BooleanField(default=False)
    plotMGStandardAcquisitionFreq = models.BooleanField(default=False)
    plotMGStandardAcquisitionAGDOverTime = models.BooleanField(default=False)

    plotMGOverTimePeriod = models.CharField(
        max_length=13,
        choices=CommonVariables.TIME_PERIOD,
        default=CommonVariables.MONTHS,
    )
    plotMGInitialSortingChoice = models.CharField(
        max_length=9,
        choices=CommonVariables.SORTING_CHOICES,
        default=CommonVariables.FREQ,
    )

    displayCT = models.BooleanField(default=True)
    displayRF = models.BooleanField(default=True)
    displayMG = models.BooleanField(default=True)
    displayDX = models.BooleanField(default=True)
    displayNM = models.BooleanField(default=True)

    plotSeriesPerSystem = models.BooleanField(default=False)

    plotHistogramBins = models.PositiveSmallIntegerField(default=20)

    plotHistograms = models.BooleanField(default=False)

    plotHistogramGlobalBins = models.BooleanField(default=False)

    plotCaseInsensitiveCategories = models.BooleanField(default=False)

    plotRemoveCategoryWhitespacePadding = models.BooleanField(default=False)

    plotLabelCharWrap = models.PositiveSmallIntegerField(default=500)

    summaryWorkloadDaysA = models.IntegerField(
        blank=True,
        null=True,
        default=7,
        verbose_name="Number of days over which to sum studies A",
    )
    summaryWorkloadDaysB = models.IntegerField(
        blank=True,
        null=True,
        default=28,
        verbose_name="Number of days over which to sum studies B",
    )


def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


post_save.connect(create_user_profile, sender=User)


class UniqueEquipmentNames(models.Model):
    """
    Table to unique equipment name information
    """

    manufacturer = models.TextField(blank=True, null=True)
    manufacturer_hash = models.CharField(max_length=64, blank=True, null=True)
    institution_name = models.TextField(blank=True, null=True)
    institution_name_hash = models.CharField(max_length=64, blank=True, null=True)
    station_name = models.CharField(max_length=32, blank=True, null=True)
    station_name_hash = models.CharField(max_length=64, blank=True, null=True)
    institutional_department_name = models.TextField(blank=True, null=True)
    institutional_department_name_hash = models.CharField(
        max_length=64, blank=True, null=True
    )
    manufacturer_model_name = models.TextField(blank=True, null=True)
    manufacturer_model_name_hash = models.CharField(
        max_length=64, blank=True, null=True
    )
    device_serial_number = models.TextField(blank=True, null=True)
    device_serial_number_hash = models.CharField(max_length=64, blank=True, null=True)
    software_versions = models.TextField(blank=True, null=True)
    software_versions_hash = models.CharField(max_length=64, blank=True, null=True)
    gantry_id = models.TextField(blank=True, null=True)
    gantry_id_hash = models.CharField(max_length=64, blank=True, null=True)
    display_name = models.TextField(blank=True, null=True)
    user_defined_modality = models.CharField(max_length=16, blank=True, null=True)
    hash_generated = models.BooleanField(default=False)
    device_observer_uid = models.TextField(blank=True, null=True)
    device_observer_uid_hash = models.CharField(max_length=64, blank=True, null=True)

    class Meta(object):
        """
        Define unique_together Meta class to enable sorting of similar devices
        """

        unique_together = (
            "manufacturer_hash",
            "institution_name_hash",
            "station_name_hash",
            "institutional_department_name_hash",
            "manufacturer_model_name_hash",
            "device_serial_number_hash",
            "software_versions_hash",
            "gantry_id_hash",
            "device_observer_uid_hash",
        )

        indexes = [
            models.Index(
                fields=[
                    "display_name",
                ]
            ),
        ]

    def __unicode__(self):
        return self.display_name


class StandardNames(models.Model):
    """
    Table to store standard study description, requested procedure, procedure or acquisition names
    """

    standard_name = models.TextField(blank=True, null=True)
    modality = models.CharField(max_length=16, blank=True, null=True)
    study_description = models.TextField(blank=True, null=True)
    requested_procedure_code_meaning = models.TextField(blank=True, null=True)
    procedure_code_meaning = models.TextField(blank=True, null=True)
    acquisition_protocol = models.TextField(blank=True, null=True)

    class Meta(object):
        """
        Define unique_together Meta class to ensure that each study description, requested procedure,
        procedure and acquisition protocol can only appear in one standard name per modality
        """

        unique_together = (
            ("modality", "study_description"),
            ("modality", "requested_procedure_code_meaning"),
            ("modality", "procedure_code_meaning"),
            ("modality", "acquisition_protocol"),
        )

    def __unicode__(self):
        return self.standard_name

    def get_absolute_url(self):
        return reverse("standard_names_view")


class StandardNameSettings(SingletonModel):
    """
    Table to store standard name mapping settings
    """

    enable_standard_names = models.BooleanField(
        default=False,
        verbose_name="Enable standard name mapping?",
    )

    def get_absolute_url(self):
        return reverse("standard_name_settings", kwargs={"pk": 1})


class SizeUpload(models.Model):
    """
    Table to store patient size information
    """

    sizefile = models.FileField(upload_to="sizeupload")
    height_field = models.TextField(blank=True, null=True)
    weight_field = models.TextField(blank=True, null=True)
    id_field = models.TextField(blank=True, null=True)
    id_type = models.TextField(blank=True, null=True)
    overwrite = models.BooleanField(default=False)
    task_id = models.TextField(blank=True, null=True)
    status = models.TextField(blank=True, null=True)
    progress = models.TextField(blank=True, null=True)
    num_records = models.IntegerField(blank=True, null=True)
    logfile = models.FileField(upload_to="sizelogs/%Y/%m/%d", null=True)
    import_date = models.DateTimeField(blank=True, null=True)
    processtime = models.FloatField(blank=True, null=True)


class Exports(models.Model):
    """Table to hold the export status and filenames"""

    task_id = models.TextField()
    filename = models.FileField(upload_to="exports/%Y/%m/%d", null=True)
    export_summary = models.TextField(blank=True, null=True)
    status = models.TextField(blank=True, null=True)
    progress = models.TextField(blank=True, null=True)
    modality = models.CharField(max_length=16, blank=True, null=True)
    num_records = models.IntegerField(blank=True, null=True)
    export_type = models.TextField(blank=True, null=True)
    export_date = models.DateTimeField(blank=True, null=True)
    processtime = models.DecimalField(
        max_digits=30, decimal_places=10, blank=True, null=True
    )
    includes_pid = models.BooleanField(default=False)
    export_user = models.ForeignKey(
        User, blank=True, null=True, on_delete=models.CASCADE
    )


class ContextID(models.Model):
    """Table to hold all the context ID code values and code meanings.

    + Could be prefilled from the tables in DICOM 3.16, but is actually populated as the codes occur. \
    This assumes they are used correctly.
    """

    code_value = models.TextField()
    code_meaning = models.TextField(blank=True, null=True)
    cid_table = models.CharField(max_length=16, blank=True)

    def __unicode__(self):
        return self.code_meaning

    class Meta(object):
        """Meta class to define ordering of objects in ContextID tables"""

        ordering = ["code_value"]


class GeneralStudyModuleAttrManager(models.Manager):
    def get_queryset(self):
        qs = (
            super(GeneralStudyModuleAttrManager, self)
            .get_queryset()
            .annotate(
                test_date_time=models.ExpressionWrapper(
                    models.F("study_date") + models.F("study_time"),
                    output_field=models.DateTimeField(),
                )
            )
        )
        return qs


class GeneralStudyModuleAttr(models.Model):  # C.7.2.1
    """General Study Module C.7.2.1

    Specifies the Attributes that describe and identify the Study
    performed upon the Patient.
    From DICOM Part 3: Information Object Definitions Table C.7-3

    Additional to the module definition:
        * performing_physician_name
        * operator_name
        * modality_type
        * procedure_code_value_and_meaning
        * requested_procedure_code_value_and_meaning
    """

    study_instance_uid = models.TextField(blank=True, null=True)
    study_date = models.DateField(blank=True, null=True)
    study_time = models.TimeField(blank=True, null=True)
    study_workload_chart_time = models.DateTimeField(blank=True, null=True)
    referring_physician_name = models.TextField(blank=True, null=True)
    referring_physician_identification = models.TextField(blank=True, null=True)
    study_id = models.CharField(max_length=16, blank=True, null=True)
    accession_number = models.TextField(blank=True, null=True)
    accession_hashed = models.BooleanField(default=False)
    study_description = models.TextField(blank=True, null=True)
    physician_of_record = models.TextField(blank=True, null=True)
    name_of_physician_reading_study = models.TextField(blank=True, null=True)
    # Possibly need a few sequences linked to this table...
    # Next three don't belong in this table, but they don't belong anywhere in a RDSR!
    performing_physician_name = models.TextField(blank=True, null=True)
    operator_name = models.TextField(blank=True, null=True)
    modality_type = models.CharField(max_length=16, blank=True, null=True)
    procedure_code_value = models.TextField(blank=True, null=True)
    procedure_code_meaning = models.TextField(blank=True, null=True)
    requested_procedure_code_value = models.TextField(blank=True, null=True)
    requested_procedure_code_meaning = models.TextField(blank=True, null=True)
    # Series and content to distinguish between multiple cumulative RDSRs
    series_instance_uid = models.TextField(blank=True, null=True)
    series_time = models.TimeField(blank=True, null=True)
    content_time = models.TimeField(blank=True, null=True)

    # Additional study summary fields
    number_of_events = models.IntegerField(blank=True, null=True)
    number_of_events_a = models.IntegerField(blank=True, null=True)
    number_of_events_b = models.IntegerField(blank=True, null=True)
    number_of_axial = models.IntegerField(blank=True, null=True)
    number_of_spiral = models.IntegerField(blank=True, null=True)
    number_of_stationary = models.IntegerField(blank=True, null=True)
    number_of_const_angle = models.IntegerField(blank=True, null=True)
    total_dlp = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    total_dap_a = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    total_dap_b = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    total_dap = models.DecimalField(max_digits=16, decimal_places=12, null=True)
    total_rp_dose_a = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    total_rp_dose_b = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    total_dap_delta_weeks = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    total_rp_dose_delta_weeks = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    total_agd_left = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    total_agd_right = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    total_agd_both = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )  # for legacy
    number_of_planes = models.IntegerField(blank=True, null=True)

    standard_names = models.ManyToManyField(StandardNames)

    def __unicode__(self):
        return self.study_instance_uid

    def dap_a_cgycm2(self):
        """Converts DAP A to cGy.cm2 from Gy.m2 for display or export"""
        if self.total_dap_a:
            return 1000000 * self.total_dap_a

    def dap_b_cgycm2(self):
        """Converts DAP B to cGy.cm2 from Gy.m2 for display or export"""
        if self.total_dap_b:
            return 1000000 * self.total_dap_b

    def dap_total_cgycm2(self):
        """Converts DAP A+B to cGy.cm2 from Gy.m2 for display or export"""
        if self.total_dap:
            return 1000000 * self.total_dap

    def dap_delta_weeks_cgycm2(self):
        """Converts DAP delta weeks to cGy.cm2 from Gy.m2 for display"""
        if self.total_dap_delta_weeks:
            return 1000000 * self.total_dap_delta_weeks

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "modality_type",
                ]
            ),
        ]

    objects = GeneralStudyModuleAttrManager()


class SkinDoseMapResults(models.Model):
    """Table to hold the results from OpenSkin"""

    general_study_module_attributes = models.ForeignKey(
        GeneralStudyModuleAttr, on_delete=models.CASCADE
    )
    phantom_width = models.DecimalField(
        max_digits=16, decimal_places=1, blank=True, null=True
    )
    phantom_height = models.DecimalField(
        max_digits=16, decimal_places=1, blank=True, null=True
    )
    phantom_depth = models.DecimalField(
        max_digits=16, decimal_places=1, blank=True, null=True
    )
    patient_mass = models.DecimalField(
        max_digits=16, decimal_places=1, blank=True, null=True
    )
    patient_mass_assumed = models.CharField(max_length=16, null=True, blank=True)
    patient_size = models.DecimalField(
        max_digits=16, decimal_places=1, blank=True, null=True
    )
    patient_size_assumed = models.CharField(max_length=16, null=True, blank=True)
    patient_orientation = models.CharField(max_length=16, blank=True, null=True)
    patient_orientation_assumed = models.CharField(max_length=16, null=True, blank=True)
    peak_skin_dose = models.DecimalField(
        max_digits=16, decimal_places=4, null=True, blank=True
    )
    dap_fraction = models.DecimalField(
        max_digits=16, decimal_places=4, null=True, blank=True
    )
    skin_map_version = models.CharField(max_length=16, null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "general_study_module_attributes",
                ]
            ),
        ]


class ObjectUIDsProcessed(models.Model):
    """Table to hold the SOP Instance UIDs of the objects that have been processed against this study to enable
    duplicate sorting.

    """

    general_study_module_attributes = models.ForeignKey(
        GeneralStudyModuleAttr, on_delete=models.CASCADE
    )
    sop_instance_uid = models.TextField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "general_study_module_attributes",
                ]
            ),
        ]


class ProjectionXRayRadiationDose(models.Model):  # TID 10001
    """Projection X-Ray Radiation Dose template TID 10001

    From DICOM Part 16:
        This template defines a container (the root) with subsidiary content items, each of which represents a
        single projection X-Ray irradiation event entry or plane-specific dose accumulations. There is a defined
        recording observer (the system or person responsible for recording the log, generally the system). A
        Biplane irradiation event will be recorded as two individual events, one for each plane. Accumulated
        values will be kept separate for each plane.

    """

    general_study_module_attributes = models.ForeignKey(
        GeneralStudyModuleAttr, on_delete=models.CASCADE
    )
    procedure_reported = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10001_procedure",
        on_delete=models.CASCADE,
    )
    has_intent = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10001_intent",
        on_delete=models.CASCADE,
    )
    acquisition_device_type_cid = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10001_type",
        on_delete=models.CASCADE,
    )
    scope_of_accumulation = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10001_scope",
        on_delete=models.CASCADE,
    )
    xray_detector_data_available = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10001_detector",
        on_delete=models.CASCADE,
    )
    xray_source_data_available = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10001_source",
        on_delete=models.CASCADE,
    )
    xray_mechanical_data_available = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10001_mech",
        on_delete=models.CASCADE,
    )
    comment = models.TextField(blank=True, null=True)
    # might need to be a table on its own as is 1-n, even though it should only list the primary source...
    source_of_dose_information = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10001_infosource",
        on_delete=models.CASCADE,
    )

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "general_study_module_attributes",
                ]
            ),
        ]


class AccumXRayDose(models.Model):  # TID 10002
    """Accumulated X-Ray Dose TID 10002

    From DICOM Part 16:
        This general template provides detailed information on projection X-Ray dose value accumulations over
        several irradiation events from the same equipment (typically a study or a performed procedure step).

    """

    projection_xray_radiation_dose = models.ForeignKey(
        ProjectionXRayRadiationDose, on_delete=models.CASCADE
    )
    acquisition_plane = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "projection_xray_radiation_dose",
                ]
            ),
        ]


class Calibration(models.Model):
    """Table to hold the calibration information

    + Container in TID 10002 Accumulated X-ray dose

    """

    accumulated_xray_dose = models.ForeignKey(AccumXRayDose, on_delete=models.CASCADE)
    dose_measurement_device = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )
    calibration_date = models.DateTimeField(blank=True, null=True)
    calibration_factor = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    calibration_uncertainty = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    calibration_responsible_party = models.TextField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "accumulated_xray_dose",
                ]
            ),
        ]


class IrradEventXRayData(models.Model):  # TID 10003
    """Irradiation Event X-Ray Data TID 10003

    From DICOM part 16:
        This template conveys the dose and equipment parameters of a single irradiation event.

    """

    projection_xray_radiation_dose = models.ForeignKey(
        ProjectionXRayRadiationDose, on_delete=models.CASCADE
    )
    acquisition_plane = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003_plane",
        on_delete=models.CASCADE,
    )  # CID 10003
    irradiation_event_uid = models.TextField(blank=True, null=True)
    irradiation_event_label = models.TextField(blank=True, null=True)
    label_type = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003_labeltype",
        on_delete=models.CASCADE,
    )  # CID 10022
    date_time_started = models.DateTimeField(blank=True, null=True)
    irradiation_event_type = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003_eventtype",
        on_delete=models.CASCADE,
    )  # CID 10002
    acquisition_protocol = models.TextField(blank=True, null=True)
    anatomical_structure = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003_anatomy",
        on_delete=models.CASCADE,
    )  # CID 4009
    laterality = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003_laterality",
        on_delete=models.CASCADE,
    )  # CID 244
    image_view = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003_view",
        on_delete=models.CASCADE,
    )  # CID 4010 "DX View" or CID 4014 "View for Mammography"
    # Lines below are incorrect, but exist in current databases. Replace with lines below them:
    projection_eponymous_name = models.CharField(
        max_length=16, blank=True, null=True
    )  # Added null to originals
    patient_table_relationship = models.CharField(max_length=16, blank=True, null=True)
    patient_orientation = models.CharField(max_length=16, blank=True, null=True)
    patient_orientation_modifier = models.CharField(
        max_length=16, blank=True, null=True
    )
    # TODO: Projection Eponymous Name should be in ImageViewModifier, not here :-(
    projection_eponymous_name_cid = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003_pojectioneponymous",
        on_delete=models.CASCADE,
    )  # CID 4012
    patient_table_relationship_cid = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003_pttablerel",
        on_delete=models.CASCADE,
    )  # CID 21
    patient_orientation_cid = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003_ptorientation",
        on_delete=models.CASCADE,
    )  # CID 19
    patient_orientation_modifier_cid = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003_ptorientationmod",
        on_delete=models.CASCADE,
    )  # CID 20
    target_region = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003_region",
        on_delete=models.CASCADE,
    )  # CID 4031
    dose_area_product = models.DecimalField(
        max_digits=16, decimal_places=10, blank=True, null=True
    )
    half_value_layer = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    patient_equivalent_thickness = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    entrance_exposure_at_rp = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    reference_point_definition_text = models.TextField(
        blank=True, null=True
    )  # in other models the code version is _code
    reference_point_definition = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003_rpdefinition",
        on_delete=models.CASCADE,
    )  # CID 10025
    # Another char field that should be a cid
    breast_composition = models.CharField(
        max_length=16, blank=True, null=True
    )  # TID 4007, CID 6000
    breast_composition_cid = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003_breastcomposition",
        on_delete=models.CASCADE,
    )  # CID 6000/6001
    percent_fibroglandular_tissue = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )  # TID 4007
    comment = models.TextField(blank=True, null=True)
    standard_protocols = models.ManyToManyField(StandardNames)

    def __unicode__(self):
        return self.irradiation_event_uid

    def convert_gym2_to_cgycm2(self):
        try:
            return 1000000 * self.dose_area_product
        except TypeError:
            return None

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "projection_xray_radiation_dose",
                ]
            ),
        ]


class ImageViewModifier(models.Model):  # EV 111032
    """Table to hold image view modifiers for the irradiation event x-ray data table

    From DICOM Part 16 Annex D DICOM controlled Terminology Definitions
        + Code Value 111032
        + Code Meaning Image View Modifier
        + Code Definition Modifier for image view
    """

    irradiation_event_xray_data = models.ForeignKey(
        IrradEventXRayData, on_delete=models.CASCADE
    )
    image_view_modifier = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )  # CID 4011 "DX View Modifier" or CID 4015 "View Modifier for Mammography"
    # TODO: Add Projection Eponymous Name

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "irradiation_event_xray_data",
                ]
            ),
            models.Index(
                fields=[
                    "image_view_modifier",
                ]
            ),
        ]


class IrradEventXRayDetectorData(models.Model):  # TID 10003a
    """Irradiation Event X-Ray Detector Data TID 10003a

    From DICOM Part 16 Correction Proposal CP-1077:
        This template contains data which is expected to be available to the X-ray detector or plate reader component of
        the equipment.
    """

    irradiation_event_xray_data = models.ForeignKey(
        IrradEventXRayData, on_delete=models.CASCADE
    )
    exposure_index = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    target_exposure_index = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    deviation_index = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    # New fields added to record the non-IEC exposure index from CR/DX image headers
    relative_xray_exposure = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    relative_exposure_unit = models.CharField(max_length=16, blank=True, null=True)
    sensitivity = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "irradiation_event_xray_data",
                ]
            ),
        ]


class IrradEventXRaySourceData(models.Model):  # TID 10003b
    """Irradiation Event X-Ray Source Data TID 10003b

    From DICOM Part 16 Correction Proposal CP-1077:
        This template contains data which is expected to be available to the X-ray source component of the equipment.

    Additional to the template:
        * ii_field_size
        * exposure_control_mode
        * grid information over and above grid type
    """

    irradiation_event_xray_data = models.ForeignKey(
        IrradEventXRayData, on_delete=models.CASCADE
    )
    dose_rp = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    reference_point_definition = models.TextField(blank=True, null=True)
    reference_point_definition_code = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003b_rpdefinition",
        on_delete=models.CASCADE,
    )  # CID 10025
    average_glandular_dose = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    fluoro_mode = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003b_fluoromode",
        on_delete=models.CASCADE,
    )  # CID 10004
    pulse_rate = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    number_of_pulses = models.DecimalField(
        max_digits=16, decimal_places=2, blank=True, null=True
    )
    # derivation should be a cid - has never been used in extractor, but was non null=True so will exist in database :-(
    derivation = models.CharField(max_length=16, blank=True, null=True)
    derivation_cid = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003b_derivation",
        on_delete=models.CASCADE,
    )  # R-10260, "Estimated"
    irradiation_duration = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    average_xray_tube_current = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    exposure_time = models.DecimalField(
        max_digits=16, decimal_places=2, blank=True, null=True
    )
    focal_spot_size = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    anode_target_material = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10003b_anodetarget",
        on_delete=models.CASCADE,
    )  # CID 10016
    collimated_field_area = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    collimated_field_height = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    collimated_field_width = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    # not in DICOM standard - 'image intensifier' field size and exposure control mode
    ii_field_size = models.IntegerField(blank=True, null=True)
    exposure_control_mode = models.CharField(max_length=16, blank=True, null=True)
    grid_absorbing_material = models.TextField(blank=True, null=True)
    grid_spacing_material = models.TextField(blank=True, null=True)
    grid_thickness = models.DecimalField(
        max_digits=16, decimal_places=6, blank=True, null=True
    )
    grid_pitch = models.DecimalField(
        max_digits=16, decimal_places=6, blank=True, null=True
    )
    grid_aspect_ratio = models.TextField(blank=True, null=True)
    grid_period = models.DecimalField(
        max_digits=16, decimal_places=6, blank=True, null=True
    )
    grid_focal_distance = models.DecimalField(
        max_digits=16, decimal_places=6, blank=True, null=True
    )

    def convert_gy_to_mgy(self):
        """Converts Gy to mGy for display in web interface"""
        if self.dose_rp:
            return 1000 * self.dose_rp

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "irradiation_event_xray_data",
                ]
            ),
        ]


class XrayGrid(models.Model):
    """Content ID 10017 X-Ray Grid

    From DICOM Part 16
    """

    irradiation_event_xray_source_data = models.ForeignKey(
        IrradEventXRaySourceData, on_delete=models.CASCADE
    )
    xray_grid = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )  # CID 10017

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "irradiation_event_xray_source_data",
                ]
            ),
            models.Index(
                fields=[
                    "xray_grid",
                ]
            ),
        ]


class PulseWidth(models.Model):  # EV 113793
    """In TID 10003b. Code value 113793 (ms)"""

    irradiation_event_xray_source_data = models.ForeignKey(
        IrradEventXRaySourceData, on_delete=models.CASCADE
    )
    pulse_width = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "irradiation_event_xray_source_data",
                ]
            ),
        ]


class Kvp(models.Model):  # EV 113733
    """In TID 10003b. Code value 113733 (kV)"""

    irradiation_event_xray_source_data = models.ForeignKey(
        IrradEventXRaySourceData, on_delete=models.CASCADE
    )
    kvp = models.DecimalField(max_digits=16, decimal_places=8, blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "irradiation_event_xray_source_data",
                ]
            ),
        ]


class XrayTubeCurrent(models.Model):  # EV 113734
    """In TID 10003b. Code value 113734 (mA)"""

    irradiation_event_xray_source_data = models.ForeignKey(
        IrradEventXRaySourceData, on_delete=models.CASCADE
    )
    xray_tube_current = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "irradiation_event_xray_source_data",
                ]
            ),
        ]


class Exposure(models.Model):  # EV 113736
    """In TID 10003b. Code value 113736 (uA.s)"""

    irradiation_event_xray_source_data = models.ForeignKey(
        IrradEventXRaySourceData, on_delete=models.CASCADE
    )
    exposure = models.DecimalField(
        max_digits=16, decimal_places=2, blank=True, null=True
    )

    def convert_uAs_to_mAs(self):
        """Converts uAs to mAs for display in web interface"""
        from decimal import Decimal
        from numbers import Number

        if isinstance(self.exposure, Number):
            return self.exposure / Decimal(1000.0)
        else:
            return None

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "irradiation_event_xray_source_data",
                ]
            ),
        ]


class XrayFilters(models.Model):  # EV 113771
    """Container in TID 10003b. Code value 113771"""

    irradiation_event_xray_source_data = models.ForeignKey(
        IrradEventXRaySourceData, on_delete=models.CASCADE
    )
    xray_filter_type = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="xrayfilters_type",
        on_delete=models.CASCADE,
    )  # CID 10007
    xray_filter_material = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="xrayfilters_material",
        on_delete=models.CASCADE,
    )  # CID 10006
    xray_filter_thickness_minimum = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    xray_filter_thickness_maximum = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "irradiation_event_xray_source_data",
                ]
            ),
        ]


class IrradEventXRayMechanicalData(models.Model):  # TID 10003c
    """Irradiation Event X-Ray Mechanical Data TID 10003c

    From DICOM Part 16 Correction Proposal CP-1077:
        This template contains data which is expected to be available to the gantry or mechanical component of the
        equipment.

    Additional to the template:
        * compression_force
        * magnification_factor
    """

    irradiation_event_xray_data = models.ForeignKey(
        IrradEventXRayData, on_delete=models.CASCADE
    )
    crdr_mechanical_configuration = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )
    positioner_primary_angle = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    positioner_secondary_angle = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    positioner_primary_end_angle = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    positioner_secondary_end_angle = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    column_angulation = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    table_head_tilt_angle = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    table_horizontal_rotation_angle = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    table_cradle_tilt_angle = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    compression_thickness = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    compression_force = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )  # Force in N - introduced in 2019b
    compression_pressure = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )  # Pressure in kPa - introduced in 2019b
    compression_contact_area = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )  # in mm2 - introduced in 2019b
    magnification_factor = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )  # Not in DICOM standard

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "irradiation_event_xray_data",
                ]
            ),
            models.Index(
                fields=[
                    "crdr_mechanical_configuration",
                ]
            ),
        ]


class DoseRelatedDistanceMeasurements(models.Model):  # CID 10008
    """Dose Related Distance Measurements Context ID 10008

    Called from TID 10003c
    """

    irradiation_event_xray_mechanical_data = models.ForeignKey(
        IrradEventXRayMechanicalData, on_delete=models.CASCADE
    )
    distance_source_to_isocenter = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    distance_source_to_reference_point = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    distance_source_to_detector = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    table_longitudinal_position = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    table_lateral_position = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    table_height_position = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    distance_source_to_table_plane = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    table_longitudinal_end_position = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    table_lateral_end_position = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    table_height_end_position = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    # not in DICOM standard - distance source to entrance surface distance in mm
    distance_source_to_entrance_surface = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    radiological_thickness = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "irradiation_event_xray_mechanical_data",
                ]
            ),
        ]


class AccumProjXRayDose(models.Model):  # TID 10004
    """Accumulated Fluoroscopy and Acquisition Projection X-Ray Dose TID 10004

    From DICOM Part 16:
        This general template provides detailed information on projection X-Ray dose value accumulations over
        several irradiation events from the same equipment (typically a study or a performed procedure step).

    """

    accumulated_xray_dose = models.ForeignKey(AccumXRayDose, on_delete=models.CASCADE)
    fluoro_dose_area_product_total = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    fluoro_dose_rp_total = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    total_fluoro_time = models.DecimalField(
        max_digits=7, decimal_places=2, blank=True, null=True
    )
    acquisition_dose_area_product_total = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    acquisition_dose_rp_total = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    total_acquisition_time = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    # The following fields should not be in this table, and are duplicated in the
    # AccumCassetteBsdProjRadiogDose and AccumIntegratedProjRadiogDose
    # tables below.
    # TODO: Ensure rdsr.py and dx.py use the other table and do not populate this one any further.
    dose_area_product_total = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    dose_rp_total = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    total_number_of_radiographic_frames = models.DecimalField(
        max_digits=6, decimal_places=0, blank=True, null=True
    )
    reference_point_definition = models.TextField(blank=True, null=True)
    reference_point_definition_code = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )

    def fluoro_gym2_to_cgycm2(self):
        """Converts fluoroscopy DAP total from Gy.m2 to cGy.cm2 for display in web interface"""
        if self.fluoro_dose_area_product_total:
            return 1000000 * self.fluoro_dose_area_product_total

    def acq_gym2_to_cgycm2(self):
        """Converts acquisition DAP total from Gy.m2 to cGy.cm2 for display in web interface"""
        if self.acquisition_dose_area_product_total:
            return 1000000 * self.acquisition_dose_area_product_total

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "accumulated_xray_dose",
                ]
            ),
        ]


class AccumMammographyXRayDose(models.Model):  # TID 10005
    """Accumulated Mammography X-Ray Dose TID 10005

    From DICOM Part 16:
        This modality specific template provides detailed information on mammography X-Ray dose value
        accumulations over several irradiation events from the same equipment (typically a study or a performed
        procedure step).
    """

    accumulated_xray_dose = models.ForeignKey(AccumXRayDose, on_delete=models.CASCADE)
    accumulated_average_glandular_dose = models.DecimalField(
        max_digits=8, decimal_places=4, blank=True, null=True
    )
    laterality = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "accumulated_xray_dose",
                ]
            ),
        ]


class AccumCassetteBsdProjRadiogDose(models.Model):  # TID 10006
    """Accumulated Cassette-based Projection Radiography Dose TID 10006

    From DICOM Part 16 Correction Proposal CP-1077:
        This template provides information on Projection Radiography dose values accumulated on Cassette-
        based systems over one or more irradiation events (typically a study or a performed procedure step) from
        the same equipment.
    """

    accumulated_xray_dose = models.ForeignKey(AccumXRayDose, on_delete=models.CASCADE)
    detector_type = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )
    total_number_of_radiographic_frames = models.DecimalField(
        max_digits=6, decimal_places=0, blank=True, null=True
    )

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "accumulated_xray_dose",
                ]
            ),
        ]


class AccumIntegratedProjRadiogDose(models.Model):  # TID 10007
    """Accumulated Integrated Projection Radiography Dose TID 10007

    From DICOM Part 16 Correction Proposal CP-1077:
        This template provides information on Projection Radiography dose values accumulated on Integrated
        systems over one or more irradiation events (typically a study or a performed procedure step) from the
        same equipment.
    """

    accumulated_xray_dose = models.ForeignKey(AccumXRayDose, on_delete=models.CASCADE)
    dose_area_product_total = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    dose_rp_total = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    total_number_of_radiographic_frames = models.DecimalField(
        max_digits=6, decimal_places=0, blank=True, null=True
    )
    reference_point_definition_code = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )
    reference_point_definition = models.TextField(blank=True, null=True)

    def convert_gym2_to_cgycm2(self):
        """Converts Gy.m2 to cGy.cm2 for display in web interface"""
        if self.dose_area_product_total:
            return 1000000 * self.dose_area_product_total

    dose_area_product_total_over_delta_weeks = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )
    dose_rp_total_over_delta_weeks = models.DecimalField(
        max_digits=16, decimal_places=12, blank=True, null=True
    )

    def total_dap_delta_gym2_to_cgycm2(self):
        """Converts total DAP over delta days from Gy.m2 to cGy.cm2 for display in web interface"""
        if self.dose_area_product_total_over_delta_weeks:
            return 1000000 * self.dose_area_product_total_over_delta_weeks

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "accumulated_xray_dose",
                ]
            ),
        ]


class PKsForSummedRFDoseStudiesInDeltaWeeks(models.Model):
    """Table to hold foreign keys of all studies that fall within the delta
    weeks of each RF study.
    """

    general_study_module_attributes = models.ForeignKey(
        GeneralStudyModuleAttr, on_delete=models.CASCADE
    )
    study_pk_in_delta_weeks = models.IntegerField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "general_study_module_attributes",
                ]
            ),
        ]


class PatientModuleAttr(models.Model):  # C.7.1.1
    """Patient Module C.7.1.1

    From DICOM Part 3: Information Object Definitions Table C.7-1:
        Specifies the Attributes of the Patient that describe and identify the Patient who is
        the subject of a diagnostic Study. This Module contains Attributes of the patient that are needed
        for diagnostic interpretation of the Image and are common for all studies performed on the
        patient. It contains Attributes that are also included in the Patient Modules in Section C.2.
    """

    general_study_module_attributes = models.ForeignKey(
        GeneralStudyModuleAttr, on_delete=models.CASCADE
    )
    patient_name = models.TextField(blank=True, null=True)
    name_hashed = models.BooleanField(default=False)
    patient_id = models.TextField(blank=True, null=True)
    id_hashed = models.BooleanField(default=False)
    patient_birth_date = models.DateField(blank=True, null=True)
    patient_sex = models.CharField(max_length=2, blank=True, null=True)
    other_patient_ids = models.TextField(blank=True, null=True)
    not_patient_indicator = models.TextField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "general_study_module_attributes",
                ]
            ),
            models.Index(
                fields=[
                    "patient_id",
                ]
            ),
        ]


class PatientStudyModuleAttr(models.Model):  # C.7.2.2
    """Patient Study Module C.7.2.2

    From DICOM Part 3: Information Object Definitions Table C.7-4a:
        Defines Attributes that provide information about the Patient at the time the Study
        started.
    """

    general_study_module_attributes = models.ForeignKey(
        GeneralStudyModuleAttr, on_delete=models.CASCADE
    )
    admitting_diagnosis_description = models.TextField(blank=True, null=True)
    admitting_diagnosis_code_sequence = models.TextField(blank=True, null=True)
    patient_age = models.CharField(max_length=4, blank=True, null=True)
    patient_age_decimal = models.DecimalField(
        max_digits=7, decimal_places=3, blank=True, null=True
    )
    patient_size = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    patient_weight = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    # TODO: Add patient size code sequence

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "general_study_module_attributes",
                ]
            ),
        ]


class GeneralEquipmentModuleAttr(models.Model):  # C.7.5.1
    """General Equipment Module C.7.5.1

    From DICOM Part 3: Information Object Definitions Table C.7-8:
        Specifies the Attributes that identify and describe the piece of equipment that
        produced a Series of Composite Instances.
    """

    general_study_module_attributes = models.ForeignKey(
        GeneralStudyModuleAttr, on_delete=models.CASCADE
    )
    manufacturer = models.TextField(blank=True, null=True)
    institution_name = models.TextField(blank=True, null=True)
    institution_address = models.TextField(blank=True, null=True)
    station_name = models.CharField(max_length=32, blank=True, null=True)
    institutional_department_name = models.TextField(blank=True, null=True)
    manufacturer_model_name = models.TextField(blank=True, null=True)
    device_serial_number = models.TextField(blank=True, null=True)
    software_versions = models.TextField(blank=True, null=True)
    gantry_id = models.TextField(blank=True, null=True)
    spatial_resolution = models.DecimalField(
        max_digits=8, decimal_places=4, blank=True, null=True
    )
    date_of_last_calibration = models.DateTimeField(blank=True, null=True)
    time_of_last_calibration = models.DateTimeField(blank=True, null=True)
    unique_equipment_name = models.ForeignKey(
        UniqueEquipmentNames, null=True, on_delete=models.CASCADE
    )

    def __unicode__(self):
        return self.station_name

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "general_study_module_attributes",
                ]
            ),
            models.Index(
                fields=[
                    "unique_equipment_name",
                ]
            ),
        ]


# Radiopharmaca


class RadiopharmaceuticalRadiationDose(models.Model):  # TID 10021
    """
    Radiopharmaceutical Radiation Dose TID 10021

    From DICOM Part 16:
       This Template defines a container (the root) with subsidiary Content Items, each of which corresponds to a
       single Radiopharmaceutical Administration Dose event entry. There is a defined recording observer (the
       system and/or person responsible for recording the assay of the radiopharmaceutical, and the person
       administered the radiopharmaceutical). Multiple Radiopharmaceutical Radiation Dose objects may be created
       for one study. Radiopharmaceutical Start DateTime in TID 10022 “Radiopharmaceutical Administration Event
       Data” will convey the order of administrations.
    """

    general_study_module_attributes = models.ForeignKey(
        GeneralStudyModuleAttr, on_delete=models.CASCADE
    )
    associated_procedure = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10021_procedure",
        on_delete=models.CASCADE,
    )  # CID 3108
    has_intent = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10021_intent",
        on_delete=models.CASCADE,
    )  # CID 3629
    comment = models.TextField(blank=True, null=True)


class LanguageofContentItemandDescendants(models.Model):  # TID 1204
    radiopharmaceutical_radiation_dose = models.ForeignKey(
        RadiopharmaceuticalRadiationDose, on_delete=models.CASCADE
    )
    language_of_contentitem_and_descendants = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid1204_language",
        on_delete=models.CASCADE,
    )  # CID 5000
    country_of_language = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid1204_country",
        on_delete=models.CASCADE,
    )  # CID 5001


class RadiopharmaceuticalAdministrationEventData(models.Model):  # TID 10022
    radiopharmaceutical_radiation_dose = models.ForeignKey(
        RadiopharmaceuticalRadiationDose, on_delete=models.CASCADE
    )
    radiopharmaceutical_agent = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10022_agent",
        on_delete=models.CASCADE,
    )  # CID 25 & CID 4021
    radiopharmaceutical_agent_string = models.TextField(
        blank=True, null=True
    )  # In NM Images the radiopharmaceutical may only be present as string
    radionuclide = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10022_radionuclide",
        on_delete=models.CASCADE,
    )  # CID 18 & CID 4020
    radionuclide_half_life = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    radiopharmaceutical_specific_activity = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    radiopharmaceutical_administration_event_uid = models.TextField(
        blank=True, null=True
    )
    estimated_extravasation_activity = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    radiopharmaceutical_start_datetime = models.DateTimeField(blank=True, null=True)
    radiopharmaceutical_stop_datetime = models.DateTimeField(blank=True, null=True)
    administered_activity = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    effective_dose = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    radiopharmaceutical_volume = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    pre_administration_measured_activity = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    pre_activity_measurement_device = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10022_device_pre",
        on_delete=models.CASCADE,
    )  # CID 10041
    post_administration_measured_activity = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    post_activity_measurement_device = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10022_device_post",
        on_delete=models.CASCADE,
    )  # CID 10041
    route_of_administration = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10022_route",
        on_delete=models.CASCADE,
    )  # CID 11
    site_of = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10022_site",
        on_delete=models.CASCADE,
    )  # CID 3746
    laterality = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10022_laterality",
        on_delete=models.CASCADE,
    )  # CID 244
    brand_name = models.TextField(blank=True, null=True)
    radiopharmaceutical_dispense_unit_identifier = models.TextField(
        blank=True, null=True
    )
    prescription_identifier = models.TextField(blank=True, null=True)
    comment = models.TextField(blank=True, null=True)


class IntravenousExtravasationSymptoms(models.Model):
    radiopharmaceutical_administration_event_data = models.ForeignKey(
        RadiopharmaceuticalAdministrationEventData, on_delete=models.CASCADE
    )
    intravenous_extravasation_symptoms = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10022_symptoms",
        on_delete=models.CASCADE,
    )  # CID 10043


class BillingCode(models.Model):
    radiopharmaceutical_administration_event_data = models.ForeignKey(
        RadiopharmaceuticalAdministrationEventData, on_delete=models.CASCADE
    )
    billing_code = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )


class DrugProductIdentifier(models.Model):
    radiopharmaceutical_administration_event_data = models.ForeignKey(
        RadiopharmaceuticalAdministrationEventData, on_delete=models.CASCADE
    )
    drug_product_identifier = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )


class RadiopharmaceuticalLotIdentifier(models.Model):
    radiopharmaceutical_administration_event_data = models.ForeignKey(
        RadiopharmaceuticalAdministrationEventData, on_delete=models.CASCADE
    )
    radiopharmaceutical_lot_identifier = models.TextField(blank=True, null=True)


class ReagentVialIdentifier(models.Model):
    radiopharmaceutical_administration_event_data = models.ForeignKey(
        RadiopharmaceuticalAdministrationEventData, on_delete=models.CASCADE
    )
    reagent_vial_identifier = models.TextField(blank=True, null=True)


class RadionuclideIdentifier(models.Model):
    radiopharmaceutical_administration_event_data = models.ForeignKey(
        RadiopharmaceuticalAdministrationEventData, on_delete=models.CASCADE
    )
    radionuclide_identifier = models.TextField(blank=True, null=True)


class OrganDose(models.Model):  # TID 10023
    radiopharmaceutical_administration_event_data = models.ForeignKey(
        RadiopharmaceuticalAdministrationEventData, on_delete=models.CASCADE
    )
    finding_site = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10023_site",
        on_delete=models.CASCADE,
    )  # CID 10044
    laterality = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10023_laterality",
        on_delete=models.CASCADE,
    )  # CID 244
    mass = models.DecimalField(max_digits=16, decimal_places=8, blank=True, null=True)
    measurement_method = models.TextField(blank=True, null=True)
    organ_dose = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    reference_authority_code = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10023_authority",
        on_delete=models.CASCADE,
    )  # CID 10040
    reference_authority_text = models.TextField(blank=True, null=True)
    type_of_detector_motion = models.TextField(blank=True, null=True)


class PETSeries(models.Model):
    radiopharmaceutical_radiation_dose = models.ForeignKey(
        RadiopharmaceuticalRadiationDose, on_delete=models.CASCADE
    )
    series_uid = models.TextField(blank=True, null=True)
    series_datetime = models.DateTimeField(blank=True, null=True)
    number_of_rr_intervals = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    number_of_time_slots = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    number_of_time_slices = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    number_of_slices = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    reconstruction_method = models.TextField(blank=True, null=True)
    coincidence_window_width = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    energy_window_lower_limit = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    energy_window_upper_limit = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    scan_progression_direction = models.TextField(blank=True, null=True)


class PETSeriesCorrection(models.Model):
    pet_series = models.ForeignKey(PETSeries, on_delete=models.CASCADE)
    corrected_image = models.TextField(blank=True, null=True)


class PETSeriesType(models.Model):
    pet_series = models.ForeignKey(PETSeries, on_delete=models.CASCADE)
    series_type = models.TextField(blank=True, null=True)


class RadiopharmaceuticalAdministrationPatientCharacteristics(models.Model):
    radiopharmaceutical_radiation_dose = models.ForeignKey(
        RadiopharmaceuticalRadiationDose, on_delete=models.CASCADE
    )
    subject_age = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    subject_sex = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10023_sex",
        on_delete=models.CASCADE,
    )  # CID 7455
    patient_height = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    patient_weight = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    body_surface_area = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    body_surface_area_formula = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10023_body_surface_area",
        on_delete=models.CASCADE,
    )  # CID 3663
    body_mass_index = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    equation = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10023_equation",
        on_delete=models.CASCADE,
    )
    glucose = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    fasting_duration = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    hydration_volume = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    recent_physical_activity = models.TextField(blank=True, null=True)
    serum_creatinine = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )


class PatientState(models.Model):  # CID 10045
    radiopharmaceutical_administration_patient_characteristics = models.ForeignKey(
        RadiopharmaceuticalAdministrationPatientCharacteristics,
        on_delete=models.CASCADE,
    )
    patient_state = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )


class GlomerularFiltrationRate(models.Model):
    radiopharmaceutical_administration_patient_characteristics = models.ForeignKey(
        RadiopharmaceuticalAdministrationPatientCharacteristics,
        on_delete=models.CASCADE,
    )
    glomerular_filtration_rate = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    measurement_method = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10023_measurement_method",
        on_delete=models.CASCADE,
    )  # CID 10047
    equivalent_meaning_of_concept_name = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10023_equivalent_meaning_of_concept",
        on_delete=models.CASCADE,
    )  # CID 10046


# CT


class CtRadiationDose(models.Model):  # TID 10011
    """CT Radiation Dose TID 10011

    From DICOM Part 16:
        This template defines a container (the root) with subsidiary content items, each of which corresponds to a
        single CT X-Ray irradiation event entry. There is a defined recording observer (the system or person
        responsible for recording the log, generally the system). Accumulated values shall be kept for a whole
        Study or at least a part of a Study, if the Study is divided in the workflow of the examination, or a
        performed procedure step. Multiple CT Radiation Dose objects may be created for one Study.
    """

    general_study_module_attributes = models.ForeignKey(
        GeneralStudyModuleAttr, on_delete=models.CASCADE
    )
    procedure_reported = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10011_procedure",
        on_delete=models.CASCADE,
    )
    has_intent = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10011_intent",
        on_delete=models.CASCADE,
    )  # CID 3629
    start_of_xray_irradiation = models.DateTimeField(blank=True, null=True)
    end_of_xray_irradiation = models.DateTimeField(blank=True, null=True)
    scope_of_accumulation = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10011_scope",
        on_delete=models.CASCADE,
    )  # CID 10000
    uid_type = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid1011_uid",
        on_delete=models.CASCADE,
    )  # CID 10001
    comment = models.TextField(blank=True, null=True)
    # does need to be a table on its own as is 1-n
    source_of_dose_information = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10011_source",
        on_delete=models.CASCADE,
    )

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "general_study_module_attributes",
                ]
            ),
        ]


class SourceOfCTDoseInformation(models.Model):  # CID 10021
    """Source of CT Dose Information"""

    # TODO: populate this table when extracting and move existing data. Task #164
    ct_radiation_dose = models.ForeignKey(CtRadiationDose, on_delete=models.CASCADE)
    source_of_dose_information = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )  # CID 10021

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "ct_radiation_dose",
                ]
            ),
            models.Index(
                fields=[
                    "source_of_dose_information",
                ]
            ),
        ]


class CtAccumulatedDoseData(models.Model):  # TID 10012
    """CT Accumulated Dose Data

    From DICOM Part 16:
        This general template provides detailed information on CT X-Ray dose value accumulations over several
        irradiation events from the same equipment and over the scope of accumulation specified for the report
        (typically a Study or a Performed Procedure Step).
    """

    ct_radiation_dose = models.ForeignKey(CtRadiationDose, on_delete=models.CASCADE)
    total_number_of_irradiation_events = models.DecimalField(
        max_digits=16, decimal_places=0, blank=True, null=True
    )
    ct_dose_length_product_total = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    ct_effective_dose_total = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    reference_authority_code = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10012_authority",
        on_delete=models.CASCADE,
    )  # CID 10015 (ICRP60/103)
    reference_authority_text = models.TextField(blank=True, null=True)
    measurement_method = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10012_method",
        on_delete=models.CASCADE,
    )  # CID 10011
    patient_model = models.TextField(blank=True, null=True)
    effective_dose_phantom_type = models.TextField(blank=True, null=True)
    dosimeter_type = models.TextField(blank=True, null=True)
    comment = models.TextField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "ct_radiation_dose",
                ]
            ),
        ]


class CtIrradiationEventData(models.Model):  # TID 10013
    """CT Irradiation Event Data TID 10013

    From DICOM Part 16:
        This template conveys the dose and equipment parameters of a single irradiation event.

    Additional to the template:
        + date_time_started
        + series_description
    """

    ct_radiation_dose = models.ForeignKey(CtRadiationDose, on_delete=models.CASCADE)
    acquisition_protocol = models.TextField(blank=True, null=True)
    target_region = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10013_region",
        on_delete=models.CASCADE,
    )  # CID 4030
    ct_acquisition_type = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10013_type",
        on_delete=models.CASCADE,
    )  # CID 10013
    procedure_context = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10013_context",
        on_delete=models.CASCADE,
    )  # CID 10014
    irradiation_event_uid = models.TextField(blank=True, null=True)
    #  TODO: Add extraction of the label and label type (Series, acquisition, instance number) Issue #167
    irradiation_event_label = models.TextField(blank=True, null=True)
    label_type = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10013_labeltype",
        on_delete=models.CASCADE,
    )  # CID 10022
    exposure_time = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    nominal_single_collimation_width = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    nominal_total_collimation_width = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    pitch_factor = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    number_of_xray_sources = models.DecimalField(
        max_digits=8, decimal_places=0, blank=True, null=True
    )
    mean_ctdivol = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    ctdiw_phantom_type = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10013_phantom",
        on_delete=models.CASCADE,
    )  # CID 4052
    ctdifreeair_calculation_factor = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    mean_ctdifreeair = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    dlp = models.DecimalField(max_digits=16, decimal_places=8, blank=True, null=True)
    effective_dose = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    measurement_method = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid10013_method",
        on_delete=models.CASCADE,
    )  # CID 10011
    effective_dose_conversion_factor = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    xray_modulation_type = models.TextField(blank=True, null=True)
    comment = models.TextField(blank=True, null=True)
    # Not in DICOM standard:
    date_time_started = models.DateTimeField(blank=True, null=True)
    series_description = models.TextField(blank=True, null=True)
    standard_protocols = models.ManyToManyField(StandardNames)

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "ct_radiation_dose",
                ]
            ),
        ]


class CtReconstructionAlgorithm(models.Model):
    """Container in TID 10013 to hold CT reconstruction methods"""

    # TODO: Add this to the rdsr extraction routines. Issue #166
    ct_irradiation_event_data = models.ForeignKey(
        CtIrradiationEventData, on_delete=models.CASCADE
    )
    reconstruction_algorithm = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )  # CID 10033

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "ct_irradiation_event_data",
                ]
            ),
            models.Index(
                fields=[
                    "reconstruction_algorithm",
                ]
            ),
        ]


class CtXRaySourceParameters(models.Model):
    """Container in TID 10013 to hold CT x-ray source parameters"""

    ct_irradiation_event_data = models.ForeignKey(
        CtIrradiationEventData, on_delete=models.CASCADE
    )
    identification_of_the_xray_source = models.TextField(blank=True, null=True)
    kvp = models.DecimalField(max_digits=16, decimal_places=8, blank=True, null=True)
    maximum_xray_tube_current = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    xray_tube_current = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    exposure_time_per_rotation = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    xray_filter_aluminum_equivalent = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "ct_irradiation_event_data",
                ]
            ),
        ]


class ScanningLength(models.Model):  # TID 10014
    """Scanning Length TID 10014

    From DICOM Part 16:
        No description
    """

    ct_irradiation_event_data = models.ForeignKey(
        CtIrradiationEventData, on_delete=models.CASCADE
    )
    scanning_length = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    length_of_reconstructable_volume = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    exposed_range = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    top_z_location_of_reconstructable_volume = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    bottom_z_location_of_reconstructable_volume = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    top_z_location_of_scanning_length = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    bottom_z_location_of_scanning_length = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    frame_of_reference_uid = models.TextField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "ct_irradiation_event_data",
                ]
            ),
        ]


class SizeSpecificDoseEstimation(models.Model):
    """Container in TID 10013 to hold size specific dose estimation details"""

    # TODO: Add this to the rdsr extraction routines. Issue #168
    ct_irradiation_event_data = models.ForeignKey(
        CtIrradiationEventData, on_delete=models.CASCADE
    )
    measurement_method = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="ssde_method",
        on_delete=models.CASCADE,
    )  # CID 10023
    measured_lateral_dimension = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    measured_ap_dimension = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    derived_effective_diameter = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    water_equivalent_diameter = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    water_equivalent_diameter_method = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="ssde_wed_method",
        on_delete=models.CASCADE,
    )  # CID 10024
    wed_estimate_location_z = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "ct_irradiation_event_data",
                ]
            ),
            models.Index(
                fields=[
                    "measurement_method",
                ]
            ),
        ]


class WEDSeriesOrInstances(models.Model):
    """From TID 10013 Series or Instance used for Water Equivalent Diameter estimation"""

    size_specific_dose_estimation = models.ForeignKey(
        SizeSpecificDoseEstimation, on_delete=models.CASCADE
    )
    wed_series_or_instance = models.TextField(blank=True, null=True)  # referenced UID

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "size_specific_dose_estimation",
                ]
            ),
        ]


class CtDoseCheckDetails(models.Model):  # TID 10015
    """CT Dose Check Details TID 10015

    From DICOM Part 16:
        This template records details related to the use of the NEMA Dose Check Standard (NEMA XR-25-2010).
    """

    ct_irradiation_event_data = models.ForeignKey(
        CtIrradiationEventData, on_delete=models.CASCADE
    )
    dlp_alert_value_configured = models.BooleanField(null=True)
    ctdivol_alert_value_configured = models.BooleanField(null=True)
    dlp_alert_value = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    ctdivol_alert_value = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    accumulated_dlp_forward_estimate = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    accumulated_ctdivol_forward_estimate = models.DecimalField(
        max_digits=16, decimal_places=8, blank=True, null=True
    )
    # alert_ added to allow two fields that are in different containers in std
    alert_reason_for_proceeding = models.TextField(blank=True, null=True)
    dlp_notification_value_configured = models.BooleanField(null=True)
    ctdivol_notification_value_configured = models.BooleanField(null=True)
    dlp_notification_value = models.DecimalField(
        max_digits=8, decimal_places=4, blank=True, null=True
    )
    ctdivol_notification_value = models.DecimalField(
        max_digits=8, decimal_places=4, blank=True, null=True
    )
    dlp_forward_estimate = models.DecimalField(
        max_digits=8, decimal_places=4, blank=True, null=True
    )
    ctdivol_forward_estimate = models.DecimalField(
        max_digits=8, decimal_places=4, blank=True, null=True
    )
    # notification_ added to allow two fields that are in different containers in std
    notification_reason_for_proceeding = models.TextField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "ct_irradiation_event_data",
                ]
            ),
        ]


# Models common to both


class ObserverContext(models.Model):  # TID 1002
    """Observer Context TID 1002

    From DICOM Part 16:
        The observer (person or device) that created the Content Items to which this context applies.
    """

    projection_xray_radiation_dose = models.ForeignKey(
        ProjectionXRayRadiationDose, blank=True, null=True, on_delete=models.CASCADE
    )
    ct_radiation_dose = models.ForeignKey(
        CtRadiationDose, blank=True, null=True, on_delete=models.CASCADE
    )
    radiopharmaceutical_administration_event_data = models.ForeignKey(
        RadiopharmaceuticalAdministrationEventData,
        blank=True,
        null=True,
        on_delete=models.CASCADE,
    )
    radiopharmaceutical_administration_is_pre_observer = models.BooleanField(
        blank=True, null=True
    )
    observer_type = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid1002_observertype",
        on_delete=models.CASCADE,
    )  # CID 270
    person_observer_name = models.TextField(blank=True, null=True)
    person_observer_organization_name = models.TextField(blank=True, null=True)
    person_observer_role_in_organization = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid1002_ptroleorg",
        on_delete=models.CASCADE,
    )  # CID 7452
    person_observer_role_in_procedure = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid1002_ptroleproc",
        on_delete=models.CASCADE,
    )  # CID 7453
    device_observer_uid = models.TextField(blank=True, null=True)
    device_observer_name = models.TextField(blank=True, null=True)
    device_observer_manufacturer = models.TextField(blank=True, null=True)
    device_observer_model_name = models.TextField(blank=True, null=True)
    device_observer_serial_number = models.TextField(blank=True, null=True)
    device_observer_physical_location_during_observation = models.TextField(
        blank=True, null=True
    )
    device_role_in_procedure = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid1002_role",
        on_delete=models.CASCADE,
    )  # CID 7445

    def __unicode__(self):
        return self.device_observer_name

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "projection_xray_radiation_dose",
                ]
            ),
            models.Index(
                fields=[
                    "ct_radiation_dose",
                ]
            ),
        ]


class DeviceParticipant(models.Model):  # TID 1021
    """Device Participant TID 1021

    From DICOM Part 16:
        This template describes a device participating in an activity as other than an observer or subject. E.g. for
        a dose report documenting an irradiating procedure, participants include the irradiating device.
    """

    accumulated_xray_dose = models.ForeignKey(
        AccumXRayDose, blank=True, null=True, on_delete=models.CASCADE
    )
    irradiation_event_xray_detector_data = models.ForeignKey(
        IrradEventXRayDetectorData, blank=True, null=True, on_delete=models.CASCADE
    )
    irradiation_event_xray_source_data = models.ForeignKey(
        IrradEventXRaySourceData, blank=True, null=True, on_delete=models.CASCADE
    )
    ct_accumulated_dose_data = models.ForeignKey(
        CtAccumulatedDoseData, blank=True, null=True, on_delete=models.CASCADE
    )
    ct_irradiation_event_data = models.ForeignKey(
        CtIrradiationEventData, blank=True, null=True, on_delete=models.CASCADE
    )
    device_role_in_procedure = models.ForeignKey(
        ContextID, blank=True, null=True, on_delete=models.CASCADE
    )
    device_name = models.TextField(blank=True, null=True)
    device_manufacturer = models.TextField(blank=True, null=True)
    device_model_name = models.TextField(blank=True, null=True)
    device_serial_number = models.TextField(blank=True, null=True)
    device_observer_uid = models.TextField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "accumulated_xray_dose",
                ]
            ),
            models.Index(
                fields=[
                    "irradiation_event_xray_detector_data",
                ]
            ),
            models.Index(
                fields=[
                    "irradiation_event_xray_source_data",
                ]
            ),
            models.Index(
                fields=[
                    "ct_accumulated_dose_data",
                ]
            ),
            models.Index(
                fields=[
                    "ct_irradiation_event_data",
                ]
            ),
            models.Index(
                fields=[
                    "device_role_in_procedure",
                ]
            ),
        ]


class PersonParticipant(models.Model):  # TID 1020
    """Person Participant TID 1020

    From DICOM Part 16:
        This template describes a person participating in an activity as other than an observer or subject. E.g. for
        a dose report documenting an irradiating procedure, participants include the person administering the
        irradiation and the person authorizing the irradiation.
    """

    projection_xray_radiation_dose = models.ForeignKey(
        ProjectionXRayRadiationDose, blank=True, null=True, on_delete=models.CASCADE
    )
    ct_radiation_dose = models.ForeignKey(
        CtRadiationDose, blank=True, null=True, on_delete=models.CASCADE
    )
    irradiation_event_xray_data = models.ForeignKey(
        IrradEventXRayData, blank=True, null=True, on_delete=models.CASCADE
    )
    ct_accumulated_dose_data = models.ForeignKey(
        CtAccumulatedDoseData, blank=True, null=True, on_delete=models.CASCADE
    )
    ct_irradiation_event_data = models.ForeignKey(
        CtIrradiationEventData, blank=True, null=True, on_delete=models.CASCADE
    )
    ct_dose_check_details_alert = models.ForeignKey(
        CtDoseCheckDetails,
        blank=True,
        null=True,
        related_name="tid1020_alert",
        on_delete=models.CASCADE,
    )
    ct_dose_check_details_notification = models.ForeignKey(
        CtDoseCheckDetails,
        blank=True,
        null=True,
        related_name="tid1020_notification",
        on_delete=models.CASCADE,
    )
    radiopharmaceutical_administration_event_data = models.ForeignKey(
        RadiopharmaceuticalAdministrationEventData,
        blank=True,
        null=True,
        on_delete=models.CASCADE,
    )
    person_name = models.TextField(blank=True, null=True)
    # CharField version is a mistake and shouldn't be used
    person_role_in_procedure = models.CharField(max_length=16, blank=True)
    person_role_in_procedure_cid = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid1020_roleproc",
        on_delete=models.CASCADE,
    )
    person_id = models.TextField(blank=True, null=True)
    person_id_issuer = models.TextField(blank=True, null=True)
    organization_name = models.TextField(blank=True, null=True)
    # TextField version is a mistake and shouldn't be used
    person_role_in_organization = models.TextField(blank=True, null=True)
    person_role_in_organization_cid = models.ForeignKey(
        ContextID,
        blank=True,
        null=True,
        related_name="tid1020_roleorg",
        on_delete=models.CASCADE,
    )  # CID 7452

    def __unicode__(self):
        return self.person_name

    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "projection_xray_radiation_dose",
                ]
            ),
            models.Index(
                fields=[
                    "ct_radiation_dose",
                ]
            ),
            models.Index(
                fields=[
                    "irradiation_event_xray_data",
                ]
            ),
            models.Index(
                fields=[
                    "ct_accumulated_dose_data",
                ]
            ),
            models.Index(
                fields=[
                    "ct_irradiation_event_data",
                ]
            ),
            models.Index(
                fields=[
                    "ct_dose_check_details_alert",
                ]
            ),
            models.Index(
                fields=[
                    "ct_dose_check_details_notification",
                ]
            ),
            models.Index(
                fields=[
                    "person_role_in_procedure_cid",
                ]
            ),
            models.Index(
                fields=[
                    "person_role_in_organization_cid",
                ]
            ),
        ]


class SummaryFields(models.Model):
    """Status and progress of populating the summary fields in GeneralStudyModuleAttr"""

    modality_type = models.CharField(max_length=2, null=True)
    complete = models.BooleanField(default=False)
    status_message = models.TextField(blank=True, null=True)
    total_studies = models.IntegerField(default=0)
    current_study = models.IntegerField(default=0)


class UpgradeStatus(SingletonModel):
    """
    Record upgrade status activity
    """

    from_0_9_1_summary_fields = models.BooleanField(default=False)

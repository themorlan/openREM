# pylint: disable=too-many-lines
# This Python file uses the following encoding: utf-8
#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2019  The Royal Marsden NHS Foundation Trust
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
..  module:: forms
    :synopsis: Django forms definitions
"""

import os
import logging
import operator
from functools import reduce

from django import forms
from django.db.models import Q
from django.contrib.admin.widgets import FilteredSelectMultiple
from django.conf import settings
from django.utils.safestring import mark_safe
from django.urls import reverse
from django.utils.translation import gettext as _
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, HTML, Div
from crispy_forms.bootstrap import (
    FormActions,
    PrependedText,
    InlineCheckboxes,
    Accordion,
    AccordionGroup,
)

from .models import (
    DicomDeleteSettings,
    DicomRemoteQR,
    DicomStoreSCP,
    SkinDoseMapCalcSettings,
    NotPatientIndicatorsName,
    NotPatientIndicatorsID,
    HighDoseMetricAlertSettings,
    CommonVariables,
    OpenSkinSafeList,
    StandardNames,
    StandardNameSettings,
    GeneralStudyModuleAttr,
    CtIrradiationEventData,
    IrradEventXRayData,
    BackgroundTaskMaximumRows,
)

logger = logging.getLogger()


class SizeUploadForm(forms.Form):
    """Form for patient size csv file upload"""

    sizefile = forms.FileField(label=_("Select a file"))


class SizeHeadersForm(forms.Form):
    """Form for csv column header patient size imports through the web interface"""

    height_field = forms.ChoiceField(choices="")
    weight_field = forms.ChoiceField(choices="")
    id_field = forms.ChoiceField(choices="")
    id_type = forms.ChoiceField(choices="")
    overwrite = forms.BooleanField(initial=False, required=False)

    def __init__(self, my_choice=None, **kwargs):
        super(SizeHeadersForm, self).__init__(**kwargs)
        if my_choice:
            self.fields["height_field"] = forms.ChoiceField(
                choices=my_choice, widget=forms.Select(attrs={"class": "form-control"})
            )
            self.fields["weight_field"] = forms.ChoiceField(
                choices=my_choice, widget=forms.Select(attrs={"class": "form-control"})
            )
            self.fields["id_field"] = forms.ChoiceField(
                choices=my_choice, widget=forms.Select(attrs={"class": "form-control"})
            )
            ID_TYPES = (
                ("acc-no", _("Accession Number")),
                ("si-uid", _("Study instance UID")),
            )
            self.fields["id_type"] = forms.ChoiceField(
                choices=ID_TYPES, widget=forms.Select(attrs={"class": "form-control"})
            )


class itemsPerPageForm(forms.Form):
    itemsPerPage = forms.ChoiceField(
        label=_("Items per page"),
        choices=CommonVariables.ITEMS_PER_PAGE,
        required=False,
    )


class DXChartOptionsForm(forms.Form):
    """Form for DX chart options"""

    plotCharts = forms.BooleanField(label=_("Plot charts?"), required=False)
    plotCharts.group = "PlotCharts"
    plotDXAcquisitionMeanDAPOverTimePeriod = forms.ChoiceField(
        label="Time period", choices=CommonVariables.TIME_PERIOD, required=False
    )
    plotDXAcquisitionMeanDAPOverTimePeriod.group = "General"
    plotAverageChoice = forms.MultipleChoiceField(
        label="Average plots",
        choices=CommonVariables.AVERAGES,
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={"class": "CheckboxSelectMultiple"}),
    )
    plotAverageChoice.group = "General"
    plotGrouping = forms.ChoiceField(
        label=mark_safe("Grouping choice"),  # nosec
        choices=CommonVariables.CHART_GROUPING,
        required=False,
    )
    plotGrouping.group = "General"
    plotSeriesPerSystem = forms.BooleanField(
        label="Plot a series per system", required=False
    )
    plotSeriesPerSystem.group = "General"
    plotHistograms = forms.BooleanField(
        label="Calculate histogram data", required=False
    )
    plotHistograms.group = "General"
    plotDXInitialSortingChoice = forms.ChoiceField(
        label="Chart sorting", choices=CommonVariables.SORTING_CHOICES, required=False
    )
    plotDXInitialSortingChoice.group = "General"
    plotInitialSortingDirection = forms.ChoiceField(
        label="Sorting direction",
        choices=CommonVariables.SORTING_DIRECTION,
        required=False,
    )
    plotInitialSortingDirection.group = "General"
    plotDXAcquisitionFreq = forms.BooleanField(
        label=_("Acquisition frequency"), required=False
    )
    plotDXAcquisitionFreq.group = "Acquisition protocol"
    plotDXAcquisitionMeanDAP = forms.BooleanField(
        label=_("Acquisition DAP"), required=False
    )
    plotDXAcquisitionMeanDAP.group = "Acquisition protocol"
    plotDXAcquisitionMeanmAs = forms.BooleanField(
        label=_("Acquisition mAs"), required=False
    )
    plotDXAcquisitionMeanmAs.group = "Acquisition protocol"
    plotDXAcquisitionMeankVp = forms.BooleanField(
        label=_("Acquisition kVp"), required=False
    )
    plotDXAcquisitionMeankVp.group = "Acquisition protocol"
    plotDXAcquisitionMeanDAPOverTime = forms.BooleanField(
        label=_("Acquisition DAP over time"), required=False
    )
    plotDXAcquisitionMeanDAPOverTime.group = "Acquisition protocol"
    plotDXAcquisitionMeanmAsOverTime = forms.BooleanField(
        label=_("Acquisition mAs over time"), required=False
    )
    plotDXAcquisitionMeanmAsOverTime.group = "Acquisition protocol"
    plotDXAcquisitionMeankVpOverTime = forms.BooleanField(
        label=_("Acquisition kVp over time"), required=False
    )
    plotDXAcquisitionMeankVpOverTime.group = "Acquisition protocol"
    plotDXAcquisitionDAPvsMass = forms.BooleanField(
        label=_("Acquisition DAP vs mass"), required=False
    )
    plotDXAcquisitionDAPvsMass.group = "Acquisition protocol"
    plotDXStudyFreq = forms.BooleanField(label=_("Study frequency"), required=False)
    plotDXStudyFreq.group = "Study description"
    plotDXStudyMeanDAP = forms.BooleanField(label=_("Study DAP"), required=False)
    plotDXStudyMeanDAP.group = "Study description"
    plotDXStudyDAPvsMass = forms.BooleanField(
        label=_("Study DAP vs mass"), required=False
    )
    plotDXStudyDAPvsMass.group = "Study description"
    plotDXStudyPerDayAndHour = forms.BooleanField(
        label=_("Study workload"), required=False
    )
    plotDXStudyPerDayAndHour.group = "Study description"
    plotDXRequestFreq = forms.BooleanField(
        label=_("Requested procedure frequency"), required=False
    )
    plotDXRequestFreq.group = "Requested procedure"
    plotDXRequestMeanDAP = forms.BooleanField(
        label=_("Requested procedure DAP"), required=False
    )
    plotDXRequestMeanDAP.group = "Requested procedure"
    plotDXRequestDAPvsMass = forms.BooleanField(
        label=_("Requested procedure DAP vs mass"), required=False
    )
    plotDXRequestDAPvsMass.group = "Requested procedure"


class DXChartOptionsFormIncStandard(DXChartOptionsForm):
    plotDXStandardAcquisitionFreq = forms.BooleanField(
        label="Standard acquisition name frequency", required=False
    )
    plotDXStandardAcquisitionFreq.group = "Standard acquisition name"
    plotDXStandardAcquisitionMeanDAP = forms.BooleanField(
        label="Standard acquisition name DAP", required=False
    )
    plotDXStandardAcquisitionMeanDAP.group = "Standard acquisition name"
    plotDXStandardAcquisitionMeanmAs = forms.BooleanField(
        label="Standard acquisition name mAs", required=False
    )
    plotDXStandardAcquisitionMeanmAs.group = "Standard acquisition name"
    plotDXStandardAcquisitionMeankVp = forms.BooleanField(
        label="Standard acquisition name kVp", required=False
    )
    plotDXStandardAcquisitionMeankVp.group = "Standard acquisition name"
    plotDXStandardAcquisitionMeanDAPOverTime = forms.BooleanField(
        label="Standard acquisition name DAP over time", required=False
    )
    plotDXStandardAcquisitionMeanDAPOverTime.group = "Standard acquisition name"
    plotDXStandardAcquisitionMeanmAsOverTime = forms.BooleanField(
        label="Standard acquisition name mAs over time", required=False
    )
    plotDXStandardAcquisitionMeanmAsOverTime.group = "Standard acquisition name"
    plotDXStandardAcquisitionMeankVpOverTime = forms.BooleanField(
        label="Standard acquisition name kVp over time", required=False
    )
    plotDXStandardAcquisitionMeankVpOverTime.group = "Standard acquisition name"
    plotDXStandardAcquisitionDAPvsMass = forms.BooleanField(
        label="Standard acquisition name DAP vs mass", required=False
    )
    plotDXStandardAcquisitionDAPvsMass.group = "Standard acquisition name"

    plotDXStandardStudyFreq = forms.BooleanField(
        label="Standard study name frequency", required=False
    )
    plotDXStandardStudyFreq.group = "Standard study name"
    plotDXStandardStudyMeanDAP = forms.BooleanField(
        label="Standard study name DAP", required=False
    )
    plotDXStandardStudyMeanDAP.group = "Standard study name"
    plotDXStandardStudyDAPvsMass = forms.BooleanField(
        label="Standard study name DAP vs mass", required=False
    )
    plotDXStandardStudyDAPvsMass.group = "Standard study name"
    plotDXStandardStudyPerDayAndHour = forms.BooleanField(
        label="Standard study name workload", required=False
    )
    plotDXStandardStudyPerDayAndHour.group = "Standard study name"


class CTChartOptionsForm(forms.Form):
    """Form for CT chart options"""

    plotCharts = forms.BooleanField(label="Plot charts?", required=False)
    plotCharts.group = "PlotCharts"
    plotCTOverTimePeriod = forms.ChoiceField(
        label="Time period", choices=CommonVariables.TIME_PERIOD, required=False
    )
    plotCTOverTimePeriod.group = "General"
    plotAverageChoice = forms.MultipleChoiceField(
        label=_("Average plots"),
        choices=CommonVariables.AVERAGES,
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={"class": "CheckboxSelectMultiple"}),
    )
    plotAverageChoice.group = "General"
    plotGrouping = forms.ChoiceField(
        label=mark_safe(_("Grouping choice")),  # nosec
        choices=CommonVariables.CHART_GROUPING,
        required=False,
    )
    plotGrouping.group = "General"
    plotSeriesPerSystem = forms.BooleanField(
        label=_("Plot a series per system"), required=False
    )
    plotSeriesPerSystem.group = "General"
    plotHistograms = forms.BooleanField(
        label=_("Calculate histogram data"), required=False
    )
    plotHistograms.group = "General"
    plotCTInitialSortingChoice = forms.ChoiceField(
        label=_("Chart sorting"),
        choices=CommonVariables.SORTING_CHOICES,
        required=False,
    )
    plotCTInitialSortingChoice.group = "General"
    plotInitialSortingDirection = forms.ChoiceField(
        label=_("Sorting direction"),
        choices=CommonVariables.SORTING_DIRECTION,
        required=False,
    )
    plotInitialSortingDirection.group = "General"

    plotCTAcquisitionFreq = forms.BooleanField(
        label=_("Acquisition frequency"), required=False
    )
    plotCTAcquisitionFreq.group = "Acquisition protocol"
    plotCTAcquisitionMeanDLP = forms.BooleanField(
        label=_("Acquisition DLP"), required=False
    )
    plotCTAcquisitionMeanDLP.group = "Acquisition protocol"
    plotCTAcquisitionMeanCTDI = forms.BooleanField(
        label=mark_safe(_("Acquisition CTDI<sub>vol</sub>")), required=False  # nosec
    )
    plotCTAcquisitionMeanCTDI.group = "Acquisition protocol"
    plotCTAcquisitionDLPOverTime = forms.BooleanField(
        label=_("Acquisition DLP over time"), required=False
    )
    plotCTAcquisitionDLPOverTime.group = "Acquisition protocol"
    plotCTAcquisitionCTDIOverTime = forms.BooleanField(
        label=mark_safe(_("Acquisition CTDI<sub>vol</sub> over time")),  # nosec
        required=False,
    )
    plotCTAcquisitionCTDIOverTime.group = "Acquisition protocol"
    plotCTAcquisitionDLPvsMass = forms.BooleanField(
        label=_("Acquisition DLP vs mass"), required=False
    )
    plotCTAcquisitionDLPvsMass.group = "Acquisition protocol"
    plotCTAcquisitionCTDIvsMass = forms.BooleanField(
        label=mark_safe(_("Acquisition CTDI<sub>vol</sub> vs mass")),  # nosec
        required=False,
    )
    plotCTAcquisitionCTDIvsMass.group = "Acquisition protocol"
    plotCTAcquisitionTypes = forms.MultipleChoiceField(
        label=mark_safe(  # nosec
            _(
                "Acquisition types to include<br/>in acquisition-level chart<br/>calculations"
            )
        ),
        choices=CommonVariables.CT_ACQUISITION_TYPES,
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={"class": "CheckboxSelectMultiple"}),
    )
    plotCTAcquisitionTypes.group = "Acquisition protocol"

    plotCTStudyFreq = forms.BooleanField(label="Study frequency", required=False)
    plotCTStudyFreq.group = "Study description"
    plotCTStudyMeanDLP = forms.BooleanField(label="Study DLP", required=False)
    plotCTStudyMeanDLP.group = "Study description"
    plotCTStudyMeanCTDI = forms.BooleanField(
        label=mark_safe(_("Study CTDI<sub>vol</sub>")), required=False  # nosec
    )
    plotCTStudyMeanCTDI.group = "Study description"
    plotCTStudyNumEvents = forms.BooleanField(label="Study events", required=False)
    plotCTStudyNumEvents.group = "Study description"
    plotCTStudyMeanDLPOverTime = forms.BooleanField(
        label=_("Study DLP over time"), required=False
    )
    plotCTStudyMeanDLPOverTime.group = "Study description"
    plotCTStudyPerDayAndHour = forms.BooleanField(
        label=_("Study workload"), required=False
    )
    plotCTStudyPerDayAndHour.group = "Study description"

    plotCTRequestFreq = forms.BooleanField(
        label=_("Requested procedure frequency"), required=False
    )
    plotCTRequestFreq.group = "Requested procedure"
    plotCTRequestMeanDLP = forms.BooleanField(
        label=_("Requested procedure DLP"), required=False
    )
    plotCTRequestMeanDLP.group = "Requested procedure"
    plotCTRequestNumEvents = forms.BooleanField(
        label=_("Requested procedure events"), required=False
    )
    plotCTRequestNumEvents.group = "Requested procedure"
    plotCTRequestDLPOverTime = forms.BooleanField(
        label=_("Requested procedure DLP over time"), required=False
    )
    plotCTRequestDLPOverTime.group = "Requested procedure"


class CTChartOptionsFormIncStandard(CTChartOptionsForm):
    plotCTStandardAcquisitionFreq = forms.BooleanField(
        label="Standard acquisition name frequency", required=False
    )
    plotCTStandardAcquisitionFreq.group = "Standard acquisition name"
    plotCTStandardAcquisitionMeanDLP = forms.BooleanField(
        label="Standard acquisition name DLP", required=False
    )
    plotCTStandardAcquisitionMeanDLP.group = "Standard acquisition name"
    plotCTStandardAcquisitionMeanCTDI = forms.BooleanField(
        label=mark_safe("Standard acquisition name CTDI<sub>vol</sub>"),  # nosec
        required=False,
    )
    plotCTStandardAcquisitionMeanCTDI.group = "Standard acquisition name"
    plotCTStandardAcquisitionDLPOverTime = forms.BooleanField(
        label="Standard acquisition name DLP over time", required=False
    )
    plotCTStandardAcquisitionDLPOverTime.group = "Standard acquisition name"
    plotCTStandardAcquisitionCTDIOverTime = forms.BooleanField(
        label=mark_safe(  # nosec
            "Standard acquisition name CTDI<sub>vol</sub> over time"
        ),
        required=False,
    )
    plotCTStandardAcquisitionCTDIOverTime.group = "Standard acquisition name"
    plotCTStandardAcquisitionDLPvsMass = forms.BooleanField(
        label="Standard acquisition name DLP vs mass", required=False
    )
    plotCTStandardAcquisitionDLPvsMass.group = "Standard acquisition name"
    plotCTStandardAcquisitionCTDIvsMass = forms.BooleanField(
        label=mark_safe(  # nosec
            "Standard acquisition name CTDI<sub>vol</sub> vs mass"
        ),
        required=False,
    )
    plotCTStandardAcquisitionCTDIvsMass.group = "Standard acquisition name"

    plotCTStandardStudyFreq = forms.BooleanField(
        label="Standard study frequency", required=False
    )
    plotCTStandardStudyFreq.group = "Standard study name"
    plotCTStandardStudyMeanDLP = forms.BooleanField(
        label="Standard study DLP", required=False
    )
    plotCTStandardStudyMeanDLP.group = "Standard study name"
    plotCTStandardStudyNumEvents = forms.BooleanField(
        label="Standard study events", required=False
    )
    plotCTStandardStudyNumEvents.group = "Standard study name"
    plotCTStandardStudyMeanDLPOverTime = forms.BooleanField(
        label="Standard study DLP over time", required=False
    )
    plotCTStandardStudyMeanDLPOverTime.group = "Standard study name"
    plotCTStandardStudyPerDayAndHour = forms.BooleanField(
        label="Standard study workload", required=False
    )
    plotCTStandardStudyPerDayAndHour.group = "Standard study name"


class NMChartOptionsForm(forms.Form):
    """
    Form for NM chart options
    """

    plotCharts = forms.BooleanField(label=_("Plot charts?"), required=False)

    plotNMStudyFreq = forms.BooleanField(label=_("Study frequency"), required=False)
    plotNMStudyPerDayAndHour = forms.BooleanField(
        label=_("Study workload"), required=False
    )
    plotNMInjectedDosePerStudy = forms.BooleanField(
        label=_("Injected dose per study"), required=False
    )
    plotNMInjectedDoseOverTime = forms.BooleanField(
        label=_("Injected dose over time"), required=False
    )
    plotNMInjectedDoseOverWeight = forms.BooleanField(
        label=_("Injected dose vs mass"), required=False
    )
    plotNMOverTimePeriod = forms.ChoiceField(
        label=_("Time period"), choices=CommonVariables.TIME_PERIOD, required=False
    )
    plotAverageChoice = forms.MultipleChoiceField(
        label=_("Average plots"),
        choices=CommonVariables.AVERAGES,
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={"class": "CheckboxSelectMultiple"}),
    )
    plotGrouping = forms.ChoiceField(
        label=mark_safe(_("Grouping choice")),  # nosec
        choices=CommonVariables.CHART_GROUPING,
        required=False,
    )
    plotSeriesPerSystem = forms.BooleanField(
        label=_("Plot a series per system"), required=False
    )
    plotHistograms = forms.BooleanField(
        label=_("Calculate histogram data"), required=False
    )
    plotNMInitialSortingChoice = forms.ChoiceField(
        label=_("Chart sorting"),
        choices=CommonVariables.SORTING_CHOICES,
        required=False,
    )
    plotInitialSortingDirection = forms.ChoiceField(
        label=_("Sorting direction"),
        choices=CommonVariables.SORTING_DIRECTION,
        required=False,
    )


class NMChartOptionsDisplayForm(forms.Form):
    """
    Form for NM chart display options
    """

    plotNMStudyFreq = forms.BooleanField(label=_("Study frequency"), required=False)
    plotNMStudyPerDayAndHour = forms.BooleanField(
        label=_("Study workload"), required=False
    )
    plotNMInjectedDosePerStudy = forms.BooleanField(
        label=_("Injected Dose per Study"), required=False
    )
    plotNMInjectedDoseOverTime = forms.BooleanField(
        label=_("Injected Dose over Time"), required=False
    )
    plotNMInjectedDoseOverWeight = forms.BooleanField(
        label=_("Injected Dose over Weight"), required=False
    )
    plotNMOverTimePeriod = forms.ChoiceField(
        label=_("Time period"), choices=CommonVariables.TIME_PERIOD, required=False
    )
    plotNMInitialSortingChoice = forms.ChoiceField(
        label=_("Chart sorting"),
        choices=CommonVariables.SORTING_CHOICES,
        required=False,
    )


class RFChartOptionsForm(forms.Form):
    """Form for RF chart options"""

    plotCharts = forms.BooleanField(label="Plot charts?", required=False)
    plotCharts.group = "PlotCharts"
    plotRFOverTimePeriod = forms.ChoiceField(
        label="Time period", choices=CommonVariables.TIME_PERIOD, required=False
    )
    plotRFOverTimePeriod.group = "General"
    plotAverageChoice = forms.MultipleChoiceField(
        label=_("Average plots"),
        choices=CommonVariables.AVERAGES,
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={"class": "CheckboxSelectMultiple"}),
    )
    plotAverageChoice.group = "General"
    plotRFSplitByPhysician = forms.BooleanField(
        label="Split plots by physician", required=False
    )
    plotRFSplitByPhysician.group = "General"
    plotGrouping = forms.ChoiceField(
        label=mark_safe(_("Grouping choice")),  # nosec
        choices=CommonVariables.CHART_GROUPING_RF,
        required=False,
    )
    plotGrouping.group = "General"
    plotSeriesPerSystem = forms.BooleanField(
        label=_("Plot a series per system"), required=False
    )
    plotSeriesPerSystem.group = "General"
    plotHistograms = forms.BooleanField(
        label=_("Calculate histogram data"), required=False
    )
    plotHistograms.group = "General"
    plotRFInitialSortingChoice = forms.ChoiceField(
        label="Chart sorting", choices=CommonVariables.SORTING_CHOICES, required=False
    )
    plotRFInitialSortingChoice.group = "General"
    plotInitialSortingDirection = forms.ChoiceField(
        label=_("Sorting direction"),
        choices=CommonVariables.SORTING_DIRECTION,
        required=False,
    )
    plotInitialSortingDirection.group = "General"

    plotRFStudyFreq = forms.BooleanField(label="Study frequency", required=False)
    plotRFStudyFreq.group = "Study description"
    plotRFStudyDAP = forms.BooleanField(label="Study DAP", required=False)
    plotRFStudyDAP.group = "Study description"
    plotRFStudyDAPOverTime = forms.BooleanField(
        label=_("Study DAP over time"), required=False
    )
    plotRFStudyDAPOverTime.group = "Study description"
    plotRFStudyPerDayAndHour = forms.BooleanField(
        label=_("Study workload"), required=False
    )
    plotRFStudyPerDayAndHour.group = "Study description"

    plotRFRequestFreq = forms.BooleanField(
        label=_("Requested procedure frequency"), required=False
    )
    plotRFRequestFreq.group = "Requested procedure"
    plotRFRequestDAP = forms.BooleanField(
        label=_("Requested procedure DAP"), required=False
    )
    plotRFRequestDAP.group = "Requested procedure"
    plotRFRequestDAPOverTime = forms.BooleanField(
        label=_("Requested procedure DAP over time"), required=False
    )
    plotRFRequestDAPOverTime.group = "Requested procedure"


class RFChartOptionsFormIncStandard(RFChartOptionsForm):
    plotRFStandardStudyFreq = forms.BooleanField(
        label="Standard study name frequency", required=False
    )
    plotRFStandardStudyFreq.group = "Standard study name"
    plotRFStandardStudyDAP = forms.BooleanField(
        label="Standard study name DAP", required=False
    )
    plotRFStandardStudyDAP.group = "Standard study name"
    plotRFStandardStudyDAPOverTime = forms.BooleanField(
        label="Standard study name DAP over time", required=False
    )
    plotRFStandardStudyDAPOverTime.group = "Standard study name"
    plotRFStandardStudyPerDayAndHour = forms.BooleanField(
        label="Standard study name workload", required=False
    )
    plotRFStandardStudyPerDayAndHour.group = "Standard study name"


class RFChartOptionsDisplayForm(forms.Form):
    """Form for RF chart display options"""

    plotRFStudyFreq = forms.BooleanField(label=_("Study frequency"), required=False)
    plotRFStudyDAP = forms.BooleanField(label=_("Study DAP"), required=False)
    plotRFStudyDAPOverTime = forms.BooleanField(
        label=_("Study DAP over time"), required=False
    )
    plotRFStudyPerDayAndHour = forms.BooleanField(
        label=_("Study workload"), required=False
    )
    plotRFRequestFreq = forms.BooleanField(
        label=_("Requested procedure frequency"), required=False
    )
    plotRFRequestDAP = forms.BooleanField(
        label=_("Requested procedure DAP"), required=False
    )
    plotRFRequestDAPOverTime = forms.BooleanField(
        label=_("Requested procedure DAP over time"), required=False
    )
    plotRFOverTimePeriod = forms.ChoiceField(
        label=_("Time period"), choices=CommonVariables.TIME_PERIOD, required=False
    )
    plotRFSplitByPhysician = forms.BooleanField(
        label=_("Split plots by physician"), required=False
    )
    plotRFInitialSortingChoice = forms.ChoiceField(
        label=_("Default chart sorting"),
        choices=CommonVariables.SORTING_CHOICES,
        required=False,
    )


class RFChartOptionsDisplayFormIncStandard(RFChartOptionsDisplayForm):
    plotRFStandardStudyFreq = forms.BooleanField(
        label="Standard study name frequency", required=False
    )
    plotRFStandardStudyDAP = forms.BooleanField(
        label="Standard study name DAP", required=False
    )
    plotRFStandardStudyDAPOverTime = forms.BooleanField(
        label="Standard study name DAP over time", required=False
    )
    plotRFStandardStudyPerDayAndHour = forms.BooleanField(
        label="Standard study name workload", required=False
    )

    field_order = [
        "plotRFStudyFreq",
        "plotRFStudyDAP",
        "plotRFStudyDAPOverTime",
        "plotRFStudyPerDayAndHour",
        "plotRFRequestFreq",
        "plotRFRequestDAP",
        "plotRFRequestDAPOverTime",
        "plotRFStandardStudyFreq",
        "plotRFStandardStudyDAP",
        "plotRFStandardStudyDAPOverTime",
        "plotRFStandardStudyPerDayAndHour",
        "plotRFOverTimePeriod",
        "plotRFSplitByPhysician",
        "plotRFInitialSortingChoice",
    ]


class MGChartOptionsForm(forms.Form):
    """Form for MG chart options"""

    plotCharts = forms.BooleanField(label="Plot charts?", required=False)
    plotCharts.group = "PlotCharts"
    plotMGOverTimePeriod = forms.ChoiceField(
        label=_("Time period"), choices=CommonVariables.TIME_PERIOD, required=False
    )
    plotMGOverTimePeriod.group = "General"
    plotAverageChoice = forms.MultipleChoiceField(
        label=_("Average plots"),
        choices=CommonVariables.AVERAGES,
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={"class": "CheckboxSelectMultiple"}),
    )
    plotAverageChoice.group = "General"
    plotGrouping = forms.ChoiceField(
        label=mark_safe(_("Grouping choice")),  # nosec
        choices=CommonVariables.CHART_GROUPING,
        required=False,
    )
    plotGrouping.group = "General"
    plotSeriesPerSystem = forms.BooleanField(
        label=_("Plot a series per system"), required=False
    )
    plotSeriesPerSystem.group = "General"
    plotHistograms = forms.BooleanField(
        label=_("Calculate histogram data"), required=False
    )
    plotHistograms.group = "General"
    plotMGInitialSortingChoice = forms.ChoiceField(
        label=_("Chart sorting"),
        choices=CommonVariables.SORTING_CHOICES,
        required=False,
    )
    plotMGInitialSortingChoice.group = "General"
    plotInitialSortingDirection = forms.ChoiceField(
        label=_("Sorting direction"),
        choices=CommonVariables.SORTING_DIRECTION,
        required=False,
    )
    plotInitialSortingDirection.group = "General"
    plotMGacquisitionFreq = forms.BooleanField(
        label="Acquisition frequency", required=False
    )
    plotMGacquisitionFreq.group = "Acquisition protocol"
    plotMGaverageAGD = forms.BooleanField(
        label="Acquisition average AGD", required=False
    )
    plotMGaverageAGD.group = "Acquisition protocol"
    plotMGaverageAGDvsThickness = forms.BooleanField(
        label="Acquisition average AGD vs. compressed thickness", required=False
    )
    plotMGaverageAGDvsThickness.group = "Acquisition protocol"
    plotMGAcquisitionAGDOverTime = forms.BooleanField(
        label="Acquisition AGD over time", required=False
    )
    plotMGAcquisitionAGDOverTime.group = "Acquisition protocol"
    plotMGAGDvsThickness = forms.BooleanField(
        label="Acquisition AGD vs. compressed thickness", required=False
    )
    plotMGAGDvsThickness.group = "Acquisition protocol"
    plotMGmAsvsThickness = forms.BooleanField(
        label="Acquisition mAs vs. compressed thickness", required=False
    )
    plotMGmAsvsThickness.group = "Acquisition protocol"
    plotMGkVpvsThickness = forms.BooleanField(
        label="Acquisition kVp vs. compressed thickness", required=False
    )
    plotMGkVpvsThickness.group = "Acquisition protocol"
    plotMGStudyPerDayAndHour = forms.BooleanField(
        label="Study workload", required=False
    )
    plotMGStudyPerDayAndHour.group = "Study description"


class MGChartOptionsFormIncStandard(MGChartOptionsForm):
    plotMGStandardAcquisitionFreq = forms.BooleanField(
        label="Standard acquisition name frequency", required=False
    )
    plotMGStandardAcquisitionFreq.group = "Standard acquisition name"
    plotMGStandardAverageAGD = forms.BooleanField(
        label="Standard acquisition name average AGD", required=False
    )
    plotMGStandardAverageAGD.group = "Standard acquisition name"
    plotMGStandardAverageAGDvsThickness = forms.BooleanField(
        label="Standard acquisition name average AGD vs. compressed thickness",
        required=False,
    )
    plotMGStandardAverageAGDvsThickness.group = "Standard acquisition name"
    plotMGStandardAcquisitionAGDOverTime = forms.BooleanField(
        label="Standard acquisition name AGD over time", required=False
    )
    plotMGStandardAcquisitionAGDOverTime.group = "Standard acquisition name"
    plotMGStandardAGDvsThickness = forms.BooleanField(
        label="Standard acquisition name AGD vs. compressed thickness", required=False
    )
    plotMGStandardAGDvsThickness.group = "Standard acquisition name"
    plotMGStandardmAsvsThickness = forms.BooleanField(
        label="Standard acquisition name mAs vs. compressed thickness", required=False
    )
    plotMGStandardmAsvsThickness.group = "Standard acquisition name"
    plotMGStandardkVpvsThickness = forms.BooleanField(
        label="Standard acquisition name kVp vs. compressed thickness", required=False
    )
    plotMGStandardkVpvsThickness.group = "Standard acquisition name"
    plotMGStandardStudyPerDayAndHour = forms.BooleanField(
        label="Standard study name workload", required=False
    )
    plotMGStandardStudyPerDayAndHour.group = "Standard study name"


class MGChartOptionsDisplayForm(forms.Form):
    """Form for MG chart display options"""

    plotMGacquisitionFreq = forms.BooleanField(
        label=_("Acquisition frequency"), required=False
    )
    plotMGaverageAGD = forms.BooleanField(
        label=_("Acquisition average AGD"), required=False
    )
    plotMGaverageAGDvsThickness = forms.BooleanField(
        label=_("Acquisition average AGD vs. compressed thickness"), required=False
    )
    plotMGAcquisitionAGDOverTime = forms.BooleanField(
        label=_("Acquisition AGD over time"), required=False
    )
    plotMGAGDvsThickness = forms.BooleanField(
        label=_("Acquisition AGD vs. compressed thickness"), required=False
    )
    plotMGmAsvsThickness = forms.BooleanField(
        label=_("Acquisition mAs vs. compressed thickness"), required=False
    )
    plotMGkVpvsThickness = forms.BooleanField(
        label=_("Acquisition kVp vs. compressed thickness"), required=False
    )
    plotMGStudyPerDayAndHour = forms.BooleanField(
        label=_("Study workload"), required=False
    )
    plotMGOverTimePeriod = forms.ChoiceField(
        label=_("Time period"), choices=CommonVariables.TIME_PERIOD, required=False
    )
    plotMGInitialSortingChoice = forms.ChoiceField(
        label=_("Chart sorting"),
        choices=CommonVariables.SORTING_CHOICES,
        required=False,
    )


class MGChartOptionsDisplayFormIncStandard(MGChartOptionsDisplayForm):
    plotMGStandardAcquisitionFreq = forms.BooleanField(
        label="Standard acquisition name frequency", required=False
    )
    plotMGStandardAverageAGD = forms.BooleanField(
        label="Standard acquisition name average AGD", required=False
    )
    plotMGStandardAverageAGDvsThickness = forms.BooleanField(
        label="Standard acquisition name average AGD vs. compressed thickness",
        required=False,
    )
    plotMGStandardAcquisitionAGDOverTime = forms.BooleanField(
        label="Standard acquisition name AGD over time", required=False
    )
    plotMGStandardAGDvsThickness = forms.BooleanField(
        label="Standard acquisition name AGD vs. compressed thickness", required=False
    )
    plotMGStandardmAsvsThickness = forms.BooleanField(
        label="Standard acquisition name mAs vs. compressed thickness", required=False
    )
    plotMGStandardkVpvsThickness = forms.BooleanField(
        label="Standard acquisition name kVp vs. compressed thickness", required=False
    )
    plotMGStandardStudyPerDayAndHour = forms.BooleanField(
        label="Standard study name", required=False
    )

    field_order = [
        "plotMGacquisitionFreq",
        "plotMGaverageAGD",
        "plotMGaverageAGDvsThickness",
        "plotMGAcquisitionAGDOverTime",
        "plotMGAGDvsThickness",
        "plotMGmAsvsThickness",
        "plotMGkVpvsThickness",
        "plotMGStandardAcquisitionFreq",
        "plotMGStandardAverageAGD",
        "plotMGStandardAverageAGDvsThickness",
        "plotMGStandardAcquisitionAGDOverTime",
        "plotMGStandardAGDvsThickness",
        "plotMGStandardmAsvsThickness",
        "plotMGStandardkVpvsThickness",
        "plotMGStudyPerDayAndHour",
        "plotMGStandardStudyPerDayAndHour",
        "plotMGOverTimePeriod",
        "plotMGInitialSortingChoice",
    ]


class DXChartOptionsDisplayForm(forms.Form):
    """Form for DX chart display options"""

    plotDXAcquisitionFreq = forms.BooleanField(
        label=_("Acquisition frequency"), required=False
    )
    plotDXAcquisitionMeanDAP = forms.BooleanField(
        label=_("Acquisition DAP"), required=False
    )
    plotDXAcquisitionMeanmAs = forms.BooleanField(
        label=_("Acquisition mAs"), required=False
    )
    plotDXAcquisitionMeankVp = forms.BooleanField(
        label=_("Acquisition kVp"), required=False
    )
    plotDXAcquisitionMeanDAPOverTime = forms.BooleanField(
        label=_("Acquisition DAP over time"), required=False
    )
    plotDXAcquisitionMeanmAsOverTime = forms.BooleanField(
        label=_("Acquisition mAs over time"), required=False
    )
    plotDXAcquisitionMeankVpOverTime = forms.BooleanField(
        label=_("Acquisition kVp over time"), required=False
    )
    plotDXAcquisitionDAPvsMass = forms.BooleanField(
        label=_("Acquisition DAP vs mass"), required=False
    )
    plotDXStudyFreq = forms.BooleanField(label=_("Study frequency"), required=False)
    plotDXStudyMeanDAP = forms.BooleanField(label=_("Study DAP"), required=False)
    plotDXStudyDAPvsMass = forms.BooleanField(
        label=_("Study DAP vs mass"), required=False
    )
    plotDXStudyPerDayAndHour = forms.BooleanField(
        label=_("Study workload"), required=False
    )
    plotDXRequestFreq = forms.BooleanField(
        label=_("Requested procedure frequency"), required=False
    )
    plotDXRequestMeanDAP = forms.BooleanField(
        label=_("Requested procedure DAP"), required=False
    )
    plotDXRequestDAPvsMass = forms.BooleanField(
        label=_("Requested procedure DAP vs mass"), required=False
    )
    plotDXAcquisitionMeanDAPOverTimePeriod = forms.ChoiceField(
        label=_("Time period"), choices=CommonVariables.TIME_PERIOD, required=False
    )
    plotDXInitialSortingChoice = forms.ChoiceField(
        label=_("Default chart sorting"),
        choices=CommonVariables.SORTING_CHOICES,
        required=False,
    )


class DXChartOptionsDisplayFormIncStandard(DXChartOptionsDisplayForm):
    plotDXStandardAcquisitionFreq = forms.BooleanField(
        label="Standard acquisition name frequency", required=False
    )
    plotDXStandardAcquisitionMeanDAP = forms.BooleanField(
        label="Standard acquisition name DAP", required=False
    )
    plotDXStandardAcquisitionMeanmAs = forms.BooleanField(
        label="Standard acquisition name mAs", required=False
    )
    plotDXStandardAcquisitionMeankVp = forms.BooleanField(
        label="Standard acquisition name kVp", required=False
    )
    plotDXStandardAcquisitionMeanDAPOverTime = forms.BooleanField(
        label="Standard acquisition name DAP over time", required=False
    )
    plotDXStandardAcquisitionMeanmAsOverTime = forms.BooleanField(
        label="Standard acquisition name mAs over time", required=False
    )
    plotDXStandardAcquisitionMeankVpOverTime = forms.BooleanField(
        label="Standard acquisition name kVp over time", required=False
    )
    plotDXStandardAcquisitionDAPvsMass = forms.BooleanField(
        label="Standard acquisition name DAP vs mass", required=False
    )
    plotDXStandardStudyFreq = forms.BooleanField(
        label="Standard study name frequency", required=False
    )
    plotDXStandardStudyMeanDAP = forms.BooleanField(
        label="Standard study name DAP", required=False
    )
    plotDXStandardStudyDAPvsMass = forms.BooleanField(
        label="Standard study name DAP vs mass", required=False
    )
    plotDXStandardStudyPerDayAndHour = forms.BooleanField(
        label="Standard study name workload", required=False
    )

    field_order = [
        "plotDXAcquisitionFreq",
        "plotDXAcquisitionMeanDAP",
        "plotDXAcquisitionMeanmAs",
        "plotDXAcquisitionMeankVp",
        "plotDXAcquisitionMeanDAPOverTime",
        "plotDXAcquisitionMeanmAsOverTime",
        "plotDXAcquisitionMeankVpOverTime",
        "plotDXAcquisitionDAPvsMass",
        "plotDXStandardAcquisitionFreq",
        "plotDXStandardAcquisitionMeanDAP",
        "plotDXStandardAcquisitionMeanmAs",
        "plotDXStandardAcquisitionMeankVp",
        "plotDXStandardAcquisitionMeanDAPOverTime",
        "plotDXStandardAcquisitionMeanmAsOverTime",
        "plotDXStandardAcquisitionMeankVpOverTime",
        "plotDXStandardAcquisitionDAPvsMass",
        "plotDXStudyFreq",
        "plotDXStudyMeanDAP",
        "plotDXStudyDAPvsMass",
        "plotDXStudyPerDayAndHour",
        "plotDXRequestFreq",
        "plotDXRequestMeanDAP",
        "plotDXRequestDAPvsMass",
        "plotDXStandardStudyFreq",
        "plotDXStandardStudyMeanDAP",
        "plotDXStandardStudyDAPvsMass",
        "plotDXStandardStudyPerDayAndHour",
        "plotDXAcquisitionMeanDAPOverTimePeriod",
        "plotDXInitialSortingChoice",
    ]


class CTChartOptionsDisplayForm(forms.Form):
    """Form for CT chart display options"""

    plotCTAcquisitionFreq = forms.BooleanField(
        label=_("Acquisition frequency"), required=False
    )
    plotCTAcquisitionMeanDLP = forms.BooleanField(
        label=_("Acquisition DLP"), required=False
    )
    plotCTAcquisitionMeanCTDI = forms.BooleanField(
        label=mark_safe(_("Acquisition CTDI<sub>vol</sub>")), required=False  # nosec
    )
    plotCTAcquisitionDLPOverTime = forms.BooleanField(
        label=_("Acquisition DLP over time"), required=False
    )
    plotCTAcquisitionCTDIOverTime = forms.BooleanField(
        label=mark_safe(_("Acquisition CTDI<sub>vol</sub> over time")),  # nosec
        required=False,
    )
    plotCTAcquisitionDLPvsMass = forms.BooleanField(
        label=_("Acquisition DLP vs mass"), required=False
    )
    plotCTAcquisitionCTDIvsMass = forms.BooleanField(
        label=_("Acquisition CTDI vs mass"), required=False
    )
    plotCTAcquisitionTypes = forms.MultipleChoiceField(
        label=mark_safe(  # nosec
            _(
                "Acquisition types to include<br/>in acquisition-level chart<br/>calculations"
            )
        ),
        choices=CommonVariables.CT_ACQUISITION_TYPES,
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={"class": "CheckboxSelectMultiple"}),
    )
    plotCTStudyFreq = forms.BooleanField(label=_("Study frequency"), required=False)
    plotCTStudyMeanDLP = forms.BooleanField(label=_("Study DLP"), required=False)
    plotCTStudyMeanCTDI = forms.BooleanField(
        label=mark_safe(_("Study CTDI<sub>vol</sub>")), required=False  # nosec
    )
    plotCTStudyNumEvents = forms.BooleanField(label=_("Study events"), required=False)
    plotCTStudyMeanDLPOverTime = forms.BooleanField(
        label=_("Study DLP over time"), required=False
    )
    plotCTStudyPerDayAndHour = forms.BooleanField(
        label=_("Study workload"), required=False
    )
    plotCTRequestFreq = forms.BooleanField(
        label=_("Requested procedure frequency"), required=False
    )
    plotCTRequestMeanDLP = forms.BooleanField(
        label=_("Requested procedure DLP"), required=False
    )
    plotCTRequestNumEvents = forms.BooleanField(
        label=_("Requested procedure events"), required=False
    )
    plotCTRequestDLPOverTime = forms.BooleanField(
        label=_("Requested procedure DLP over time"), required=False
    )
    plotCTOverTimePeriod = forms.ChoiceField(
        label=_("Time period"), choices=CommonVariables.TIME_PERIOD, required=False
    )
    plotCTInitialSortingChoice = forms.ChoiceField(
        label=_("Chart sorting"),
        choices=CommonVariables.SORTING_CHOICES,
        required=False,
    )


class CTChartOptionsDisplayFormIncStandard(CTChartOptionsDisplayForm):
    plotCTStandardAcquisitionFreq = forms.BooleanField(
        label="Standard acquisition name frequency", required=False
    )
    plotCTStandardAcquisitionMeanDLP = forms.BooleanField(
        label="Standard acquisition DLP", required=False
    )
    plotCTStandardAcquisitionMeanCTDI = forms.BooleanField(
        label=mark_safe("Standard acquisition CTDI<sub>vol</sub>"),  # nosec
        required=False,
    )
    plotCTStandardAcquisitionDLPOverTime = forms.BooleanField(
        label="Standard acquisition name DLP over time", required=False
    )
    plotCTStandardAcquisitionCTDIOverTime = forms.BooleanField(
        label=mark_safe(  # nosec
            "Standard acquisition name CTDI<sub>vol</sub> over time"
        ),
        required=False,
    )
    plotCTStandardAcquisitionCTDIvsMass = forms.BooleanField(
        label=mark_safe(  # nosec
            "Standard acquisition name CTDI<sub>vol</sub> vs mass"
        ),
        required=False,
    )
    plotCTStandardAcquisitionDLPvsMass = forms.BooleanField(
        label="Standard acquisition name DLP vs mass", required=False
    )

    plotCTStandardStudyMeanDLP = forms.BooleanField(
        label="Standard study DLP", required=False
    )
    plotCTStandardStudyNumEvents = forms.BooleanField(
        label="Standard study events", required=False
    )
    plotCTStandardStudyFreq = forms.BooleanField(
        label="Standard study frequency", required=False
    )
    plotCTStandardStudyMeanDLPOverTime = forms.BooleanField(
        label="Standard study DLP over time", required=False
    )
    plotCTStandardStudyPerDayAndHour = forms.BooleanField(
        label="Standard study workload", required=False
    )

    field_order = [
        "plotCTAcquisitionFreq",
        "plotCTAcquisitionMeanDLP",
        "plotCTAcquisitionMeanCTDI",
        "plotCTAcquisitionDLPOverTime",
        "plotCTAcquisitionCTDIOverTime",
        "plotCTAcquisitionDLPvsMass",
        "plotCTAcquisitionCTDIvsMass",
        "plotCTAcquisitionTypes",
        "plotCTStandardAcquisitionFreq",
        "plotCTStandardAcquisitionMeanDLP",
        "plotCTStandardAcquisitionMeanCTDI",
        "plotCTStandardAcquisitionDLPOverTime",
        "plotCTStandardAcquisitionCTDIOverTime",
        "plotCTStandardAcquisitionDLPvsMass",
        "plotCTStandardAcquisitionCTDIvsMass",
        "plotCTStudyFreq",
        "plotCTStudyMeanDLP",
        "plotCTStudyMeanCTDI",
        "plotCTStudyNumEvents",
        "plotCTStudyMeanDLPOverTime",
        "plotCTStudyPerDayAndHour",
        "plotCTRequestFreq",
        "plotCTRequestMeanDLP",
        "plotCTRequestNumEvents",
        "plotCTRequestDLPOverTime",
        "plotCTStandardStudyFreq",
        "plotCTStandardStudyMeanDLP",
        "plotCTStandardStudyNumEvents",
        "plotCTStandardStudyMeanDLPOverTime",
        "plotCTStandardStudyPerDayAndHour",
        "plotCTOverTimePeriod",
        "plotCTInitialSortingChoice",
    ]


class GeneralChartOptionsDisplayForm(forms.Form):
    """Form for general chart display options"""

    plotCharts = forms.BooleanField(label=_("Plot charts?"), required=False)
    plotAverageChoice = forms.MultipleChoiceField(
        label=_("Average plots"),
        choices=CommonVariables.AVERAGES,
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={"class": "CheckboxSelectMultiple"}),
    )
    plotInitialSortingDirection = forms.ChoiceField(
        label=_("Sorting direction"),
        choices=CommonVariables.SORTING_DIRECTION,
        required=False,
    )
    plotSeriesPerSystem = forms.BooleanField(
        label=_("Plot a series per system"), required=False
    )
    plotHistograms = forms.BooleanField(
        label=_("Calculate histogram data"), required=False
    )
    plotHistogramBins = forms.IntegerField(
        label=_("Number of histogram bins"), min_value=2, max_value=40, required=False
    )
    plotHistogramGlobalBins = forms.BooleanField(
        label=_("Fixed histogram bins across subplots"), required=False
    )
    plotCaseInsensitiveCategories = forms.BooleanField(
        label=_("Case-insensitive categories"), required=False
    )
    plotRemoveCategoryWhitespacePadding = forms.BooleanField(
        label=_("Remove category whitespace padding"), required=False
    )
    plotLabelCharWrap = forms.IntegerField(
        label=_("Chart label character wrap length"),
        min_value=10,
        max_value=500,
        required=False,
    )
    plotGrouping = forms.ChoiceField(
        label=_("Chart grouping"),
        choices=CommonVariables.CHART_GROUPING,
        required=False,
    )
    plotThemeChoice = forms.ChoiceField(
        label=_("Chart theme"), choices=CommonVariables.CHART_THEMES, required=False
    )
    plotColourMapChoice = forms.ChoiceField(
        label=_("Colour map choice"),
        choices=CommonVariables.CHART_COLOUR_MAPS,
        required=False,
        widget=forms.RadioSelect(attrs={"id": "value"}),
    )
    plotFacetColWrapVal = forms.IntegerField(
        label=_("Number of sub-charts per row"),
        min_value=1,
        max_value=10,
        required=False,
    )


class UpdateDisplayNamesForm(forms.Form):
    display_names = forms.CharField()


class RFHighDoseFluoroAlertsForm(forms.ModelForm):
    """Form for displaying and changing fluoroscopy high dose alert settings"""

    def __init__(self, *args, **kwargs):
        from crispy_forms.layout import Button

        super(RFHighDoseFluoroAlertsForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_class = "form-horizontal"

        # If HighDoseMetricAlertSettings.changed_accum_dose_delta_weeks is True then the summed DAP and dose at RP
        # values have not yet been recalculated - display the recalculate button on the form.
        if self.instance.changed_accum_dose_delta_weeks:
            self.helper.add_input(
                Button(
                    "recalc_all_summed_data",
                    "Recalculate all summed data",
                    css_class="btn btn-warning",
                )
            )

        # If there is nothing in self.data and accum_dose_delta_weeks is in self.changed_data then the user must have
        # changed the accum_dose_delta_weeks value: set the changed_accum_dose_delta_weeks flag to True. This updates
        # the HighDoseMetricAlertSettings.changed_accum_dose_delta_weeks to True.
        if len(self.data):
            if self.has_changed():
                if "accum_dose_delta_weeks" in self.changed_data:
                    self.instance.changed_accum_dose_delta_weeks = True
                    self.save()
        self.helper.layout = Layout(
            Div(
                "alert_total_dap_rf",
                "alert_total_rp_dose_rf",
                "alert_skindose",
                "accum_dose_delta_weeks",
                "show_accum_dose_over_delta_weeks",
                "calc_accum_dose_over_delta_weeks_on_import",
                "send_high_dose_metric_alert_emails_ref",
                "send_high_dose_metric_alert_emails_skin",
            ),
            FormActions(Submit("submit", "Submit")),
        )

    class Meta(object):
        model = HighDoseMetricAlertSettings
        fields = [
            "alert_total_dap_rf",
            "alert_total_rp_dose_rf",
            "alert_skindose",
            "accum_dose_delta_weeks",
            "show_accum_dose_over_delta_weeks",
            "calc_accum_dose_over_delta_weeks_on_import",
            "send_high_dose_metric_alert_emails_ref",
            "send_high_dose_metric_alert_emails_skin",
        ]


class HomepageOptionsForm(forms.Form):
    """Form for displaying and changing the home page options"""

    dayDeltaA = forms.IntegerField(
        label=_("Primary time period to sum studies (days)"), required=False
    )
    dayDeltaB = forms.IntegerField(
        label=_("Secondary time period to sum studies (days)"), required=False
    )
    enable_workload_stats = forms.BooleanField(
        label=_("Enable calculation and display of workload stats on home page?"),
        required=False,
    )


class MergeOnDeviceObserverUIDForm(forms.Form):
    """Form for displaying and changing the option for merging on Device Observer UID"""

    match_on_device_observer_uid = forms.BooleanField(
        label=_(
            "Set Display Name and Modality type if Device Observer UID is matching"
        ),
        required=False,
    )


class DicomQueryForm(forms.Form):
    """Form for launching DICOM Query"""

    from datetime import date

    MODALITIES = (
        ("CT", _("CT")),
        ("FL", _("Fluoroscopy (XA and RF)")),
        ("DX", _("DX, including CR")),
        ("MG", _("Mammography")),
        ("NM", _("Nuclear Medicine")),
    )

    remote_host_field = forms.ChoiceField(
        choices=[], widget=forms.Select(attrs={"class": "form-control"})
    )
    store_scp_field = forms.ChoiceField(
        choices=[], widget=forms.Select(attrs={"class": "form-control"})
    )
    date_from_field = forms.DateField(
        label=_("Date from"),
        widget=forms.DateInput(attrs={"class": "form-control datepicker"}),
        required=False,
        initial=date.today().isoformat(),
        help_text=_("Format yyyy-mm-dd, restrict as much as possible for best results"),
    )
    date_until_field = forms.DateField(
        label=_("Date until"),
        widget=forms.DateInput(attrs={"class": "form-control datepicker"}),
        required=False,
        help_text=_("Format yyyy-mm-dd, restrict as much as possible for best results"),
    )
    modality_field = forms.MultipleChoiceField(
        choices=MODALITIES,
        widget=forms.CheckboxSelectMultiple(attrs={"checked": ""}),
        required=False,
        help_text=(
            _(
                "At least one modality must be ticked - if SR only is ticked (Advanced) these "
                "modalities will be ignored"
            )
        ),
    )
    inc_sr_field = forms.BooleanField(
        label=_("Include SR only studies?"),
        required=False,
        initial=False,
        help_text=_(
            "Only use with stores containing only RDSRs, with no accompanying images"
        ),
    )
    duplicates_field = forms.BooleanField(
        label=_("Ignore studies already in the database?"),
        required=False,
        initial=True,
        help_text=_(
            "Objects that have already been processed won't be imported, "
            "so there isn't any point getting them!"
        ),
    )
    desc_exclude_field = forms.CharField(
        required=False,
        label=_("Exclude studies with these terms in the study description:"),
        help_text=_("Comma separated list of terms"),
    )
    desc_include_field = forms.CharField(
        required=False,
        label=_("Only keep studies with these terms in the study description:"),
        help_text=_("Comma separated list of terms"),
    )
    stationname_exclude_field = forms.CharField(
        required=False,
        label=_("Exclude studies or series with these terms in the station name:"),
        help_text=_(
            "Comma separated list of terms, tested at series level — see Advanced"
        ),
    )
    stationname_include_field = forms.CharField(
        required=False,
        label=_("Only keep studies or series with these terms in the station name:"),
        help_text=_(
            "Comma separated list of terms, tested at series level — see Advanced"
        ),
    )
    get_toshiba_images_field = forms.BooleanField(
        label=_("Attempt to get Toshiba dose images"),
        required=False,
        help_text=_(
            "Only applicable if using Toshiba RDSR generator extension, see docs"
        ),
    )
    get_empty_sr_field = forms.BooleanField(
        label=_("Get SR series that return nothing at image level query"),
        help_text=_("Only use if suggested in qrscu log, see docs"),
        required=False,
    )
    stationname_study_level_field = forms.BooleanField(
        label=_("Check station name include/exclude at study level"),
        help_text=_("Default from v1.0 is to check at series level only"),
        required=False,
    )

    def __init__(self, *args, **kwargs):
        super(DicomQueryForm, self).__init__(*args, **kwargs)

        self.fields["remote_host_field"].choices = [
            (x.pk, x.name) for x in DicomRemoteQR.objects.all()
        ]
        self.fields["store_scp_field"].choices = [
            (x.pk, x.name) for x in DicomStoreSCP.objects.all()
        ]
        self.helper = FormHelper(self)
        self.helper.form_id = "post-form"
        self.helper.form_method = "post"
        self.helper.form_action = "queryprocess"
        self.helper.add_input(Submit("submit", "Submit"))
        self.helper.layout = Layout(
            Div(
                Div(
                    Div("remote_host_field", css_class="col-md-6"),
                    Div("store_scp_field", css_class="col-md-6"),
                ),
                InlineCheckboxes("modality_field"),
                Div(
                    Div("date_from_field", css_class="col-md-6"),
                    Div("date_until_field", css_class="col-md-6"),
                ),
                "desc_exclude_field",
                "desc_include_field",
                "stationname_exclude_field",
                "stationname_include_field",
                Accordion(
                    AccordionGroup(
                        "Advanced",
                        "get_toshiba_images_field",
                        "duplicates_field",
                        "inc_sr_field",
                        "get_empty_sr_field",
                        "stationname_study_level_field",
                        active=False,
                    )
                ),
            )
        )

    def clean(self):
        """
        Validate the form data to clear modality selections if sr_only is selected.
        :return: Form with modalities _or_ sr_only selected
        """
        qr_logger = logging.getLogger("remapp.netdicom.qrscu")

        cleaned_data = super(DicomQueryForm, self).clean()
        mods = cleaned_data.get("modality_field")
        inc_sr = cleaned_data.get("inc_sr_field")
        qr_logger.debug("Form mods are {0}, inc_sr is {1}".format(mods, inc_sr))
        qr_logger.debug("All form modes are {0}".format(cleaned_data))
        if inc_sr:
            self.cleaned_data["modality_field"] = None
        elif not mods:
            raise forms.ValidationError(
                "You must select at least one modality (or Advanced SR Only)"
            )
        return cleaned_data


class DicomDeleteSettingsForm(forms.ModelForm):
    """Form for configuring whether DICOM objects are stored or deleted once processed"""

    def __init__(self, *args, **kwargs):
        super(DicomDeleteSettingsForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_class = "form-horizontal"
        self.helper.label_class = "col-lg-2"
        self.helper.field_class = "col-lg-8"
        self.helper.layout = Layout(
            Div(
                HTML(
                    """
                     <h4>Do you want objects that we can't do anything with to be deleted?</h4>
                """
                ),
                "del_no_match",
                HTML(
                    """
                     <h4>The remaining choices are for DICOM objects we have processed and attempted to import to the
                     database:</h4>
                """
                ),
                "del_rdsr",
                "del_mg_im",
                "del_dx_im",
                "del_ct_phil",
                "del_nm_im",
            ),
            FormActions(Submit("submit", "Submit")),
            Div(
                HTML(
                    """
                <div class="col-lg-4 col-lg-offset-2">
                    <a href='"""
                    + reverse("dicom_summary")
                    + """#delete' role="button" class="btn btn-default">
                        Cancel and return to the DICOM configuration and DICOM object delete summary page
                    </a>
                </div>
                """
                )
            ),
        )

    class Meta(object):
        model = DicomDeleteSettings
        fields = [
            "del_no_match",
            "del_rdsr",
            "del_mg_im",
            "del_dx_im",
            "del_ct_phil",
            "del_nm_im",
        ]


class DicomQRForm(forms.ModelForm):
    """Form for configuring remote Query Retrieve nodes"""

    def __init__(self, *args, **kwargs):
        super(DicomQRForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_class = "form-horizontal"
        self.helper.label_class = "col-md-8"
        self.helper.field_class = "col-md-4"
        self.helper.layout = Layout(
            Div("name", "aetitle", "callingaet", "port", "ip", "hostname"),
            Accordion(
                AccordionGroup(
                    "Non-standard configuration options",
                    Div(
                        HTML(
                            """
                        <p>
                          Some PACS systems (like Impax 6.6) need modality at study level for correct filtering. 
                          Others will return no results if modality is included at study level. See
                            <a href="https://docs.openrem.org/en/{{ admin.docsversion }}/netdicom-qr-config.html"
                                target="_blank" data-toggle="tooltip"
                                title="DICOM query-retrieve node config documentation - opens in a new tab">
                                DICOM query-retrieve node config documentation
                            </a>
                        </p>
                        """
                        )
                    ),
                    PrependedText(
                        "use_modality_tag", ""
                    ),  # Trick to force label to join the other labels,
                    # otherwise sits to right
                    active=False,
                )
            ),
            FormActions(Submit("submit", "Submit")),
            Div(
                HTML(
                    """
                <div class="col-lg-4 col-lg-offset-4">
                    <a href='"""
                    + reverse("dicom_summary")
                    + """' role="button" class="btn btn-default">
                        Cancel and return to DICOM configuration summary page
                    </a>
                </div>
                """
                )
            ),
        )

    class Meta(object):
        model = DicomRemoteQR
        fields = [
            "name",
            "aetitle",
            "callingaet",
            "port",
            "ip",
            "hostname",
            "use_modality_tag",
        ]


class DicomStoreForm(forms.ModelForm):
    """Form for configuring local Store nodes"""

    def __init__(self, *args, **kwargs):
        super(DicomStoreForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_class = "form-horizontal"
        self.helper.label_class = "col-md-8"
        self.helper.field_class = "col-md-4"
        if not settings.DOCKER_INSTALL:
            self.helper.layout = Layout(
                Div("name", "aetitle", "peer", "port"),
                FormActions(Submit("submit", "Submit")),
                Div(
                    HTML(
                        """
                    <div class="col-lg-4 col-lg-offset-4">
                        <a href='"""
                        + reverse("dicom_summary")
                        + """' role="button" class="btn btn-default">
                            Cancel and return to DICOM configuration summary page
                        </a>
                    </div>
                    """
                    )
                ),
            )
        else:
            self.helper.layout = Layout(
                Div("name", "aetitle", "peer", "port"),
                FormActions(Submit("submit", "Submit")),
                Div(
                    HTML(
                        """
                    <div class="col-lg-4 col-lg-offset-4">
                        <a href='"""
                        + reverse("dicom_summary")
                        + """' role="button" class="btn btn-default">
                            Cancel and return to DICOM configuration summary page
                        </a>
                    </div>
                    """
                    )
                ),
            )

    class Meta(object):
        model = DicomStoreSCP
        fields = ["name", "aetitle", "peer", "port"]
        labels = {
            "peer": "Peer: Set this to localhost",
            "port": "Port: port 104 is standard for DICOM but ports higher than 1024 require fewer admin rights",
        }
        if settings.DOCKER_INSTALL:
            labels["peer"] = "Docker container name: initial default is orthanc_1"
            labels["port"] = (
                "Port: set to the same as the DICOM_PORT setting in docker-compose.yml"
            )


class StandardNameFormBase(forms.ModelForm):
    """For configuring standard names for study description, requested procedure, procedure and acquisition name."""

    class Meta(object):
        model = StandardNames
        fields = [
            "standard_name",
            "modality",
            "study_description",
            "requested_procedure_code_meaning",
            "procedure_code_meaning",
            "acquisition_protocol",
        ]
        widgets = {
            "standard_name": forms.TextInput,
            "modality": forms.HiddenInput,
        }

    def clean_study_description(self):
        if self.cleaned_data["study_description"] == "":
            return None
        else:
            return self.cleaned_data["study_description"]

    def clean_requested_procedure_code_meaning(self):
        if self.cleaned_data["requested_procedure_code_meaning"] == "":
            return None
        else:
            return self.cleaned_data["requested_procedure_code_meaning"]

    def clean_procedure_code_meaning(self):
        if self.cleaned_data["procedure_code_meaning"] == "":
            return None
        else:
            return self.cleaned_data["procedure_code_meaning"]

    def clean_acquisition_protocol(self):
        if self.cleaned_data["acquisition_protocol"] == "":
            return None
        else:
            return self.cleaned_data["acquisition_protocol"]


class StandardNameFormCT(StandardNameFormBase):
    """Form for configuring standard names for study description, requested procedure, procedure and acquisition name"""

    ctdi_limit = forms.DecimalField(
        required=False,
        max_digits=5,
        decimal_places=4,
        help_text="CTDIvol limit in mGy for this standard name",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    def __init__(self, *args, **kwargs):
        super(StandardNameFormCT, self).__init__(*args, **kwargs)
        self.fields['modality'].initial = "CT"

        # Get initial CTDI limit value if editing existing standard name
        if 'instance' in kwargs and kwargs['instance']:
            self.fields['ctdi_limit'].initial = kwargs['instance'].ctdi_limit

        all_studies = GeneralStudyModuleAttr.objects.filter(modality_type__iexact="CT")

        field_names = [
            ("study_description", "Study description"),
            ("requested_procedure_code_meaning", "Requested procedure name"),
            ("procedure_code_meaning", "Procedure name"),
        ]

        for field_name, label_name in field_names:
            # Exclude items already in the CT standard names entries except for the current value of the field
            items_to_exclude = (
                StandardNames.objects.all()
                .filter(modality="CT")
                .values(field_name)
                .exclude(**{field_name: None})
            )
            if "standard_name" in self.initial:
                items_to_exclude = items_to_exclude.exclude(
                    standard_name=self.initial["standard_name"]
                )

            query = (
                all_studies.values_list(field_name, flat=True)
                .exclude(**{field_name + "__in": items_to_exclude})
                .distinct()
                .order_by(field_name)
            )
            query_choices = [("", "None")] + [(item, item) for item in query]

            initial_choices = (
                StandardNames.objects.all()
                .filter(modality="CT")
                .exclude(**{field_name: None})
                .order_by(field_name)
            )
            if "standard_name" in self.initial:
                initial_choices = initial_choices.filter(
                    standard_name=self.initial["standard_name"]
                )

            self.initial[field_name] = list(
                initial_choices.values_list(field_name, flat=True)
            )

            self.fields[field_name] = forms.MultipleChoiceField(
                choices=query_choices,
                required=False,
                widget=FilteredSelectMultiple(
                    label_name.lower() + "s", is_stacked=False
                ),
            )

        field_name, label_name = ("acquisition_protocol", "Acquisition protocol name")
        items_to_exclude = (
            StandardNames.objects.all().values(field_name).exclude(**{field_name: None})
        )
        if "standard_name" in self.initial:
            items_to_exclude = items_to_exclude.exclude(
                standard_name=self.initial["standard_name"]
            )
        query = (
            CtIrradiationEventData.objects.values_list(field_name, flat=True)
            .exclude(**{field_name + "__in": items_to_exclude})
            .distinct()
            .order_by(field_name)
        )
        query_choices = [("", "None")] + [(item, item) for item in query]

        initial_choices = (
            StandardNames.objects.all()
            .filter(modality="CT")
            .exclude(**{field_name: None})
            .order_by(field_name)
        )
        if "standard_name" in self.initial:
            initial_choices = initial_choices.filter(
                standard_name=self.initial["standard_name"]
            )

        self.initial[field_name] = list(
            initial_choices.values_list(field_name, flat=True)
        )

        self.fields[field_name] = forms.MultipleChoiceField(
            choices=query_choices,
            required=False,
            widget=FilteredSelectMultiple(label_name.lower() + "s", is_stacked=False),
        )

        class Media:
            css = {
                "all": (
                    os.path.join(settings.BASE_DIR, "/static/admin/css/widgets.css"),
                ),
            }
            js = ("/admin/jsi18n",)


class StandardNameFormDX(StandardNameFormBase):
    """Form for configuring standard names for study description, requested procedure, procedure and acquisition name"""

    def __init__(self, *args, **kwargs):
        super(StandardNameFormDX, self).__init__(*args, **kwargs)
        self.fields["modality"].initial = "DX"

        all_studies = GeneralStudyModuleAttr.objects.filter(
            Q(modality_type__in=["DX", "CR", "PX"])
        )

        field_names = [
            ("study_description", "Study description"),
            ("requested_procedure_code_meaning", "Requested procedure name"),
            ("procedure_code_meaning", "Procedure name"),
        ]

        for field_name, label_name in field_names:
            # Exclude items already in the DX standard names entries except for the current value of the field
            items_to_exclude = (
                StandardNames.objects.all()
                .filter(modality="DX")
                .values(field_name)
                .exclude(**{field_name: None})
            )
            if "standard_name" in self.initial:
                items_to_exclude = items_to_exclude.exclude(
                    standard_name=self.initial["standard_name"]
                )

            query = (
                all_studies.values_list(field_name, flat=True)
                .exclude(**{field_name + "__in": items_to_exclude})
                .distinct()
                .order_by(field_name)
            )
            query_choices = [("", "None")] + [(item, item) for item in query]

            initial_choices = (
                StandardNames.objects.all()
                .filter(modality="DX")
                .exclude(**{field_name: None})
                .order_by(field_name)
            )
            if "standard_name" in self.initial:
                initial_choices = initial_choices.filter(
                    standard_name=self.initial["standard_name"]
                )

            self.initial[field_name] = list(
                initial_choices.values_list(field_name, flat=True)
            )

            self.fields[field_name] = forms.MultipleChoiceField(
                choices=query_choices,
                required=False,
                widget=FilteredSelectMultiple(
                    label_name.lower() + "s", is_stacked=False
                ),
            )

        q = ["DX", "CR", "PX"]
        q_criteria = reduce(
            operator.or_,
            (
                Q(
                    projection_xray_radiation_dose__general_study_module_attributes__modality_type__icontains=item
                )
                for item in q
            ),
        )
        field_name, label_name = ("acquisition_protocol", "Acquisition protocol name")
        items_to_exclude = (
            StandardNames.objects.all().values(field_name).exclude(**{field_name: None})
        )
        if "standard_name" in self.initial:
            items_to_exclude = items_to_exclude.exclude(
                standard_name=self.initial["standard_name"]
            )
        query = (
            IrradEventXRayData.objects.filter(q_criteria)
            .values_list(field_name, flat=True)
            .exclude(**{field_name + "__in": items_to_exclude})
            .distinct()
            .order_by(field_name)
        )
        query_choices = [("", "None")] + [(item, item) for item in query]

        initial_choices = (
            StandardNames.objects.all()
            .filter(modality="DX")
            .exclude(**{field_name: None})
            .order_by(field_name)
        )
        if "standard_name" in self.initial:
            initial_choices = initial_choices.filter(
                standard_name=self.initial["standard_name"]
            )

        self.initial[field_name] = list(
            initial_choices.values_list(field_name, flat=True)
        )

        self.fields[field_name] = forms.MultipleChoiceField(
            choices=query_choices,
            required=False,
            widget=FilteredSelectMultiple(label_name.lower() + "s", is_stacked=False),
        )

        class Media:
            css = {
                "all": (
                    os.path.join(settings.BASE_DIR, "/static/admin/css/widgets.css"),
                ),
            }
            js = ("/admin/jsi18n",)


class StandardNameFormMG(StandardNameFormBase):
    """Form for configuring standard names for study description, requested procedure, procedure and acquisition name"""

    def __init__(self, *args, **kwargs):
        super(StandardNameFormMG, self).__init__(*args, **kwargs)
        self.fields["modality"].initial = "MG"

        all_studies = GeneralStudyModuleAttr.objects.filter(modality_type__iexact="MG")

        field_names = [
            ("study_description", "Study description"),
            ("requested_procedure_code_meaning", "Requested procedure name"),
            ("procedure_code_meaning", "Procedure name"),
        ]

        for field_name, label_name in field_names:
            # Exclude items already in the MG standard names entries except for the current value of the field
            items_to_exclude = (
                StandardNames.objects.all()
                .filter(modality="MG")
                .values(field_name)
                .exclude(**{field_name: None})
            )
            if "standard_name" in self.initial:
                items_to_exclude = items_to_exclude.exclude(
                    standard_name=self.initial["standard_name"]
                )

            query = (
                all_studies.values_list(field_name, flat=True)
                .exclude(**{field_name + "__in": items_to_exclude})
                .distinct()
                .order_by(field_name)
            )
            query_choices = [("", "None")] + [(item, item) for item in query]

            initial_choices = (
                StandardNames.objects.all()
                .filter(modality="MG")
                .exclude(**{field_name: None})
                .order_by(field_name)
            )
            if "standard_name" in self.initial:
                initial_choices = initial_choices.filter(
                    standard_name=self.initial["standard_name"]
                )

            self.initial[field_name] = list(
                initial_choices.values_list(field_name, flat=True)
            )

            self.fields[field_name] = forms.MultipleChoiceField(
                choices=query_choices,
                required=False,
                widget=FilteredSelectMultiple(
                    label_name.lower() + "s", is_stacked=False
                ),
            )

        q = ["MG"]
        q_criteria = reduce(
            operator.or_,
            (
                Q(
                    projection_xray_radiation_dose__general_study_module_attributes__modality_type__icontains=item
                )
                for item in q
            ),
        )
        field_name, label_name = ("acquisition_protocol", "Acquisition protocol name")
        items_to_exclude = (
            StandardNames.objects.all().values(field_name).exclude(**{field_name: None})
        )
        if "standard_name" in self.initial:
            items_to_exclude = items_to_exclude.exclude(
                standard_name=self.initial["standard_name"]
            )
        query = (
            IrradEventXRayData.objects.filter(q_criteria)
            .values_list(field_name, flat=True)
            .exclude(**{field_name + "__in": items_to_exclude})
            .distinct()
            .order_by(field_name)
        )
        query_choices = [("", "None")] + [(item, item) for item in query]

        initial_choices = (
            StandardNames.objects.all()
            .filter(modality="MG")
            .exclude(**{field_name: None})
            .order_by(field_name)
        )
        if "standard_name" in self.initial:
            initial_choices = initial_choices.filter(
                standard_name=self.initial["standard_name"]
            )

        self.initial[field_name] = list(
            initial_choices.values_list(field_name, flat=True)
        )

        self.fields[field_name] = forms.MultipleChoiceField(
            choices=query_choices,
            required=False,
            widget=FilteredSelectMultiple(label_name.lower() + "s", is_stacked=False),
        )

        class Media:
            css = {
                "all": (
                    os.path.join(settings.BASE_DIR, "/static/admin/css/widgets.css"),
                ),
            }
            js = ("/admin/jsi18n",)


class StandardNameFormRF(StandardNameFormBase):
    """Form for configuring standard names for study description, requested procedure, procedure and acquisition name"""

    def __init__(self, *args, **kwargs):
        super(StandardNameFormRF, self).__init__(*args, **kwargs)
        self.fields["modality"].initial = "RF"

        all_studies = GeneralStudyModuleAttr.objects.filter(modality_type__iexact="RF")

        field_names = [
            ("study_description", "Study description"),
            ("requested_procedure_code_meaning", "Requested procedure name"),
            ("procedure_code_meaning", "Procedure name"),
        ]

        for field_name, label_name in field_names:
            # Exclude items already in the RF standard names entries except for the current value of the field
            items_to_exclude = (
                StandardNames.objects.all()
                .filter(modality="RF")
                .values(field_name)
                .exclude(**{field_name: None})
            )
            if "standard_name" in self.initial:
                items_to_exclude = items_to_exclude.exclude(
                    standard_name=self.initial["standard_name"]
                )

            query = (
                all_studies.values_list(field_name, flat=True)
                .exclude(**{field_name + "__in": items_to_exclude})
                .distinct()
                .order_by(field_name)
            )
            query_choices = [("", "None")] + [(item, item) for item in query]

            initial_choices = (
                StandardNames.objects.all()
                .filter(modality="RF")
                .exclude(**{field_name: None})
                .order_by(field_name)
            )
            if "standard_name" in self.initial:
                initial_choices = initial_choices.filter(
                    standard_name=self.initial["standard_name"]
                )

            self.initial[field_name] = list(
                initial_choices.values_list(field_name, flat=True)
            )

            self.fields[field_name] = forms.MultipleChoiceField(
                choices=query_choices,
                required=False,
                widget=FilteredSelectMultiple(
                    label_name.lower() + "s", is_stacked=False
                ),
            )

        q = ["RF"]
        q_criteria = reduce(
            operator.or_,
            (
                Q(
                    projection_xray_radiation_dose__general_study_module_attributes__modality_type__icontains=item
                )
                for item in q
            ),
        )
        field_name, label_name = ("acquisition_protocol", "Acquisition protocol name")
        items_to_exclude = (
            StandardNames.objects.all().values(field_name).exclude(**{field_name: None})
        )
        if "standard_name" in self.initial:
            items_to_exclude = items_to_exclude.exclude(
                standard_name=self.initial["standard_name"]
            )
        query = (
            IrradEventXRayData.objects.filter(q_criteria)
            .values_list(field_name, flat=True)
            .exclude(**{field_name + "__in": items_to_exclude})
            .distinct()
            .order_by(field_name)
        )
        query_choices = [("", "None")] + [(item, item) for item in query]

        initial_choices = (
            StandardNames.objects.all()
            .filter(modality="RF")
            .exclude(**{field_name: None})
            .order_by(field_name)
        )
        if "standard_name" in self.initial:
            initial_choices = initial_choices.filter(
                standard_name=self.initial["standard_name"]
            )

        self.initial[field_name] = list(
            initial_choices.values_list(field_name, flat=True)
        )

        self.fields[field_name] = forms.MultipleChoiceField(
            choices=query_choices,
            required=False,
            widget=FilteredSelectMultiple(label_name.lower() + "s", is_stacked=False),
        )

        class Media:
            css = {
                "all": (
                    os.path.join(settings.BASE_DIR, "/static/admin/css/widgets.css"),
                ),
            }
            js = ("/admin/jsi18n",)


class StandardNameSettingsForm(forms.ModelForm):
    """Form for configuring whether standard names are shown / used"""

    def __init__(self, *args, **kwargs):
        super(StandardNameSettingsForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_class = "form-horizontal"
        self.helper.layout = Layout(
            Div("enable_standard_names"),
            FormActions(Submit("submit", "Submit")),
        )

    class Meta(object):
        model = StandardNameSettings
        fields = ["enable_standard_names"]


class SkinDoseMapCalcSettingsForm(forms.ModelForm):
    """Form for configuring whether skin dose maps are shown / calculated"""

    def __init__(self, *args, **kwargs):
        super(SkinDoseMapCalcSettingsForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_class = "form-horizontal"
        self.helper.layout = Layout(
            Div("enable_skin_dose_maps", "calc_on_import", "allow_safelist_modify"),
            FormActions(Submit("submit", "Submit")),
        )

    class Meta(object):
        model = SkinDoseMapCalcSettings
        fields = ["enable_skin_dose_maps", "calc_on_import", "allow_safelist_modify"]


class NotPatientNameForm(forms.ModelForm):
    """Form for configuring not-patient name patterns"""

    def __init__(self, *args, **kwargs):
        super(NotPatientNameForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_class = "form-horizontal"
        self.helper.label_class = "col-md-8"
        self.helper.field_class = "col-md-4"
        self.helper.layout = Layout(
            Div("not_patient_name"),
            FormActions(Submit("submit", "Submit")),
            Div(
                HTML(
                    """
                <div class="col-lg-4 col-lg-offset-4">
                    <a href='"""
                    + reverse("not_patient_indicators")
                    + """' role="button" class="btn btn-default">
                        Cancel and return to not-patient indicator summary page
                    </a>
                </div>
                """
                )
            ),
        )

    class Meta(object):
        model = NotPatientIndicatorsName
        fields = ["not_patient_name"]
        labels = {"not_patient_name": "pattern for name matching"}


class NotPatientIDForm(forms.ModelForm):
    """Form for configuring not-patient ID patterns"""

    def __init__(self, *args, **kwargs):
        super(NotPatientIDForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_class = "form-horizontal"
        self.helper.label_class = "col-md-8"
        self.helper.field_class = "col-md-4"
        self.helper.layout = Layout(
            Div("not_patient_id"),
            FormActions(Submit("submit", "Submit")),
            Div(
                HTML(
                    """
                <div class="col-lg-4 col-lg-offset-4">
                    <a href='"""
                    + reverse("not_patient_indicators")
                    + """' role="button" class="btn btn-default">
                        Cancel and return to not-patient indicator summary page
                    </a>
                </div>
                """
                )
            ),
        )

    class Meta(object):
        model = NotPatientIndicatorsID
        fields = ["not_patient_id"]
        labels = {"not_patient_id": "pattern for ID matching"}


class SkinSafeListForm(forms.ModelForm):
    """Form for adding/updating/removing system from openSkin safe list"""

    class Meta(object):
        model = OpenSkinSafeList
        fields = ["manufacturer", "manufacturer_model_name", "software_version"]


class BackgroundTaskMaximumRowsForm(forms.ModelForm):
    """Form for configuring the maximum number of rows in the BackgroundTask table"""

    def __init__(self, *args, **kwargs):
        super(BackgroundTaskMaximumRowsForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_class = "form-horizontal"
        self.helper.layout = Layout(
            Div("max_background_task_rows"),
            FormActions(Submit("submit", "Submit")),
        )

    class Meta(object):
        model = BackgroundTaskMaximumRows
        fields = ["max_background_task_rows"]

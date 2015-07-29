from django import forms
from django.utils.safestring import mark_safe
from openremproject import settings

DAYS = 'days'
WEEKS = 'weeks'
MONTHS = 'months'
YEARS = 'years'
TIME_PERIOD = (
    (DAYS, 'Days'),
    (WEEKS, 'Weeks'),
    (MONTHS, 'Months'),
    (YEARS, 'Years'),
)

MEAN = 'mean'
MEDIAN = 'median'
BOTH = 'both'
AVERAGES = (
    (MEAN, 'Mean'),
    (MEDIAN, 'Median'),
    (BOTH, 'Both'),
)

DLP = 'dlp'
CTDI = 'ctdi'
FREQ = 'freq'
NAME = 'name'
SORTING_CHOICES_CT = (
    (DLP, 'DLP'),
    (CTDI, 'CTDI'),
    (FREQ, 'Frequency'),
    (NAME, 'Name'),
)

ASCENDING = 1
DESCENDING = -1
SORTING_DIRECTION = (
    (ASCENDING, 'Ascending'),
    (DESCENDING, 'Descending'),
)

class SizeUploadForm(forms.Form):
    """Form for patient size csv file upload
    """
    sizefile = forms.FileField(
        label='Select a file'
    )

class SizeHeadersForm(forms.Form):
    """Form for csv column header patient size imports through the web interface
    """

    height_field = forms.ChoiceField(choices='')
    weight_field = forms.ChoiceField(choices='')
    id_field = forms.ChoiceField(choices='')
    id_type = forms.ChoiceField(choices='')

    def __init__(self, my_choice=None, **kwargs):
        super(SizeHeadersForm, self).__init__(**kwargs)
        if my_choice:

            self.fields['height_field'] = forms.ChoiceField(choices=my_choice, widget=forms.Select(attrs={"class":"form-control"}))
            self.fields['weight_field'] = forms.ChoiceField(choices=my_choice, widget=forms.Select(attrs={"class":"form-control"}))
            self.fields['id_field'] = forms.ChoiceField(choices=my_choice, widget=forms.Select(attrs={"class":"form-control"}))
            ID_TYPES = (("acc-no", "Accession Number"),("si-uid", "Study instance UID"))
            self.fields['id_type'] = forms.ChoiceField(choices=ID_TYPES, widget=forms.Select(attrs={"class":"form-control"}))

class DXChartOptionsForm(forms.Form):
    plotCharts = forms.BooleanField(label='Plot charts?',required=False)
    plotDXAcquisitionMeanDAP = forms.BooleanField(label='DAP per acquisition',required=False)
    plotDXAcquisitionFreq = forms.BooleanField(label='Acquisition frequency',required=False)
    plotDXAcquisitionMeankVp = forms.BooleanField(label='kVp per acquisition',required=False)
    plotDXAcquisitionMeanmAs = forms.BooleanField(label='mAs per acquisition',required=False)
    plotDXStudyPerDayAndHour = forms.BooleanField(label='Study workload',required=False)
    plotDXAcquisitionMeanDAPOverTime = forms.BooleanField(label='Acquisition DAP over time',required=False)
    plotDXAcquisitionMeanDAPOverTimePeriod = forms.ChoiceField(label='Time period', choices=TIME_PERIOD, required=False)
    if 'postgresql' in settings.DATABASES['default']['ENGINE']:
        plotMeanMedianOrBoth = forms.ChoiceField(label='Average to use', choices=AVERAGES, required=False)


class CTChartOptionsForm(forms.Form):
    plotCharts = forms.BooleanField(label='Plot charts?',required=False)
    plotCTAcquisitionMeanDLP = forms.BooleanField(label='DLP per acquisition',required=False)
    plotCTAcquisitionMeanCTDI = forms.BooleanField(label=mark_safe('CTDI<sub>vol</sub> per acquisition'),required=False)
    plotCTAcquisitionFreq = forms.BooleanField(label='Acquisition frequency',required=False)
    plotCTStudyMeanDLP = forms.BooleanField(label='DLP per study',required=False)
    plotCTStudyFreq = forms.BooleanField(label='Study frequency',required=False)
    plotCTRequestMeanDLP = forms.BooleanField(label='DLP per requested procedure',required=False)
    plotCTRequestFreq = forms.BooleanField(label='Requested procedure frequency',required=False)
    plotCTStudyPerDayAndHour = forms.BooleanField(label='Study workload',required=False)
    plotCTStudyMeanDLPOverTime = forms.BooleanField(label='Study DLP over time',required=False)
    plotCTStudyMeanDLPOverTimePeriod = forms.ChoiceField(label='Time period', choices=TIME_PERIOD, required=False)
    if 'postgresql' in settings.DATABASES['default']['ENGINE']:
        plotMeanMedianOrBoth = forms.ChoiceField(label='Average to use', choices=AVERAGES, required=False)


class DXChartOptionsDisplayForm(forms.Form):
    plotDXAcquisitionMeanDAP = forms.BooleanField(label='DAP per acquisition',required=False)
    plotDXAcquisitionFreq = forms.BooleanField(label='Acquisition frequency',required=False)
    plotDXAcquisitionMeankVp = forms.BooleanField(label='kVp per acquisition',required=False)
    plotDXAcquisitionMeanmAs = forms.BooleanField(label='mAs per acquisition',required=False)
    plotDXStudyPerDayAndHour = forms.BooleanField(label='Study workload',required=False)
    plotDXAcquisitionMeanDAPOverTime = forms.BooleanField(label='Acquisition DAP over time',required=False)
    plotDXAcquisitionMeanDAPOverTimePeriod = forms.ChoiceField(label='Time period', choices=TIME_PERIOD, required=False)
    if 'postgresql' in settings.DATABASES['default']['ENGINE']:
        plotMeanMedianOrBoth = forms.ChoiceField(label='Average to use', choices=AVERAGES, required=False)


class CTChartOptionsDisplayForm(forms.Form):
    plotCTAcquisitionMeanDLP = forms.BooleanField(label='DLP per acquisition',required=False)
    plotCTAcquisitionMeanCTDI = forms.BooleanField(label=mark_safe('CTDI<sub>vol</sub> per acquisition'),required=False)
    plotCTAcquisitionFreq = forms.BooleanField(label='Acquisition frequency',required=False)
    plotCTStudyMeanDLP = forms.BooleanField(label='DLP per study',required=False)
    plotCTStudyFreq = forms.BooleanField(label='Study frequency',required=False)
    plotCTRequestMeanDLP = forms.BooleanField(label='DLP per requested procedure',required=False)
    plotCTRequestFreq = forms.BooleanField(label='Requested procedure frequency',required=False)
    plotCTStudyPerDayAndHour = forms.BooleanField(label='Study workload',required=False)
    plotCTStudyMeanDLPOverTime = forms.BooleanField(label='Study DLP over time',required=False)
    plotCTStudyMeanDLPOverTimePeriod = forms.ChoiceField(label='Time period', choices=TIME_PERIOD, required=False)
    if 'postgresql' in settings.DATABASES['default']['ENGINE']:
        plotMeanMedianOrBoth = forms.ChoiceField(label='Average to use', choices=AVERAGES, required=False)


class GeneralChartOptionsDisplayForm(forms.Form):
    plotCharts = forms.BooleanField(label='Plot charts?',required=False)
    if 'postgresql' in settings.DATABASES['default']['ENGINE']:
        plotMeanMedianOrBoth = forms.ChoiceField(label='Average to use', choices=AVERAGES, required=False)
    plotCTInitialSortingChoice = forms.ChoiceField(label='Default chart sorting', choices=SORTING_CHOICES_CT, required=False)
    plotCTInitialSortingDirection = forms.ChoiceField(label='Default sorting direction', choices=SORTING_DIRECTION, required=False)


class UpdateDisplayNameForm(forms.Form):
        display_name = forms.CharField()
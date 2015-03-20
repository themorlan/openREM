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
#
#    8/10/2014: DJP added new DX section and added DX to home page.
#    9/10/2014: DJP changed DX to CR
#
"""
..  module:: views.
    :synopsis: Module to render appropriate content according to request.

..  moduleauthor:: Ed McDonagh

"""
# Following two lines added so that sphinx autodocumentation works.
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'openremproject.settings'

from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User, Group, Permission
from django.contrib.contenttypes.models import ContentType
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404
from django.template import RequestContext
from django.shortcuts import render_to_response
from django.core.urlresolvers import reverse, reverse_lazy
import json
from django.views.decorators.csrf import csrf_exempt
import datetime
from remapp.models import GeneralStudyModuleAttr, create_user_profile
from django.core.context_processors import csrf

try:
    from numpy import *
    plotting = 1
except ImportError:
    plotting = 0


def logout_page(request):
    """
    Log users out and re-direct them to the main page.
    """
    logout(request)
    return HttpResponseRedirect('/openrem/')


@login_required
def dx_summary_list_filter(request):
    if plotting: import numpy as np
    from remapp.interface.mod_filters import DXSummaryListFilter
    from django.db.models import Q, Avg, Count, Min # For the Q "OR" query used for DX and CR
    import pkg_resources # part of setuptools
    import datetime, qsstats
    from remapp.forms import DXChartOptionsForm

    if request.method == 'POST':
        requestResults = request.POST
        request.GET = request.POST
    else:
        requestResults = request.GET

    try:
        # See if the user has plot settings in userprofile
        userProfile = request.user.userprofile
    except:
        # Create a default userprofile for the user if one doesn't exist
        create_user_profile(sender=request.user, instance=request.user, created=True)
        userProfile = request.user.userprofile


    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        chartOptionsForm = DXChartOptionsForm(request.POST)
        # check whether it's valid:
        if chartOptionsForm.is_valid():
            # process the data in form.cleaned_data as required
            userProfile.plotCharts = chartOptionsForm.cleaned_data['plotCharts']
            userProfile.plotDXAcquisitionMeanDAP = chartOptionsForm.cleaned_data['plotDXAcquisitionMeanDAP']
            userProfile.plotDXAcquisitionFreq = chartOptionsForm.cleaned_data['plotDXAcquisitionFreq']
            userProfile.plotDXAcquisitionMeankVp = chartOptionsForm.cleaned_data['plotDXAcquisitionMeankVp']
            userProfile.plotDXStudyPerDayAndHour = chartOptionsForm.cleaned_data['plotDXStudyPerDayAndHour']
            userProfile.plotDXAcquisitionMeanDAPOverTime = chartOptionsForm.cleaned_data['plotDXAcquisitionMeanDAPOverTime']
            userProfile.plotDXAcquisitionMeanDAPOverTimePeriod = chartOptionsForm.cleaned_data['plotDXAcquisitionMeanDAPOverTimePeriod']
            userProfile.save()

    # if a GET (or any other method) we'll create a form populated with options from userProfile
    else:
        formData = {'plotCharts': userProfile.plotCharts,
                    'plotDXAcquisitionMeanDAP': userProfile.plotDXAcquisitionMeanDAP,
                    'plotDXAcquisitionFreq': userProfile.plotDXAcquisitionFreq,
                    'plotDXAcquisitionMeankVp': userProfile.plotDXAcquisitionMeankVp,
                    'plotDXStudyPerDayAndHour': userProfile.plotDXStudyPerDayAndHour,
                    'plotDXAcquisitionMeanDAPOverTime': userProfile.plotDXAcquisitionMeanDAPOverTime,
                    'plotDXAcquisitionMeanDAPOverTimePeriod': userProfile.plotDXAcquisitionMeanDAPOverTimePeriod}
        chartOptionsForm = DXChartOptionsForm(formData)

    plotCharts = userProfile.plotCharts
    plotDXAcquisitionMeanDAP = userProfile.plotDXAcquisitionMeanDAP
    plotDXAcquisitionFreq = userProfile.plotDXAcquisitionFreq
    plotDXAcquisitionMeankVp = userProfile.plotDXAcquisitionMeankVp
    plotDXStudyPerDayAndHour = userProfile.plotDXStudyPerDayAndHour
    plotDXAcquisitionMeanDAPOverTime = userProfile.plotDXAcquisitionMeanDAPOverTime
    plotDXAcquisitionMeanDAPOverTimePeriod = userProfile.plotDXAcquisitionMeanDAPOverTimePeriod

    f = DXSummaryListFilter(requestResults, queryset=GeneralStudyModuleAttr.objects.filter(Q(modality_type__exact = 'DX') | Q(modality_type__exact = 'CR')).distinct())
    expInclude = [o.study_instance_uid for o in f]

    if plotting and plotCharts:
        # Required for mean DAP per acquisition plot and mean DAP over time
        acquisitionSummary = f.qs.exclude(projectionxrayradiationdose__irradeventxraydata__dose_area_product__isnull=True).filter(projectionxrayradiationdose__general_study_module_attributes__study_instance_uid__in = expInclude).values('projectionxrayradiationdose__irradeventxraydata__acquisition_protocol').annotate(mean_dap = Avg('projectionxrayradiationdose__irradeventxraydata__dose_area_product'), num_acq = Count('projectionxrayradiationdose__irradeventxraydata__dose_area_product')).order_by('projectionxrayradiationdose__irradeventxraydata__acquisition_protocol')
        acquisitionHistogramData = [[None for i in xrange(2)] for i in xrange(len(acquisitionSummary))]

        # may have to add a filter(projectionxrayradiationdose__general_study_module_attributes__study_instance_uid__in = expInclude) to the line below
        acquisitionkVpSummary = f.qs.exclude(projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp__isnull=True).filter(projectionxrayradiationdose__general_study_module_attributes__study_instance_uid__in = expInclude).values('projectionxrayradiationdose__irradeventxraydata__acquisition_protocol').annotate(mean_kVp = Avg('projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp'), num_acq = Count('projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp')).order_by('projectionxrayradiationdose__irradeventxraydata__acquisition_protocol')
        acquisitionHistogramkVpData = [[None for i in xrange(2)] for i in xrange(len(acquisitionkVpSummary))]

        if plotDXAcquisitionMeanDAPOverTime:
            # Required for mean DAP per month plot
            acquisitionDAPoverTime = [None] * len(acquisitionSummary)
            startDate = f.qs.aggregate(Min('study_date')).get('study_date__min')
            today = datetime.date.today()

        for idx, protocol in enumerate(acquisitionSummary):
            # Required for mean DAP per acquisition plot and mean DAP per month plot
            subqs = f.qs.filter(projectionxrayradiationdose__irradeventxraydata__acquisition_protocol__exact = protocol.get('projectionxrayradiationdose__irradeventxraydata__acquisition_protocol')).filter(projectionxrayradiationdose__general_study_module_attributes__study_instance_uid__in = expInclude)

            if plotDXAcquisitionMeanDAP:
                # Required for mean DAP per acquisition plot
                dapValues = subqs.exclude(projectionxrayradiationdose__irradeventxraydata__dose_area_product__isnull=True).values_list('projectionxrayradiationdose__irradeventxraydata__dose_area_product', flat=True)
                acquisitionHistogramData[idx][0], acquisitionHistogramData[idx][1] = np.histogram([float(x)*1000000 for x in dapValues], bins=20)

            if plotDXAcquisitionMeankVp:
                # Required for mean kVp per acquisition plot
                kVpValues = subqs.exclude(projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp__isnull=True).values_list('projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp', flat=True)
                acquisitionHistogramkVpData[idx][0], acquisitionHistogramkVpData[idx][1] = np.histogram([float(x) for x in kVpValues], bins=20)

            if plotDXAcquisitionMeanDAPOverTime:
                # Required for mean DAP over time
                qss = qsstats.QuerySetStats(subqs, 'projectionxrayradiationdose__irradeventxraydata__date_time_started', aggregate=Avg('projectionxrayradiationdose__irradeventxraydata__dose_area_product'))
                acquisitionDAPoverTime[idx] = qss.time_series(startDate, today, interval=plotDXAcquisitionMeanDAPOverTimePeriod)

        if plotDXStudyPerDayAndHour:
            # Required for studies per weekday and studies per hour in each weekday plot
            studiesPerHourInWeekdays = [[0 for x in range(24)] for x in range(7)]
            for day in range(7):
                studyTimesOnThisWeekday = f.qs.filter(study_date__week_day=day+1).values('study_time')
                if studyTimesOnThisWeekday:
                    for hour in range(24):
                        try:
                            studiesPerHourInWeekdays[day][hour] = studyTimesOnThisWeekday.filter(study_time__gte = str(hour)+':00').filter(study_time__lte = str(hour)+':59').values('study_time').count()
                        except:
                            studiesPerHourInWeekdays[day][hour] = 0

    try:
        vers = pkg_resources.require("openrem")[0].version
    except:
        vers = ''
    admin = {'openremversion' : vers}

    if request.user.groups.filter(name="exportgroup"):
        admin['exportperm'] = True
    if request.user.groups.filter(name="admingroup"):
        admin['adminperm'] = True

    returnStructure = {'filter': f, 'admin':admin, 'chartOptionsForm':chartOptionsForm}

    returnStructure.update(csrf(request))

    if plotting and plotCharts:
        returnStructure['acquisitionSummary'] = acquisitionSummary

    if plotting and plotCharts and plotDXAcquisitionMeanDAP:
        returnStructure['acquisitionHistogramData'] = acquisitionHistogramData

    if plotting and plotCharts and plotDXAcquisitionMeankVp:
        returnStructure['acquisitionkVpSummary'] = acquisitionkVpSummary
        returnStructure['acquisitionHistogramkVpData'] = acquisitionHistogramkVpData

    if plotting and plotCharts and plotDXAcquisitionMeanDAPOverTime:
        returnStructure['acquisitionDAPoverTime'] = acquisitionDAPoverTime

    if plotting and plotCharts and plotDXStudyPerDayAndHour:
        returnStructure['studiesPerHourInWeekdays'] = studiesPerHourInWeekdays

    return render_to_response(
        'remapp/dxfiltered.html',
        returnStructure,
        context_instance=RequestContext(request)
        )

@login_required
def dx_histogram_list_filter(request):
    if plotting: import numpy as np
    from remapp.interface.mod_filters import DXSummaryListFilter
    from django.db.models import Q, Avg, Count, Min # For the Q "OR" query used for DX and CR
    import pkg_resources # part of setuptools
    import datetime, qsstats
    from remapp.forms import DXChartOptionsForm

    if request.method == 'POST':
        requestResults = request.POST
        request.GET = request.POST
    else:
        requestResults = request.GET

    if requestResults.get('acquisitionhist'):
        f = DXSummaryListFilter(requestResults, queryset=GeneralStudyModuleAttr.objects.filter(
            Q(modality_type__exact = 'DX') | Q(modality_type__exact = 'CR'),
            projectionxrayradiationdose__irradeventxraydata__acquisition_protocol=requestResults.get('acquisition_protocol'),
            projectionxrayradiationdose__irradeventxraydata__dose_area_product__gte=requestResults.get('acquisition_dap_min'),
            projectionxrayradiationdose__irradeventxraydata__dose_area_product__lte=requestResults.get('acquisition_dap_max')
            ).order_by().distinct())
        if requestResults.get('study_description') : f.qs.filter(study_description=requestResults.get('study_description'))
        if requestResults.get('study_dap_max')     : f.qs.filter(projectionxrayradiationdose__accumulatedxraydose__accumulatedprojectionxraydose__dose_area_product_total__lte=requestResults.get('study_dap_max'))
        if requestResults.get('study_dap_min')     : f.qs.filter(projectionxrayradiationdose__accumulatedxraydose__accumulatedprojectionxraydose__dose_area_product_total__gte=requestResults.get('study_dap_min'))

    elif requestResults.get('studyhist'):
        f = DXSummaryListFilter(requestResults, queryset=GeneralStudyModuleAttr.objects.filter(
            Q(modality_type__exact = 'DX') | Q(modality_type__exact = 'CR'),
            projectionxrayradiationdose__irradeventxraydata__acquisition_protocol=requestResults.get('study_description'),
            projectionxrayradiationdose__accumulatedxraydose__accumulatedprojectionxraydose__dose_area_product_total__gte=requestResults.get('study_dap_min'),
            projectionxrayradiationdose__accumulatedxraydose__accumulatedprojectionxraydose__dose_area_product_total__lte=requestResults.get('study_dap_max')
            ).order_by().distinct())
        if requestResults.get('acquisition_protocol') : f.qs.filter(study_description=requestResults.get('acquisition_protocol'))
        if requestResults.get('acquisition_dap_max')  : f.qs.filter(projectionxrayradiationdose__irradeventxraydata__dose_area_product__lte=requestResults.get('acquisition_dap_max'))
        if requestResults.get('acquisition_dap_min')  : f.qs.filter(projectionxrayradiationdose__irradeventxraydata__dose_area_product__gte=requestResults.get('acquisition_dap_min'))

    else:
        f = DXSummaryListFilter(requestResults, queryset=GeneralStudyModuleAttr.objects.filter(
            Q(modality_type__exact = 'DX') | Q(modality_type__exact = 'CR')).order_by().distinct())
        if requestResults.get('study_description')    : f.qs.filter(projectionxrayradiationdose__irradeventxraydata__acquisition_protocol=requestResults.get('study_description'))
        if requestResults.get('study_dap_min')        : f.qs.filter(projectionxrayradiationdose__accumulatedxraydose__accumulatedprojectionxraydose__dose_area_product_total__gte=requestResults.get('study_dap_min'))
        if requestResults.get('study_dap_max')        : f.qs.filter(projectionxrayradiationdose__accumulatedxraydose__accumulatedprojectionxraydose__dose_area_product_total__lte=requestResults.get('study_dap_max'))
        if requestResults.get('acquisition_protocol') : f.qs.filter(study_description=requestResults.get('acquisition_protocol'))
        if requestResults.get('acquisition_dap_max')  : f.qs.filter(projectionxrayradiationdose__irradeventxraydata__dose_area_product__lte=requestResults.get('acquisition_dap_max'))
        if requestResults.get('acquisition_dap_min')  : f.qs.filter(projectionxrayradiationdose__irradeventxraydata__dose_area_product__gte=requestResults.get('acquisition_dap_min'))

    if requestResults.get('accession_number')  : f.qs.filter(accession_number=requestResults.get('accession_number'))
    if requestResults.get('date_after')        : f.qs.filter(study_date__gt=requestResults.get('date_after'))
    if requestResults.get('date_before')       : f.qs.filter(study_date__lt=requestResults.get('date_before'))
    if requestResults.get('institution_name')  : f.qs.filter(generalequipmentmoduleattr__institution_name=requestResults.get('institution_name'))
    if requestResults.get('manufacturer')      : f.qs.filter(generalequipmentmoduleattr__manufacturer=requestResults.get('manufacturer'))
    if requestResults.get('model_name')        : f.qs.filter(generalequipmentmoduleattr__model_name=requestResults.get('model_name'))
    if requestResults.get('patient_age_max')   : f.qs.filter(patientstudymoduleattr__patient_age_decimal__lte=requestResults.get('patient_age_max'))
    if requestResults.get('patient_age_min')   : f.qs.filter(patientstudymoduleattr__patient_age_decimal__gte=requestResults.get('patient_age_min'))
    if requestResults.get('station_name')      : f.qs.filter(generalequipmentmoduleattr__station_name=requestResults.get('station_name'))

    try:
        # See if the user has plot settings in userprofile
        userProfile = request.user.userprofile
    except:
        # Create a default userprofile for the user if one doesn't exist
        create_user_profile(sender=request.user, instance=request.user, created=True)
        userProfile = request.user.userprofile


    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        chartOptionsForm = DXChartOptionsForm(request.POST)
        # check whether it's valid:
        if chartOptionsForm.is_valid():
            # process the data in form.cleaned_data as required
            userProfile.plotCharts = chartOptionsForm.cleaned_data['plotCharts']
            userProfile.plotDXAcquisitionMeanDAP = chartOptionsForm.cleaned_data['plotDXAcquisitionMeanDAP']
            userProfile.plotDXAcquisitionFreq = chartOptionsForm.cleaned_data['plotDXAcquisitionFreq']
            userProfile.plotDXAcquisitionMeankVp = chartOptionsForm.cleaned_data['plotDXAcquisitionMeankVp']
            userProfile.plotDXStudyPerDayAndHour = chartOptionsForm.cleaned_data['plotDXStudyPerDayAndHour']
            userProfile.plotDXAcquisitionMeanDAPOverTime = chartOptionsForm.cleaned_data['plotDXAcquisitionMeanDAPOverTime']
            userProfile.plotDXAcquisitionMeanDAPOverTimePeriod = chartOptionsForm.cleaned_data['plotDXAcquisitionMeanDAPOverTimePeriod']
            userProfile.save()
    else:
        formData = {'plotCharts': userProfile.plotCharts,
                    'plotDXAcquisitionMeanDAP': userProfile.plotDXAcquisitionMeanDAP,
                    'plotDXAcquisitionFreq': userProfile.plotDXAcquisitionFreq,
                    'plotDXAcquisitionMeankVp': userProfile.plotDXAcquisitionMeankVp,
                    'plotDXStudyPerDayAndHour': userProfile.plotDXStudyPerDayAndHour,
                    'plotDXAcquisitionMeanDAPOverTime': userProfile.plotDXAcquisitionMeanDAPOverTime,
                    'plotDXAcquisitionMeanDAPOverTimePeriod': userProfile.plotDXAcquisitionMeanDAPOverTimePeriod}
        chartOptionsForm = DXChartOptionsForm(formData)

    plotCharts = userProfile.plotCharts
    plotDXAcquisitionMeanDAP = userProfile.plotDXAcquisitionMeanDAP
    plotDXAcquisitionFreq = userProfile.plotDXAcquisitionFreq
    plotDXAcquisitionMeankVp = userProfile.plotDXAcquisitionMeankVp
    plotDXStudyPerDayAndHour = userProfile.plotDXStudyPerDayAndHour
    plotDXAcquisitionMeanDAPOverTime = userProfile.plotDXAcquisitionMeanDAPOverTime
    plotDXAcquisitionMeanDAPOverTimePeriod = userProfile.plotDXAcquisitionMeanDAPOverTimePeriod

    expInclude = [o.study_instance_uid for o in f]

    if plotting and plotCharts:
        # Required for mean DAP per acquisition plot and mean DAP over time
        #acquisitionSummary = f.qs.exclude(projectionxrayradiationdose__irradeventxraydata__dose_area_product__isnull=True).values('projectionxrayradiationdose__irradeventxraydata__acquisition_protocol').distinct().annotate(mean_dap = Avg('projectionxrayradiationdose__irradeventxraydata__dose_area_product'), num_acq = Count('projectionxrayradiationdose__irradeventxraydata__dose_area_product')).order_by('projectionxrayradiationdose__irradeventxraydata__acquisition_protocol')
        acquisitionSummary = f.qs.exclude(projectionxrayradiationdose__irradeventxraydata__dose_area_product__isnull=True).filter(projectionxrayradiationdose__general_study_module_attributes__study_instance_uid__in = expInclude).values('projectionxrayradiationdose__irradeventxraydata__acquisition_protocol').annotate(mean_dap = Avg('projectionxrayradiationdose__irradeventxraydata__dose_area_product'), num_acq = Count('projectionxrayradiationdose__irradeventxraydata__dose_area_product')).order_by('projectionxrayradiationdose__irradeventxraydata__acquisition_protocol')
        acquisitionHistogramData = [[None for i in xrange(2)] for i in xrange(len(acquisitionSummary))]

        # may have to add a filter(projectionxrayradiationdose__general_study_module_attributes__study_instance_uid__in = expInclude) to the line below
        #acquisitionkVpSummary = f.qs.exclude(projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp__isnull=True).values('projectionxrayradiationdose__irradeventxraydata__acquisition_protocol').distinct().annotate(mean_kVp = Avg('projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp'), num_acq = Count('projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp')).order_by('projectionxrayradiationdose__irradeventxraydata__acquisition_protocol')
        acquisitionkVpSummary = f.qs.exclude(projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp__isnull=True).filter(projectionxrayradiationdose__general_study_module_attributes__study_instance_uid__in = expInclude).values('projectionxrayradiationdose__irradeventxraydata__acquisition_protocol').annotate(mean_kVp = Avg('projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp'), num_acq = Count('projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp')).order_by('projectionxrayradiationdose__irradeventxraydata__acquisition_protocol')
        acquisitionHistogramkVpData = [[None for i in xrange(2)] for i in xrange(len(acquisitionkVpSummary))]

        if plotDXAcquisitionMeanDAPOverTime:
            # Required for mean DAP per month plot
            acquisitionDAPoverTime = [None] * len(acquisitionSummary)
            startDate = f.qs.aggregate(Min('study_date')).get('study_date__min')
            today = datetime.date.today()

        for idx, protocol in enumerate(acquisitionSummary):
            # Required for mean DAP per acquisition plot AND mean DAP per month plot
            subqs = f.qs.filter(projectionxrayradiationdose__irradeventxraydata__acquisition_protocol=protocol.get('projectionxrayradiationdose__irradeventxraydata__acquisition_protocol'))

            if plotDXAcquisitionMeanDAP:
                # Required for mean DAP per acquisition plot
                dapValues = subqs.exclude(projectionxrayradiationdose__irradeventxraydata__dose_area_product__isnull=True).values_list('projectionxrayradiationdose__irradeventxraydata__dose_area_product', flat=True)
                acquisitionHistogramData[idx][0], acquisitionHistogramData[idx][1] = np.histogram([float(x)*1000000 for x in dapValues], bins=20)

            if plotDXAcquisitionMeankVp:
                # Required for mean kVp per acquisition plot
                kVpValues = subqs.exclude(projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp__isnull=True).values_list('projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp', flat=True)
                acquisitionHistogramkVpData[idx][0], acquisitionHistogramkVpData[idx][1] = np.histogram([float(x) for x in kVpValues], bins=20)

            if plotDXAcquisitionMeanDAPOverTime:
                # Required for mean DAP per time period plot
                qss = qsstats.QuerySetStats(subqs, 'projectionxrayradiationdose__irradeventxraydata__date_time_started', aggregate=Avg('projectionxrayradiationdose__irradeventxraydata__dose_area_product'))
                acquisitionDAPoverTime[idx] = qss.time_series(startDate, today,interval=plotDXAcquisitionMeanDAPOverTimePeriod)

        if plotDXStudyPerDayAndHour:
            # Required for studies per weekday and studies per hour in each weekday plot
            studiesPerHourInWeekdays = [[0 for x in range(24)] for x in range(7)]
            for day in range(7):
                studyTimesOnThisWeekday = f.qs.filter(study_date__week_day=day+1).values('study_time')
                if studyTimesOnThisWeekday:
                    for hour in range(24):
                        try:
                            studiesPerHourInWeekdays[day][hour] = studyTimesOnThisWeekday.filter(study_time__gte = str(hour)+':00').filter(study_time__lte = str(hour)+':59').values('study_time').count()
                        except:
                            studiesPerHourInWeekdays[day][hour] = 0

    try:
        vers = pkg_resources.require("openrem")[0].version
    except:
        vers = ''
    admin = {'openremversion' : vers}

    if request.user.groups.filter(name="exportgroup"):
        admin['exportperm'] = True
    if request.user.groups.filter(name="admingroup"):
        admin['adminperm'] = True

    returnStructure = {'filter': f, 'admin':admin, 'chartOptionsForm':chartOptionsForm}

    if plotting and plotCharts:
        returnStructure['acquisitionSummary'] = acquisitionSummary

    if plotting and plotCharts and plotDXAcquisitionMeanDAP:
        returnStructure['acquisitionHistogramData'] = acquisitionHistogramData

    if plotting and plotCharts and plotDXAcquisitionMeankVp:
        returnStructure['acquisitionkVpSummary'] = acquisitionkVpSummary
        returnStructure['acquisitionHistogramkVpData'] = acquisitionHistogramkVpData

    if plotting and plotCharts and plotDXAcquisitionMeanDAPOverTime:
        returnStructure['acquisitionDAPoverTime'] = acquisitionDAPoverTime

    if plotting and plotCharts and plotDXStudyPerDayAndHour:
        returnStructure['studiesPerHourInWeekdays'] = studiesPerHourInWeekdays

    return render_to_response(
        'remapp/dxfiltered.html',
        returnStructure,
        context_instance=RequestContext(request)
        )


@login_required
def rf_summary_list_filter(request):
    from remapp.interface.mod_filters import RFSummaryListFilter
    import pkg_resources # part of setuptools
    f = RFSummaryListFilter(request.GET, queryset=GeneralStudyModuleAttr.objects.filter(modality_type__contains = 'RF'))

    try:
        vers = pkg_resources.require("openrem")[0].version
    except:
        vers = ''
    admin = {'openremversion' : vers}

    if request.user.groups.filter(name="exportgroup"):
        admin['exportperm'] = True
    if request.user.groups.filter(name="admingroup"):
        admin['adminperm'] = True

    return render_to_response(
        'remapp/rffiltered.html',
        {'filter': f, 'admin':admin},
        context_instance=RequestContext(request)
        )

@login_required
def ct_summary_list_filter(request):
    if plotting: import numpy as np
    from remapp.interface.mod_filters import CTSummaryListFilter
    from django.db.models import Q, Avg, Count, Min # For the Q "OR" query used for DX and CR
    import pkg_resources # part of setuptools
    import datetime, qsstats
    from remapp.forms import CTChartOptionsForm

    if request.method == 'POST':
        requestResults = request.POST
        request.GET = request.POST
    else:
        requestResults = request.GET

    try:
        # See if the user has plot settings in userprofile
        userProfile = request.user.userprofile
    except:
        # Create a default userprofile for the user if one doesn't exist
        create_user_profile(sender=request.user, instance=request.user, created=True)
        userProfile = request.user.userprofile


    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        chartOptionsForm = CTChartOptionsForm(request.POST)
        # check whether it's valid:
        if chartOptionsForm.is_valid():
            # process the data in form.cleaned_data as required
            userProfile.plotCharts = chartOptionsForm.cleaned_data['plotCharts']
            userProfile.plotCTAcquisitionMeanDLP = chartOptionsForm.cleaned_data['plotCTAcquisitionMeanDLP']
            userProfile.plotCTAcquisitionFreq = chartOptionsForm.cleaned_data['plotCTAcquisitionFreq']
            userProfile.plotCTStudyMeanDLP = chartOptionsForm.cleaned_data['plotCTStudyMeanDLP']
            userProfile.plotCTStudyFreq = chartOptionsForm.cleaned_data['plotCTStudyFreq']
            userProfile.plotCTStudyPerDayAndHour = chartOptionsForm.cleaned_data['plotCTStudyPerDayAndHour']
            userProfile.plotCTStudyMeanDLPOverTime = chartOptionsForm.cleaned_data['plotCTStudyMeanDLPOverTime']
            userProfile.plotCTStudyMeanDLPOverTimePeriod = chartOptionsForm.cleaned_data['plotCTStudyMeanDLPOverTimePeriod']
            userProfile.save()

    # if a GET (or any other method) we'll create a blank form
    else:
        formData = {'plotCharts': userProfile.plotCharts,
                    'plotCTAcquisitionMeanDLP': userProfile.plotCTAcquisitionMeanDLP,
                    'plotCTAcquisitionFreq': userProfile.plotCTAcquisitionFreq,
                    'plotCTStudyMeanDLP': userProfile.plotCTStudyMeanDLP,
                    'plotCTStudyFreq': userProfile.plotCTStudyFreq,
                    'plotCTStudyPerDayAndHour': userProfile.plotCTStudyPerDayAndHour,
                    'plotCTStudyMeanDLPOverTime': userProfile.plotCTStudyMeanDLPOverTime,
                    'plotCTStudyMeanDLPOverTimePeriod': userProfile.plotCTStudyMeanDLPOverTimePeriod}
        chartOptionsForm = CTChartOptionsForm(formData)

    plotCharts = userProfile.plotCharts
    plotCTAcquisitionMeanDLP = userProfile.plotCTAcquisitionMeanDLP
    plotCTAcquisitionFreq = userProfile.plotCTAcquisitionFreq
    plotCTStudyMeanDLP = userProfile.plotCTStudyMeanDLP
    plotCTStudyFreq = userProfile.plotCTStudyFreq
    plotCTStudyPerDayAndHour = userProfile.plotCTStudyPerDayAndHour
    plotCTStudyMeanDLPOverTime = userProfile.plotCTStudyMeanDLPOverTime
    plotCTStudyMeanDLPOverTimePeriod = userProfile.plotCTStudyMeanDLPOverTimePeriod


    f = CTSummaryListFilter(requestResults, queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact = 'CT').distinct())

    if plotting and plotCharts:
        # Required for mean DLP per acquisition plot
        acquisitionSummary = f.qs.exclude(ctradiationdose__ctirradiationeventdata__dlp__isnull=True).values('ctradiationdose__ctirradiationeventdata__acquisition_protocol').distinct().annotate(mean_dlp = Avg('ctradiationdose__ctirradiationeventdata__dlp'), num_acq = Count('ctradiationdose__ctirradiationeventdata__dlp')).order_by('ctradiationdose__ctirradiationeventdata__acquisition_protocol')
        acquisitionHistogramData = [[None for i in xrange(2)] for i in xrange(len(acquisitionSummary))]
        for idx, protocol in enumerate(acquisitionSummary):
            dlpValues = f.qs.exclude(ctradiationdose__ctirradiationeventdata__dlp__isnull=True).filter(ctradiationdose__ctirradiationeventdata__acquisition_protocol=protocol.get('ctradiationdose__ctirradiationeventdata__acquisition_protocol')).values_list('ctradiationdose__ctirradiationeventdata__dlp', flat=True)
            acquisitionHistogramData[idx][0], acquisitionHistogramData[idx][1] = np.histogram([float(x) for x in dlpValues], bins=20)

        # Required for mean DLP per study type plot
        studySummary = f.qs.exclude(ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total__isnull=True).values('study_description').distinct().annotate(mean_dlp = Avg('ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total'), num_acq = Count('ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total')).order_by('study_description')
        studyHistogramData = [[None for i in xrange(2)] for i in xrange(len(studySummary))]

        if plotCTStudyMeanDLPOverTime:
            # Required for mean DLP per study type per week plot
            studyDLPoverTime = [None] * len(studySummary)
            startDate = f.qs.aggregate(Min('study_date')).get('study_date__min')
            today = datetime.date.today()

        # Required for all plots
        qs = f.qs.exclude(ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total__isnull=True)

        for idx, study in enumerate(studySummary):
            # Required for mean DLP per study type plot AND mean DLP per study type per week plot
            subqs = qs.filter(study_description=study.get('study_description'))

            # Required for mean DLP per study type plot
            dlpValues = subqs.values_list('ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total', flat=True)
            studyHistogramData[idx][0], studyHistogramData[idx][1] = np.histogram([float(x) for x in dlpValues], bins=20)

            if plotCTStudyMeanDLPOverTime:
                # Required for mean DLP per study type per time period plot
                qss = qsstats.QuerySetStats(subqs, 'study_date', aggregate=Avg('ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total'))
                studyDLPoverTime[idx] = qss.time_series(startDate, today,interval=plotCTStudyMeanDLPOverTimePeriod)

        if plotCTStudyPerDayAndHour:
            # Required for studies per weekday and studies per hour in each weekday plot
            studiesPerHourInWeekdays = [[0 for x in range(24)] for x in range(7)]
            for day in range(7):
                studyTimesOnThisWeekday = f.qs.filter(study_date__week_day=day+1).values('study_time')
                if studyTimesOnThisWeekday:
                    for hour in range(24):
                        try:
                            studiesPerHourInWeekdays[day][hour] = studyTimesOnThisWeekday.filter(study_time__gte = str(hour)+':00').filter(study_time__lte = str(hour)+':59').values('study_time').count()
                        except:
                            studiesPerHourInWeekdays[day][hour] = 0

    try:
        vers = pkg_resources.require("openrem")[0].version
    except:
        vers = ''
    admin = {'openremversion' : vers}

    if request.user.groups.filter(name="exportgroup"):
        admin['exportperm'] = True
    if request.user.groups.filter(name="admingroup"):
        admin['adminperm'] = True

    returnStructure = {'filter': f, 'admin':admin, 'chartOptionsForm':chartOptionsForm}

    returnStructure.update(csrf(request))

    if plotting and plotCharts:
        returnStructure['studySummary'] = studySummary
        returnStructure['studyHistogramData'] = studyHistogramData
        returnStructure['acquisitionSummary'] = acquisitionSummary
        returnStructure['acquisitionHistogramData'] = acquisitionHistogramData

    if plotting and plotCharts and plotCTStudyPerDayAndHour:
        returnStructure['studiesPerHourInWeekdays'] = studiesPerHourInWeekdays

    if plotting and plotCharts and plotCTStudyMeanDLPOverTime:
        returnStructure['studyDLPoverTime'] = studyDLPoverTime

    return render_to_response(
        'remapp/ctfiltered.html',
        returnStructure,
        context_instance=RequestContext(request)
        )


@login_required
def ct_histogram_list_filter(request):
    if plotting: import numpy as np
    from remapp.interface.mod_filters import CTSummaryListFilter
    from django.db.models import Q, Avg, Count, Min # For the Q "OR" query used for DX and CR
    import pkg_resources # part of setuptools
    import datetime, qsstats
    from remapp.forms import CTChartOptionsForm

    if request.method == 'POST':
        requestResults = request.POST
        request.GET = request.POST
    else:
        requestResults = request.GET

    if requestResults.get('acquisitionhist'):
        f = CTSummaryListFilter(requestResults, queryset=GeneralStudyModuleAttr.objects.filter(
            modality_type__exact = 'CT',
            ctradiationdose__ctirradiationeventdata__acquisition_protocol=requestResults.get('acquisition_protocol'),
            ctradiationdose__ctirradiationeventdata__dlp__gte=requestResults.get('acquisition_dlp_min'),
            ctradiationdose__ctirradiationeventdata__dlp__lte=requestResults.get('acquisition_dlp_max')
            ).order_by().distinct())
        if requestResults.get('study_description') : f.qs.filter(study_description=requestResults.get('study_description'))
        if requestResults.get('study_dlp_max')     : f.qs.filter(ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total__lte=requestResults.get('study_dlp_max'))
        if requestResults.get('study_dlp_min')     : f.qs.filter(ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total__gte=requestResults.get('study_dlp_min'))

    elif requestResults.get('studyhist'):
        f = CTSummaryListFilter(requestResults, queryset=GeneralStudyModuleAttr.objects.filter(
            modality_type__exact = 'CT',
            study_description=requestResults.get('study_description'),
            ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total__gte=requestResults.get('study_dlp_min'),
            ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total__lte=requestResults.get('study_dlp_max')
            ).order_by().distinct())
        if requestResults.get('acquisition_protocol') : f.qs.filter(ctradiationdose__ctirradiationeventdata__acquisition_protocol=requestResults.get('acquisition_protocol'))
        if requestResults.get('acquisition_dlp_max')  : f.qs.filter(ctradiationdose__ctirradiationeventdata__dlp__lte=requestResults.get('study_dlp_max'))
        if requestResults.get('acquisition_dlp_min')  : f.qs.filter(ctradiationdose__ctirradiationeventdata__dlp__gte=requestResults.get('study_dlp_min'))

    else:
        f = CTSummaryListFilter(requestResults, queryset=GeneralStudyModuleAttr.objects.filter(
            modality_type__exact = 'CT').order_by().distinct())
        if requestResults.get('study_description')    : f.qs.filter(study_description=requestResults.get('study_description'))
        if requestResults.get('study_dlp_min')        : f.qs.filter(ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total__gte=requestResults.get('study_dlp_min'))
        if requestResults.get('study_dlp_max')        : f.qs.filter(ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total__lte=requestResults.get('study_dlp_max'))
        if requestResults.get('acquisition_protocol') : f.qs.filter(ctradiationdose__ctirradiationeventdata__acquisition_protocol=requestResults.get('acquisition_protocol'))
        if requestResults.get('study_dlp_max')        : f.qs.filter(ctradiationdose__ctirradiationeventdata__dlp__lte=requestResults.get('study_dlp_max'))
        if requestResults.get('study_dlp_min')        : f.qs.filter(ctradiationdose__ctirradiationeventdata__dlp__gte=requestResults.get('study_dlp_min'))

    if requestResults.get('accession_number')  : f.qs.filter(accession_number=requestResults.get('accession_number'))
    if requestResults.get('date_after')        : f.qs.filter(study_date__gt=requestResults.get('date_after'))
    if requestResults.get('date_before')       : f.qs.filter(study_date__lt=requestResults.get('date_before'))
    if requestResults.get('institution_name')  : f.qs.filter(generalequipmentmoduleattr__institution_name=requestResults.get('institution_name'))
    if requestResults.get('manufacturer')      : f.qs.filter(generalequipmentmoduleattr__manufacturer=requestResults.get('manufacturer'))
    if requestResults.get('model_name')        : f.qs.filter(generalequipmentmoduleattr__model_name=requestResults.get('model_name'))
    if requestResults.get('patient_age_max')   : f.qs.filter(patientstudymoduleattr__patient_age_decimal__lte=requestResults.get('patient_age_max'))
    if requestResults.get('patient_age_min')   : f.qs.filter(patientstudymoduleattr__patient_age_decimal__gte=requestResults.get('patient_age_min'))
    if requestResults.get('station_name')      : f.qs.filter(generalequipmentmoduleattr__station_name=requestResults.get('station_name'))

    try:
        # See if the user has plot settings in userprofile
        userProfile = request.user.userprofile
    except:
        # Create a default userprofile for the user if one doesn't exist
        create_user_profile(sender=request.user, instance=request.user, created=True)
        userProfile = request.user.userprofile


    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        chartOptionsForm = CTChartOptionsForm(request.POST)
        # check whether it's valid:
        if chartOptionsForm.is_valid():
            # process the data in form.cleaned_data as required
            userProfile.plotCharts = chartOptionsForm.cleaned_data['plotCharts']
            userProfile.plotCTAcquisitionMeanDLP = chartOptionsForm.cleaned_data['plotCTAcquisitionMeanDLP']
            userProfile.plotCTAcquisitionFreq = chartOptionsForm.cleaned_data['plotCTAcquisitionFreq']
            userProfile.plotCTStudyMeanDLP = chartOptionsForm.cleaned_data['plotCTStudyMeanDLP']
            userProfile.plotCTStudyFreq = chartOptionsForm.cleaned_data['plotCTStudyFreq']
            userProfile.plotCTStudyPerDayAndHour = chartOptionsForm.cleaned_data['plotCTStudyPerDayAndHour']
            userProfile.plotCTStudyMeanDLPOverTime = chartOptionsForm.cleaned_data['plotCTStudyMeanDLPOverTime']
            userProfile.plotCTStudyMeanDLPOverTimePeriod = chartOptionsForm.cleaned_data['plotCTStudyMeanDLPOverTimePeriod']
            userProfile.save()
    else:
        formData = {'plotCharts': userProfile.plotCharts,
                    'plotCTAcquisitionMeanDLP': userProfile.plotCTAcquisitionMeanDLP,
                    'plotCTAcquisitionFreq': userProfile.plotCTAcquisitionFreq,
                    'plotCTStudyMeanDLP': userProfile.plotCTStudyMeanDLP,
                    'plotCTStudyFreq': userProfile.plotCTStudyFreq,
                    'plotCTStudyPerDayAndHour': userProfile.plotCTStudyPerDayAndHour,
                    'plotCTStudyMeanDLPOverTime': userProfile.plotCTStudyMeanDLPOverTime,
                    'plotCTStudyMeanDLPOverTimePeriod': userProfile.plotCTStudyMeanDLPOverTimePeriod}
        chartOptionsForm = CTChartOptionsForm(formData)

    plotCharts = userProfile.plotCharts
    plotCTAcquisitionMeanDLP = userProfile.plotCTAcquisitionMeanDLP
    plotCTAcquisitionFreq = userProfile.plotCTAcquisitionFreq
    plotCTStudyMeanDLP = userProfile.plotCTStudyMeanDLP
    plotCTStudyFreq = userProfile.plotCTStudyFreq
    plotCTStudyPerDayAndHour = userProfile.plotCTStudyPerDayAndHour
    plotCTStudyMeanDLPOverTime = userProfile.plotCTStudyMeanDLPOverTime
    plotCTStudyMeanDLPOverTimePeriod = userProfile.plotCTStudyMeanDLPOverTimePeriod

    if plotting and plotCharts:
        # Required for mean DLP per acquisition plot
        acquisitionSummary = f.qs.exclude(Q(ctradiationdose__ctirradiationeventdata__acquisition_protocol__isnull=True)|Q(ctradiationdose__ctirradiationeventdata__acquisition_protocol='')).values('ctradiationdose__ctirradiationeventdata__acquisition_protocol').distinct().annotate(mean_dlp = Avg('ctradiationdose__ctirradiationeventdata__dlp'), num_acq = Count('ctradiationdose__ctirradiationeventdata__dlp')).order_by('ctradiationdose__ctirradiationeventdata__acquisition_protocol')
        acquisitionHistogramData = [[None for i in xrange(2)] for i in xrange(len(acquisitionSummary))]
        for idx, protocol in enumerate(acquisitionSummary):
            dlpValues = f.qs.filter(ctradiationdose__ctirradiationeventdata__acquisition_protocol=protocol.get('ctradiationdose__ctirradiationeventdata__acquisition_protocol')).values_list('ctradiationdose__ctirradiationeventdata__dlp', flat=True)
            acquisitionHistogramData[idx][0], acquisitionHistogramData[idx][1] = np.histogram([float(x) for x in dlpValues], bins=20)

        # Required for mean DLP per study type plot
        studySummary = f.qs.exclude(Q(study_description__isnull=True)|Q(study_description='')).values('study_description').distinct().annotate(mean_dlp = Avg('ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total'), num_acq = Count('ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total')).order_by('study_description')
        studyHistogramData = [[None for i in xrange(2)] for i in xrange(len(studySummary))]

        if plotCTStudyMeanDLPOverTime:
            # Required for mean DLP per study type per week plot
            studyDLPoverTime = [None] * len(studySummary)
            startDate = f.qs.aggregate(Min('study_date')).get('study_date__min')
            today = datetime.date.today()

        for idx, study in enumerate(studySummary):
            # Required for Mean DLP per study type plot AND mean DLP per study type per week plot
            subqs = f.qs.filter(study_description=study.get('study_description'))

            # Required for mean DLP per study type plot
            dlpValues = subqs.values_list('ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total', flat=True)
            studyHistogramData[idx][0], studyHistogramData[idx][1] = np.histogram([float(x) for x in dlpValues], bins=20)

            if plotCTStudyMeanDLPOverTime:
                # Required for mean DLP per study type per time period plot
                qss = qsstats.QuerySetStats(subqs, 'study_date', aggregate=Avg('ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total'))
                studyDLPoverTime[idx] = qss.time_series(startDate, today,interval=plotCTStudyMeanDLPOverTimePeriod)

        if plotCTStudyPerDayAndHour:
            # Required for studies per weekday and studies per hour in each weekday plot
            studiesPerHourInWeekdays = [[0 for x in range(24)] for x in range(7)]
            for day in range(7):
                studyTimesOnThisWeekday = f.qs.filter(study_date__week_day=day+1).values('study_time')
                if studyTimesOnThisWeekday:
                    for hour in range(24):
                        try:
                            studiesPerHourInWeekdays[day][hour] = studyTimesOnThisWeekday.filter(study_time__gte = str(hour)+':00').filter(study_time__lte = str(hour)+':59').values('study_time').count()
                        except:
                            studiesPerHourInWeekdays[day][hour] = 0

    try:
        vers = pkg_resources.require("openrem")[0].version
    except:
        vers = ''
    admin = {'openremversion' : vers}

    if request.user.groups.filter(name="exportgroup"):
        admin['exportperm'] = True
    if request.user.groups.filter(name="admingroup"):
        admin['adminperm'] = True

    returnStructure = {'filter': f, 'admin':admin, 'chartOptionsForm':chartOptionsForm}

    if plotting and plotCharts:
        returnStructure['studySummary'] = studySummary
        returnStructure['studyHistogramData'] = studyHistogramData
        returnStructure['acquisitionSummary'] = acquisitionSummary
        returnStructure['acquisitionHistogramData'] = acquisitionHistogramData

    if plotting and plotCharts and plotCTStudyPerDayAndHour:
        returnStructure['studiesPerHourInWeekdays'] = studiesPerHourInWeekdays

    if plotting and plotCharts and plotCTStudyMeanDLPOverTime:
        returnStructure['studyDLPoverTime'] = studyDLPoverTime

    return render_to_response(
        'remapp/ctfiltered.html',
        returnStructure,
        context_instance=RequestContext(request)
        )


@login_required
def mg_summary_list_filter(request):
    from remapp.interface.mod_filters import MGSummaryListFilter
    import pkg_resources # part of setuptools
    filter_data = request.GET.copy()
    if 'page' in filter_data:
        del filter_data['page']
    f = MGSummaryListFilter(filter_data, queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact = 'MG'))

    try:
        vers = pkg_resources.require("openrem")[0].version
    except:
        vers = ''
    admin = {'openremversion' : vers}

    if request.user.groups.filter(name="exportgroup"):
        admin['exportperm'] = True
    if request.user.groups.filter(name="admingroup"):
        admin['adminperm'] = True

    return render_to_response(
        'remapp/mgfiltered.html',
        {'filter': f, 'admin':admin},
        context_instance=RequestContext(request)
        )


def openrem_home(request):
    from remapp.models import GeneralStudyModuleAttr
    from django.db.models import Q # For the Q "OR" query used for DX and CR
    from datetime import datetime
    import pytz
    from collections import OrderedDict
    import pkg_resources # part of setuptools
    utc = pytz.UTC
    
    if not Group.objects.filter(name="viewgroup"):
        vg = Group(name="viewgroup")
        vg.save()
    if not Group.objects.filter(name="exportgroup"):
        eg = Group(name="exportgroup")
        eg.save()
    if not Group.objects.filter(name="admingroup"):
        ag = Group(name="admingroup")
        ag.save()
    
    allstudies = GeneralStudyModuleAttr.objects.all()
    homedata = { 
        'total' : allstudies.count(),
        'mg' : allstudies.filter(modality_type__exact = 'MG').count(),
        'ct' : allstudies.filter(modality_type__exact = 'CT').count(),
        'rf' : allstudies.filter(modality_type__contains = 'RF').count(),
        #'dx' : allstudies.filter(modality_type__contains = 'CR').count(),
        'dx' : allstudies.filter(Q(modality_type__exact = 'DX') | Q(modality_type__exact = 'CR')).count(),
        }

    try:
        vers = pkg_resources.require("openrem")[0].version
    except:
        vers = ''
    admin = {'openremversion' : vers}

    if request.user.groups.filter(name="exportgroup"):
        admin['exportperm'] = True
    if request.user.groups.filter(name="admingroup"):
        admin['adminperm'] = True

    modalities = ('MG','CT','RF','DX')
    for modality in modalities:
        # 10/10/2014, DJP: added code to combine DX with CR
        if modality == 'DX':
            #studies = allstudies.filter(modality_type__contains = modality).all()
            studies = allstudies.filter(Q(modality_type__exact = 'DX') | Q(modality_type__exact = 'CR')).all()
        else:
            studies = allstudies.filter(modality_type__contains = modality).all()
        # End of 10/10/2014 DJP code changes

        stations = studies.values_list('generalequipmentmoduleattr__station_name').distinct()
        modalitydata = {}
        for station in stations:
            latestdate = studies.filter(
                generalequipmentmoduleattr__station_name__exact = station[0]
                ).latest('study_date').study_date
            latestuid = studies.filter(generalequipmentmoduleattr__station_name__exact = station[0]
                ).filter(study_date__exact = latestdate).latest('study_time')
            latestdatetime = datetime.combine(latestuid.study_date, latestuid.study_time)
            
            inst_name = studies.filter(
                generalequipmentmoduleattr__station_name__exact = station[0]
                ).latest('study_date').generalequipmentmoduleattr_set.get().institution_name
                
            model_name = studies.filter(
                generalequipmentmoduleattr__station_name__exact = station[0]
                ).latest('study_date').generalequipmentmoduleattr_set.get().manufacturer_model_name
            
            institution = '{0}, {1}'.format(inst_name,model_name)
                       
            modalitydata[station[0]] = {
                'total' : studies.filter(
                    generalequipmentmoduleattr__station_name__exact = station[0]
                    ).count(),
                'latest' : latestdatetime,
                'institution' : institution
            }
        ordereddata = OrderedDict(sorted(modalitydata.items(), key=lambda t: t[1]['latest'], reverse=True))
        homedata[modality] = ordereddata
    
    
    return render(request,"remapp/home.html",{'homedata':homedata, 'admin':admin})

@login_required
def study_delete(request, pk, template_name='remapp/study_confirm_delete.html'):
    study = get_object_or_404(GeneralStudyModuleAttr, pk=pk)

    if request.method=='POST':
        if request.user.groups.filter(name="admingroup"):
            study.delete()
        return redirect("/openrem/")

    if request.user.groups.filter(name="admingroup"):
        return render(request, template_name, {'exam':study})

    return redirect("/openrem/")

import os, sys, csv
from django.shortcuts import render_to_response
from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.contrib import messages

from openremproject.settings import MEDIA_ROOT
from remapp.models import SizeUpload
from remapp.forms import SizeUploadForm

@login_required
def size_upload(request):
    """Form for upload of csv file containing patient size information. POST request passes database entry ID to size_process

    :param request: If POST, contains the file upload information
    """
    # Handle file upload
    if request.method == 'POST':
        form = SizeUploadForm(request.POST, request.FILES)
        if form.is_valid():
            newcsv = SizeUpload(sizefile = request.FILES['sizefile'])
            newcsv.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect("/openrem/admin/sizeprocess/{0}/".format(newcsv.id))
    else:
        form = SizeUploadForm() # A empty, unbound form


    try:
        vers = pkg_resources.require("openrem")[0].version
    except:
        vers = ''
    admin = {'openremversion' : vers}

    if request.user.groups.filter(name="exportgroup"):
        admin['exportperm'] = True
    if request.user.groups.filter(name="admingroup"):
        admin['adminperm'] = True

    # Render list page with the documents and the form
    return render_to_response(
        'remapp/sizeupload.html',
        {'form': form, 'admin':admin},
        context_instance=RequestContext(request)
    )

from remapp.forms import SizeHeadersForm

@login_required
def size_process(request, *args, **kwargs):
    """Form for csv column header patient size imports through the web interface. POST request launches import task

    :param request: If POST, contains the field header information
    :param pk: From URL, identifies database patient size import record
    :type pk: kwarg
    """
    from remapp.extractors.ptsizecsv2db import websizeimport

    if request.method == 'POST': 
              
        itemsInPost = len(request.POST.values())
        uniqueItemsInPost = len(set(request.POST.values()))
        
        if itemsInPost == uniqueItemsInPost:
            csvrecord = SizeUpload.objects.all().filter(id__exact = kwargs['pk'])[0]
            
            if not csvrecord.sizefile:
                messages.error(request, "File to be processed doesn't exist. Do you wish to try again?")
                return HttpResponseRedirect("/openrem/admin/sizeupload")
            
            csvrecord.height_field = request.POST['height_field']
            csvrecord.weight_field = request.POST['weight_field']
            csvrecord.id_field = request.POST['id_field']
            csvrecord.id_type = request.POST['id_type']
            csvrecord.save()

            job = websizeimport.delay(csv_pk = kwargs['pk'])

            return HttpResponseRedirect("/openrem/admin/sizeimports")

        else:
            messages.error(request, "Duplicate column header selection. Each field must have a different header.")
            return HttpResponseRedirect("/openrem/admin/sizeprocess/{0}/".format(kwargs['pk']))
            

    else:
    
        csvrecord = SizeUpload.objects.all().filter(id__exact = kwargs['pk'])
        with open(os.path.join(MEDIA_ROOT, csvrecord[0].sizefile.name), 'rb') as csvfile:
            try:
                dialect = csv.Sniffer().sniff(csvfile.read(1024))
                csvfile.seek(0)
                if csv.Sniffer().has_header(csvfile.read(1024)):
                    csvfile.seek(0)
                    dataset = csv.DictReader(csvfile)
                    messages.success(request, "CSV file with column headers found.")
                    fieldnames = tuple(zip(dataset.fieldnames, dataset.fieldnames))
                    form = SizeHeadersForm(my_choice = fieldnames)
                else:
                    csvfile.seek(0)
                    messages.error(request, "Doesn't appear to have a header row. First row: {0}. The uploaded file has been deleted.".format(next(csvfile)))
                    csvrecord[0].sizefile.delete()
                    return HttpResponseRedirect("/openrem/admin/sizeupload")
            except csv.Error as e:
                messages.error(request, "Doesn't appear to be a csv file. Error({0}). The uploaded file has been deleted.".format(e))
                csvrecord[0].sizefile.delete()
                return HttpResponseRedirect("/openrem/admin/sizeupload")
            except:
                messages.error(request, "Unexpected error - please contact an administrator: {0}.".format(sys.exc_info()[0]))
                csvrecord[0].sizefile.delete()
                return HttpResponseRedirect("/openrem/admin/sizeupload")

    try:
        vers = pkg_resources.require("openrem")[0].version
    except:
        vers = ''
    admin = {'openremversion' : vers}

    if request.user.groups.filter(name="exportgroup"):
        admin['exportperm'] = True
    if request.user.groups.filter(name="admingroup"):
        admin['adminperm'] = True

    return render_to_response(
        'remapp/sizeprocess.html',
        {'form':form, 'csvid':kwargs['pk'], 'admin':admin},
        context_instance=RequestContext(request)
    )

def size_imports(request, *args, **kwargs):
    """Lists patient size imports in the web interface

    :param request:
    """
    import os
    import pkg_resources # part of setuptools
    from django.template import RequestContext  
    from django.shortcuts import render_to_response
    from remapp.models import SizeUpload

    imports = SizeUpload.objects.all().order_by('-import_date')
    
    current = imports.filter(status__contains = 'CURRENT')
    complete = imports.filter(status__contains = 'COMPLETE')
    errors = imports.filter(status__contains = 'ERROR')
    
    try:
        vers = pkg_resources.require("openrem")[0].version
    except:
        vers = ''
    admin = {'openremversion' : vers}

    if request.user.groups.filter(name="exportgroup"):
        admin['exportperm'] = True
    if request.user.groups.filter(name="admingroup"):
        admin['adminperm'] = True


    return render_to_response(
        'remapp/sizeimports.html',
        {'admin': admin, 'current': current, 'complete': complete, 'errors': errors},
        context_instance = RequestContext(request)
    )
    

@csrf_exempt
@login_required
def size_delete(request):
    """Task to delete records of patient size imports through the web interface

    :param request: Contains the task ID
    :type request: POST
    """
    from django.http import HttpResponseRedirect
    from django.core.urlresolvers import reverse
    from django.contrib import messages
    from remapp.models import SizeUpload

    for task in request.POST:
        uploads = SizeUpload.objects.filter(task_id__exact = request.POST[task])
        for upload in uploads:
            try:
                upload.logfile.delete()
                upload.delete()
                messages.success(request, "Export file and database entry deleted successfully.")
            except OSError as e:
                messages.error(request, "Export file delete failed - please contact an administrator. Error({0}): {1}".format(e.errno, e.strerror))
            except:
                messages.error(request, "Unexpected error - please contact an administrator: {0}".format(sys.exc_info()[0]))

    return HttpResponseRedirect(reverse(size_imports))

@login_required
def size_abort(request, pk):
    """View to abort current patient size imports

    :param request: Contains the task primary key
    :type request: POST
    """
    from celery.task.control import revoke
    from django.http import HttpResponseRedirect
    from django.shortcuts import render, redirect, get_object_or_404
    from remapp.models import SizeUpload

    size = get_object_or_404(SizeUpload, pk=pk)

    if request.user.groups.filter(name="admingroup"):
        revoke(size.task_id, terminate=True)
        size.logfile.delete()
        size.sizefile.delete()
        size.delete()

    return HttpResponseRedirect("/openrem/admin/sizeimports/")

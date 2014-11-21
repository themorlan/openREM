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
from remapp.models import General_study_module_attributes

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
    from django.db.models import Q, Avg, Count # For the Q "OR" query used for DX and CR
    import pkg_resources # part of setuptools

    f = DXSummaryListFilter(request.GET, queryset=General_study_module_attributes.objects.filter(Q(modality_type__exact = 'DX') | Q(modality_type__exact = 'CR')).order_by().distinct())

    if plotting:
        acquisitionSummary = f.qs.exclude(Q(projection_xray_radiation_dose__irradiation_event_xray_data__acquisition_protocol__isnull=True)|Q(projection_xray_radiation_dose__irradiation_event_xray_data__acquisition_protocol='')).values('projection_xray_radiation_dose__irradiation_event_xray_data__acquisition_protocol').order_by().distinct().annotate(mean_dap = Avg('projection_xray_radiation_dose__irradiation_event_xray_data__dose_area_product'), num_acq = Count('projection_xray_radiation_dose__irradiation_event_xray_data__dose_area_product'))
        acquisitionHistogramData = [[None for i in xrange(2)] for i in xrange(len(acquisitionSummary))]
        for idx, protocol in enumerate(acquisitionSummary):
            dapValues = f.qs.filter(projection_xray_radiation_dose__irradiation_event_xray_data__acquisition_protocol=protocol.values()[2]).values_list('projection_xray_radiation_dose__irradiation_event_xray_data__dose_area_product', flat=True)
            acquisitionHistogramData[idx][0], acquisitionHistogramData[idx][1] = np.histogram([float(x)*1000000 for x in dapValues], bins=20)

    try:
        vers = pkg_resources.require("openrem")[0].version
    except:
        vers = ''
    admin = {'openremversion' : vers}

    if request.user.groups.filter(name="exportgroup"):
        admin['exportperm'] = True
    if request.user.groups.filter(name="admingroup"):
        admin['adminperm'] = True

    if plotting:
        return render_to_response(
            'remapp/dxfiltered.html',
            {'filter': f, 'admin':admin,
             'acquisitionSummary': acquisitionSummary,
             'acquisitionHistogramData': acquisitionHistogramData},
            context_instance=RequestContext(request)
            )
    else:
        return render_to_response(
            'remapp/dxfiltered.html',
            {'filter': f, 'admin':admin},
            context_instance=RequestContext(request)
            )


@login_required
def rf_summary_list_filter(request):
    from remapp.interface.mod_filters import RFSummaryListFilter
    import pkg_resources # part of setuptools
    f = RFSummaryListFilter(request.GET, queryset=General_study_module_attributes.objects.filter(modality_type__contains = 'RF'))

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
    from django.db.models import Q, Avg, Count # For the Q "OR" query used for DX and CR
    import pkg_resources # part of setuptools

    f = CTSummaryListFilter(request.GET, queryset=General_study_module_attributes.objects.filter(modality_type__exact = 'CT').order_by().distinct())

    if plotting:
        # New approach, DJP 21/11/2014
        # In this approach the names, mean DLP and number of data points making up each mean are
        # contained in a pair of variables. I haven't looked at how ctfiltered.html accesses
        # these values - it will be different to how it's currently done. I will replace
        # the old method with this new one once I've looked at how to access these variables.
        # The pair of lines below returns the name of each acquisition and study protocol, its
        # mean DLP and the number of data points contributing to each mean.

        # The next line and the following for loop obtains the histogram counts and bins for each acquisition protocol
        acquisitionSummary = f.qs.exclude(Q(ct_radiation_dose__ct_irradiation_event_data__acquisition_protocol__isnull=True)|Q(ct_radiation_dose__ct_irradiation_event_data__acquisition_protocol='')).values('ct_radiation_dose__ct_irradiation_event_data__acquisition_protocol').order_by().distinct().annotate(mean_dlp = Avg('ct_radiation_dose__ct_irradiation_event_data__dlp'), num_acq = Count('ct_radiation_dose__ct_irradiation_event_data__dlp'))
        acquisitionHistogramData = [[None for i in xrange(2)] for i in xrange(len(acquisitionSummary))]
        for idx, protocol in enumerate(acquisitionSummary):
            dlpValues = f.qs.filter(ct_radiation_dose__ct_irradiation_event_data__acquisition_protocol=protocol.values()[2]).values_list('ct_radiation_dose__ct_irradiation_event_data__dlp', flat=True)
            acquisitionHistogramData[idx][0], acquisitionHistogramData[idx][1] = np.histogram([float(x) for x in dlpValues], bins=20)

        # The next line and the following for loop obtains the histogram counts and bins for each study
        studySummary = f.qs.exclude(Q(study_description__isnull=True)|Q(study_description='')).values('study_description').order_by().distinct().annotate(mean_dlp = Avg('ct_radiation_dose__ct_accumulated_dose_data__ct_dose_length_product_total'), num_acq = Count('ct_radiation_dose__ct_accumulated_dose_data__ct_dose_length_product_total'))
        studyHistogramData = [[None for i in xrange(2)] for i in xrange(len(studySummary))]
        for idx, study in enumerate(studySummary):
            dlpValues = f.qs.filter(study_description=study.values()[2]).values_list('ct_radiation_dose__ct_accumulated_dose_data__ct_dose_length_product_total', flat=True)
            studyHistogramData[idx][0], studyHistogramData[idx][1] = np.histogram([float(x) for x in dlpValues], bins=20)
        # End of new approach, DJP 21/11/2014


        # Data for plot of mean DLP per acquisition protocol and also drilldown histogram for each
        uniqueProtocols = f.qs.exclude(Q(ct_radiation_dose__ct_irradiation_event_data__acquisition_protocol__isnull=True)|Q(ct_radiation_dose__ct_irradiation_event_data__acquisition_protocol='')).values('ct_radiation_dose__ct_irradiation_event_data__acquisition_protocol').order_by().distinct()

        protocolMeanDLP = [[None for i in xrange(2)] for i in xrange(len(uniqueProtocols))]
        protocolNames   = [None] * len(uniqueProtocols)
        protocolHistogramCounts   = [None] * len(uniqueProtocols)
        protocolHistogramBinEdges = [None] * len(uniqueProtocols)

        for idx, protocol in enumerate(protocolMeanDLP):
            protocolMeanDLP[idx][0] = f.qs.filter(ct_radiation_dose__ct_irradiation_event_data__acquisition_protocol=(uniqueProtocols[idx].values())[0]).aggregate(Avg('ct_radiation_dose__ct_irradiation_event_data__dlp')).values()[0]
            protocolMeanDLP[idx][1] = f.qs.filter(ct_radiation_dose__ct_irradiation_event_data__acquisition_protocol=(uniqueProtocols[idx].values())[0]).count()

            protocolNames[idx]   = uniqueProtocols[idx].values()[0]
            dlpValues = f.qs.filter(ct_radiation_dose__ct_irradiation_event_data__acquisition_protocol=(uniqueProtocols[idx].values())[0]).values_list('ct_radiation_dose__ct_irradiation_event_data__dlp', flat=True)
            dlpValuesFloatArray = []
            for idx2, dlpValue in enumerate(dlpValues):
                try:
                    dlpValuesFloatArray.append(float(dlpValue))
                except:
                    pass
        
            protocolHistogramCounts[idx], protocolHistogramBinEdges[idx] = np.histogram(dlpValuesFloatArray, bins=20)

        # Data for plot of mean DLP per study description and also drilldown histogram for each
        uniqueStudies = f.qs.exclude(Q(study_description__isnull=True)|Q(study_description='')).values('study_description').order_by().distinct()
        
        studyMeanDLP = [None] * len(uniqueStudies)
        studyMeanDLP = [[None for i in xrange(2)] for i in xrange(len(uniqueStudies))]
        studyNames   = [None] * len(uniqueStudies)
        studyHistogramCounts   = [None] * len(uniqueStudies)
        studyHistogramBinEdges = [None] * len(uniqueStudies)

        for idx, study in enumerate(studyMeanDLP):
            studyMeanDLP[idx][0] = f.qs.filter(study_description=(uniqueStudies[idx].values())[0]).aggregate(Avg('ct_radiation_dose__ct_accumulated_dose_data__ct_dose_length_product_total')).values()[0]
            studyMeanDLP[idx][1] = f.qs.filter(study_description=(uniqueStudies[idx].values())[0]).count()
            studyNames[idx]   = uniqueStudies[idx].values()[0]
            dlpValues = f.qs.filter(study_description=(uniqueStudies[idx].values())[0]).values_list('ct_radiation_dose__ct_accumulated_dose_data__ct_dose_length_product_total', flat=True)
            dlpValuesFloatArray = []
            for idx2, dlpValue in enumerate(dlpValues):
                try:
                    dlpValuesFloatArray.append(float(dlpValue))
                except:
                    pass
        
            studyHistogramCounts[idx], studyHistogramBinEdges[idx] = np.histogram(dlpValuesFloatArray, bins=20)

    try:
        vers = pkg_resources.require("openrem")[0].version
    except:
        vers = ''
    admin = {'openremversion' : vers}

    if request.user.groups.filter(name="exportgroup"):
        admin['exportperm'] = True
    if request.user.groups.filter(name="admingroup"):
        admin['adminperm'] = True

    if plotting:
        return render_to_response(
            'remapp/ctfiltered.html',
            {'filter': f, 'admin':admin,
             'plotNames': protocolNames,
             'plotData':  protocolMeanDLP,
             'histogramCounts':   protocolHistogramCounts,
             'histogramBinEdges': protocolHistogramBinEdges,
             'studyPlotNames': studyNames,
             'studyPlotData':  studyMeanDLP,
             'studyHistogramCounts':   studyHistogramCounts,
             'studyHistogramBinEdges': studyHistogramBinEdges,
             'studySummary': studySummary,
             'acquisitionSummary': acquisitionSummary,
             'studyHistogramData': studyHistogramData,
             'acquisitionHistogramData': acquisitionHistogramData},
            context_instance=RequestContext(request)
            )
    else:
        return render_to_response(
            'remapp/ctfiltered.html',
            {'filter': f, 'admin':admin},
            context_instance=RequestContext(request)
            )

@login_required
def mg_summary_list_filter(request):
    from remapp.interface.mod_filters import MGSummaryListFilter
    import pkg_resources # part of setuptools
    filter_data = request.GET.copy()
    if 'page' in filter_data:
        del filter_data['page']
    f = MGSummaryListFilter(filter_data, queryset=General_study_module_attributes.objects.filter(modality_type__exact = 'MG'))

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
    from remapp.models import General_study_module_attributes
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
    
    allstudies = General_study_module_attributes.objects.all()
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

        stations = studies.values_list('general_equipment_module_attributes__station_name').distinct()
        modalitydata = {}
        for station in stations:
            latestdate = studies.filter(
                general_equipment_module_attributes__station_name__exact = station[0]
                ).latest('study_date').study_date
            latestuid = studies.filter(general_equipment_module_attributes__station_name__exact = station[0]
                ).filter(study_date__exact = latestdate).latest('study_time')
            latestdatetime = datetime.combine(latestuid.study_date, latestuid.study_time)
            
            inst_name = studies.filter(
                general_equipment_module_attributes__station_name__exact = station[0]
                ).latest('study_date').general_equipment_module_attributes_set.get().institution_name
                
            model_name = studies.filter(
                general_equipment_module_attributes__station_name__exact = station[0]
                ).latest('study_date').general_equipment_module_attributes_set.get().manufacturer_model_name
            
            institution = '{0}, {1}'.format(inst_name,model_name)
                       
            modalitydata[station[0]] = {
                'total' : studies.filter(
                    general_equipment_module_attributes__station_name__exact = station[0]
                    ).count(),
                'latest' : latestdatetime,
                'institution' : institution
            }
        ordereddata = OrderedDict(sorted(modalitydata.items(), key=lambda t: t[1]['latest'], reverse=True))
        homedata[modality] = ordereddata
    
    
    return render(request,"remapp/home.html",{'homedata':homedata, 'admin':admin})

@login_required
def study_delete(request, pk, template_name='remapp/study_confirm_delete.html'):
    study = get_object_or_404(General_study_module_attributes, pk=pk)    

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
from remapp.models import Size_upload
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
            newcsv = Size_upload(sizefile = request.FILES['sizefile'])
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
            csvrecord = Size_upload.objects.all().filter(id__exact = kwargs['pk'])[0]
            
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
    
        csvrecord = Size_upload.objects.all().filter(id__exact = kwargs['pk'])
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
    from remapp.models import Size_upload

    imports = Size_upload.objects.all().order_by('-import_date')
    
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
    from remapp.models import Size_upload

    for task in request.POST:
        uploads = Size_upload.objects.filter(task_id__exact = request.POST[task])
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
    from remapp.models import Size_upload

    size = get_object_or_404(Size_upload, pk=pk)

    if request.user.groups.filter(name="admingroup"):
        revoke(size.task_id, terminate=True)
        size.logfile.delete()
        size.sizefile.delete()
        size.delete()

    return HttpResponseRedirect("/openrem/admin/sizeimports/")

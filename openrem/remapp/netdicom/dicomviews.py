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
..  module:: dicomviews.py
    :synopsis: To manage the DICOM servers

..  moduleauthor:: Ed McDonagh

"""
import json
import os
import uuid

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.db.models import Count
from django.http import HttpResponse
from django.shortcuts import (redirect, render)
from django.urls import reverse_lazy
from django.views.decorators.csrf import csrf_exempt
from django.views.generic.edit import CreateView, UpdateView, DeleteView

from .qrscu import (movescu, qrscu)
from .storescp import start_store
from .tools import echoscu
from .. import (__docs_version__, __version__)
from ..forms import (DicomQueryForm, DicomQRForm, DicomStoreForm)
from ..models import (DicomDeleteSettings, DicomQuery, DicomStoreSCP, DicomRemoteQR)
from ..views import _create_admin_dict

os.environ['DJANGO_SETTINGS_MODULE'] = 'openremproject.settings'


@csrf_exempt
@login_required
def run_store(request, pk):
    """View to start built-in STORE-SCP"""
    if request.user.groups.filter(name="admingroup"):
        store = DicomStoreSCP.objects.get(pk__exact = pk)
        store.run = True
        store.save()
        start_store(store_pk=pk)
    return redirect(reverse_lazy('dicom_summary'))


@csrf_exempt
@login_required
def stop_store(request, pk):
    """View to stop built-in STORE-SCP"""
    if request.user.groups.filter(name="admingroup"):
        store = DicomStoreSCP.objects.filter(pk__exact = pk)
        if store:
            store[0].run = False
            store[0].save()
            store[0].status = u"Quit signal sent"
            store[0].save()
        else:
            print(u"Can't stop store SCP: Invalid primary key")
    return redirect(reverse_lazy('dicom_summary'))


@csrf_exempt
def status_update_store(request):
    """View to check if store is running using DICOM ECHO"""

    resp = {}
    data = request.POST
    scp_pk = data.get('scp_pk')

    echo_response = echoscu(scp_pk=scp_pk, store_scp=True)

    store = DicomStoreSCP.objects.get(pk=scp_pk)

    if echo_response == "Success":
        resp['message'] = u"<div>Last status: {0}</div>".format(store.status)
        resp['statusindicator'] = u"<h3 class='pull-right panel-title'>" \
                                  u"<span class='glyphicon glyphicon-ok' aria-hidden='true'></span>" \
                                  u"<span class='sr-only'>OK:</span> Server is alive</h3>"
        resp['delbutton'] = u"<button type='button' class='btn btn-primary' disabled='disabled'>Delete</button>"
        resp['startbutton'] = u""
        resp['stopbutton'] = u"<a class='btn btn-danger' href='{0}' role='button'>Stop server</a>".format(
            reverse_lazy('stop_store', kwargs={'pk': scp_pk}))
    else:
        resp['message'] = u"<div>Last status: {0}</div>".format(store.status)
        resp['statusindicator'] = u"<h3 class='pull-right panel-title status-red'>" \
                                  u"<span class='glyphicon glyphicon-exclamation-sign' aria-hidden='true'></span>" \
                                  u"<span class='sr-only'>Error:</span> {0}</h3>".format(
            echo_response)
        resp['delbutton'] = u"<a class='btn btn-primary' href='{0}' role='button'>Delete</a>".format(
            reverse_lazy('dicomstore_delete', kwargs={'pk': scp_pk}))
        resp['startbutton'] = u"<a class='btn btn-success' href='{0}' role='button'>Start server</a>".format(
            reverse_lazy('run_store', kwargs={'pk': scp_pk}))
        resp['stopbutton'] = u""

    return HttpResponse(json.dumps(resp), content_type="application/json")


@csrf_exempt
def q_update(request):
    """View to update query status"""

    resp = {}
    data = request.POST
    query_id = data.get('queryID')
    resp['queryID'] = query_id
    try:
        query = DicomQuery.objects.get(query_id=query_id)
    except ObjectDoesNotExist:
        resp['status'] = 'not complete'
        resp['message'] = '<h4>Query {0} not yet started</h4>'.format(query_id)
        resp['subops'] = ''
        return HttpResponse(json.dumps(resp), content_type='application/json')

    if query.failed:
        resp['status'] = u'failed'
        resp['message'] ='<h4>Query Failed</h4> {0}'.format(query.message)
        resp['subops'] = ''
        return HttpResponse(json.dumps(resp), content_type='application/json')

    study_rsp = query.dicomqrrspstudy_set.all()
    if not query.complete:
        modalities = study_rsp.values('modalities_in_study').annotate(count=Count('pk'))
        table = [u'<table class="table table-bordered">'
                 u'<tr><th>Modalities in study</th><th>Number of responses</th></tr>']
        for m in modalities:
            table.append(u'<tr><td>')
            if m['modalities_in_study']:
                table.append(u', '.join(json.loads(m['modalities_in_study'])))
            else:
                table.append(u'Unknown')
            table.append(u'</td><td>')
            table.append(str(m['count']))
            table.append(u'</tr></td>')
        table.append(u'</table>')
        tablestr = ''.join(table)
        resp['status'] = u'not complete'
        resp['message'] = u'<h4>{0}</h4><p>Responses so far:</p> {1}'.format(query.stage, tablestr)
        resp['subops'] = ''
    else:
        modalities = study_rsp.values('modality').annotate(count=Count('pk'))
        table = [u'<table class="table table-bordered"><tr><th>Modality</th><th>Number of responses</th></tr>']
        for m in modalities:
            table.append(u'<tr><td>')
            if m['modality']:
                table.append(m['modality'])
            else:
                table.append(u'Unknown - SR only study?')
            table.append(u'</td><td>')
            table.append(str(m['count']))
            table.append(u'</tr></td>')
        table.append(u'</table>')
        tablestr = u''.join(table)
        resp['status'] = u'complete'
        query_details_text = u"<div class='panel-group' id='accordion'>" \
                             u"<div class='panel panel-default'>" \
                             u"<div class='panel-heading'>  "\
                             u"<h4 class='panel-title'>" \
                             u"<a data-toggle='collapse' data-parent='#accordion' href='#query-details'>" \
                             u"Query details</h4>" \
                             u"</a></h4></div>" \
                             u"<div id='query-details' class='panel-collapse collapse'>" \
                             u"<div class='panel-body'>" \
                             u"<p>{0}</p></div></div></div></div>".format(query.stage)
        not_as_expected_help_text = u"<div class='panel-group' id='accordion'>" \
                                    u"<div class='panel panel-default'>" \
                                    u"<div class='panel-heading'>  "\
                                    u"<h4 class='panel-title'>" \
                                    u"<a data-toggle='collapse' data-parent='#accordion' href='#not-expected'>" \
                                    u"Not what you expected?</h4>" \
                                    u"</a></h4></div>" \
                                    u"<div id='not-expected' class='panel-collapse collapse'>" \
                                    u"<div class='panel-body'>" \
                                    u"<p>For DX and mammography, the query will look for Radiation Dose Structured " \
                                    u"Reports, or images if the RDSR is not available. For Fluoroscopy, RDSRs are " \
                                    u"required. For CT RDSRs are preferred, but Philips dose images can be used and " \
                                    u"for some scanners, particularly older Toshiba scanners that can't create RDSR " \
                                    u"OpenREM can process the data to create an RDSR to import.</p>" \
                                    u"<p>If you haven't got the results you expect, it may be that the imaging system" \
                                    u" is not creating RDSRs or not sending them to the PACS you are querying. In " \
                                    u"either case you will need to have the system reconfigured to create and/or send" \
                                    u" them. If it is a CT scanner that can't create an RDSR (it is too old), it is " \
                                    u"worth trying the 'Toshiba' option, but you will need to be using Orthanc and " \
                                    u"configure your scanner in the " \
                                    u"<a href='https://docs.openrem.org/en/{0}/netdicom-orthanc-config.html" \
                                    u"#guide-to-customising-orthanc-configuration' target='_blank'>" \
                                    u"toshiba_extractor_systems</a> list" \
                                    u". You will need to verify the resulting data to confirm accuracy.</p>" \
                                    u"</div></div></div></div>".format(__docs_version__)
        resp['message'] = u'<h4>Query complete - there are {1} studies we can move</h4> {0} {2} {3}'.format(
            tablestr, study_rsp.count(), query_details_text, not_as_expected_help_text)
        resp['subops'] = ''

    return HttpResponse(json.dumps(resp), content_type='application/json')


@csrf_exempt
@login_required
def q_process(request, *args, **kwargs):
    """View to process query form POST"""

    if request.method == 'POST':
        form = DicomQueryForm(request.POST)
        if form.is_valid():
            rh_pk = form.cleaned_data.get('remote_host_field')
            store_pk = form.cleaned_data.get('store_scp_field')
            date_from = form.cleaned_data.get('date_from_field')
            date_until = form.cleaned_data.get('date_until_field')
            modalities = form.cleaned_data.get('modality_field')
            inc_sr = form.cleaned_data.get('inc_sr_field')
            remove_duplicates = form.cleaned_data.get('duplicates_field')
            desc_exclude = form.cleaned_data.get('desc_exclude_field')
            desc_include = form.cleaned_data.get('desc_include_field')
            stationname_exclude = form.cleaned_data.get('stationname_exclude_field')
            stationname_include = form.cleaned_data.get('stationname_include_field')
            get_toshiba_images = form.cleaned_data.get('get_toshiba_images_field')
            get_empty_sr = form.cleaned_data.get('get_empty_sr_field')

            query_id = str(uuid.uuid4())

            if date_from:
                date_from = date_from.isoformat()
            if date_until:
                date_until = date_until.isoformat()

            if desc_exclude:
                study_desc_exc = list(map(str.lower, list(map(str.strip, desc_exclude.split(',')))))
            else:
                study_desc_exc = None
            if desc_include:
                study_desc_inc = list(map(str.lower, list(map(str.strip, desc_include.split(',')))))
            else:
                study_desc_inc = None
            if stationname_exclude:
                stationname_exc = list(map(str.lower, list(map(str.strip, stationname_exclude.split(',')))))
            else:
                stationname_exc = None
            if stationname_include:
                stationname_inc = list(map(str.lower, list(map(str.strip, stationname_include.split(',')))))
            else:
                stationname_inc = None

            filters = {
                'stationname_inc': stationname_inc,
                'stationname_exc': stationname_exc,
                'study_desc_inc': study_desc_inc,
                'study_desc_exc': study_desc_exc,
            }

            qrscu.delay(qr_scp_pk=rh_pk, store_scp_pk=store_pk, query_id=query_id, date_from=date_from,
                               date_until=date_until, modalities=modalities, inc_sr=inc_sr,
                               remove_duplicates=remove_duplicates, filters=filters,
                               get_toshiba_images=get_toshiba_images, get_empty_sr=get_empty_sr,
                               )

            resp = {}
            resp['message'] = u'Request created'
            resp['status'] = u'not complete'
            resp['queryID'] = query_id

            return HttpResponse(json.dumps(resp), content_type='application/json')
        else:
            print(u"Bother, form wasn't valid")
            errors = form.errors
            print(errors)
            print(form)

            # Need to find a way to deal with this event
#            render_to_response('remapp/dicomqr.html', {'form': form}, context_instance=RequestContext(request))
            resp = {}
            resp['message'] = errors
            resp['status'] = 'not complete'

            admin = {'openremversion': __version__, 'docsversion': __docs_version__}

            for group in request.user.groups.all():
                admin[group.name] = True

            return render(request, 'remapp/dicomqr.html', {'form': form, 'admin': admin})


@login_required
def dicom_qr_page(request, *args, **kwargs):
    """View for DICOM Query Retrieve page"""

    if not request.user.groups.filter(name="importqrgroup"):
        messages.error(request, u"You are not in the importqrgroup - please contact your administrator")
        return redirect(reverse_lazy('home'))

    form = DicomQueryForm

    store_nodes = DicomStoreSCP.objects.all()
    qr_nodes = DicomRemoteQR.objects.all()

    admin = {'openremversion': __version__, 'docsversion': __docs_version__}

    for group in request.user.groups.all():
        admin[group.name] = True

    return render(request, 'remapp/dicomqr.html',
                  {'form': form, 'admin': admin, 'qr_nodes': qr_nodes, 'store_nodes': store_nodes},)


@csrf_exempt
@login_required
def r_start(request):
    """View to trigger move following successful query"""
    resp = {}
    data = request.POST
    query_id = data.get('queryID')
    resp['queryID'] = query_id

    movescu.delay(query_id)

    return HttpResponse(json.dumps(resp), content_type='application/json')


@csrf_exempt
def r_update(request):
    """View to update progress of QR move (retrieval)"""

    resp = {}
    data = request.POST
    query_id = data.get('queryID')
    resp['queryID'] = query_id
    try:
        query = DicomQuery.objects.get(query_id=query_id)
    except ObjectDoesNotExist:
        resp['status'] = 'not complete'
        resp['message'] = f'<h4>Move request {query_id} not yet started</h4>'
        resp['subops'] = ''
        return HttpResponse(json.dumps(resp), content_type='application/json')

    resp['subops'] = f'<h4>Cumulative Sub-operations for move request:</h4>' \
                     f'<table class="table">' \
                     f'<tr><th>Completed</th><th>Failed</th><th>Warnings</th></tr>' \
                     f'<tr>' \
                     f'<td>{query.move_completed_sub_ops}</td>' \
                     f'<td>{query.move_failed_sub_ops}</td>' \
                     f'<td>{query.move_warning_sub_ops}</td>' \
                     f'</tr>' \
                     f'</table>'
    # query.move_summary = f'Cumulative Sub-operations for move request: Completed {query.move_completed_sub_ops},' \
    #                      f'Failed {query.move_failed_sub_ops}, Warnings {query.move_warning_sub_ops}.'
    query.save()

    if query.failed:
        resp['status'] = 'failed'
        resp['message'] = f'<h4>Move request failed</h4> {query.message}'
        query.move_summary = f'Move request failed: {query.message}'
        query.save()
        return HttpResponse(json.dumps(resp), content_type='application/json')

    if not query.move_complete:
        resp['status'] = 'not complete'
        resp['message'] = '<h4>{0}</h4>'.format(query.move_summary)
    else:
        resp['status'] = u'move complete'
        resp['message'] = u'<h4>Move request complete</h4>'

    return HttpResponse(json.dumps(resp), content_type='application/json')


def get_qr_status(request):
    """View to get query-retrieve node status for query page"""

    data = request.POST
    echo_response = echoscu(scp_pk=data.get('node'), qr_scp=True)
    if echo_response == "Success":
        status = u"<span class='glyphicon glyphicon-ok' aria-hidden='true'></span>" \
                             u"<span class='sr-only'>OK:</span> responding to DICOM echo"
    else:
        status = "<span class='glyphicon glyphicon-exclamation-sign' aria-hidden='true'></span>" \
                             "<span class='sr-only'>Error:</span> {0}".format(echo_response)
    return HttpResponse(json.dumps(status), content_type='application/json')


def get_store_status(request):
    """View to get store node status for query page"""

    data = request.POST
    echo_response = echoscu(scp_pk=data.get('node'), store_scp=True)
    if echo_response == "Success":
        status = u"<span class='glyphicon glyphicon-ok' aria-hidden='true'></span>" \
                             u"<span class='sr-only'>OK:</span> responding to DICOM echo"
    else:
        status = "<span class='glyphicon glyphicon-exclamation-sign' aria-hidden='true'></span>" \
                             "<span class='sr-only'>Error:</span> {0}".format(echo_response)
    return HttpResponse(json.dumps(status), content_type='application/json')


@login_required
def dicom_summary(request):
    """Displays current DICOM configuration
    """

    try:
        del_settings = DicomDeleteSettings.objects.get()
    except ObjectDoesNotExist:
        DicomDeleteSettings.objects.create()
        del_settings = DicomDeleteSettings.objects.get()

    store = DicomStoreSCP.objects.all()
    remoteqr = DicomRemoteQR.objects.all()

    admin = _create_admin_dict(request)

    # Render list page with the documents and the form
    return render(request,
        'remapp/dicomsummary.html',
        {'store': store, 'remoteqr': remoteqr, 'admin': admin, 'del_settings': del_settings},
    )


class DicomStoreCreate(CreateView):  # pylint: disable=unused-variable
    """CreateView to add details of a DICOM Store to the database

    """

    model = DicomStoreSCP
    form_class = DicomStoreForm

    def get_context_data(self, **context):
        context = super(DicomStoreCreate, self).get_context_data(**context)
        admin = {'openremversion': __version__, 'docsversion': __docs_version__}
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context['admin'] = admin
        return context


class DicomStoreUpdate(UpdateView):  # pylint: disable=unused-variable
    """UpdateView to update details of a DICOM store in the database

    """

    model = DicomStoreSCP
    form_class = DicomStoreForm

    def get_context_data(self, **context):
        context = super(DicomStoreUpdate, self).get_context_data(**context)
        admin = {'openremversion': __version__, 'docsversion': __docs_version__}
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context['admin'] = admin
        return context


class DicomStoreDelete(DeleteView):  # pylint: disable=unused-variable
    """DeleteView to delete DICOM store information from the database

    """

    model = DicomStoreSCP
    success_url = reverse_lazy('dicom_summary')

    def get_context_data(self, **context):
        context[self.context_object_name] = self.object
        admin = {'openremversion': __version__, 'docsversion': __docs_version__}
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context['admin'] = admin
        return context


class DicomQRCreate(CreateView):  # pylint: disable=unused-variable
    """CreateView to add details of a DICOM query-retrieve node

    """

    model = DicomRemoteQR
    form_class = DicomQRForm

    def get_context_data(self, **context):
        context = super(DicomQRCreate, self).get_context_data(**context)
        admin = {'openremversion': __version__, 'docsversion': __docs_version__}
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context['admin'] = admin
        return context


class DicomQRUpdate(UpdateView):  # pylint: disable=unused-variable
    """UpdateView to update details of a DICOM query-retrieve node

    """

    model = DicomRemoteQR
    form_class = DicomQRForm

    def get_context_data(self, **context):
        context = super(DicomQRUpdate, self).get_context_data(**context)
        admin = {'openremversion': __version__, 'docsversion': __docs_version__}
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context['admin'] = admin
        return context


class DicomQRDelete(DeleteView):  # pylint: disable=unused-variable
    """DeleteView to delete details of a DICOM query-retrieve node

    """

    model = DicomRemoteQR
    success_url = reverse_lazy('dicom_summary')

    def get_context_data(self, **context):
        context[self.context_object_name] = self.object
        admin = {'openremversion': __version__, 'docsversion': __docs_version__}
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context['admin'] = admin
        return context

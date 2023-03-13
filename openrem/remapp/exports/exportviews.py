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
..  module:: exportviews.py
    :synopsis: Module to render appropriate content according to request, specific to the exports.

..  moduleauthor:: Ed McDonagh

"""

# Following two lines added so that sphinx autodocumentation works.
import os

os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"

import logging
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import render
from django.urls import reverse_lazy
import remapp
from openrem.remapp.tools.background import (  # pylint: disable=wrong-import-position
    run_in_background,
    terminate_background,
    get_queued_tasks,
    remove_task_from_queue,
)

logger = logging.getLogger(__name__)


def include_pid(request, name, pat_id):
    """
    Check if user is allowed to export PID, then check if they have asked to.
    :param request: request so we can determine the user and therefore groups
    :param name: string, 0 or 1 from URL indicating if names should be exported
    :param pat_id: string, 0 or 1 from URL indicating if patient ID should be exported
    :return: dict, with pidgroup, include_names and include_pat_id as bools
    """

    pid = bool(request.user.groups.filter(name="pidgroup"))

    include_names = False
    include_pat_id = False
    if pid:
        try:
            if int(name):  # Will be unicode from URL
                include_names = True
        except ValueError:  # If anything else comes in, just don't export that column
            pass
        try:
            if int(pat_id):
                include_pat_id = True
        except ValueError:
            pass

    return {
        "pidgroup": pid,
        "include_names": include_names,
        "include_pat_id": include_pat_id,
    }


@csrf_exempt
@login_required
def ctcsv1(request, name=None, pat_id=None):
    """
    View to launch  task to export CT studies to csv file

    :param request: Contains the database filtering parameters. Also used to get user group.
    :param name: string, 0 or 1 from URL indicating if names should be exported
    :param pat_id: string, 0 or 1 from URL indicating if patient ID should be exported
    :type request: GET
    """
    from django.shortcuts import redirect
    from remapp.exports.ct_export import ct_csv

    pid = include_pid(request, name, pat_id)

    if request.user.groups.filter(name="exportgroup"):
        # The 'ct_acquisition_type' filter values must be passed to ct_csv as a list, so convert the GET to a dict and
        # then update the 'ct_acquisition_type' value with a list.
        filter_dict = request.GET.dict()
        if "ct_acquisition_type" in filter_dict:
            filter_dict["ct_acquisition_type"] = request.GET.getlist(
                "ct_acquisition_type"
            )

        job = run_in_background(
            ct_csv,
            "export_ct",
            filter_dict,
            pid["pidgroup"],
            pid["include_names"],
            pid["include_pat_id"],
            request.user.id,
        )
        logger.debug(f"Export CT to CSV job is {job.id}")
    return redirect(reverse_lazy("export"))


@csrf_exempt
@login_required
def ctxlsx1(request, name=None, pat_id=None):
    """
    View to launch  task to export CT studies to xlsx file

    :param request: Contains the database filtering parameters. Also used to get user group.
    :param name: string, 0 or 1 from URL indicating if names should be exported
    :param pat_id: string, 0 or 1 from URL indicating if patient ID should be exported
    :type request: GET
    """
    from django.shortcuts import redirect
    from remapp.exports.ct_export import ctxlsx

    pid = include_pid(request, name, pat_id)

    if request.user.groups.filter(name="exportgroup"):
        # The 'ct_acquisition_type' filter values must be passed to ctxlsx as a list, so convert the GET to a dict and
        # then update the 'ct_acquisition_type' value with a list.
        filter_dict = request.GET.dict()
        if "ct_acquisition_type" in filter_dict:
            filter_dict["ct_acquisition_type"] = request.GET.getlist(
                "ct_acquisition_type"
            )

        job = run_in_background(
            ctxlsx,
            "export_ct",
            filter_dict,
            pid["pidgroup"],
            pid["include_names"],
            pid["include_pat_id"],
            request.user.id,
        )
        logger.debug("Export CT to XLSX job is {0}".format(job.id))

    return redirect(reverse_lazy("export"))


@csrf_exempt
@login_required
def ct_xlsx_phe2019(request):
    """
    View to launch  task to export CT studies to xlsx file in PHE 2019 CT survey format

    :param request: Contains the database filtering parameters and user details.
    """
    from django.shortcuts import redirect
    from remapp.exports.ct_export import ct_phe_2019

    if request.user.groups.filter(name="exportgroup"):
        job = run_in_background(ct_phe_2019, "export_ct", request.GET, request.user.id)
        logger.debug("Export CT to XLSX job is {0}".format(job.id))
    return redirect(reverse_lazy("export"))


@csrf_exempt
@login_required
def nmcsv1(request, name=None, pat_id=None):
    """
    View to launch task to export NM studies to csv file

    :param request: Contains the database filtering parameters. Also used to get user group.
    :param name: string, 0 or 1 from URL indicating if names should be exported
    :param pat_id: string, 0 or 1 from URL indicating if patient ID should be exported
    :type request: GET
    """
    from django.shortcuts import redirect
    from remapp.exports.nm_export import exportNM2csv

    pid = include_pid(request, name, pat_id)

    if request.user.groups.filter(name="exportgroup"):
        job = run_in_background(
            exportNM2csv,
            "export_nm",
            request.GET,
            pid["pidgroup"],
            pid["include_names"],
            pid["include_pat_id"],
            request.user.id,
        )
        logger.debug(f"Export NM to CSV job is {job.id}")

    return redirect(reverse_lazy("export"))


@csrf_exempt
@login_required
def nmxlsx1(request, name=None, pat_id=None):
    """
    View to launch celery task to export NM studies to excel file

    :param request: Contains the database filtering parameters. Also used to get user group.
    :param name: string, 0 or 1 from URL indicating if names should be exported
    :param pat_id: string, 0 or 1 from URL indicating if patient ID should be exported
    :type request: GET
    """
    from django.shortcuts import redirect
    from remapp.exports.nm_export import exportNM2excel

    pid = include_pid(request, name, pat_id)

    if request.user.groups.filter(name="exportgroup"):
        job = run_in_background(
            exportNM2excel,
            "export_nm",
            request.GET,
            pid["pidgroup"],
            pid["include_names"],
            pid["include_pat_id"],
            request.user.id,
        )
        logger.debug(f"Exprt NM to Excel job is {job.id}")

    return redirect(reverse_lazy("export"))


@csrf_exempt
@login_required
def dxcsv1(request, name=None, pat_id=None):
    """
    View to launch  task to export DX and CR studies to csv file

    :param request: Contains the database filtering parameters. Also used to get user group.
    :param name: string, 0 or 1 from URL indicating if names should be exported
    :param pat_id: string, 0 or 1 from URL indicating if patient ID should be exported
    :type request: GET
    """
    from django.shortcuts import redirect
    from remapp.exports.dx_export import exportDX2excel

    pid = include_pid(request, name, pat_id)

    if request.user.groups.filter(name="exportgroup"):
        job = run_in_background(
            exportDX2excel,
            "export_dx",
            request.GET,
            pid["pidgroup"],
            pid["include_names"],
            pid["include_pat_id"],
            request.user.id,
        )
        logger.debug("Export DX to CSV job is {0}".format(job.id))

    return redirect(reverse_lazy("export"))


@csrf_exempt
@login_required
def dxxlsx1(request, name=None, pat_id=None):
    """
    View to launch  task to export DX and CR studies to xlsx file

    :param request: Contains the database filtering parameters. Also used to get user group.
    :param name: string, 0 or 1 from URL indicating if names should be exported
    :param pat_id: string, 0 or 1 from URL indicating if patient ID should be exported
    :type request: GET
    """
    from django.shortcuts import redirect
    from remapp.exports.dx_export import dxxlsx

    pid = include_pid(request, name, pat_id)

    if request.user.groups.filter(name="exportgroup"):
        job = run_in_background(
            dxxlsx,
            "export_dx",
            request.GET,
            pid["pidgroup"],
            pid["include_names"],
            pid["include_pat_id"],
            request.user.id,
        )
        logger.debug("Export DX to XLSX job is {0}".format(job.id))

    return redirect(reverse_lazy("export"))


@csrf_exempt
@login_required
def dx_xlsx_phe2019(request, export_type=None):
    """
    View to launch  task to export DX studies to xlsx file in PHE 2019 DX survey format

    :param request: Contains the database filtering parameters and user details.
    :param export_type: string, 'projection' or 'exam'
    """
    import urllib
    from django.db.models import Max
    from django.shortcuts import redirect
    from remapp.exports.dx_export import dx_phe_2019
    from remapp.interface.mod_filters import dx_acq_filter

    if request.user.groups.filter(name="exportgroup"):
        if export_type in ("exam", "projection"):
            bespoke = False
            exams = dx_acq_filter(request.GET, pid=False).qs
            if not exams.count():
                messages.error(request, "No studies in export, nothing to do!")
                return redirect(
                    "{0}?{1}".format(
                        reverse_lazy("dx_summary_list_filter"),
                        urllib.urlencode(request.GET),
                    )
                )
            max_events_dict = exams.aggregate(Max("number_of_events"))
            max_events = max_events_dict["number_of_events__max"]
            if "projection" in export_type:
                if max_events > 1:
                    messages.warning(
                        request,
                        "PHE 2019 DX Projection export is expecting one exposure per study - "
                        "some studies selected have more than one. Only the first exposure will "
                        "be considered.",
                    )
                else:
                    messages.info(
                        request, "PHE 2019 DX single projection export started."
                    )
                job = run_in_background(
                    dx_phe_2019,
                    "export_dx",
                    request.GET,
                    request.user.id,
                    projection=True,
                )
                logger.debug(
                    "Export PHE 2019 DX survey format job is {0}".format(job.id)
                )
                return redirect(reverse_lazy("export"))
            elif "exam" in export_type:
                if max_events > 6:
                    bespoke = True
                    if max_events > 20:
                        messages.warning(
                            request,
                            "PHE 2019 DX Study sheets expect a maximum of six projections. You "
                            "need to request a bespoke workbook from PHE. This export has a "
                            "maximum of {0} projections, but only the first 20 will be included "
                            "in the main columns of the bespoke worksheet.".format(
                                max_events
                            ),
                        )
                    else:
                        messages.warning(
                            request,
                            "PHE 2019 DX Study sheets expect a maximum of six projections. This "
                            "export has a maximum of {0} projections so you will need to request"
                            " a bespoke workbook from PHE. This has space for 20 "
                            "projections.".format(max_events),
                        )
                else:
                    messages.info(request, "PHE 2019 DX Study export started.")
                job = run_in_background(
                    dx_phe_2019,
                    "export_dx",
                    request.GET,
                    request.user.id,
                    projection=False,
                    bespoke=bespoke,
                )
                logger.debug(
                    "Export PHE 2019 DX survey format job is {0}".format(job.id)
                )
                return redirect(reverse_lazy("export"))
        else:
            messages.error(request, "Malformed export URL {0}".format(type))
            return redirect(
                "{0}?{1}".format(
                    reverse_lazy("dx_summary_list_filter"),
                    urllib.urlencode(request.GET),
                )
            )
    else:
        messages.error(request, "Only users in the Export group can launch exports")
        return redirect(
            "{0}?{1}".format(
                reverse_lazy("dx_summary_list_filter"), urllib.urlencode(request.GET)
            )
        )


@csrf_exempt
@login_required
def flcsv1(request, name=None, pat_id=None):
    """
    View to launch  task to export fluoroscopy studies to csv file

    :param request: Contains the database filtering parameters. Also used to get user group.
    :param name: string, 0 or 1 from URL indicating if names should be exported
    :param pat_id: string, 0 or 1 from URL indicating if patient ID should be exported
    :type request: GET
    """
    from django.shortcuts import redirect
    from remapp.exports.rf_export import exportFL2excel

    pid = include_pid(request, name, pat_id)

    if request.user.groups.filter(name="exportgroup"):
        job = run_in_background(
            exportFL2excel,
            "export_fl",
            request.GET,
            pid["pidgroup"],
            pid["include_names"],
            pid["include_pat_id"],
            request.user.id,
        )
        logger.debug("Export Fluoro to CSV job is {0}".format(job.id))

    return redirect(reverse_lazy("export"))


@csrf_exempt
@login_required
def rfxlsx1(request, name=None, pat_id=None):
    """
    View to launch  task to export fluoroscopy studies to xlsx file

    :param request: Contains the database filtering parameters. Also used to get user group.
    :param name: string, 0 or 1 from URL indicating if names should be exported
    :param pat_id: string, 0 or 1 from URL indicating if patient ID should be exported
    :type request: GET
    """
    from django.shortcuts import redirect
    from remapp.exports.rf_export import rfxlsx

    pid = include_pid(request, name, pat_id)

    if request.user.groups.filter(name="exportgroup"):
        job = run_in_background(
            rfxlsx,
            "export_rf",
            request.GET,
            pid["pidgroup"],
            pid["include_names"],
            pid["include_pat_id"],
            request.user.id,
        )
        logger.debug("Export Fluoro to XLSX job is {0}".format(job.id))

    return redirect(reverse_lazy("export"))


@csrf_exempt
@login_required
def rfopenskin(request, pk):
    """
    Create csv export suitable for import to standalone openSkin
    :param request: request object
    :param pk: primary key of study in GeneralStudyModuleAttr table
    """
    from django.shortcuts import redirect, get_object_or_404
    from remapp.exports.rf_export import rfopenskin
    from remapp.models import GeneralStudyModuleAttr

    export = get_object_or_404(GeneralStudyModuleAttr, pk=pk)

    if request.user.groups.filter(name="exportgroup"):
        job = run_in_background(rfopenskin, "export_rf", export.pk)
        logger.debug("Export Fluoro to openSkin CSV job is {0}".format(job.id))

    return redirect(reverse_lazy("export"))


@csrf_exempt
@login_required
def rf_xlsx_phe2019(request):
    """
    View to launch  task to export fluoro studies to xlsx file in PHE 2019 IR/fluoro survey format

    :param request: Contains the database filtering parameters and user details.
    """
    import urllib
    from django.shortcuts import redirect
    from remapp.exports.rf_export import rf_phe_2019

    if request.user.groups.filter(name="exportgroup"):
        job = run_in_background(rf_phe_2019, "export_rf", request.GET, request.user.id)
        logger.debug(
            "Export PHE 2019 IR/fluoro survey format job is {0}.".format(job.id)
        )
        return redirect(reverse_lazy("export"))
    else:
        messages.error(request, "Only users in the Export group can launch exports")
        return redirect(
            "{0}?{1}".format(
                reverse_lazy("rf_summary_list_filter"), urllib.urlencode(request.GET)
            )
        )


@csrf_exempt
@login_required
def mgcsv1(request, name=None, pat_id=None):
    """
    Launches export of mammo data to CSV
    :param request: Contains the database filtering parameters. Also used to get user group.
    :param name: string, 0 or 1 from URL indicating if names should be exported
    :param pat_id: string, 0 or 1 from URL indicating if patient ID should be exported
    :return:
    """
    from django.shortcuts import redirect
    from remapp.exports.mg_export import exportMG2excel

    pid = include_pid(request, name, pat_id)

    if request.user.groups.filter(name="exportgroup"):
        job = run_in_background(
            exportMG2excel,
            "export_mg",
            request.GET,
            pid["pidgroup"],
            pid["include_names"],
            pid["include_pat_id"],
            request.user.id,
        )
        logger.debug("Export MG to CSV job is {0}".format(job.id))

    return redirect(reverse_lazy("export"))


@csrf_exempt
@login_required
def mgxlsx1(request, name=None, pat_id=None):
    """
    Launches export of mammo data to xlsx
    :param request: Contains the database filtering parameters. Also used to get user group.
    :param name: string, 0 or 1 from URL indicating if names should be exported
    :param pat_id: string, 0 or 1 from URL indicating if patient ID should be exported
    :return:
    """
    from django.shortcuts import redirect
    from remapp.exports.mg_export import exportMG2excel

    pid = include_pid(request, name, pat_id)

    if request.user.groups.filter(name="exportgroup"):
        job = run_in_background(
            exportMG2excel,
            "export_mg",
            request.GET,
            pid=pid["pidgroup"],
            name=pid["include_names"],
            patid=pid["include_pat_id"],
            user=request.user.id,
            xlsx=True,
        )
        logger.debug("Export MG to xlsx job is {0}".format(job.id))

    return redirect(reverse_lazy("export"))


@csrf_exempt
@login_required
def mgnhsbsp(request):
    """
    View to launch  task to export mammography studies to csv file using a NHSBSP template

    :param request: Contains the database filtering parameters. Also used to get user group.
    :type request: GET
    """
    from django.shortcuts import redirect
    from remapp.exports.mg_csv_nhsbsp import mg_csv_nhsbsp

    if request.user.groups.filter(name="exportgroup"):
        job = run_in_background(
            mg_csv_nhsbsp, "export_mg", request.GET, request.user.id
        )
        logger.debug("Export MG to CSV NHSBSP job is {0}".format(job.id))

    return redirect(reverse_lazy("export"))


@csrf_exempt
@login_required
def export(request):
    """
    View to list current and completed exports to track progress, download and delete

    :param request: Used to get user group.
    """
    from remapp.models import Exports

    try:
        complete = Exports.objects.filter(status__contains="COMPLETE").order_by(
            "-export_date"
        )
        latest_complete_pk = complete[0].pk
    except IndexError:
        complete = None
        latest_complete_pk = 0

    admin = {
        "openremversion": remapp.__version__,
        "docsversion": remapp.__docs_version__,
    }
    for group in request.user.groups.all():
        admin[group.name] = True
    template = "remapp/exports.html"

    return render(
        request,
        template,
        {
            "admin": admin,
            "latest_complete_pk": latest_complete_pk,
            "complete": complete,
        },
    )


@login_required
def download(request, task_id):
    """
    View to handle downloads of files from the server

    Originally used for download of the export spreadsheets, now also used
    for downloading the patient size import logfiles.

    :param request: Used to get user group.
    :param task_id: ID of the export or logfile

    """
    import mimetypes
    from django.core.exceptions import ObjectDoesNotExist
    from wsgiref.util import FileWrapper
    from django.utils.encoding import smart_str
    from django.shortcuts import redirect
    from django.conf import settings
    from remapp.models import Exports

    exportperm = False
    pidperm = False
    if request.user.groups.filter(name="exportgroup"):
        exportperm = True
    if request.user.groups.filter(name="pidgroup"):
        pidperm = True
    try:
        exp = Exports.objects.get(task_id__exact=task_id)
    except ObjectDoesNotExist:
        messages.error(request, "Can't match the task ID, download aborted")
        return redirect(reverse_lazy("export"))

    if not exportperm:
        messages.error(request, "You don't have permission to download exported data")
        return redirect(reverse_lazy("export"))

    if exp.includes_pid and not pidperm:
        messages.error(
            request,
            "You don't have permission to download export data that includes patient identifiable information",
        )
        return redirect(reverse_lazy("export"))

    file_path = os.path.join(settings.MEDIA_ROOT, exp.filename.name)
    file_wrapper = FileWrapper(open(file_path, mode="rb"))
    file_mimetype = mimetypes.guess_type(file_path)
    response = HttpResponse(file_wrapper, content_type=file_mimetype)
    response["X-Sendfile"] = file_path
    response["Content-Length"] = os.stat(file_path).st_size
    response["Content-Disposition"] = "attachment; filename=%s" % smart_str(
        exp.filename
    )
    return response


@csrf_exempt
@login_required
def deletefile(request):
    """
    View to delete export files from the server

    :param request: Contains the task ID
    :type request: POST
    """
    import sys
    from django.http import HttpResponseRedirect
    from django.urls import reverse
    from remapp.models import Exports

    for task in request.POST:
        exports = Exports.objects.filter(task_id__exact=request.POST[task])
        for export_object in exports:
            try:
                export_object.filename.delete()
                export_object.delete()
                messages.success(
                    request, "Export file and database entry deleted successfully."
                )
            except OSError as e:
                messages.error(
                    request,
                    "Export file delete failed - please contact an administrator. Error({0}): {1}".format(
                        e.errno, e.strerror
                    ),
                )
            except Exception:
                messages.error(
                    request,
                    "Unexpected error - please contact an administrator: {0}".format(
                        sys.exc_info()[0]
                    ),
                )

    return HttpResponseRedirect(reverse(export))


@login_required
def export_abort(request, pk):
    """
    View to abort current export job

    :param request: Contains the task primary key
    :type request: POST
    """
    from django.http import HttpResponseRedirect
    from django.shortcuts import get_object_or_404
    from remapp.models import Exports, BackgroundTask

    export_task = get_object_or_404(Exports, pk=pk)
    task = get_object_or_404(BackgroundTask, uuid=export_task.task_id)

    if request.user.groups.filter(name="exportgroup"):
        terminate_background(task)
        export_task.delete()
        logger.info(
            "Export task {0} terminated from the Exports interface".format(
                export_task.task_id
            )
        )

    return HttpResponseRedirect(reverse_lazy("export"))


@login_required
def export_remove(request, task_id=None):
    """
    Function to remove export task from queue

    :param request: Contains the task primary key
    :param task_id: UUID of task in question
    :type request: POST
    """
    from django.http import HttpResponseRedirect

    if task_id and request.user.groups.filter(name="exportgroup"):
        remove_task_from_queue(task_id)
        logger.info("Export task {0} removed from queue".format(task_id))

    return HttpResponseRedirect(reverse_lazy("export"))


@csrf_exempt
@login_required
def update_queue(request):
    """
    AJAX function to return queued exports

    :param request: Request object
    :return: HTML table of active exports
    """
    template = "remapp/exports-queue.html"
    if request.is_ajax():
        queued_export_tasks = get_queued_tasks(task_type="export")
        return render(request, template, {"queued": queued_export_tasks})

    return render(request, template)


@csrf_exempt
@login_required
def update_active(request):
    """
    AJAX function to return active exports

    :param request: Request object
    :return: HTML table of active exports
    """
    from remapp.models import Exports

    if request.is_ajax():
        current_export_tasks = Exports.objects.filter(
            status__contains="CURRENT"
        ).order_by("-export_date")
        template = "remapp/exports-active.html"

        return render(request, template, {"current": current_export_tasks})


@csrf_exempt
@login_required
def update_error(request):
    """
    AJAX function to return exports in error state

    :param request: Request object
    :return: HTML table of exports in error state
    """
    from remapp.models import Exports

    if request.is_ajax():
        error_export_tasks = Exports.objects.filter(status__contains="ERROR").order_by(
            "-export_date"
        )
        template = "remapp/exports-error.html"

        return render(request, template, {"errors": error_export_tasks})


@csrf_exempt
@login_required
def update_complete(request):
    """
    AJAX function to return recently completed exports

    :param request: Request object, including pk of latest complete export at initial page load
    :return: HTML table of completed exports
    """
    from remapp.models import Exports

    if request.is_ajax():
        data = request.POST
        latest_complete_pk = data.get("latest_complete_pk")
        in_pid_group = data.get("in_pid_group")
        complete_export_tasks = Exports.objects.filter(
            status__contains="COMPLETE"
        ).filter(pk__gt=latest_complete_pk)
        template = "remapp/exports-complete.html"

        return render(
            request,
            template,
            {"complete": complete_export_tasks, "in_pid_group": in_pid_group},
        )

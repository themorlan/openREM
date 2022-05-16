#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2020  The Royal Marsden NHS Foundation Trust
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
..  module:: import_views.
    :synopsis: Views to enable imports to be triggered via a URL call from the Orthanc docker container

..  moduleauthor:: Ed McDonagh
"""

import csv
import logging
import os
import sys

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import redirect, render, get_object_or_404
from django.conf import settings
from django.urls import reverse, reverse_lazy
from django.views.decorators.csrf import csrf_exempt
from openrem.remapp.tools.background import (
    run_in_background,
    run_in_background_with_limits,
    terminate_background,
)

from remapp.models import BackgroundTask, SizeUpload
from .rdsr import rdsr
from .dx import dx
from .mam import mam
from .nm_image import nm_image
from .ct_philips import ct_philips
from .ct_toshiba import ct_toshiba
from .. import __docs_version__, __version__
from ..forms import SizeHeadersForm, SizeUploadForm

logger = logging.getLogger(__name__)


@csrf_exempt
def import_from_docker(request):
    """
    View to consume the local path of an object ot import and pass to import scripts. To be used by Orthanc Docker
    container.

    :param request: Request object containing local path and script name in POST data
    :return: Text detailing what was run
    """
    data = request.POST
    dicom_path = data.get("dicom_path")
    import_type = data.get("import_type")

    # This violates the rules for running background processes at the moment (may block),
    # but at least it ensures correctness
    if dicom_path:
        if import_type == "rdsr":
            run_in_background_with_limits(
                rdsr, "import_rdsr", 0, {"import_rdsr": 1}, dicom_path
            )
            return_type = "RDSR"
        elif import_type == "dx":
            run_in_background_with_limits(
                dx,
                "import_dx",
                0,
                {"import_dx": 1},
                dicom_path,
            )
            return_type = "DX"
        elif import_type == "nm":
            run_in_background_with_limits(
                nm_image,
                "import_nm",
                0,
                {"import_nm": 1},
                dicom_path,
            )
            return_type = "NM image"
        elif import_type == "mam":
            run_in_background_with_limits(
                mam,
                "import_mam",
                0,
                {"import_mam": 1},
                dicom_path,
            )
            return_type = "Mammography"
        elif import_type == "ct_philips":
            run_in_background_with_limits(
                ct_philips,
                "import_ct_philips",
                0,
                {"import_ct_philips": 1},
                dicom_path,
            )
            return_type = "CT Philips"
        elif import_type == "ct_toshiba":
            run_in_background_with_limits(
                ct_toshiba,
                "import_ct_toshiba",
                0,
                {"import_ct_toshiba": 1},
                dicom_path,
            )
            return HttpResponse(f"{dicom_path} passed to CT Toshiba import")
        else:
            return HttpResponse("Import script name not recognised")
        return HttpResponse(f"{return_type} import run on {dicom_path}")
    return HttpResponse("No dicom_path, import not carried out")


@login_required
def size_upload(request):
    """
    Form for upload of csv file containing patient size information. POST request passes database entry ID to
    size_process

    :param request: If POST, contains the file upload information
    """

    if not request.user.groups.filter(name="importsizegroup"):
        messages.error(
            request,
            "You are not in the import size group - please contact your administrator",
        )
        return redirect(reverse_lazy("home"))

    # Handle file upload
    if request.method == "POST" and request.user.groups.filter(name="importsizegroup"):
        form = SizeUploadForm(request.POST, request.FILES)
        if form.is_valid():
            newcsv = SizeUpload(sizefile=request.FILES["sizefile"])
            newcsv.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(
                reverse_lazy("size_process", kwargs={"pk": newcsv.id})
            )

    else:
        form = SizeUploadForm()  # A empty, unbound form

    admin = {"openremversion": __version__, "docsversion": __docs_version__}

    for group in request.user.groups.all():
        admin[group.name] = True

    # Render list page with the documents and the form
    return render(request, "remapp/sizeupload.html", {"form": form, "admin": admin})


@login_required
def size_process(request, *args, **kwargs):
    """
    Form for csv column header patient size imports through the web interface. POST request launches import task

    :param request: If POST, contains the field header information
    :param pk: From URL, identifies database patient size import record
    :type pk: kwarg
    """
    from .ptsizecsv2db import websizeimport

    if not request.user.groups.filter(name="importsizegroup"):
        messages.error(
            request,
            "You are not in the import size group - please contact your administrator",
        )
        return redirect(reverse_lazy("home"))

    if request.method == "POST":

        items_in_post = len(list(request.POST.values()))
        unique_items_in_post = len(set(request.POST.values()))

        if items_in_post == unique_items_in_post:
            csvrecord = SizeUpload.objects.all().filter(id__exact=kwargs["pk"])[0]

            if not csvrecord.sizefile:
                messages.error(
                    request,
                    "File to be processed doesn't exist. Do you wish to try again?",
                )
                return HttpResponseRedirect(reverse_lazy("size_upload"))

            csvrecord.height_field = request.POST["height_field"]
            csvrecord.weight_field = request.POST["weight_field"]
            csvrecord.id_field = request.POST["id_field"]
            csvrecord.id_type = request.POST["id_type"]
            if "overwrite" in request.POST:
                csvrecord.overwrite = True
            csvrecord.save()

            run_in_background(
                websizeimport,
                "import_size",
                csv_pk=kwargs["pk"],
            )

            return HttpResponseRedirect(reverse_lazy("size_imports"))

        else:
            messages.error(
                request,
                "Duplicate column header selection. Each field must have a different header.",
            )
            return HttpResponseRedirect(
                reverse_lazy("size_process", kwargs={"pk": kwargs["pk"]})
            )

    else:

        csvrecord = SizeUpload.objects.all().filter(id__exact=kwargs["pk"])
        with open(
            os.path.join(settings.MEDIA_ROOT, csvrecord[0].sizefile.name), "r"
        ) as csvfile:
            try:
                # dialect = csv.Sniffer().sniff(csvfile.read(1024))
                csvfile.seek(0)
                if csv.Sniffer().has_header(csvfile.read(1024)):
                    csvfile.seek(0)
                    dataset = csv.DictReader(csvfile)
                    messages.success(request, "CSV file with column headers found.")
                    fieldnames = tuple(zip(dataset.fieldnames, dataset.fieldnames))
                    form = SizeHeadersForm(my_choice=fieldnames)
                else:
                    csvfile.seek(0)
                    messages.error(
                        request,
                        "Doesn't appear to have a header row. First row: {0}. The uploaded "
                        "file has been deleted.".format(next(csvfile)),
                    )
                    csvrecord[0].sizefile.delete()
                    return HttpResponseRedirect(reverse_lazy("size_upload"))
            except csv.Error as csv_error:
                messages.error(
                    request,
                    "Doesn't appear to be a csv file. Error({0}). The uploaded file has been "
                    "deleted.".format(csv_error),
                )
                csvrecord[0].sizefile.delete()
                return HttpResponseRedirect(reverse_lazy("size_upload"))
            except:
                messages.error(
                    request,
                    "Unexpected error - please contact an administrator: {0}.".format(
                        sys.exc_info()[0]
                    ),
                )
                csvrecord[0].sizefile.delete()
                return HttpResponseRedirect(reverse_lazy("size_upload"))

    admin = {"openremversion": __version__, "docsversion": __docs_version__}

    for group in request.user.groups.all():
        admin[group.name] = True

    return render(
        request,
        "remapp/sizeprocess.html",
        {"form": form, "csvid": kwargs["pk"], "admin": admin},
    )


def size_imports(request, *args, **kwargs):
    """
    Lists patient size imports in the web interface

    :param request:
    """

    if not request.user.groups.filter(
        name="importsizegroup"
    ) and not request.user.groups.filter(name="admingroup"):
        messages.error(
            request,
            "You are not in the import size group - please contact your administrator",
        )
        return redirect(reverse_lazy("home"))

    imports = SizeUpload.objects.all().order_by("-import_date")

    current = imports.filter(status__contains="CURRENT")
    complete = imports.filter(status__contains="COMPLETE")
    errors = imports.filter(status__contains="ERROR")

    admin = {"openremversion": __version__, "docsversion": __docs_version__}

    for group in request.user.groups.all():
        admin[group.name] = True

    return render(
        request,
        "remapp/sizeimports.html",
        {"admin": admin, "current": current, "complete": complete, "errors": errors},
    )


@csrf_exempt
@login_required
def size_delete(request):
    """
    Task to delete records of patient size imports through the web interface

    :param request: Contains the task ID
    :type request: POST
    """

    for task in request.POST:
        uploads = SizeUpload.objects.filter(task_id__exact=request.POST[task])
        for upload in uploads:
            try:
                upload.logfile.delete()
                upload.delete()
                messages.success(
                    request, "Export file and database entry deleted successfully."
                )
            except OSError as delete_error:
                messages.error(
                    request,
                    "Export file delete failed - please contact an administrator. Error({0}): {1}".format(
                        delete_error.errno, delete_error.strerror
                    ),
                )
            except:
                messages.error(
                    request,
                    "Unexpected error - please contact an administrator: {0}".format(
                        sys.exc_info()[0]
                    ),
                )

    return HttpResponseRedirect(reverse("size_imports"))


@login_required
def size_abort(request, pk):
    """
    View to abort current patient size imports

    :param pk: Size upload task primary key
    """
    size_import = get_object_or_404(SizeUpload, pk=pk)
    task = get_object_or_404(BackgroundTask, uuid=size_import.task_id)

    if request.user.groups.filter(
        name="importsizegroup"
    ) or request.users.groups.filter(name="admingroup"):

        terminate_background(task)
        size_import.logfile.delete()
        size_import.sizefile.delete()
        size_import.delete()
        logger.info(
            "Size import task {0} terminated from the patient size imports interface".format(
                size_import.task_id
            )
        )
    else:
        messages.error(
            request,
            "Only members of the importsizegroup or admingroup can abort a size import task",
        )

    return HttpResponseRedirect(reverse_lazy("size_imports"))


@login_required
def size_download(request, task_id):
    """
    View to handle downloads of files from the server

    For downloading the patient size import logfiles.

    :param request: Used to get user group.
    :param task_id: Size import task ID.

    """
    import mimetypes
    from django.utils.encoding import smart_str
    from wsgiref.util import FileWrapper

    import_permission = False
    if request.user.groups.filter(name="importsizegroup"):
        import_permission = True
    try:
        export_log = SizeUpload.objects.get(task_id__exact=task_id)
    except ObjectDoesNotExist:
        messages.error(request, "Can't match the task ID, download aborted")
        return redirect(reverse_lazy("size_imports"))

    if not import_permission:
        messages.error(request, "You don't have permission to download import logs")
        return redirect(reverse_lazy("size_imports"))

    file_path = os.path.join(settings.MEDIA_ROOT, export_log.logfile.name)
    file_wrapper = FileWrapper(open(file_path, "rb"))
    file_mimetype = mimetypes.guess_type(file_path)
    response = HttpResponse(file_wrapper, content_type=file_mimetype)
    response["X-Sendfile"] = file_path
    response["Content-Length"] = os.stat(file_path).st_size
    response["Content-Disposition"] = "attachment; filename=%s" % smart_str(
        export_log.logfile
    )
    return response

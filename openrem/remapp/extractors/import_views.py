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

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from .rdsr import rdsr
from .dx import dx
from .mam import mam
from .ct_philips import ct_philips
from .ct_toshiba import ct_toshiba


@csrf_exempt
def import_from_docker(request):
    """View to consume the local path of an object ot import and pass to import scripts. To be used by Orthanc Docker
    container.

    :param request: Request object containing local path and script name in POST data
    :return: Text detailing what was run
    """
    data = request.POST
    dicom_path = data.get('dicom_path')
    import_type = data.get('import_type')
    print(f'In import_from_docker, dicom_path is {dicom_path}, import type is {import_type}')

    if dicom_path:
        if import_type == 'rdsr':
            rdsr(dicom_path)
            return_type = "RDSR"
        elif import_type == 'dx':
            dx(dicom_path)
            return_type = "DX"
        elif import_type == 'mam':
            mam(dicom_path)
            return_type = "Mammography"
        elif import_type == 'ct_philips':
            ct_philips(dicom_path)
            return_type = "CT Philips"
        elif import_type == 'ct_toshiba':
            ct_toshiba(dicom_path)
            return HttpResponse(f'{dicom_path} passed to CT Toshiba import')
        else:
            return HttpResponse('Import script name not recognised')
        return HttpResponse(f"{return_type} import run on {dicom_path}")
    return HttpResponse('No dicom_path, import not carried out')

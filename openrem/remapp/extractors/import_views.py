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

from urllib.parse import  unquote

from django.http import HttpResponseRedirect
from django.urls import reverse_lazy
from django.views.decorators.csrf import csrf_exempt

from .rdsr import rdsr


@csrf_exempt
def import_rdsr(request):
    """

    :param request:
    :return:
    """
    data = request.POST
    dicom_path = data.get('dicom_path')

    if dicom_path:
        dicom_path = unquote(dicom_path)
        rdsr(dicom_path)
    return HttpResponseRedirect(reverse_lazy('home'))
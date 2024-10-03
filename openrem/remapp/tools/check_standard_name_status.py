# This Python file uses the following encoding: utf-8
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
..  module:: check_standard_name_status.
    :synopsis: Method to check if standard name mapping is enabled

..  moduleauthor:: David Platten

"""

from django.core.exceptions import ObjectDoesNotExist
from ..models import StandardNameSettings


def are_standard_names_enabled():
    """Return a Boolean indicating if standard name mapping is enabled."""

    # Obtain the system-level enable_standard_names setting
    try:
        StandardNameSettings.objects.get()
    except ObjectDoesNotExist:
        StandardNameSettings.objects.create()
    enable_standard_names = StandardNameSettings.objects.values_list(
        "enable_standard_names", flat=True
    )[0]
    return enable_standard_names

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

# Following two lines added so that sphinx autodocumentation works.
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'openremproject.settings'

import json
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse

import time
from threading import Thread
from remapp.netdicom.storescp import web_store


class DICOMStoreSCP(Thread):
    def __init__(self):
        self.__running = True
        # self.pk = store_pk
        # super(DICOMStoreSCP, self).__init__()

    def terminate(self):
        self.__running = False

    def run(self, store_pk):
        n = 30
        while self.__running and n > 0:
            web_store(store_pk=store_pk)
            print('T-minus', n)
            n -= 1
            time.sleep(1)


@csrf_exempt
@login_required
def storescp(request, pk):
    from django.shortcuts import redirect

    if request.user.groups.filter(name="exportgroup") or request.user.groups.filter(name="admingroup"):
        t = DICOMStoreSCP(store_pk=pk)
        t.daemon = True
        t.start()
        print "Thread ident is {0}".format(t.ident)

    return redirect('/openrem/admin/dicomsummary/')
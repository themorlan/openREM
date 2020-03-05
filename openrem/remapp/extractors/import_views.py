from urllib.parse import  unquote

from django.http import HttpResponseRedirect
from django.urls import reverse_lazy

from .rdsr import rdsr


def import_rdsr(request):
    if 'dicom_path' in request.GET:
        dicom_path = request.GET.get('dicom_path')
        if dicom_path:
            dicom_path = unquote(dicom_path)
            rdsr(dicom_path)
    return HttpResponseRedirect(reverse_lazy('home'))
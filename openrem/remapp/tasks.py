"""
..  module:: tasks.py
    :synopsis: Module to import all the functions that run as Celery tasks.
"""
from celery import shared_task  # pylint: disable=unused-import

from .exports.mg_export import exportMG2excel  # pylint: disable=unused-import
from .exports.ct_export import ctxlsx, ct_csv  # pylint: disable=unused-import
from .exports.mg_csv_nhsbsp import mg_csv_nhsbsp  # pylint: disable=unused-import
from .exports.dx_export import (
    exportDX2excel,
    dxxlsx,
    dx_phe_2019,
)  # pylint: disable=unused-import
from .exports.rf_export import (
    exportFL2excel,
    rfxlsx,
    rf_phe_2019,
)  # pylint: disable=unused-import
from .extractors.ct_philips import ct_philips  # pylint: disable=unused-import
from .extractors.dx import dx  # pylint: disable=unused-import
from .extractors.mam import mam  # pylint: disable=unused-import
from .extractors.rdsr import rdsr  # pylint: disable=unused-import
from .extractors.ptsizecsv2db import (
    websizeimport,
)  # pylint: disable=unused-import
from .netdicom.qrscu import qrscu, movescu  # pylint: disable=unused-import
from .netdicom.keepalive import keep_alive  # pylint: disable=unused-import
from .tools.make_skin_map import make_skin_map  # pylint: disable=unused-import
from .tools.populate_summary import (
    populate_summary_ct,
)  # pylint: disable=unused-import
from .tools.populate_summary import (
    populate_summary_rf,
)  # pylint: disable=unused-import
from .tools.populate_summary import (
    populate_summary_mg,
)  # pylint: disable=unused-import
from .tools.populate_summary import (
    populate_summary_dx,
)  # pylint: disable=unused-import
from .tools.populate_summary import (
    populate_summary_study_level,
)  # pylint: disable=unused-import

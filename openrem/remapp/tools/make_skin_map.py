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
..  module:: make_skin_map.
    :synopsis: Module to calculate skin dose map from study data.

..  moduleauthor:: Ed McDonagh, David Platten, Wens Kong

"""
import os
import sys
import logging

import django
from django.core.exceptions import ObjectDoesNotExist
import numpy as np

# Setup django. This is required on windows, because process is created via spawn and
# django will not be initialized anymore then (On Linux this will only be executed once)
basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"
django.setup()

from remapp.models import (  # pylint: disable=wrong-import-position
    GeneralStudyModuleAttr,
    SkinDoseMapResults,
    OpenSkinSafeList,
)
from .background import record_task_info  # pylint: disable=wrong-import-position
from .save_skin_map_structure import (  # pylint: disable=wrong-import-position
    save_openskin_structure,
)
from .openskin.calc_exp_map import CalcExpMap  # pylint: disable=wrong-import-position
from ..version import __skin_map_version__  # pylint: disable=wrong-import-position

# Explicitly name logger so that it is still handled when using __main__
logger = logging.getLogger("remapp.tools.make_skin_map")


def make_skin_map(study_pk=None):
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-locals
    # noqa: C901

    if study_pk:
        study = GeneralStudyModuleAttr.objects.get(pk=study_pk)
        record_task_info(
            f"Unit: {study.generalequipmentmoduleattr_set.get().unique_equipment_name.display_name} | "
            f"PK: {study_pk} | Study UID: {study.study_instance_uid.replace('.', '. ')}"
        )

        # Get all OpenSkinSafeList table entries that match the manufacturer and model name of the current study
        entries = OpenSkinSafeList.objects.all().filter(
            manufacturer=study.generalequipmentmoduleattr_set.get().manufacturer,
            manufacturer_model_name=study.generalequipmentmoduleattr_set.get().manufacturer_model_name,
        )

        # Look for an entry which has a matching software version with the current study,
        # or an entry where the software version is blank (any software version)
        entry = None
        for current_entry in entries:
            if (
                current_entry.software_version
                == study.generalequipmentmoduleattr_set.get().software_versions
                or current_entry.software_version is None
                or not current_entry.software_version
            ):
                entry = current_entry
                break

        if entry is None:
            # There is no match, so return a blank dummy openSkin structure without trying
            # to calculate a skin dose map
            return_structure = {
                "skin_map": [0, 0],
                "skin_map_version": __skin_map_version__,
            }
            save_openskin_structure(study, return_structure)
            return

        pat_mass_source = "assumed"
        try:
            pat_mass = float(study.patientstudymoduleattr_set.get().patient_weight)
            pat_mass_source = "extracted"
        except (ValueError, TypeError):
            pat_mass = 73.2

        if pat_mass == 0.0:
            pat_mass = 73.2
            pat_mass_source = "assumed"

        pat_height_source = "assumed"
        try:
            pat_height = (
                float(study.patientstudymoduleattr_set.get().patient_size) * 100
            )

            pat_height_source = "extracted"
        except (ValueError, TypeError):
            pat_height = 178.6

        if pat_height == 0.0:
            pat_height = 178.6
            pat_height_source = "assumed"

        ptr = None
        orientation_modifier = None
        try:
            ptr_meaning = (
                study.projectionxrayradiationdose_set.get()
                .irradeventxraydata_set.all()[0]
                .patient_table_relationship_cid.code_meaning.lower()
            )
            if ptr_meaning in "headfirst":
                ptr = "H"
            elif ptr_meaning in "feet-first":
                ptr = "F"
            else:
                logger.info(
                    f"Study PK {study_pk}: Patient table relationship not recognised ({ptr_meaning}). "
                    f"Assuming head first."
                )
        except AttributeError:
            logger.info(
                f"Study PK {study_pk}: Patient table relationship not found. Assuming head first."
            )
        except IndexError:
            logger.info(
                f"Study PK {study_pk}: No irradiation event x-ray data found. Assuming head first."
            )
        try:
            orientation_modifier_meaning = (
                study.projectionxrayradiationdose_set.get()
                .irradeventxraydata_set.all()[0]
                .patient_orientation_modifier_cid.code_meaning.lower()
            )
            if orientation_modifier_meaning in "supine":
                orientation_modifier = "S"
            elif orientation_modifier_meaning in "prone":
                orientation_modifier = "P"
            else:
                logger.info(
                    f"Study PK {study_pk}: Orientation modifier not recognised ({orientation_modifier_meaning}). "
                    f"Assuming supine."
                )
        except AttributeError:
            logger.info(
                f"Study PK {study_pk}: Orientation modifier not found. Assuming supine."
            )
        except IndexError:
            logger.info(
                f"Study PK {study_pk}: No irradiation event x-ray data found. Assuming supine."
            )
        if ptr and orientation_modifier:
            pat_pos_source = "extracted"
            pat_pos = ptr + "F" + orientation_modifier
        elif ptr:
            pat_pos_source = "supine assumed"
            pat_pos = ptr + "FS"
        elif orientation_modifier:
            pat_pos_source = "head first assumed"
            pat_pos = "HF" + orientation_modifier
        else:
            pat_pos_source = "assumed"
            pat_pos = "HFS"
        logger.debug(f"patPos is {pat_pos} and source is {pat_pos_source}")

        my_exp_map = CalcExpMap(
            phantom_type="3D",
            pat_pos=pat_pos,
            pat_mass=pat_mass,
            pat_height=pat_height,
            table_thick=0.5,
            table_width=45.0,
            table_length=150.0,
            matt_thick=4.0,
        )

        for (
            irrad
        ) in study.projectionxrayradiationdose_set.get().irradeventxraydata_set.all():
            try:
                delta_x = (
                    float(
                        irrad.irradeventxraymechanicaldata_set.get()
                        .doserelateddistancemeasurements_set.get()
                        .table_longitudinal_position
                    )
                    / 10.0
                )
            except (ObjectDoesNotExist, TypeError):
                delta_x = 0.0
            try:
                delta_y = (
                    float(
                        irrad.irradeventxraymechanicaldata_set.get()
                        .doserelateddistancemeasurements_set.get()
                        .table_lateral_position
                    )
                    / 10.0
                )
            except (ObjectDoesNotExist, TypeError):
                delta_y = 0.0
            try:
                delta_z = (
                    float(
                        irrad.irradeventxraymechanicaldata_set.get()
                        .doserelateddistancemeasurements_set.get()
                        .table_height_position
                    )
                    / 10.0
                )
            except (ObjectDoesNotExist, TypeError):
                delta_z = 0.0
            if irrad.irradeventxraymechanicaldata_set.get().positioner_primary_angle:
                angle_x = float(
                    irrad.irradeventxraymechanicaldata_set.get().positioner_primary_angle
                )
            else:
                angle_x = 0.0
            try:
                angle_y = float(
                    irrad.irradeventxraymechanicaldata_set.get().positioner_secondary_angle
                )
            except (ObjectDoesNotExist, TypeError):
                angle_y = 0.0
            try:
                d_ref = (
                    float(
                        irrad.irradeventxraymechanicaldata_set.get()
                        .doserelateddistancemeasurements_set.get()
                        .distance_source_to_isocenter
                    )
                    / 10.0
                    - 15.0
                )
            except (ObjectDoesNotExist, TypeError):
                # This will result in failure to calculate skin dose map. Need a sensible default, or a lookup to a
                # user-entered value
                d_ref = None
            try:
                dap = float(irrad.dose_area_product)
            except (ObjectDoesNotExist, TypeError):
                dap = None
            try:
                ref_ak = float(irrad.irradeventxraysourcedata_set.get().dose_rp)
            except (ObjectDoesNotExist, TypeError):
                ref_ak = None
            try:
                kvp = np.mean(
                    irrad.irradeventxraysourcedata_set.get()
                    .kvp_set.all()
                    .exclude(kvp__isnull=True)
                    .exclude(kvp__exact=0)
                    .values_list("kvp", flat=True)
                )
                kvp = float(kvp)
                if np.isnan(kvp):
                    kvp = None
            except (ObjectDoesNotExist, TypeError):
                kvp = None

            filter_cu = 0.0
            if irrad.irradeventxraysourcedata_set.get().xrayfilters_set.all():
                for (
                    xray_filter
                ) in irrad.irradeventxraysourcedata_set.get().xrayfilters_set.all():
                    try:
                        if xray_filter.xray_filter_material.code_value == "C-127F9":
                            filter_cu += float(
                                xray_filter.xray_filter_thickness_minimum
                            )
                    except AttributeError:
                        pass

            if irrad.irradiation_event_type:
                run_type = irrad.irradiation_event_type.code_meaning
            else:
                run_type = None
            try:
                frames = float(
                    irrad.irradeventxraysourcedata_set.get().number_of_pulses
                )
            except TypeError:
                try:
                    frames = float(
                        irrad.irradeventxraysourcedata_set.get().exposure_time
                        / irrad.irradeventxraysourcedata_set.get()
                        .pulsewidth_set.get()
                        .pulse_width
                    )
                except (ObjectDoesNotExist, TypeError):
                    frames = None
            except ObjectDoesNotExist:
                frames = None
            try:
                end_angle = float(
                    irrad.irradeventxraymechanicaldata_set.get().positioner_primary_end_angle
                )
            except (ObjectDoesNotExist, TypeError):
                end_angle = None
            if ref_ak and d_ref:
                my_exp_map.add_view(
                    delta_x=delta_x,
                    delta_y=delta_y,
                    delta_z=delta_z,
                    angle_x=angle_x,
                    angle_y=angle_y,
                    d_ref=d_ref,
                    dap=dap,
                    ref_ak=ref_ak,
                    kvp=kvp,
                    filter_cu=filter_cu,
                    run_type=run_type,
                    frames=frames,
                    end_angle=end_angle,
                    pat_pos=pat_pos,
                )

        # Flip the skin dose map left-right so the view is from the front
        # my_exp_map.my_dose.fliplr()
        my_exp_map.my_dose.total_dose = np.roll(
            my_exp_map.my_dose.total_dose,
            int(my_exp_map.phantom.phantom_flat_dist // 2),
            axis=0,
        )
        try:
            my_exp_map.my_dose.total_dose = np.rot90(my_exp_map.my_dose.total_dose)
        except ValueError:
            pass
        try:
            SkinDoseMapResults.objects.get(
                general_study_module_attributes=study
            ).delete()
        except ObjectDoesNotExist:
            pass
        # assume that calculation failed if max(peak_skin_dose) == 0 ==> set peak_skin_dose to None
        max_skin_dose = np.max(my_exp_map.my_dose.total_dose, initial=0)
        max_skin_dose = max_skin_dose if max_skin_dose > 0 else None
        try:
            dap_fraction = my_exp_map.my_dose.dap_count / float(study.total_dap)
        except ZeroDivisionError:
            dap_fraction = 1.0
        SkinDoseMapResults(
            general_study_module_attributes=study,
            patient_orientation=pat_pos,
            patient_mass=pat_mass,
            patient_mass_assumed=pat_mass_source,
            patient_size_assumed=pat_height_source,
            patient_orientation_assumed=pat_pos_source,
            phantom_width=my_exp_map.phantom.phantom_width,
            phantom_height=my_exp_map.phantom.phantom_height,
            phantom_depth=my_exp_map.phantom.phantom_depth,
            patient_size=pat_height,
            skin_map_version=__skin_map_version__,
            peak_skin_dose=max_skin_dose,
            dap_fraction=dap_fraction,
        ).save()
        return_structure = {
            "skin_map": my_exp_map.my_dose.total_dose.flatten().tolist(),
            "width": my_exp_map.phantom.width,
            "height": my_exp_map.phantom.height,
            "phantom_width": my_exp_map.phantom.phantom_width,
            "phantom_height": my_exp_map.phantom.phantom_height,
            "phantom_head_height": my_exp_map.phantom.phantom_head_height,
            "phantom_head_radius": my_exp_map.phantom.phantom_head_radius,
            "phantom_depth": my_exp_map.phantom.phantom_depth,
            "phantom_flat_dist": my_exp_map.phantom.phantom_flat_dist,
            "phantom_curved_dist": my_exp_map.phantom.phantom_curved_dist,
            "patient_height": pat_height,
            "patient_mass": pat_mass,
            "patient_orientation": pat_pos,
            "patient_height_source": pat_height_source,
            "patient_mass_source": pat_mass_source,
            "patient_orientation_source": pat_pos_source,
            "fraction_DAP": my_exp_map.my_dose.dap_count / float(study.total_dap),
            "skin_map_version": __skin_map_version__,
        }

        # Save the return_structure as a pickle in a skin_maps sub-folder of the MEDIA_ROOT folder
        save_openskin_structure(study, return_structure)

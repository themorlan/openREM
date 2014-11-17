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
#
#
#    This file (dx.py) is intended to extract radiation dose related data from
#    DX images. It is based on mam.py.
#    David Platten, 28/3/2014
#
"""
..  module:: dx.
    :synopsis: Module to extract radiation dose related data from DX image objects.

..  moduleauthor:: David Platten

"""
def _xrayfilters(dataset,source):
    from remapp.models import Xray_filters
    from remapp.tools.get_values import get_value_kw, get_or_create_cid
    filters = Xray_filters.objects.create(irradiation_event_xray_source_data=source)
    xray_filter_material = get_value_kw('FilterMaterial',dataset)
    if xray_filter_material:
        if xray_filter_material.strip().lower() == 'molybdenum':
            filters.xray_filter_material = get_or_create_cid('C-150F9','Molybdenum or Molybdenum compound')
        if xray_filter_material.strip().lower() == 'rhodium':
            filters.xray_filter_material = get_or_create_cid('C-167F9','Rhodium or Rhodium compound')
        if xray_filter_material.strip().lower() == 'silver':
            filters.xray_filter_material = get_or_create_cid('C-137F9','Silver or Silver compound')
        if xray_filter_material.strip().lower() == 'aluminum' or xray_filter_material.strip().lower() == 'aluminium':
            filters.xray_filter_material = get_or_create_cid('C-120F9','Aluminum or Aluminum compound')
        # Added by DJP on 28/3/2014 to ensure that all possible filters are looked for. Data taken
        # from https://www.dabsoft.ch/dicom/16/CID_10006/
        if xray_filter_material.strip().lower() == 'copper':
            filters.xray_filter_material = get_or_create_cid('C-127F9','Copper or Copper compound')
        if xray_filter_material.strip().lower() == 'niobium':
            filters.xray_filter_material = get_or_create_cid('C-1190E','Niobium or Niobium compound')
        if xray_filter_material.strip().lower() == 'europium':
            filters.xray_filter_material = get_or_create_cid('C-1190F','Europium or Europium compound')
        if xray_filter_material.strip().lower() == 'lead':
            filters.xray_filter_material = get_or_create_cid('C-132F9','Lead or Lead compound')
        if xray_filter_material.strip().lower() == 'tantalum':
            filters.xray_filter_material = get_or_create_cid('C-156F9','Tantalum or Tantalum compound')
        # End of 28/3/2014 DJP additions
        
        filters.save()
    

def _kvp(dataset,source):
    from remapp.models import Kvp
    from remapp.tools.get_values import get_value_kw
    kv = Kvp.objects.create(irradiation_event_xray_source_data=source)
    kv.kvp = get_value_kw('KVP',dataset)
    kv.save()


def _exposure(dataset,source):
    from remapp.models import Exposure
    exp = Exposure.objects.create(irradiation_event_xray_source_data=source)
    from remapp.tools.get_values import get_value_kw
     exp.exposure = get_value_kw('ExposureInuAs', dataset) # uAs
     if not exp.exposure:
         exposure = get_value_kw('Exposure', dataset)
         if exposure:
             exp.exposure = exposure * 1000
    exp.save()


def _xraygrid(gridcode,source):
    from remapp.models import Xray_grid
    from remapp.tools.get_values import get_or_create_cid
    grid = Xray_grid.objects.create(irradiation_event_xray_source_data=source)
    if gridcode == '111646':
        grid.xray_grid = get_or_create_cid('111646','No grid')
    if gridcode == '111642':
        grid.xray_grid = get_or_create_cid('111642','Focused grid')
    if gridcode == '111643':
        grid.xray_grid = get_or_create_cid('111643','Reciprocating grid')
    grid.save()


# 28/3/2014 Added by DJP to collect exposure index
def _irradiationeventxraydetectordata(dataset,event):
    from remapp.models import Irradiation_event_xray_detector_data
    from remapp.tools.get_values import get_value_kw, get_or_create_cid
    detector = Irradiation_event_xray_detector_data.objects.create(irradiation_event_xray_data=event)
    detector.exposure_index = get_value_kw('ExposureIndex',dataset)
    detector.relative_xray_exposure = get_value_kw('RelativeXRayExposure',dataset)
    manufacturer = detector.irradiation_event_xray_data.projection_xray_radiation_dose.general_study_module_attributes.general_equipment_module_attributes_set.all()[0].manufacturer.lower()
    if   'fuji'       in manufacturer: detector.relative_exposure_unit = 'S ()'
    elif 'carestream' in manufacturer: detector.relative_exposure_unit = 'EI (Mbels)'
    elif 'agfa'       in manufacturer: detector.relative_exposure_unit = 'lgM (Bels)'
    elif 'konica'     in manufacturer: detector.relative_exposure_unit = 'S ()'
    elif 'canon'      in manufacturer: detector.relative_exposure_unit = 'REX ()'
    elif 'swissray'   in manufacturer: detector.relative_exposure_unit = 'DI ()'
    elif 'philips'    in manufacturer: detector.relative_exposure_unit = 'EI ()'
    elif 'siemens'    in manufacturer: detector.relative_exposure_unit = 'EXI (uGy)'
    detector.sensitivity = get_value_kw('Sensitivity',dataset)
    detector.target_exposure_index = get_value_kw('TargetExposureIndex',dataset)
    detector.deviation_index = get_value_kw('DeviationIndex',dataset)
    detector.save()


def _irradiationeventxraysourcedata(dataset,event):
    from remapp.models import Irradiation_event_xray_source_data
    from remapp.tools.get_values import get_value_kw, get_or_create_cid
    source = Irradiation_event_xray_source_data.objects.create(irradiation_event_xray_data=event)
    source.average_xray_tube_current = get_value_kw('XRayTubeCurrent',dataset)
    if not source.average_xray_tube_current: source.average_xray_tube_current = get_value_kw('AverageXRayTubeCurrent',dataset)
    source.exposure_time = get_value_kw('ExposureTime',dataset)
    source.irradiation_duration = get_value_kw('IrradiationDuration',dataset)
    source.focal_spot_size = get_value_kw('FocalSpots',dataset)
    collimated_field_area = get_value_kw('FieldOfViewDimensions',dataset)
    if collimated_field_area:
        source.collimated_field_area = float(collimated_field_area[0]) * float(collimated_field_area[1]) / 1000000
    exp_ctrl_mode = get_value_kw('ExposureControlMode',dataset)
    if exp_ctrl_mode:
        source.exposure_control_mode = exp_ctrl_mode
    
    source.save()
    
    _xrayfilters(dataset,source)
    _kvp(dataset,source)
    _exposure(dataset,source)
    xray_grid = get_value_kw('Grid',dataset)
    if xray_grid:
        if xray_grid == 'NONE':
            _xraygrid('111646',source)
        elif xray_grid == ['RECIPROCATING', 'FOCUSED']:
            _xraygrid('111642',source)
            _xraygrid('111643',source)


def _doserelateddistancemeasurements(dataset,mech):
    from remapp.models import Dose_related_distance_measurements
    from remapp.tools.get_values import get_value_kw, get_value_num
    dist = Dose_related_distance_measurements.objects.create(irradiation_event_xray_mechanical_data=mech)
    dist.distance_source_to_detector = get_value_kw('DistanceSourceToDetector',dataset)
    dist.distance_source_to_entrance_surface = get_value_kw('DistanceSourceToEntrance',dataset)
    dist.distance_source_to_isocenter = get_value_kw('DistanceSourceToIsocenter',dataset)
    dist.distance_source_to_reference_point = get_value_kw('DistanceSourceToReferencePoint',dataset)
    dist.table_longitudinal_position = get_value_kw('TableLongitudinalPosition',dataset)
    dist.table_lateral_position = get_value_kw('TableLateralPosition',dataset)
    dist.table_height_position = get_value_kw('TableHeightPosition',dataset)
    dist.distance_source_to_table_plane = get_value_kw('DistanceSourceToTablePlane',dataset)
    dist.radiological_thickness = get_value_num(0x00451049,dataset)
	
    dist.save()        


def _irradiationeventxraymechanicaldata(dataset,event):
    from remapp.models import Irradiation_event_xray_mechanical_data
    from remapp.tools.get_values import get_value_kw
    mech = Irradiation_event_xray_mechanical_data.objects.create(irradiation_event_xray_data=event)
    mech.magnification_factor = get_value_kw('EstimatedRadiographicMagnificationFactor',dataset)
    mech.dxdr_mechanical_configuration = get_value_kw('DX/DRMechanicalConfiguration',dataset)
    mech.primary_angle = get_value_kw('PositionerPrimaryAngle',dataset)
    mech.secondary_angle = get_value_kw('PositionerSecondaryAngle',dataset)
    mech.primary_end_angle = get_value_kw('PositionerPrimaryEndAngle',dataset)
    mech.secondary_angle = get_value_kw('PositionerSecondaryEndAngle',dataset)
    mech.column_angulation = get_value_kw('ColumnAngulation',dataset)
    mech.table_head_tilt_angle = get_value_kw('TableHeadTiltAngle',dataset)
    mech.table_horizontal_rotation_angle = get_value_kw('TableHorizontalRotationAngle',dataset)
    mech.table_cradle_tilt_angle = get_value_kw('TableCradleTiltAngle',dataset)
    mech.compression_thickness = get_value_kw('CompressionThickness',dataset)
    comp_force = get_value_kw('CompressionForce',dataset)
    if comp_force:
        mech.compression_force = float(comp_force)/10 # GE Conformance statement says in N, treating as dN
    
    mech.save()
    _doserelateddistancemeasurements(dataset,mech)


# 28/3/2014 DJP commented the below routine out as it's not relevant to DX images
#def _accumulatedmammo_update(dataset,event): # TID 10005
#    from remapp.tools.get_values import get_value_kw, get_or_create_cid
#    accummam = event.projection_xray_radiation_dose.accumulated_xray_dose_set.get().accumulated_mammography_xray_dose_set.get()
#    if event.irradiation_event_xray_source_data_set.get().average_glandular_dose:
#        accummam.accumulated_average_glandular_dose += event.irradiation_event_xray_source_data_set.get().average_glandular_dose
#    if event.laterality:
#        if accummam.laterality:
#            if accummam.laterality.code_meaning == 'Left breast':
#                if event.laterality.code_meaning == 'Right':
#                    accummam.laterality = get_or_create_cid('T-04080','Both breasts')
#            if accummam.laterality.code_meaning == 'Right breast':
#                if event.laterality.code_meaning == 'Left':
#                    accummam.laterality = get_or_create_cid('T-04080','Both breasts')
#        else:
#            if event.laterality.code_meaning == 'Right':
#                accummam.laterality = get_or_create_cid('T-04020','Right breast')
#            if event.laterality.code_meaning == 'Left':
#                accummam.laterality = get_or_create_cid('T-04030','Left breast')
#    accummam.save()


def _irradiationeventxraydata(dataset,proj): # TID 10003
    from remapp.models import Irradiation_event_xray_data
    from remapp.tools.get_values import get_value_kw, get_or_create_cid, get_seq_code_value, get_seq_code_meaning
    from remapp.tools.dcmdatetime import make_date_time
    event = Irradiation_event_xray_data.objects.create(projection_xray_radiation_dose=proj)
    event.acquisition_plane = get_or_create_cid('113622', 'Single Plane')
    event.irradiation_event_uid = get_value_kw('SOPInstanceUID',dataset)
    event_time = get_value_kw('AcquisitionTime',dataset)
    if not event_time: event_time = get_value_kw('ContentTime',dataset)
    if not event_time: event_time = get_value_kw('StudyTime',dataset)
    event_date = get_value_kw('AcquisitionDate',dataset)
    if not event_date: event_date = get_value_kw('ContentDate',dataset)
    if not event_date: event_date = get_value_kw('StudyDate',dataset)
    event.date_time_started = make_date_time('{0}{1}'.format(event_date,event_time))
    event.irradiation_event_type = get_or_create_cid('113611','Stationary Acquisition')
    event.acquisition_protocol = get_value_kw('ProtocolName',dataset)
    if not event.acquisition_protocol: event.acquisition_protocol = get_value_kw('SeriesDescription',dataset)
    try:
        event.anatomical_structure = get_or_create_cid(get_seq_code_value('AnatomicRegionSequence',dataset),get_seq_code_meaning('AnatomicRegionSequence',dataset))
    except:
        print "Error creating AnatomicRegionSequence. Continuing."
    laterality = get_value_kw('ImageLaterality',dataset)
    if laterality:
        if laterality.strip() == 'R':
            event.laterality = get_or_create_cid('G-A100','Right')
        if laterality.strip() == 'L':
            event.laterality = get_or_create_cid('G-A101','Left')

    event.image_view = get_or_create_cid(get_seq_code_value('ViewCodeSequence',dataset),get_seq_code_meaning('ViewCodeSequence',dataset))
    if not event.image_view:
        projection = get_value_kw('ViewPosition',dataset)
        if   projection == 'AP': event.image_view = get_or_create_cid('R-10206','antero-posterior')
        elif projection == 'PA': event.image_view = get_or_create_cid('R-10214','postero-anterior')
        elif projection == 'LL': event.image_view = get_or_create_cid('R-10236','left lateral')
        elif projection == 'RL': event.image_view = get_or_create_cid('R-10232','right lateral')
        # http://dicomlookup.com/lookup.asp?sw=Tnumber&q=(0018,5101) lists four other views: RLD (Right Lateral Decubitus),
        # LLD (Left Lateral Decubitus), RLO (Right Lateral Oblique) and LLO (Left Lateral Oblique). There isn't an exact
        # match for these views in the CID 4010 DX View (http://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_4010.html)

    # image view modifier?
    if event.anatomical_structure:
        event.target_region = event.anatomical_structure
    event.entrance_exposure_at_rp = get_value_kw('EntranceDoseInmGy',dataset)
    # reference point definition?
    pc_fibroglandular = get_value_kw('CommentsOnRadiationDose',dataset)
    if pc_fibroglandular:
        if '%' in pc_fibroglandular:
            event.percent_fibroglandular_tissue = pc_fibroglandular.replace('%','').strip()
    event.comment = get_value_kw('ExposureControlModeDescription',dataset)

    dap = get_value_kw('ImageAndFluoroscopyAreaDoseProduct',dataset)
    if dap: event.dose_area_product = dap / 100000 # Value of DICOM tag (0018,115e) in dGy.cm2, converted to Gy.m2
    event.save()
    
    # 28/3/2014 DJP put the line below in
    _irradiationeventxraydetectordata(dataset,event)
    _irradiationeventxraysourcedata(dataset,event)
    _irradiationeventxraymechanicaldata(dataset,event)
    _accumulatedxraydose_update(dataset,event)


# 28/3/2014 DJP: removed the mammo bits from the routine below
def _accumulatedxraydose(dataset,proj):
    from remapp.models import Accumulated_xray_dose, Accumulated_projection_xray_dose
    from remapp.tools.get_values import get_value_kw, get_or_create_cid
    accum = Accumulated_xray_dose.objects.create(projection_xray_radiation_dose=proj)
    accum.acquisition_plane = get_or_create_cid('113622','Single Plane')
    accum.save()
    accumdx = Accumulated_projection_xray_dose.objects.create(accumulated_xray_dose=accum)
    accumdx.dose_area_product_total = 0.0
    accumdx.total_number_of_radiographic_frames = 0
    accumdx.save()


def _accumulatedxraydose_update(dataset,event):
    from remapp.tools.get_values import get_value_kw, get_or_create_cid
    from decimal import Decimal
    accumdx = event.projection_xray_radiation_dose.accumulated_xray_dose_set.get().accumulated_projection_xray_dose_set.get()
    accumdx.total_number_of_radiographic_frames = accumdx.total_number_of_radiographic_frames + 1
    if event.dose_area_product:
        accumdx.dose_area_product_total += Decimal(event.dose_area_product)
    accumdx.save()


def _projectionxrayradiationdose(dataset,g):
    from remapp.models import Projection_xray_radiation_dose
    from remapp.tools.get_values import get_or_create_cid
    proj = Projection_xray_radiation_dose.objects.create(general_study_module_attributes=g)
    proj.procedure_reported = get_or_create_cid('113704','Projection X-Ray')
    proj.has_intent = get_or_create_cid('R-408C3','Diagnostic Intent')
    proj.scope_of_accumulation = get_or_create_cid('113014','Study')
    proj.source_of_dose_information = get_or_create_cid('113866','Copied From Image Attributes')
    proj.xray_detector_data_available = get_or_create_cid('R-00339','No')
    proj.xray_source_data_available = get_or_create_cid('R-0038D','Yes')
    proj.xray_mechanical_data_available = get_or_create_cid('R-0038D','Yes')
    proj.save()
    _accumulatedxraydose(dataset,proj)
    _irradiationeventxraydata(dataset,proj)


def _generalequipmentmoduleattributes(dataset,study):
    from remapp.models import General_equipment_module_attributes
    from remapp.tools.get_values import get_value_kw
    from remapp.tools.dcmdatetime import get_date, get_time
    equip = General_equipment_module_attributes.objects.create(general_study_module_attributes=study)
    equip.manufacturer = get_value_kw("Manufacturer",dataset)
    equip.institution_name = get_value_kw("InstitutionName",dataset)
    equip.institution_address = get_value_kw("InstitutionAddress",dataset)
    equip.station_name = get_value_kw("StationName",dataset)
    equip.institutional_department_name = get_value_kw("InstitutionalDepartmentName",dataset)
    equip.manufacturer_model_name = get_value_kw("ManufacturerModelName",dataset)
    equip.device_serial_number = get_value_kw("DeviceSerialNumber",dataset)
    equip.software_versions = get_value_kw("SoftwareVersions",dataset)
    equip.gantry_id = get_value_kw("GantryID",dataset)
    equip.spatial_resolution = get_value_kw("SpatialResolution",dataset)
    equip.date_of_last_calibration = get_date("DateOfLastCalibration",dataset)
    equip.time_of_last_calibration = get_time("TimeOfLastCalibration",dataset)
    equip.save()


def _patientstudymoduleattributes(dataset,g): # C.7.2.2
    from remapp.models import Patient_study_module_attributes
    from remapp.tools.get_values import get_value_kw
    patientatt = Patient_study_module_attributes.objects.create(general_study_module_attributes=g)
    patientatt.patient_age = get_value_kw('PatientAge',dataset)
    patientatt.save()


def _patientmoduleattributes(dataset,g): # C.7.1.1
    from remapp.models import Patient_module_attributes, Patient_study_module_attributes
    from remapp.tools.get_values import get_value_kw
    from remapp.tools.dcmdatetime import get_date
    from remapp.tools.not_patient_indicators import get_not_pt
    from datetime import timedelta
    from decimal import Decimal
    pat = Patient_module_attributes.objects.create(general_study_module_attributes=g)
    pat.patient_sex = get_value_kw('PatientSex',dataset)
    patient_birth_date = get_date('PatientBirthDate',dataset) # Not saved to database
    pat.not_patient_indicator = get_not_pt(dataset)
    patientatt = Patient_study_module_attributes.objects.get(general_study_module_attributes=g)
    if patient_birth_date:
        patientatt.patient_age_decimal = Decimal((g.study_date.date() - patient_birth_date.date()).days)/Decimal('365.25')
    elif patientatt.patient_age:
        if patientatt.patient_age[-1:]=='Y':
            patientatt.patient_age_decimal = Decimal(patientatt.patient_age[:-1])
        elif patientatt.patient_age[-1:]=='M':
            patientatt.patient_age_decimal = Decimal(patientatt.patient_age[:-1])/Decimal('12')
        elif patientatt.patient_age[-1:]=='D':
            patientatt.patient_age_decimal = Decimal(patientatt.patient_age[:-1])/Decimal('365.25') 
    if patientatt.patient_age_decimal:
        patientatt.patient_age_decimal = patientatt.patient_age_decimal.quantize(Decimal('.1'))
    patientatt.save()
    pat.save()


def _generalstudymoduleattributes(dataset,g):
    from remapp.tools.get_values import get_value_kw, get_seq_code_meaning, get_seq_code_value
    from remapp.tools.dcmdatetime import get_date, get_time
    g.study_instance_uid = get_value_kw('StudyInstanceUID',dataset)
    g.study_date = get_date('StudyDate',dataset)
    g.study_time = get_time('StudyTime',dataset)
    g.referring_physician_name = get_value_kw('ReferringPhysicianName',dataset)
    g.referring_physician_identification = get_value_kw('ReferringPhysicianIdentification',dataset)
    g.study_id = get_value_kw('StudyID',dataset)
    g.accession_number = get_value_kw('AccessionNumber',dataset)
    g.study_description = get_value_kw('StudyDescription',dataset)
    if not g.study_description: g.study_description = get_value_kw('SeriesDescription',dataset)
    g.modality_type = get_value_kw('Modality',dataset)
    g.physician_of_record = get_value_kw('PhysicianOfRecord',dataset)
    g.name_of_physician_reading_study = get_value_kw('NameOfPhysicianReadingStudy',dataset)
    g.performing_physician_name = get_value_kw('PerformingPhysicianName',dataset)
    g.operator_name = get_value_kw('OperatorName',dataset)
    g.procedure_code_meaning = get_value_kw('ProtocolName',dataset) # Being used to summarise protocol for study
    if not g.procedure_code_meaning: g.procedure_code_meaning = get_value_kw('SeriesDescription',dataset)
    g.requested_procedure_code_value = get_seq_code_value('RequestedProcedureCodeSequence',dataset)
    g.requested_procedure_code_meaning = get_seq_code_meaning('RequestedProcedureCodeSequence',dataset)
    g.save()
    
    _generalequipmentmoduleattributes(dataset,g)
    _projectionxrayradiationdose(dataset,g)
    _patientstudymoduleattributes(dataset,g)
    _patientmoduleattributes(dataset,g)

    
# 28/3/2014 DJP renamed the routine below to "_test_if_dx" from "_test_if_mammo" and made code
# changes to make it DX rather than mammography specific. The routine will accept three types
# of image:
# DX image storage                               (SOP UID = '1.2.840.10008.5.1.4.1.1.1')
# Digital x-ray image storage - for presentation (SOP UID = '1.2.840.10008.5.1.4.1.1.1.1')
# Digital x-ray image storage - for processing   (SOP UID = '1.2.840.10008.5.1.4.1.1.1.1.1')
# These SOP UIDs were taken from http://www.dicomlibrary.com/dicom/sop/
def _test_if_dx(dataset):
    """ Test if dicom object passed is a DX or CR radiographic file by looking at SOP Class UID"""
    if dataset.SOPClassUID != '1.2.840.10008.5.1.4.1.1.1' and dataset.SOPClassUID != '1.2.840.10008.5.1.4.1.1.1.1' and dataset.SOPClassUID != '1.2.840.10008.5.1.4.1.1.1.1.1':
        return 0
    return 1


# 28/3/2014 DJP renamed this "_dx2db" from "_mammo2db" and made code changes
# to make it DX rather than mammography specific.
def _dx2db(dataset):
    import os, sys
    import openrem_settings
    
    openrem_settings.add_project_to_path()
#    os.environ['DJANGO_SETTINGS_MODULE'] = '{0}.settings'.format(openrem_settings.openremproject())
    os.environ['DJANGO_SETTINGS_MODULE'] = 'openrem.openremproject.settings'

    from django.db import models
    from remapp.models import General_study_module_attributes
    from remapp.tools import check_uid
    from remapp.tools.get_values import get_value_kw
    from remapp.tools.dcmdatetime import make_date_time
    
    study_uid = get_value_kw('StudyInstanceUID',dataset)
    if not study_uid:
        sys.exit('No UID returned')  
    study_in_db = check_uid.check_uid(study_uid)
    if study_in_db:
        event_uid = get_value_kw('SOPInstanceUID',dataset)
        inst_in_db = check_uid.check_uid(event_uid,'Event')
        if inst_in_db:
            return 0
        # further check required to ensure 'for processing' and 'for presentation' 
        # versions of the same irradiation event don't get imported twice
        same_study_uid = General_study_module_attributes.objects.filter(study_instance_uid__exact = study_uid)
        event_time = get_value_kw('AcquisitionTime',dataset)
        if not event_time: event_time = get_value_kw('StudyTime',dataset)
        event_date = get_value_kw('AcquisitionDate',dataset)
        if not event_date: event_date = get_value_kw('StudyDate',dataset)
        event_date_time = make_date_time('{0}{1}'.format(event_date,event_time))
        for events in same_study_uid.get().projection_xray_radiation_dose_set.get().irradiation_event_xray_data_set.all():
            if event_date_time == events.date_time_started:
                return 0
        # study exists, but event doesn't
        _irradiationeventxraydata(dataset,same_study_uid.get().projection_xray_radiation_dose_set.get())
        # update the accumulated tables
        return 0
    
    # study doesn't exist, start from scratch
    g = General_study_module_attributes.objects.create()
    _generalstudymoduleattributes(dataset,g)


def dx(dig_file):
    """Extract radiation dose structured report related data from DX radiographic images
    
    :param filename: relative or absolute path to DICOM DX radiographic image file.
    :type filename: str.

    Tested with:
        Nothing yet
    
    """
    
    import sys
    import dicom
    
    dataset = dicom.read_file(dig_file)
    isdx = _test_if_dx(dataset)
    if not isdx:
        return '{0} is not a DICOM DX radiographic image'.format(dig_file)
    
    _dx2db(dataset)
    
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit('Error: Supply exactly one argument - the DICOM DX radiographic image file')
    
    sys.exit(dx(sys.argv[1]))

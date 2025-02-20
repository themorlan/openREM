# This Python file uses the following encoding: utf-8
# test_import_dx_rdsr.py

import os
import pydicom
from django.test import TestCase
from remapp.extractors.dx import _dx2db
from remapp.extractors.rdsr import _rdsr2db
from remapp.models import (
    GeneralStudyModuleAttr,
    PatientIDSettings,
    UniqueEquipmentNames,
    MergeOnDeviceObserverUIDSettings,
)


class DislayNamesAndMatchOnObsUID(TestCase):
    """Class to test the use of custom display names and match_on_device_observer_uid setting"""

    def test_display_Name_permutations(self):
        """
        Import DX RDSRs, same Device Observer UIDs, vary station name and department name, add custom display name,
        turn on matching by device observer UID.
        """

        PatientIDSettings.objects.create()

        dicom_file = "test_files/DX-RDSR-Canon_CXDI.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dicom_path = os.path.join(root_tests, dicom_file)

        # Initial import of DX RDSR without any modifications
        dataset = pydicom.dcmread(dicom_path)
        dataset.decode()
        _rdsr2db(dataset)
        study = GeneralStudyModuleAttr.objects.order_by("id")[0]
        # Test that auto-generated display name is as expected
        display_name = (
            study.generalequipmentmoduleattr_set.get().unique_equipment_name.display_name
        )
        self.assertEqual(display_name, "OpenREM CANONDaRt")
        # Check that matching is currently false
        self.assertEqual(
            MergeOnDeviceObserverUIDSettings.get_solo().match_on_device_observer_uid,
            False,
        )

        # Copy the dataset, change some unique fields, change SOP and Study Instance UID so it is seen as a new file
        dataset_2 = dataset
        dataset_2.InstitutionalDepartmentName = "New Name"
        dataset_2.StationName = "StnName2"
        dataset_2.SOPInstanceUID = "1.2.3.4"
        dataset_2.StudyInstanceUID = "1.2.3.4"
        # Import the second dataset, same device observer UID but should create new display name
        _rdsr2db(dataset_2)
        study_2 = GeneralStudyModuleAttr.objects.order_by("id")[1]
        # Test that the second study has a new autogenerated display name
        display_name_2 = (
            study_2.generalequipmentmoduleattr_set.get().unique_equipment_name.display_name
        )
        self.assertEqual(display_name_2, "OpenREM StnName2")

        # Set a customised display name
        unique_equipment_names = UniqueEquipmentNames.objects.order_by("pk")
        # Test there are two entries, and set them both to the same display name
        self.assertEqual(unique_equipment_names.count(), 2)
        for display_name_data in unique_equipment_names:
            display_name_data.display_name = "Custom Display Name"
            display_name_data.save()
        # Test that study_2 has the new name
        study_2 = GeneralStudyModuleAttr.objects.order_by("id")[1]
        display_name_2 = (
            study_2.generalequipmentmoduleattr_set.get().unique_equipment_name.display_name
        )
        self.assertEqual(display_name_2, "Custom Display Name")

        # Import a third dataset with a different department name, should create new display name again
        dataset_3 = dataset
        dataset_3.InstitutionalDepartmentName = "Third Department"
        dataset_3.StationName = "StnName2"
        dataset_3.SOPInstanceUID = "1.2.3.5"
        dataset_3.StudyInstanceUID = "1.2.3.5"
        _rdsr2db(dataset_3)
        study_3 = GeneralStudyModuleAttr.objects.order_by("id")[2]
        # Test that the third study has a new autogenerated display name
        display_name_3 = (
            study_3.generalequipmentmoduleattr_set.get().unique_equipment_name.display_name
        )
        self.assertEqual(display_name_3, "OpenREM StnName2")

        # Set match on device observer UID as True and import again
        device_uid_settings = MergeOnDeviceObserverUIDSettings.get_solo()
        device_uid_settings.match_on_device_observer_uid = True
        device_uid_settings.save()
        self.assertEqual(
            MergeOnDeviceObserverUIDSettings.get_solo().match_on_device_observer_uid,
            True,
        )

        # Import a fourth dataset with a different department name, should automatically match first instance name
        dataset_4 = dataset
        dataset_4.InstitutionalDepartmentName = "Fourth Department"
        dataset_4.StationName = "StnName4"
        dataset_4.SOPInstanceUID = "1.2.3.6"
        dataset_4.StudyInstanceUID = "1.2.3.6"
        _rdsr2db(dataset_4)
        study_4 = GeneralStudyModuleAttr.objects.order_by("id")[3]
        # Test that the third study has a new autogenerated display name
        display_name_4 = (
            study_4.generalequipmentmoduleattr_set.get().unique_equipment_name.display_name
        )
        self.assertEqual(display_name_4, "Custom Display Name")

    def test_0_9_0_upgrade(self):
        """
        Import RDSR and set device_observer_uid to None (existing studies on upgrade condition), then import again
        to test if display name is matched
        """

        PatientIDSettings.objects.create()

        dicom_file_1 = "test_files/DX-RDSR-Canon_CXDI.dcm"
        dicom_file_2 = "test_files/DX-RDSR-Carestream_DRXEvolution.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dicom_path_1 = os.path.join(root_tests, dicom_file_1)
        dicom_path_2 = os.path.join(root_tests, dicom_file_2)

        dataset_1 = pydicom.dcmread(dicom_path_1)
        dataset_1.decode()
        _rdsr2db(dataset_1)
        dataset_2 = pydicom.dcmread(dicom_path_2)
        dataset_2.decode()
        _rdsr2db(dataset_2)

        display_name_0 = UniqueEquipmentNames.objects.order_by("pk")[0]
        display_name_0.device_observer_uid = None
        display_name_0.device_observer_uid_hash = None
        display_name_0.display_name = "System One"
        display_name_0.save()

        display_name_1 = UniqueEquipmentNames.objects.order_by("pk")[1]
        display_name_1.device_observer_uid = None
        display_name_1.device_observer_uid_hash = None
        display_name_1.display_name = "System Two"
        display_name_1.save()

        dataset_1_a = dataset_1
        dataset_1_a.SOPInstanceUID = "1.1.1.1"
        dataset_1_a.StudyInstanceUID = "1.1.1.1"
        _rdsr2db(dataset_1_a)

        study_1_a = GeneralStudyModuleAttr.objects.order_by("id")[2]
        display_name_1_a = (
            study_1_a.generalequipmentmoduleattr_set.get().unique_equipment_name.display_name
        )
        self.assertEqual(display_name_1_a, "System One")

        # Set match on device observer UID as True and import second study
        device_uid_settings = MergeOnDeviceObserverUIDSettings.get_solo()
        device_uid_settings.match_on_device_observer_uid = True
        device_uid_settings.save()
        self.assertEqual(
            MergeOnDeviceObserverUIDSettings.get_solo().match_on_device_observer_uid,
            True,
        )

        dataset_2_a = dataset_2
        dataset_2_a.SOPInstanceUID = "1.1.1.2"
        dataset_2_a.StudyInstanceUID = "1.1.1.2"
        _rdsr2db(dataset_2_a)

        study_2_a = GeneralStudyModuleAttr.objects.order_by("id")[3]
        display_name_2_a = (
            study_2_a.generalequipmentmoduleattr_set.get().unique_equipment_name.display_name
        )
        self.assertEqual(display_name_2_a, "System Two")

    def test_0_9_0_rdsr_images(self):
        """
        Import image and import rdsr of the same system. There should be no errors (but won't be assumed to be same
        system)
        """

        PatientIDSettings.objects.create()

        img_file = "test_files/DX-Im-GE_XR220-1.dcm"
        rdsr_file = "test_files/DX-RDSR-Canon_CXDI.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(root_tests, img_file)
        rdsr_path = os.path.join(root_tests, rdsr_file)

        img = pydicom.dcmread(img_path)
        rdsr = pydicom.dcmread(rdsr_path)
        img.decode()
        rdsr.decode()

        # Reset all matching values to match between image and RDSR from two different systems!
        img.Manufacturer = "Manufacturer"
        img.InstitutionName = "Institution"
        img.StationName = "Station Name"
        img.InstitutionalDepartmentName = "Institutional Department Name"
        img.ManufacturerModelName = "Manufacturer Model Name"
        img.DeviceSerialNumber = "Device Serial Number"
        img.SoftwareVersions = "Software Versions"
        img.GantryID = "GantryID"

        rdsr.Manufacturer = "Manufacturer"
        rdsr.InstitutionName = "Institution"
        rdsr.StationName = "Station Name"
        rdsr.InstitutionalDepartmentName = "Institutional Department Name"
        rdsr.ManufacturerModelName = "Manufacturer Model Name"
        rdsr.DeviceSerialNumber = "Device Serial Number"
        rdsr.SoftwareVersions = "Software Versions"
        rdsr.GantryID = "GantryID"

        _dx2db(img)

        display_name_img = UniqueEquipmentNames.objects.order_by("pk")[0]
        display_name_img.display_name = "Custom name for img"
        display_name_img.save()

        _rdsr2db(rdsr)

        # Check both studies have imported ok
        studies = GeneralStudyModuleAttr.objects.order_by("pk")
        self.assertEqual(studies.count(), 2)

        img_2 = img
        img_2.SOPInstanceUID = "1.1.1.2"
        img_2.StudyInstanceUID = "1.1.1.2"
        _dx2db(img_2)

        studies = GeneralStudyModuleAttr.objects.order_by("pk")
        self.assertEqual(studies.count(), 3)

        # Should be just two rows in UniqueEquipmentNames table
        self.assertEqual(UniqueEquipmentNames.objects.all().count(), 2)

        # RDSR import won't know the difference between img entry and pre-0.9.0 RDSR entry, so will adopt same name
        name_1 = UniqueEquipmentNames.objects.order_by("pk")[0].display_name
        name_2 = UniqueEquipmentNames.objects.order_by("pk")[1].display_name
        self.assertEqual(name_1, "Custom name for img")
        self.assertEqual(name_2, "Custom name for img")

        # Now do the same, but RDSR first, expect two different names
        rdsr_2 = rdsr
        rdsr_2.DeviceSerialNumber = "0003"
        rdsr_2.SOPInstanceUID = "1.1.2.1"
        rdsr_2.StudyInstanceUID = "1.1.2.1"
        img_3 = img
        img_3.DeviceSerialNumber = "0003"
        img_3.SOPInstanceUID = "1.1.3.2"
        img_3.StudyInstanceUID = "1.1.3.2"
        img_4 = img
        img_4.DeviceSerialNumber = "0003"
        img_4.SOPInstanceUID = "1.1.3.3"
        img_4.StudyInstanceUID = "1.1.3.3"

        _rdsr2db(rdsr_2)

        self.assertEqual(UniqueEquipmentNames.objects.all().count(), 3)
        display_name_3 = UniqueEquipmentNames.objects.order_by("pk")[2]
        self.assertEqual(display_name_3.display_name, "Institution Station Name")
        display_name_3.display_name = "Custom RDSR name"
        display_name_3.save()

        _dx2db(img_3)
        _dx2db(img_4)

        self.assertEqual(UniqueEquipmentNames.objects.all().count(), 4)
        display_name_3 = UniqueEquipmentNames.objects.order_by("pk")[2]
        display_name_4 = UniqueEquipmentNames.objects.order_by("pk")[3]
        self.assertEqual(display_name_3.display_name, "Custom RDSR name")
        self.assertEqual(display_name_4.display_name, "Institution Station Name")

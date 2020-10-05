import os
import pydicom
from django.core.exceptions import ObjectDoesNotExist

root_this_file = os.path.dirname(os.path.abspath(__file__))

test_files_path = os.path.abspath(os.path.join(root_this_file, "..\\openrem\\remapp\\tests\\test_files"))

tags = [
    "InstitutionName",
    "StationName",
    "StudyDescription",
    "ProtocolName",
    "ImageAndFluoroscopyAreaDoseProduct",
    "KVP",
    "Exposure",
    "StudyDate",
    "StudyTime",
]

files = []
files.append(os.path.join(test_files_path, "DX-Im-Carestream_DR7500-1.dcm"))
files.append(os.path.join(test_files_path, "DX-Im-Carestream_DR7500-2.dcm"))
files.append(os.path.join(test_files_path, "DX-Im-Carestream_DRX.dcm"))
files.append(os.path.join(test_files_path, "DX-Im-GE_XR220-1.dcm"))
files.append(os.path.join(test_files_path, "DX-Im-GE_XR220-2.dcm"))
files.append(os.path.join(test_files_path, "DX-Im-GE_XR220-3.dcm"))

output_file = open("dx_tags.csv", "w+")

output_file.write("Filename,")
for tag in tags:
    output_file.write(tag + ",")

output_file.write("\n")

for file in files:
    output_file.write(file + ",")

    dataset = pydicom.dcmread(file)
    dataset.decode()

    for tag in tags:
        try:
            output_file.write(str(dataset[tag].value) + ",")
        except KeyError:
            output_file.write(",")

    output_file.write("\n")

output_file.close()
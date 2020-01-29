"""
    Copyright 2015 Jonathan Cole

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import math
import numpy as np
from .geomclass import Triangle3, Segment3
from .geomfunc import collimate, check_orthogonal, intersect, rotate_ray_y, get_bsf
import time
from decimal import *


def skin_map(x_ray, phantom, area, ref_ak, kv, cu_thickness, d_ref, table_length, table_width, transmission,
             table_mattress_thickness):
    """ This function calculates a skin dose map.

    Args:
        x_ray: the x-ray beam as a Segment_3
        phantom: the phantom object representing the surface to map on
        area: the area of the beam at the reference point in square cm
        ref_ak: the air kerma at the reference point
        kv: the peak kilovoltage
        cu_thickness: the copper filter thickness in mm
        d_ref: the distance to the interventional reference point in cm
        table_length: the length of the table in cm from head to foot
        table_width: the width of the table in cm
        transmission: the table and/or mattress transmission as a decimal (0 to 1.0)
        table_mattress_thickness: the table and/or mattress thickess in cm

    Returns:
        An array containing doses for each cell in the phantom.
    """
    ref_length_squared = math.pow(d_ref, 2)

    skinmap = np.zeros((phantom.width, phantom.height), dtype=np.dtype(Decimal))
    focus = x_ray.source
    table1 = Triangle3(np.array([-table_width / 2, 0, 0]), np.array([table_width / 2, 0, 0]),
                       np.array([-table_width / 2, table_length, 0]))
    table2 = Triangle3(np.array([-table_width / 2, table_length, 0]), np.array([table_width / 2, table_length, 0]),
                       np.array([table_width / 2, 0, 0]))

    it = np.nditer(skinmap, op_flags=['readwrite'], flags=['multi_index', 'refs_ok'])

    (myTriangle1, myTriangle2) = collimate(x_ray, area, d_ref)

    while not it.finished:

        lookup_row = it.multi_index[0]
        lookup_col = it.multi_index[1]
        my_x = phantom.phantomMap[lookup_row, lookup_col][0]
        my_y = phantom.phantomMap[lookup_row, lookup_col][1]
        my_z = phantom.phantomMap[lookup_row, lookup_col][2]
        my_ray = Segment3(focus, np.array([my_x, my_y, my_z]))
        reverse_normal = phantom.normalMap[lookup_row, lookup_col]

        if check_orthogonal(reverse_normal, my_ray):
            # Check to see if the beam hits the patient
            hit1 = intersect(my_ray, myTriangle1)
            hit2 = intersect(my_ray, myTriangle2)
            if hit1 is "hit" or hit2 is "hit":

                # Check to see if the beam passes through the table
                table_normal = Segment3(np.array([0, 0, 0]), np.array([0, 0, 1]))
                hit_table1 = intersect(my_ray, table1)
                hit_table2 = intersect(my_ray, table2)
                # If the beam passes the table and does so on the way in to the patient, correct the AK
                if hit_table1 is "hit" or hit_table2 is "hit":
                    if check_orthogonal(table_normal, my_ray):
                        sin_alpha = x_ray.vector[2] / x_ray.length
                        path_length = table_mattress_thickness / sin_alpha
                        mu = np.log(transmission) / (-table_mattress_thickness)
                        table_cor = np.exp(-mu * path_length)
                        ref_ak_cor = ref_ak * table_cor
                    # If the beam is more than 90 degrees (ie above the table) leave the AK alone
                    else:
                        ref_ak_cor = ref_ak
                # If the beam doesn't pass through the table, leave the AK alone
                else:
                    ref_ak_cor = ref_ak

                # Calculate the dose at the skin point by correcting for distance and BSF
                mylength_squared = pow(my_ray.length, 2)
                it[0] = Decimal(ref_length_squared / mylength_squared * ref_ak_cor *
                                get_bsf(kv, cu_thickness, math.sqrt(mylength_squared / ref_length_squared))).quantize(
                    Decimal('0.000000001'), rounding=ROUND_HALF_UP)

        it.iternext()

    return skinmap


def rotational(xray, start_angle, end_angle, frames, phantom, area, ref_ak, kv, cu_thickness, d_ref, table_length,
               table_width, transmission, table_mattress_thickness):
    """ This function computes the dose from a rotational exposure.

    Args:
        xray: the initial ray
        start_angle: the initial angle in degrees
        end_angle: the stop angle in degrees
        frames: the number of frames in the rotation
        phantom: the geomclass.phantom class being exposed
        area: the area of the beam
        ref_ak: the air kerma at the reference point
        kv: the kV used for the exposure
        cu_thickness: the copper filter used, if any
        d_ref: the reference distance
        table_length: the length of the table in cm from head to foot
        table_width: the width of the table in cm
        transmission: the table and/or mattress transmission as a decimal (0 to 1.0)
        table_mattress_thickness: the table and/or mattress thickess in cm

    Returns:
        A skin dose map.

    """
    try:
        rotation_angle = (end_angle - start_angle) / frames
    except TypeError as e:
        # We assume that it is Philips Allura XPer FD10 or FD20 data if start angle = -120 (propeller mode) or
        # -45 (roll mode) and endAngle is not available.
        if (end_angle is None) and (start_angle > -120.5) and (start_angle < -119.5):
            end_angle = 120
        elif (end_angle is None) and (start_angle > -45.5) and (start_angle < -44.5):
            end_angle = 135
        else:
            raise e
        rotation_angle = (end_angle - start_angle) / frames

    my_dose = skin_map(xray, phantom, area, ref_ak / frames, kv, cu_thickness, d_ref, table_length, table_width,
                       transmission, table_mattress_thickness)
    for i in range(1, frames - 1):
        xray = rotate_ray_y(xray, rotation_angle)
        my_dose = my_dose + skin_map(xray, phantom, area, ref_ak / frames, kv, cu_thickness, d_ref, table_length,
                                     table_width, transmission, table_mattress_thickness)
    return my_dose


def skinmap_to_png(colour, total_dose, filename, test_phantom, encode_16_bit_colour=None):
    """ Writes a dose map to a PNG file.

    Args:
        colour: a boolean choice of colour or black and white
        filename: the file name to write the PNG to
        test_phantom: the phantom used for calculations
        total_dose: the dose map to write
        encode_16_bit_colour: a boolean choice if colour output should be encoded 16 bit

    Returns:
        Nothing.

    """

    import png

    if colour:
        thresh_dose = 5.

        blue = np.zeros((test_phantom.width, test_phantom.height))

        red = total_dose * (255. / thresh_dose)
        red[total_dose[:, :] > thresh_dose] = 255

        green = (total_dose - thresh_dose) * (-255. / thresh_dose) + 255.
        green[green[:, :] > 255] = 255
        green[total_dose[:, :] == 0] = 0

        image_3d = np.dstack((red, green, blue))
        image_3d = np.reshape(image_3d, (-1, test_phantom.height * 3))

        f = open(filename, 'wb')

        w = png.Writer(test_phantom.height, test_phantom.width, greyscale=False, bitdepth=8)
        w.write(f, image_3d)
        f.close()

    elif encode_16_bit_colour:
        # White at 10 Gy
        thresh_dose = Decimal(10)
        total_dose = (total_dose * Decimal(65535)) / thresh_dose

        r, g = divmod(total_dose, 255)
        # r is the number of times 255 goes into totalDose; g is the remainder

        b = np.empty([test_phantom.width, test_phantom.height])
        b.fill(255)

        # To reconstruct the 16-bit value, do (r * b) + g

        image_3d = np.dstack((r, g, b))
        image_3d = np.reshape(image_3d, (-1, test_phantom.height * 3))

        f = open(filename, 'wb')

        w = png.Writer(test_phantom.height, test_phantom.width, greyscale=False, bitdepth=8)
        w.write(f, image_3d)
        f.close()

    else:
        # White at 10 Gy
        thresh_dose = Decimal(10)
        total_dose = (total_dose * Decimal(65535)) / thresh_dose

        f = open(filename, 'wb')

        w = png.Writer(test_phantom.height, test_phantom.width, greyscale=True, bitdepth=16)
        w.write(f, total_dose)
        f.close()


def write_results_to_txt(txtfile, csvfile, test_phantom, my_dose):
    """ This function writes useful skin dose results to a text file.

    Args:
        txtfile: the destination filename with path
        csvfile: the original data file name
        test_phantom: the phantom used for calculations
        my_dose: the skinDose object holding the results

    Returns:
        Nothing.

    """
    total_dose = my_dose.totalDose
    phantom_txt = str(test_phantom.width) + 'x' + str(test_phantom.height) + ' ' + test_phantom.phantomType + ' phantom'
    f = open(txtfile, 'w')
    f.write('{0:15} : {1:30}\n'.format('File created', time.strftime("%c")))
    f.write('{0:15} : {1:30}\n'.format('Data file', csvfile))
    f.write('{0:15} : {1:30}\n'.format('Phantom', phantom_txt))
    f.write('{0:15} : {1:30}\n'.format('Peak dose (Gy)', np.amax(total_dose)))
    f.write('{0:15} : {1:30}\n'.format('Cells > 3 Gy', np.sum(total_dose >= 3)))
    f.write('{0:15} : {1:30}\n'.format('Cells > 5 Gy', np.sum(total_dose >= 5)))
    f.write('{0:15} : {1:30}\n'.format('Cells > 10 Gy', np.sum(total_dose >= 10)))
    f.close()

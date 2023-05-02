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
from decimal import Decimal, ROUND_HALF_UP


def skin_map(
    x_ray,
    phantom,
    area,
    ref_ak,
    tube_voltage,
    cu_thickness,
    d_ref,
    table_length,
    table_width,
    transmission,
    table_mattress_thickness,
    angle_x,
    angle_y,
):
    """This function calculates a skin dose map.

    Args:
        x_ray: the x-ray beam as a Segment3
        phantom: the phantom object representing the surface to map on
        area: the area of the beam at the reference point in square cm
        ref_ak: the air kerma at the reference point
        tube_voltage: the peak kilovoltage
        cu_thickness: the copper filter thickness in mm
        d_ref: the distance to the interventional reference point in cm
        table_length: the length of the table in cm from head to foot
        table_width: the width of the table in cm
        transmission: the table and/or mattress transmission as a decimal (0 to 1.0)
        table_mattress_thickness: the table and/or mattress thickness in cm

    Returns:
        An array containing doses for each cell in the phantom.
    """
    ref_length_squared = math.pow(d_ref, 2)

    skin_dose_map = np.zeros(
        (int(phantom.width), int(phantom.height + phantom.phantom_head_height)),
        dtype=np.dtype(Decimal),
    )
    focus = x_ray.source
    table1 = Triangle3(
        np.array([-table_width / 2, 0, 0]),
        np.array([table_width / 2, 0, 0]),
        np.array([-table_width / 2, table_length, 0]),
    )
    table2 = Triangle3(
        np.array([-table_width / 2, table_length, 0]),
        np.array([table_width / 2, table_length, 0]),
        np.array([table_width / 2, 0, 0]),
    )

    (triangle1, triangle2) = collimate(x_ray, area, d_ref, angle_x, angle_y)

    ###########################################################################################
    # Count how many 1 cm^2 cells the field hits - this is the total skin area that is exposed
    # and will include any curved areas of the phantom and also account for change in the field
    # area at the skin due to non-zero secondary beam angles.
    area_count = 0
    iterator = np.nditer(
        skin_dose_map, op_flags=["readwrite"], flags=["multi_index", "refs_ok"]
    )
    while not iterator.finished:
        lookup_row = iterator.multi_index[0]
        lookup_col = iterator.multi_index[1]
        my_x = phantom.phantom_map[lookup_row, lookup_col][0]
        my_y = phantom.phantom_map[lookup_row, lookup_col][1]
        my_z = phantom.phantom_map[lookup_row, lookup_col][2]
        my_ray = Segment3(focus, np.array([my_x, my_y, my_z]))
        reverse_normal = phantom.normal_map[lookup_row, lookup_col]
        if check_orthogonal(reverse_normal, my_ray):
            # Check to see if the beam hits the patient
            hit1 = intersect(my_ray, triangle1)
            hit2 = intersect(my_ray, triangle2)
            if hit1 == "hit" or hit2 == "hit":
                area_count = area_count + 1
        iterator.iternext()
    equiv_field_size_on_skin = math.sqrt(area_count)
    ###########################################################################################

    # Reset the existing iterator so we can iterate through the map to calculate the dose to each cell
    iterator.reset()

    index0_length = iterator.shape[0]

    while not iterator.finished:

        # DJP note: myRay describes a single path from the source to the corner of the cell, defined by myX, myY, myZ.
        # When a cell is only partially irradiated by the x-ray field myRay may not intersect with the field because
        # myRay is positioned in one corner of the cell. For this to be more robust a check is now made for four rays -
        # one at each corner of the cell. This will ensure that cells that are only partially irradiated are
        # always registered as being hit. Thanks to my colleague Hannah Thurlbeck for this idea.

        my_rays = []
        my_reverse_normals = []

        # Ray to first cell corner
        lookup_row = iterator.multi_index[0]
        lookup_col = iterator.multi_index[1]
        my_x = phantom.phantom_map[lookup_row, lookup_col][0]
        my_y = phantom.phantom_map[lookup_row, lookup_col][1]
        my_z = phantom.phantom_map[lookup_row, lookup_col][2]
        my_ray = Segment3(focus, np.array([my_x, my_y, my_z]))
        reverse_normal = phantom.normal_map[lookup_row, lookup_col]
        my_rays.append(my_ray)
        my_reverse_normals.append(reverse_normal)

        # Ray to second cell corner
        if lookup_row < index0_length - 1:
            next_row = lookup_row + 1
        else:
            next_row = 0
        my_x = phantom.phantom_map[next_row, lookup_col][0]
        my_y = phantom.phantom_map[next_row, lookup_col][1]
        my_z = phantom.phantom_map[next_row, lookup_col][2]
        my_ray = Segment3(focus, np.array([my_x, my_y, my_z]))
        reverse_normal = phantom.normal_map[next_row, lookup_col]
        my_rays.append(my_ray)
        my_reverse_normals.append(reverse_normal)

        # Ray to third cell corner
        if lookup_col > 0:
            prev_col = lookup_col - 1
        else:
            prev_col = 0
        my_x = phantom.phantom_map[next_row, prev_col][0]
        my_y = phantom.phantom_map[next_row, prev_col][1]
        my_z = phantom.phantom_map[next_row, prev_col][2]
        my_ray = Segment3(focus, np.array([my_x, my_y, my_z]))
        reverse_normal = phantom.normal_map[next_row, prev_col]
        my_rays.append(my_ray)
        my_reverse_normals.append(reverse_normal)

        # Ray to fourth cell corner
        my_x = phantom.phantom_map[lookup_row, prev_col][0]
        my_y = phantom.phantom_map[lookup_row, prev_col][1]
        my_z = phantom.phantom_map[lookup_row, prev_col][2]
        my_ray = Segment3(focus, np.array([my_x, my_y, my_z]))
        reverse_normal = phantom.normal_map[lookup_row, prev_col]
        my_rays.append(my_ray)
        my_reverse_normals.append(reverse_normal)

        for my_ray in my_rays:

            if check_orthogonal(reverse_normal, my_ray):
                # Check to see if the beam hits the patient
                hit1 = intersect(my_ray, triangle1)
                hit2 = intersect(my_ray, triangle2)
                if hit1 == "hit" or hit2 == "hit":

                    # Check to see if the beam passes through the table
                    table_normal = Segment3(np.array([0, 0, 0]), np.array([0, 0, 1]))
                    hit_table1 = intersect(my_ray, table1)
                    hit_table2 = intersect(my_ray, table2)
                    # If the beam passes the table and does so on the way in to the patient, correct the AK
                    if hit_table1 == "hit" or hit_table2 == "hit":
                        if check_orthogonal(table_normal, my_ray):
                            sin_alpha = x_ray.vector[2] / x_ray.length
                            path_length = table_mattress_thickness / sin_alpha
                            mu_table = np.log(transmission) / (
                                -table_mattress_thickness
                            )
                            table_cor = np.exp(-mu_table * path_length)
                            ref_ak_cor = ref_ak * table_cor
                        # If the beam is more than 90 degrees (ie above the table) leave the AK alone
                        else:
                            ref_ak_cor = ref_ak
                    # If the beam doesn't pass through the table, leave the AK alone
                    else:
                        ref_ak_cor = ref_ak

                    # Only attribute a quarter of the refAKcor to the cell because we are going to check four rays.
                    # This will account, to some extent, for partial irradiation of a cell.
                    ref_ak_cor = ref_ak_cor / 4.0

                    # Calculate the dose at the skin point by correcting for distance and BSF
                    mylength_squared = pow(my_ray.length, 2)

                    iterator[0] = iterator[0] + Decimal(
                        ref_length_squared
                        / mylength_squared
                        * ref_ak_cor
                        * get_bsf(
                            tube_voltage,
                            cu_thickness,
                            equiv_field_size_on_skin,
                        )
                    ).quantize(Decimal("0.000000001"), rounding=ROUND_HALF_UP)
        iterator.iternext()

    return skin_dose_map


def rotational(
    xray,
    start_angle,
    end_angle,
    frames,
    phantom,
    area,
    ref_ak,
    tube_voltage,
    cu_thickness,
    d_ref,
    table_length,
    table_width,
    transmission,
    table_mattress_thickness,
    angle_x,
    angle_y,
):
    """This function computes the dose from a rotational exposure.

    Args:
        xray: the initial ray
        start_angle: the initial angle in degrees
        end_angle: the stop angle in degrees
        frames: the number of frames in the rotation
        phantom: the geomclass.phantom class being exposed
        area: the area of the beam
        ref_ak: the air kerma at the reference point
        tube_voltage: the kV used for the exposure
        cu_thickness: the copper filter used, if any
        d_ref: the reference distance
        table_length: the length of the table in cm from head to foot
        table_width: the width of the table in cm
        transmission: the table and/or mattress transmission as a decimal (0 to 1.0)
        table_mattress_thickness: the table and/or mattress thickess in cm
        angle_x: primary angle
        angle_y: secondary angle

    Returns:
        A skin dose map.

    """
    try:
        rotation_angle = (end_angle - start_angle) / frames
    except TypeError as type_error:
        # We assume that it is Philips Allura XPer FD10 or FD20 data if start angle = -120 (propeller mode) or
        # -45 (roll mode) and endAngle is not available.
        if (end_angle is None) and (-120.5 < start_angle < -119.5):
            end_angle = 120
        elif (end_angle is None) and (-45.5 < start_angle < -44.5):
            end_angle = 135
        else:
            raise type_error
        rotation_angle = (end_angle - start_angle) / frames

    my_dose = skin_map(
        xray,
        phantom,
        area,
        ref_ak / frames,
        tube_voltage,
        cu_thickness,
        d_ref,
        table_length,
        table_width,
        transmission,
        table_mattress_thickness,
        angle_x,
        angle_y,
    )
    for i in range(1, frames - 1):
        xray = rotate_ray_y(xray, rotation_angle)
        my_dose = my_dose + skin_map(
            xray,
            phantom,
            area,
            ref_ak / frames,
            tube_voltage,
            cu_thickness,
            d_ref,
            table_length,
            table_width,
            transmission,
            table_mattress_thickness,
            angle_x,
            angle_y,
        )
    return my_dose


def skinmap_to_png(
    colour, total_dose, filename, test_phantom, encode_16_bit_colour=None
):
    """Writes a dose map to a PNG file.

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

    # LO: a bit strange usage of parameters. I would expect encode_16_bit_colour to be used only with colour set
    # to True. But it seems it is mutually exclusive. Next to that, I don't see the 16 bit data is built.
    # changed flow somewhat to keep Codacy happy, but didn't change behaviour in order not to break down anything.
    if colour or encode_16_bit_colour:
        if colour:
            thresh_dose = 5.0

            blue = np.zeros((test_phantom.width, test_phantom.height))

            red = total_dose * (255.0 / thresh_dose)
            red[total_dose[:, :] > thresh_dose] = 255

            green = (total_dose - thresh_dose) * (-255.0 / thresh_dose) + 255.0
            green[green[:, :] > 255] = 255
            green[total_dose[:, :] == 0] = 0

        else:
            # encode_16_bit_colour is True
            # White at 10 Gy
            thresh_dose = Decimal(10)
            total_dose = (total_dose * Decimal(65535)) / thresh_dose

            red, green = divmod(total_dose, 255)
            # red is the number of times 255 goes into total_dose; green is the remainder

            blue = np.empty([test_phantom.width, test_phantom.height])
            blue.fill(255)

            # To reconstruct the 16-bit value, do (red * blue) + green
            # LO: this doesn't seem to happen anywhere

        image_3d = np.dstack((red, green, blue))
        image_3d = np.reshape(image_3d, (-1, test_phantom.height * 3))

        with open(filename, "wb") as png_file:
            png_writer = png.Writer(
                test_phantom.height, test_phantom.width, greyscale=False, bitdepth=8
            )
            png_writer.write(png_file, image_3d)

    else:
        # White at 10 Gy
        thresh_dose = Decimal(10)
        total_dose = (total_dose * Decimal(65535)) / thresh_dose

        with open(filename, "wb") as png_file:
            png_writer = png.Writer(
                test_phantom.height, test_phantom.width, greyscale=True, bitdepth=16
            )
            png_writer.write(png_file, total_dose)


def write_results_to_txt(txtfile, csvfile, test_phantom, my_dose):
    """This function writes useful skin dose results to a text file.

    Args:
        txtfile: the destination filename with path
        csvfile: the original data file name
        test_phantom: the phantom used for calculations
        my_dose: the skinDose object holding the results

    Returns:
        Nothing.

    """
    total_dose = my_dose.total_dose
    phantom_txt = (
        str(test_phantom.width)
        + "x"
        + str(test_phantom.height)
        + " "
        + test_phantom.phantom_type
        + " phantom"
    )
    with open(txtfile, "w") as text_file:
        text_file.write("{0:15} : {1:30}\n".format("File created", time.strftime("%c")))
        text_file.write("{0:15} : {1:30}\n".format("Data file", csvfile))
        text_file.write("{0:15} : {1:30}\n".format("Phantom", phantom_txt))
        text_file.write(
            "{0:15} : {1:30}\n".format("Peak dose (Gy)", np.amax(total_dose))
        )
        text_file.write(
            "{0:15} : {1:30}\n".format("Cells > 3 Gy", np.sum(total_dose >= 3))
        )
        text_file.write(
            "{0:15} : {1:30}\n".format("Cells > 5 Gy", np.sum(total_dose >= 5))
        )
        text_file.write(
            "{0:15} : {1:30}\n".format("Cells > 10 Gy", np.sum(total_dose >= 10))
        )

"""
    Copyright 2016 Jonathan Cole

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


def intersect(a_ray, a_triangle):
    """ Derived from example code at http://geomalgorithms.com/a06-_intersect-2.html
    provided under the following license:

    Copyright 2001 softSurfer, 2012 Dan Sunday
    This code may be freely used and modified for any purpose
    providing that this copyright notice is included with it.
    SoftSurfer makes no warranty for this code, and cannot be held
    liable for any real or imagined damage resulting from its use.
    Users of this code must verify correctness for their application.

    This function checks if a ray intersects a triangle

    Args:
        a_ray: the ray (Segment3) being projected
        a_triangle: the triangle (Triangle3) to hit

    Returns:
        A string describing the status of the hit.
    """

    # Get triangle plane normal
    plane_normal = np.cross(a_triangle.vector_ab, a_triangle.vector_ac)
    if np.array_equal(plane_normal, [0, 0, 0]):
        output = "degenerate"
        return output

    # Determine if ray intersects with triangle plane
    w0 = a_ray.source - a_triangle.point_a
    a = -np.dot(plane_normal, w0)
    b = np.dot(plane_normal, a_ray.vector)

    # Get the intersection point of the ray with the triangle plane
    if abs(b) < 0.00000001:
        output = "same plane"
        return output

    r = a / b
    if r < 0.0:
        output = "away from triangle"
        return output

    intersect_point = a_ray.source + r * a_ray.vector

    # Determine if intersection point is within inside the triangle
    uu = np.dot(a_triangle.vector_ab, a_triangle.vector_ab)
    uv = np.dot(a_triangle.vector_ab, a_triangle.vector_ac)
    vv = np.dot(a_triangle.vector_ac, a_triangle.vector_ac)
    w = intersect_point - a_triangle.point_a
    wu = np.dot(w, a_triangle.vector_ab)
    wv = np.dot(w, a_triangle.vector_ac)
    d = uv * uv - uu * vv

    s = (uv * wv - vv * wu) / d
    t = (uv * wu - uu * wv) / d

    # Hit some precision problems so either use this fix or use an exact maths library. This seems easier for now.

    if s < 0.0 or s > 1.000000000001:  # Technically >1 but for the rounding errors.
        output = "outside test 1"
    elif t < 0.0 or (s + t) > 1.000000000001:
        output = "outside test 2" + "S:" + str(s) + " t:" + str(t)
    else:
        output = "hit"

    return output


def collimate(a_ray, area, d_ref):
    """ This function produces a pair of triangles representing a square field
    of a collimated x-ray beam. These are then used for intersection checks to
    see if the phantom cell sees radiation.

    Args:
        a_ray: the x-ray beam from focus to isoncentre as a Segment_3
        area: an area of the beam in square centimetres at any arbitrary distance
        d_ref: the reference distance the area is defined at

    Returns:
        A tuple of two touching triangles making a square field oriented
        perpendicular to the beam direction.
    """
    side_length = math.sqrt(area) * 10 / d_ref  # Side at 10 cm

    centre_point = a_ray.source + a_ray.vector / a_ray.length * 10  # point at 10 cm up on the midline of the ray

    xvector = np.array([np.sin(a_ray.xangle), 0, -np.cos(a_ray.xangle)])
    yvector = np.array([0, np.sin(a_ray.yangle), np.cos(a_ray.yangle)])
    point_a = centre_point + ((side_length / 2) * xvector) + ((side_length / 2) * yvector)
    point_b = centre_point + ((side_length / 2) * xvector) - ((side_length / 2) * yvector)
    point_c = centre_point - ((side_length / 2) * xvector) + ((side_length / 2) * yvector)
    point_d = centre_point - ((side_length / 2) * xvector) - ((side_length / 2) * yvector)

    triangle_1 = Triangle3(point_d, point_b, point_c)
    triangle_2 = Triangle3(point_a, point_b, point_c)
    return triangle_1, triangle_2


def build_ray(table_longitudinal, table_lateral, table_height, lr_angle, cc_angle, d_ref):
    """ This function takes RDSR geometry information and uses it to build
    an x-ray (Segment_3) taking into account translation and rotation.

    Args:
        table_longitudinal: the table longitudinal offset as defined in the DICOM statement
        table_lateral: the table lateral offset as defined in the DICOM statement
        table_height: the table height offset as defined in the DICOM statement
        lr_angle: the left-right angle. +90 is detector to the patient's left
        cc_angle: the cranial-caudal angle in degrees. +90 is detector to the head
        d_ref: the reference distance to the isocentre

    Returns:
        A ray (Segment_3) representing the x-ray beam.
    """
    x = 0
    y = 0
    z = -d_ref

    lr_rads = (lr_angle / 360.) * 2. * math.pi
    cc_rads = (cc_angle / 360.) * 2. * math.pi

    sin_lr = math.sin(lr_rads)
    cos_lr = math.cos(lr_rads)
    sin_cc = math.sin(cc_rads)
    cos_cc = math.cos(cc_rads)

    # Full maths: x_new = z*sin_lr + x*cos_lr
    x_new = z * sin_lr

    # Full maths: z_step = z*cos_lr - x*sin_lr
    z_step = z * cos_lr

    # Full maths: y_new = y*cos_cc - z_step*sin_cc
    y_new = -z_step * sin_cc

    # Full maths: z_new = y*sin_cc + z_step*cos_cc
    z_new = z_step * cos_cc

    z_translated = z_new + table_height

    x_translated = x_new - table_longitudinal

    y_translated = y_new + table_lateral

    focus = np.array([x_translated, y_translated, z_translated])
    isocentre = np.array([x - table_longitudinal, y + table_lateral, 0 + table_height])

    my_ray = Segment3(focus, isocentre)

    return my_ray


def check_orthogonal(segment1, segment2):
    """ This function checks whether two segments are within 90 degrees

    Args:
        segment1: A Segment_3 line segment
        segment2: Another Segment_3 line segment

    Returns:
        A boolean: true if the segments are within 90 degrees,
        false if outside.
    """
    return np.dot(segment1.vector, segment2.vector) >= 0


def check_miss(source, centre, target1, target2):
    """ This function compares two angles between a source and two targets.
    If the second target is at a steeper angle than the first, it misses.

    Args:
        source: the shared start point
        centre: the reference point to angle against
        target1: the triangle corner
        target2: the ray cell target

    Returns:
        A boolean: true if the second target misses.
    """

    main_line = centre - source
    main_length = np.linalg.norm(main_line)
    target1_vec = target1 - source
    # target1_length = np.linalg.norm(target1_vec)
    target1_length = math.sqrt(math.pow(target1_vec[0], 2) + math.pow(target1_vec[1], 2) + math.pow(target1_vec[2], 2))
    target2_vec = target2 - source
    # target2_length = np.linalg.norm(target2_vec)
    target2_length = math.sqrt(math.pow(target2_vec[0], 2) + math.pow(target2_vec[1], 2) + math.pow(target2_vec[2], 2))

    angle1 = np.arccos(np.dot(main_line, target1_vec) / (main_length * target1_length))
    angle2 = np.arccos(np.dot(main_line, target2_vec) / (main_length * target2_length))

    return abs(angle2) > abs(angle1)


def find_nearest(array, value):
    """ This function finds the closest match to a value from an array.

    Args:
        The array to search and the value to compare.

    Returns:
        The index of the matching value.
    """
    return (np.abs(array - value)).argmin()


def get_bsf(tube_voltage, cu_thickness, size):
    """ This function gives a BSF and f-factor combined. Data from:
    Backscatter factors and mass energy-absorption coefficient ratios for diagnostic radiology dosimetry
    Hamza Benmakhlouf et al 2011 Phys. Med. Biol. 56 7179 doi:10.1088/0031-9155/56/22/012

    Args:
        tube_voltage: The peak kilovoltage
        cu_thickness: the added copper filtration in mm. In addition, 3.1 mm Al is assumed by default
        size: The side of the square field incident on the patient

    Returns:
        A combined backscatter factor and f-factor.
    """
    kv_table = np.array([50, 80, 110, 150])
    cu_table = np.array([0, 0.1, 0.2, 0.3, 0.6, 0.9])
    size_table = np.array([5, 10, 20, 35])

    lookup_kv = find_nearest(kv_table, tube_voltage)
    lookup_cu = find_nearest(cu_table, cu_thickness)
    lookup_size = find_nearest(size_table, size)

    lookup_array = np.array([
        [[1.2, 1.3, 1.3, 1.3], [1.3, 1.3, 1.4, 1.4], [1.3, 1.4, 1.4, 1.4], [1.3, 1.4, 1.4, 1.5], [1.3, 1.4, 1.5, 1.5],
         [1.3, 1.5, 1.5, 1.6]],
        [[1.3, 1.4, 1.4, 1.5], [1.3, 1.4, 1.5, 1.5], [1.3, 1.5, 1.6, 1.6], [1.4, 1.5, 1.6, 1.7], [1.4, 1.5, 1.7, 1.7],
         [1.4, 1.5, 1.7, 1.7]],
        [[1.3, 1.4, 1.5, 1.5], [1.3, 1.5, 1.6, 1.6], [1.3, 1.5, 1.6, 1.7], [1.4, 1.5, 1.6, 1.7], [1.4, 1.5, 1.7, 1.7],
         [1.3, 1.5, 1.7, 1.7]],
        [[1.3, 1.5, 1.5, 1.6], [1.3, 1.5, 1.6, 1.6], [1.3, 1.5, 1.6, 1.7], [1.3, 1.5, 1.6, 1.7], [1.3, 1.5, 1.6, 1.7],
         [1.3, 1.5, 1.6, 1.7]]
    ])

    return lookup_array[lookup_kv, lookup_cu, lookup_size]


def rotate_ray_y(segment1, angle):
    """ This function rotates a ray around the end point of the ray by angle degrees.

    Args:
        segment1: the ray to rotate_ray_y
        angle: rotation angle in degrees

    Returns:
        A new ray with the same end point but the start point rotated.
    """
    isocentre = segment1.target
    translate_source = segment1.source - isocentre
    angle_rads = angle / 360 * 2. * math.pi
    my_y = translate_source[1]
    my_x = translate_source[2] * math.sin(angle_rads) + translate_source[0] * math.cos(angle_rads)
    my_z = translate_source[2] * math.cos(angle_rads) - translate_source[0] * math.sin(angle_rads)
    new_source = np.array([my_x, my_y, my_z])
    return Segment3(new_source + isocentre, isocentre)


def get_table_trans(tube_voltage, cu_thickness):
    """ This function gives just the table transmission factor based
    on measurements made at the Royal Free Hospital on a Siemens Artis Zeego
    in early 2016.

    Args:
        tube_voltage: The peak kilovoltage
        cu_thickness: the added copper filtration in mm. In addition, 3.1 mm Al is assumed by default

    Returns:
        A transmission factor for the table without a mattress.
    """
    kv_table = np.array([60, 80, 110, 125])
    cu_table = np.array([0, 0.1, 0.2, 0.3, 0.6, 0.9])

    lookup_kv = find_nearest(kv_table, tube_voltage)
    lookup_cu = find_nearest(cu_table, cu_thickness)

    lookup_array = np.array([
        [0.80, 0.82, 0.82, 0.82],
        [0.84, 0.84, 0.86, 0.87],
        [0.86, 0.86, 0.88, 0.88],
        [0.84, 0.86, 0.88, 0.89],
        [0.86, 0.87, 0.88, 0.90],
        [0.86, 0.87, 0.89, 0.90]
    ])

    return lookup_array[lookup_cu, lookup_kv]


def get_table_mattress_trans(tube_voltage, cu_thickness):
    """ This function gives a table and mattress transmission factor based
    on measurements made at the Royal Free Hospital on a Siemens Artis Zeego
    in early 2016.

    Args:
        tube_voltage: The peak kilovoltage
        cu_thickness: the added copper filtration in mm. In addition, 3.1 mm Al is assumed by default

    Returns:
        A combined transmission factor for table and mattress.
    """
    kv_table = np.array([60, 80, 110, 125])
    cu_table = np.array([0, 0.1, 0.2, 0.3, 0.6, 0.9])

    lookup_kv = find_nearest(kv_table, tube_voltage)
    lookup_cu = find_nearest(cu_table, cu_thickness)

    lookup_array = np.array([
        [0.66, 0.68, 0.71, 0.72],
        [0.73, 0.75, 0.78, 0.78],
        [0.75, 0.78, 0.81, 0.81],
        [0.76, 0.79, 0.83, 0.83],
        [0.79, 0.81, 0.85, 0.85],
        [0.80, 0.82, 0.85, 0.86]
    ])

    return lookup_array[lookup_cu, lookup_kv]

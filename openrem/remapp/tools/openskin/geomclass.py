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

import numpy as np
import math


class Triangle3:
    """This is a class to construct triangles in 3 dimensional Cartesian coordinate space.

    These triangles are used to construct target objects for intersection

    Constructor Args:
        Three (3x1) numpy arrays representing the coordinates of the vertices.

    Properties:
        a: The first vertex
        b: The second vertex
        c: The third vertex
        u: First of two vectors defining the triangle by making a v shape. Used for intersection calculations.
        v: Second of two vectors defining the triangle by making a v shape. Used for intersection calculations.
    """

    def __init__(self, point_3a, point_3b, point_3c):
        self.point_a = point_3a
        self.point_b = point_3b
        self.point_c = point_3c
        self.vector_ab = point_3b - point_3a
        self.vector_ac = point_3c - point_3a


class Segment3:
    """ This is a class to construct line segments in 3 dimensional Cartesian coordinate space.

    These segments are used to represent rays from the focus to isocentre or from the focus to the skin cell
    under evaluation.

    Constructor Args:
        Two (3x1) numpy arrays representing the coordinates of the start and end of the segment.

    Properties:
        source: the start point
        target: the end point
        vector: the vector from start to end
        length: the magnitude of the segment
        xangle: the angle between the segment and the x-axis
        yangle: the angle between the segment and the y-axis
    """

    def __init__(self, point_3a, point_3b):
        self.source = point_3a
        self.target = point_3b
        self.vector = point_3b - point_3a
        self.length = np.linalg.norm(self.vector)
        denom0 = self.vector[0]
        if abs(denom0) < 0.00000001:
            denom0 = 0.00000001
        denom1 = self.vector[1]
        if abs(denom1) < 0.00000001:
            denom1 = 0.00000001
        self.xangle = np.arctan(self.vector[2] / denom0)
        self.yangle = np.arctan(self.vector[2] / denom1)


class PhantomFlat:
    """ This class defines a surface to which dose will be delivered.

    Constructor Args:
        phantom_type: the type of phantom being assembled
        origin: the coordinate system for the phantom. For example, some systems
            use the head end, table plane in the patient mid-line. So the origin
            would be [25,0,0] on a 50 cm wide phantom.
        width: the number of cells the phantom is wide
        height: the number of cells the phantom is long
        scale: the steps (in cm) to make between cells

    Properties:
        width: the width of the phantom in cells
        height: the height of the phantom in cells
        phantom_map: an array containing a list of points which represent each cell in the phantom surface to be
        evaluated
        normal_map: an array containing line segments (Segment_3) indicating the outward facing surface of the cell
    """

    def __init__(self, phantom_type, origin, width, height, scale):
        self.phantom_type = phantom_type
        self.width = width
        self.height = height
        if phantom_type == "flat":
            z_offset = -origin[2]
            self.phantom_map = np.empty((width, height), dtype=object)
            self.normal_map = np.empty((width, height), dtype=object)
            iterator = np.nditer(self.phantom_map, op_flags=['readwrite'], flags=['multi_index', 'refs_ok'])
            while not iterator.finished:
                my_x = iterator.multi_index[0] * scale - origin[0]
                my_y = iterator.multi_index[1] * scale - origin[1]
                self.phantom_map[iterator.multi_index[0], iterator.multi_index[1]] = np.array([my_x, my_y, z_offset])

                plane_point = np.array([my_x, my_y, z_offset])
                outside_point = np.array([my_x, my_y, z_offset - 1])
                # The normal is defined going back in to the plane, to make checking alignment easier
                normal = Segment3(outside_point, plane_point)
                self.normal_map[iterator.multi_index[0], iterator.multi_index[1]] = normal
                iterator.iternext()


class Phantom3:
    """ This class defines a surface in 3d to project dose onto.
    It is formed of a central cuboid with two semi cylinders on the sides.

    Constructor Args:
        origin: the coordinate system for the phantom. For example, some systems
            use the head end, table plane in the patient mid-line. This phantom
            assumes the origin is at the head, on the mid-line and on the table
            for [0,0,0].
        width: the number of cells the phantom is wide. Includes the wrap around
        height: the number of cells the phantom is long
        scale: the steps (in cm) to make between cells. Not used (yet?)

    Properties:
        width: the total distance around the phantom (distance around both curved edges,
            plus the distance across the flat front and back - not really the width...)
        height: the height of the phantom in cells
        phantom_width: the horizontal distance across the 3D phantom
        phantom_height: the height of the 3D phantom
        phantom_depth: the depth of the 3D phantom
        phantom_flat_dist: the width of the flat part of the phantom (same for front and back)
        phantom_curved_dist: the distance around one curved side of the phantom (same for left and right sides)
        phantom_map: an array containing a list of points which represent each cell in the phantom surface to be
        evaluated
        normal_map: an array containing line segments (Segment_3) indicating the outward facing surface of the cell
        phantom_type: set to "3d"
    """

    def __init__(self, origin, scale=1, mass=73.2, height=178.6, pat_pos="HFS"):

        ref_height = 178.6
        ref_mass = 73.2
        ref_torso = 70.
        ref_radius = 10.
        ref_width = 14.4
        torso = ref_torso * height / ref_height
        radius = ref_radius / math.sqrt(height / ref_height) * math.sqrt(mass / ref_mass)

        if pat_pos == "HFS":
            prone = False
            pat_pos_z = 1.
            pat_pos_y = 1.
        elif pat_pos == "FFS":
            prone = False
            pat_pos_z = 1.
            pat_pos_y = -1.
            origin[1] = origin[1] - 174 * height / ref_height
        elif pat_pos == "HFP":
            prone = True
            pat_pos_z = -1.
            pat_pos_y = 1.
        elif pat_pos == "FFP":
            prone = True
            pat_pos_z = -1.
            pat_pos_y = -1.
            origin[1] = origin[1] - 174 * height / ref_height
        else:
            raise ValueError('patient position has an unknown value ({patpos})'.format(patpos=pat_pos))

        part_circumference = math.pi * radius
        round_circumference = round(part_circumference, 0)
        flat_width = ref_width / ref_radius * radius
        round_flat = round(flat_width, 0)
        flat_spacing = flat_width / round_flat
        head_height = round(24 * height / ref_height)
        head_circumference = 58
        radius_head = head_circumference / (2 * math.pi)

        # The three properties were added by DJP to describe
        # the dimensions of the 3D phantom.
        self.phantom_width = int(round(flat_width + 2 * radius, 0))
        self.phantom_height = int(round(torso, 0))
        self.phantom_depth = round(radius * 2, 0)
        self.phantom_flat_dist = round_flat
        self.phantom_curved_dist = round_circumference
        self.phantom_head_radius = radius_head
        self.phantom_head_height = head_height

        self.width = int(2 * round_circumference + 2 * round_flat)
        self.height = int(round(torso, 0))
        self.phantom_type = "3d"
        self.phantom_map = np.empty((self.width, self.height + self.phantom_head_height), dtype=object)
        self.normal_map = np.empty((self.width, self.height + self.phantom_head_height), dtype=object)
        transition1 = (round_flat / 2.) + 0.5  # Centre line flat to start of curve.
        transition2 = transition1 + round_circumference  # End of first curve to table flat
        transition3 = transition2 + round_flat  # End of table flat to second curve
        transition4 = transition3 + round_circumference  # End of second curve to flat back to centre line
        iterator = np.nditer(self.phantom_map, op_flags=['readwrite'], flags=['multi_index', 'refs_ok'])

        while not iterator.finished:
            # Start top, centre line.
            row_index = iterator.multi_index[0] - origin[0]
            col_index = iterator.multi_index[1] - origin[1]
            angle_step = math.pi / round_circumference
            angle_step_head = 2 * math.pi / head_circumference
            z_offset = -origin[2]

            if row_index < transition1 and col_index > self.phantom_head_height - origin[1]:
                my_z = (2. * radius + z_offset) * pat_pos_z
                my_x = row_index * flat_spacing - (round_flat / 2.) + round(round_flat / 2., 0)
                my_y = col_index * pat_pos_y
                normal = Segment3(np.array([my_x, my_y, my_z + pat_pos_z]), np.array([my_x, my_y, my_z]))
            elif transition1 <= row_index < transition2 and col_index > self.phantom_head_height - origin[1]:
                my_y = col_index * pat_pos_y
                my_x = flat_spacing * round(transition1, 0) - 1 + radius * math.sin(
                    angle_step * (row_index - round(transition1, 0) + 1)) - (round_flat / 2.) + round(round_flat / 2.,
                                                                                                      0)
                my_z = (2. * radius + z_offset + radius * math.cos(
                    angle_step * (row_index - round(transition1, 0) + 1)) - radius) * pat_pos_z
                normal_x = my_x + math.sin(angle_step * (row_index - round(transition1, 0) + 1))
                normal_z = my_z + pat_pos_z * math.cos(angle_step * (row_index - round(transition1, 0) + 1))
                normal = Segment3(np.array([normal_x, my_y, normal_z]), np.array([my_x, my_y, my_z]))
            elif transition2 <= row_index < transition3 and col_index > self.phantom_head_height - origin[1]:
                my_z = z_offset * pat_pos_z
                my_x = flat_width - (row_index - round_circumference) * flat_spacing + ((round_flat / 2.) - round(
                    round_flat / 2., 0)) * (row_index - round_circumference) / abs(row_index - round_circumference)
                my_y = col_index * pat_pos_y
                normal = Segment3(np.array([my_x, my_y, my_z - pat_pos_z]), np.array([my_x, my_y, my_z]))
            elif transition3 <= row_index < transition4 and col_index > self.phantom_head_height - origin[1]:
                my_y = col_index * pat_pos_y
                my_x = -flat_spacing * round(round_flat / 2, 0) - radius * math.sin(
                    angle_step * (row_index - round(transition3, 0) + 1)) - (round_flat / 2.) + round(round_flat / 2.,
                                                                                                      0)
                my_z = (z_offset - radius * math.cos(
                    angle_step * (row_index - round(transition3, 0) + 1)) + radius) * pat_pos_z
                normal_x = my_x - math.sin(angle_step * (row_index - round(transition3, 0) + 1))
                normal_z = my_z - pat_pos_z * math.cos(angle_step * (row_index - round(transition3, 0) + 1))
                normal = Segment3(np.array([normal_x, my_y, normal_z]), np.array([my_x, my_y, my_z]))
            elif row_index >= transition4 and col_index > self.phantom_head_height - origin[1]:
                my_z = (2. * radius + z_offset) * pat_pos_z
                my_x = (row_index - self.width) * flat_spacing - (round_flat / 2.) + round(round_flat / 2., 0)
                my_y = col_index * pat_pos_y
                normal = Segment3(np.array([my_x, my_y, my_z + pat_pos_z]), np.array([my_x, my_y, my_z]))
            elif row_index < head_circumference and col_index <= self.phantom_head_height - origin[1]:  #phantom head map
                my_y = (col_index) * pat_pos_y
                my_x = radius_head * math.cos(angle_step_head * (row_index)) - (round_flat / 2.) + round(round_flat / 2., 0)
                my_z = (z_offset + radius_head * (math.sin(angle_step_head * row_index) + 1))* pat_pos_z
                normal_x = my_x + math.sin(angle_step_head * (row_index))
                normal_z = my_y + math.cos(angle_step_head * (row_index + 0))
                normal = Segment3(np.array([normal_x, my_y, normal_z]), np.array([my_x, my_y, my_z]))
            else:
                my_y, my_x, my_z = [-999, -999, -999] 
                normal = Segment3(np.array([-999, -999, -999]), np.array([-999, -999, -999]))
                #for now a trick to have single phantom map for both head and torso, it would be neater to
                #implement a phantom map for both head and torso.    
                #this region of the phantom map will never intersect with ray segments and will not be used in JS             
            self.phantom_map[iterator.multi_index[0], iterator.multi_index[1]] = np.array([my_x, my_y, my_z])
            self.normal_map[iterator.multi_index[0], iterator.multi_index[1]] = normal
            iterator.iternext()
        # Flip to correct left and right so iterator becomes a view of the back.
        # self.phantom_map = np.flipud(self.phantom_map)
        # self.normal_map = np.flipud(self.normal_map)
        self.phantom_map = np.fliplr(self.phantom_map)
        self.normal_map = np.fliplr(self.normal_map)


        if prone:
            self.normal_map = np.roll(self.normal_map, int(self.phantom_flat_dist + self.phantom_curved_dist), axis=0)
            self.phantom_map = np.roll(self.phantom_map, int(self.phantom_flat_dist + self.phantom_curved_dist), axis=0)
            self.phantom_map = np.flipud(self.phantom_map)
            self.normal_map = np.flipud(self.normal_map)


class SkinDose:
    """ This class holds dose maps for a defined phantom. It is intended
    to simplify combining multiple views.

    Constructor Args:
        phantom: the phantom being irradiated.

    Properties:
        phantom: the phantom being irradiated
        views: a list of the irradiations included
        dose_array: an array of doses delivered to the phantom
        total_dose: a summed array of doses
        fliplr: flip the left and right of the dose map to provide a view from behind the patient
    """

    def __init__(self, phantom):
        self.phantom = phantom
        self.views = []
        self.dose_array = []
        self.total_dose = []

    def add_view(self, view_str):
        """
        Add a view (irradiation event) to the list of views (irradiation events)
        :param view_str: the view number
        :return: Nothing
        """
        if len(self.views) == 0:
            self.views = view_str
        else:
            self.views = np.vstack((self.views, view_str))

    def add_dose(self, skin_map):
        """
        Add the skin-dose of a specific view/irradiation event to the "summed" skin-dose map

        :param skin_map: the skin-dose to add
        :return: Nothing
        """
        if len(self.dose_array) == 0:
            self.dose_array = skin_map
            self.total_dose = skin_map
        else:
            self.dose_array = np.dstack((self.dose_array, skin_map))
            self.total_dose = self.total_dose + skin_map

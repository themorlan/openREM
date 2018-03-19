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
..  module:: get_rgb_hex_colours.
    :synopsis: Module to return strings of rgb colours in hex format.

..  moduleauthor:: David Platten

"""
def get_rgb_hex_colours(n_colours):
    """Get a list of rgb colours in hex format, e.g. ['#00FF00', '#FF0000'].

    :param n_colours:   The number of colours required
    :type tag:          int
    :returns:           list
    """
    import colorsys

    hsv_tuples = [(x * 1.0 / n_colours, 0.5, 0.5) for x in range(n_colours)]
    rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
    colours = []
    for (r, g, b) in rgb_tuples:
        colours.append('#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)))
    return colours

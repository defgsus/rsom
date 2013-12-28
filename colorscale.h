/*  This is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either
    version 3.0 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this software; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*/
/** @file
    @brief colorscale container (using QColor)

    @version 2013/12/18 init

    copyright 2013 stefan.berke @ modular-audio-graphics.com
*/
#ifndef COLORSCALE_H
#define COLORSCALE_H

#include <QColor>

/** some arbitrary but unique color scale for values of 0.0 to 1.0 */
struct ColorScale
{
    /** some precalculated color scale (256) */
    std::vector<QColor> color_map;

    /** constructor calculates colors */
    ColorScale() { calc_scale(); }

    /** (re-)calc the colors.
        Currently, there is no need to do that. */
    void calc_scale();

    /** Gets the color for value 'f' (0-1) */
    QColor get(const float f) const;

    /** Gets a color from a number of vectors */
    QColor get_spectral(const float * f, size_t num, float amp = 1.f) const;
};


#endif // COLORSCALE_H

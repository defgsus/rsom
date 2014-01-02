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
#include "colorscale.h"


void ColorScale::calc_scale()
{
    // create color-scale
    color_map.resize(256);

    for (int i=0; i<256; ++i)
    {
        float t = (float)i/255;

        color_map[i].setHslF(1.0-t*0.835, 0.6 + 0.4 * t, 0.3 + 0.2 * t);

        /*
        t *= 7.0;
        color_map[i] = QColor(
            255 * ( 0.55-0.45*cosf(t    ) ),
            255 * ( 0.55-0.45*cosf(t*1.3) ),
            255 * ( 0.55-0.45*cosf(t*1.7) )
            );
        */
    }
}

QColor ColorScale::get(const float f) const
{
    const int i = std::max(0, std::min(255, (int)(f * 255.f + 0.5f) ));
    return color_map[i];
}

float specf(float t, float v) { return v * 2.5f * (cosf(t*t*19.f) * 0.5f + 0.3f + 0.2f * cosf(v*29.f)); }

/** get a color from a number of vectors */
QColor ColorScale::get_spectral(const float * f, size_t num, float amp) const
{
    if (!num) return QColor(0,0,0);

    float r=0, g=0, b=0;
    for (size_t k=0; k<num; ++k)
    {
        const float
            t = (float)k/(num-1),
            v = f[k];
        r += specf(1.f - t, v);
        g += specf(std::max(0.f, fabsf(0.5f - t)), v);
        b += specf(t, v);
    }
    const float a = 255.f / num * amp;

    return QColor(
           std::max(0, std::min(255, (int)(r * a) )),
           std::max(0, std::min(255, (int)(g * a) )),
           std::max(0, std::min(255, (int)(b * a) ))
           );
}

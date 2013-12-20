#ifndef COLORSCALE_H
#define COLORSCALE_H

#include <QColor>

/** some arbitrary but unique color scale for values of 0.0 to 1.0 */
struct ColorScale
{
    // some precalculated color scale (256)
    std::vector<QColor> color_map;

    ColorScale()
    {
        // create color-scale
        color_map.resize(256);
        for (int i=0; i<256; ++i)
        {
            float t = (float)i/255;
            t *= 7.0;
            color_map[i] = QColor(
                255 * ( 0.55-0.45*cosf(t    ) ),
                255 * ( 0.55-0.45*cosf(t*1.3) ),
                255 * ( 0.55-0.45*cosf(t*1.7) )
                );
        }
    }

    // get the color for value 'f' (0-1)
    QColor get(const float f) const
    {
        const int i = std::max(0, std::min(255, (int)(f * 255.f + 0.5f) ));
        return color_map[i];
    }
};


#endif // COLORSCALE_H

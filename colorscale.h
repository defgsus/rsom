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

    /* get the color for value 'f' (0-1) */
    QColor get(const float f) const
    {
        const int i = std::max(0, std::min(255, (int)(f * 255.f + 0.5f) ));
        return color_map[i];
    }

    /** get a color from a number of vectors */
    QColor get_spectral(const float * f, size_t num, float amp = 1.f) const
    {
        if (!num) return QColor(0,0,0);

        float r=0, g=0, b=0;
        for (size_t k=0; k<num; ++k)
        {
            const float
                t = (float)k/(num-1),
                v = f[k];
            r += v * (1.f - t);
            g += v * std::max(0.f, fabsf(0.5f - t));
            b += v * t;
        }
        const float a = 255.f / num * amp;

        return QColor(
               std::min(255, (int)(r * a)),
               std::min(255, (int)(g * a)),
               std::min(255, (int)(b * a))
               );
    }
};


#endif // COLORSCALE_H

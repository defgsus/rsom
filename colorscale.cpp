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

float sinc(float x) { return (cosf(x*x*11.f) * 0.5f + 0.4f + 0.1f * cos(x*29.f)); }

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
        r += v * sinc(1.f - t);
        g += v * sinc(std::max(0.f, fabsf(0.5f - t)));
        b += v * sinc(t);
    }
    const float a = 255.f / num * amp;

    return QColor(
           std::min(255, (int)(r * a)),
           std::min(255, (int)(g * a)),
           std::min(255, (int)(b * a))
           );
}

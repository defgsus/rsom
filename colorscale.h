#ifndef COLORSCALE_H
#define COLORSCALE_H

#include <QColor>

/** some arbitrary but unique color scale for values of 0.0 to 1.0 */
struct ColorScale
{
    // some precalculated color scale (256)
    std::vector<QColor> color_map;

    ColorScale() { calc_scale(); }

    void calc_scale();

    /* get the color for value 'f' (0-1) */
    QColor get(const float f) const;

    /** get a color from a number of vectors */
    QColor get_spectral(const float * f, size_t num, float amp = 1.f) const;
};


#endif // COLORSCALE_H

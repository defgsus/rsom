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
    @brief View around core/som.cpp

    @version 2013/12/18 init

    copyright 2013 stefan.berke @ modular-audio-graphics.com
*/
#ifndef SOMVIEW_H
#define SOMVIEW_H

#include <QFrame>

#include "colorscale.h"

namespace RSOM { class Som; }

class SomView : public QFrame
{
    Q_OBJECT
public:
    enum PaintMode
    {
        /** draw a single band */
        PM_Band,
        /** draw something like the spectral color */
        PM_MultiBand,
        /** draw the umap (whatever is in there) */
        PM_UMap,
        /** draw the index map */
        PM_IMap
    };


    explicit SomView(QWidget *parent = 0);

    /** Sets the Som to draw on the next paintEvent.
        Make sure that the data does not reallocate while the Som
        is connected to this View. Set to NULL to disconnect. */
    void setSom(const RSOM::Som * som) { som_ = som; }

    /** set scale for colors */
    void paintMultiplier(float pm) { paint_mult_ = pm; update(); }

    /** Sets what to paint. */
    void paintMode(PaintMode mode) { pmode_ = mode; update(); }

    /** Selects PM_Band, and sets the band to draw */
    void paintBandNr(size_t band_nr);

signals:

    void map_clicked(int index);

public slots:

protected:
    virtual void paintEvent(QPaintEvent *);

    virtual void mousePressEvent(QMouseEvent *);

    void paint_band_();
    void paint_multi_band_();
    void paint_umap_();
    void paint_imap_();

    const RSOM::Som * som_;

    PaintMode pmode_;
    int band_sel_;
    float paint_mult_;

    ColorScale colors_;
};


#endif // SOMVIEW_H

/** @file
    @brief View around core/wave.cpp

    @version 2013/12/18 init

    copyright 2013 stefan.berke @ modular-audio-graphics.com

    This program is coverd by the GNU General Public License
*/
#ifndef SOMVIEW_H
#define SOMVIEW_H

#include <QFrame>

#include "colorscale.h"

class Som;

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
        PM_UMap
    };


    explicit SomView(QWidget *parent = 0);

    /** Sets the Som to draw on the next paintEvent.
        Make sure that the data does not reallocate while the Som
        is connected to this View. Set to NULL to disconnect. */
    void setSom(const Som * som) { som_ = som; }

    /** set scale for colors */
    void paintMultiplier(float pm) { paint_mult_ = pm; }

    /** Sets what to paint. */
    void paintMode(PaintMode mode) { pmode_ = mode; }
    /** Selects PM_Band, and sets the band to draw */
    void paintBandNr(size_t band_nr);

signals:

public slots:

protected:
    virtual void paintEvent(QPaintEvent *);

    void paint_band_();
    void paint_multi_band_();
    void paint_umap_();

    const Som * som_;

    PaintMode pmode_;
    size_t band_sel_;
    float paint_mult_;

    ColorScale colors_;
};


#endif // SOMVIEW_H

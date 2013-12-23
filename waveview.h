/** @file
    @brief View around core/wave.cpp

    @version 2013/12/18 init

    copyright 2013 stefan.berke @ modular-audio-graphics.com

    This program is coverd by the GNU General Public License
*/
#ifndef WAVEVIEW_H
#define WAVEVIEW_H

#include <QFrame>

#include "colorscale.h"

class Wave;


class WaveView : public QFrame
{
    Q_OBJECT
public:
    explicit WaveView(QWidget *parent = 0);

    virtual QSize minSizeHint() const { return QSize(256,100); }
    virtual QSize sizeHint() const { return QSize(1024,300); }

    // ---- Wave link ----------

    /** Sets the Wave to draw on the next paintEvent.
        Make sure that the data does not reallocate while the Wave
        is connected to this View. Set to NULL to disconnect. */
    void setWave(const Wave * wave) { wave_ = wave; }

    // ------- properties ------

    bool draw_spec_colors() const { return draw_spec_colors_; }
    void draw_spec_colors(bool do_it) { draw_spec_colors_ = do_it; update(); }

    bool draw_waveform() const { return draw_waveform_; }
    void draw_waveform(bool do_it) { draw_waveform_ = do_it; update(); }

signals:

public slots:

protected:
    virtual void paintEvent(QPaintEvent *);

    void paint_waveform();
    void paint_bands();
    void paint_color_scale();

    const Wave * wave_;

    ColorScale colors_;

    bool draw_spec_colors_,
        draw_waveform_;

    int csheight_;
};

#endif // WAVEVIEW_H

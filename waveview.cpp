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
#include "waveview.h"

#include "core/log.h"
#include "core/wavefile.h"

#include <QPainter>
#include <QRect>
#include <QPalette>
#include <QColor>




WaveView::WaveView(QWidget *parent) :
    QFrame(parent),
    wave_   (0),
    draw_spec_colors_ (false),
    draw_waveform_    (true),
    csheight_         (16)
{
    setFrameStyle(QFrame::Panel | QFrame::Sunken);
    setLineWidth(2);

    setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Preferred);
}



void WaveView::paintEvent(QPaintEvent * event)
{
    QFrame::paintEvent(event);

    if (wave_)
    {
        paint_bands();
        if (draw_waveform_) paint_waveform();
    }

    paint_color_scale();
}

void WaveView::paint_waveform()
{
    SOM_DEBUGN(1, "WaveView::paint_waveform()");

    QPainter p(this);

    p.setPen(QColor(255,255,255));

    // simply draw enough points from the sample data
    // with a bit oversampling
    size_t w = width() * 3,
           w1 = width() - frameWidth() * 2,
           h1 = (height() - frameWidth() * 2) / 2;
    qreal x0=0,y0=0;
    for (size_t i=0; i<w; ++i)
    {
        qreal
            sam = wave_->wave[
                std::min((size_t)((float)i/w * wave_->wave.size()), wave_->wave.size())],
            x = frameWidth() + (float)i / w * w1,
            y = frameWidth() + (sam + 1.f) * h1;

        if (i>0) p.drawLine(x0,y0,x,y);

        x0 = x;
        y0 = y;
    }

}

void WaveView::paint_bands()
{
    SOM_DEBUGN(1, "WaveView::paint_bands()");

    QPainter p(this);
    p.setPen(Qt::NoPen);

    // draw each grain in each band as rectangle

    int w = width() - frameWidth() * 2,
        h = height() - frameWidth() * 2 - csheight_;
    const
    qreal sx = (qreal)w / wave_->nr_grains + 1,
          sy = (qreal)h / wave_->nr_bands + 1;

    if (!draw_spec_colors_)
        for (size_t x=0; x<wave_->nr_grains; ++x)
        for (size_t y=0; y<wave_->nr_bands; ++y)
        {
            p.setBrush(QBrush(colors_.get(wave_->band[x][wave_->nr_bands-1-y])));

            p.drawRect( (qreal)x / wave_->nr_grains * w + frameWidth(),
                        (qreal)y / wave_->nr_bands * h + frameWidth(),
                        sx,sy);

        }
    else
        for (size_t x=0; x<wave_->nr_grains; ++x)
        {
            p.setBrush(QBrush(
                colors_.get_spectral(&wave_->band[x][0], wave_->nr_bands)
                       ));

            p.drawRect( (qreal)x / wave_->nr_grains * w + frameWidth(),
                        frameWidth(),
                        sx, h);

        }

}

void WaveView::paint_color_scale()
{
    SOM_DEBUGN(1, "WaveView::paint_color_scale()");

    QPainter p(this);
    p.setPen(Qt::NoPen);

    const size_t w = width() - frameWidth() * 2,
                 h = csheight_ - 1,
                 num = 255;
    const qreal  sx = (qreal)w / 255 + 1;

    for (size_t x=0; x<num; ++x)
    {
        p.setBrush(QBrush(colors_.get((float)x/(num-1))));

        p.drawRect( (qreal)x / num * w + frameWidth(),
                    height() - 1 - h - frameWidth(),
                    sx, h);
    }

}

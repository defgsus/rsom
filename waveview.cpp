#include "waveview.h"

#include "core/log.h"
#include "core/wavefile.h"

#include <QPainter>
#include <QRect>
#include <QPalette>
#include <QColor>




WaveView::WaveView(QWidget *parent) :
    QFrame(parent),
    wave_   (0)
{
    setFrameStyle(QFrame::Panel | QFrame::Sunken);
    setLineWidth(1);

    setFixedSize(700,300);
}



void WaveView::paintEvent(QPaintEvent * event)
{
    QFrame::paintEvent(event);

    if (!wave_) return;

    paint_bands();
    paint_waveform();
}

void WaveView::paint_waveform()
{
    QPainter p(this);

    p.setPen(QColor(255,255,255));

    // simply draw enough points from the sample data
    // with a bit oversampling
    size_t w = width() * 3,
           w1 = width()-2,
           h1 = (height() - 2) / 2;
    qreal x0=0,y0=0;
    for (size_t i=0; i<w; ++i)
    {
        qreal
            sam = wave_->wave[
                std::min((size_t)((float)i/w * wave_->wave.size()), wave_->wave.size())],
            x = 1.f + (float)i / w * w1,
            y = 1.f + (sam + 1.f) * h1;

        if (i>0) p.drawLine(x0,y0,x,y);

        x0 = x;
        y0 = y;
    }

}

void WaveView::paint_bands()
{
    //SOM_DEBUG("WaveView::paint_bands()");

    QPainter p(this);
    p.setPen(Qt::NoPen);

    // draw each grain in each band as rectangle

    int w = width()-2,
        h = height()-2;
    const
    qreal sx = (qreal)w / wave_->nr_grains + 1,
          sy = (qreal)h / wave_->nr_bands + 1;

    for (size_t x=0; x<wave_->nr_grains; ++x)
    for (size_t y=0; y<wave_->nr_bands; ++y)
    {
        p.setBrush(QBrush(colors_.get(wave_->band[x][y])));

        p.drawRect( (qreal)x / wave_->nr_grains * w + 1,
                    (1.0 - (qreal)y / wave_->nr_bands) * h - sy + 2,
                    sx,sy);

    }

}

#include "somview.h"

#include "core/log.h"
#include "core/som.h"

#include <QPainter>


SomView::SomView(QWidget * parent)
    :   QFrame  (parent),
        som_    (0),
        pmode_  (PM_Band),
        band_sel_(0),
        paint_mult_(1.f)
{
    SOM_DEBUG("SomView::SomView()");

    setFrameStyle(QFrame::Panel | QFrame::Sunken);
    setLineWidth(1);

    setFixedSize(386,386);
}

void SomView::paintBandNr(size_t band_nr)
{
    band_sel_ = band_nr;
    pmode_ = PM_Band;
}

void SomView::paintEvent(QPaintEvent * event)
{
    QFrame::paintEvent(event);

    if (!som_ || !som_->dim)
    {
        SOM_ERROR("SomView::paintEvent called without SOM");
        return;
    }

    if (pmode_ == PM_Band && band_sel_ >= som_->dim)
    {
        SOM_DEBUG("SomView::paintEvent:: band_sel_ was out of range");

        band_sel_ = som_->dim - 1;
    }

    switch (pmode_)
    {
        case PM_Band: paint_band_(); break;
        case PM_MultiBand: paint_multi_band_(); break;
        case PM_UMap: paint_umap_(); break;
    }
}


void SomView::paint_band_()
{
    SOM_DEBUGN("SomView::paint_band()");

    QPainter p(this);
    p.setPen(Qt::NoPen);

    int w = width() - frameWidth()*2,
        h = height() - frameWidth()*2;
    const
    qreal sx = (qreal)w / som_->sizex + frameWidth(),
          sy = (qreal)h / som_->sizey + frameWidth();

    for (size_t y=0; y<som_->sizey; ++y)
    for (size_t x=0; x<som_->sizex; ++x)
    {
        p.setBrush(QBrush(colors_.get(som_->data[y*som_->sizex+x][band_sel_])));

        p.drawRect( (qreal)x / som_->sizex * w + frameWidth(),
                    (qreal)y / som_->sizey * h + frameWidth(),
                    sx, sy );

    }
}

void SomView::paint_multi_band_()
{
    SOM_DEBUGN("SomView::paint_multi_band()");

    QPainter p(this);
    p.setPen(Qt::NoPen);

    int w = width() - frameWidth() * 2,
        h = height() - frameWidth() * 2;
    const
    qreal sx = (qreal)w / som_->sizex + frameWidth(),
          sy = (qreal)h / som_->sizey + frameWidth();

    for (size_t y=0; y<som_->sizey; ++y)
    for (size_t x=0; x<som_->sizex; ++x)
    {
        // get spectral color from vector
        p.setBrush(QBrush(
            colors_.get_spectral(&som_->data[y*som_->sizex+x][0], som_->dim, paint_mult_)
                   ));

        p.drawRect( (qreal)x / som_->sizex * w + frameWidth(),
                    (qreal)y / som_->sizey * h + frameWidth(),
                    sx, sy );

    }
}

void SomView::paint_umap_()
{
    SOM_DEBUGN("SomView::paint_umap()");

    QPainter p(this);
    p.setPen(Qt::NoPen);

    int w = width() - frameWidth() * 2,
        h = height() - frameWidth() * 2;
    const
    qreal sx = (qreal)w / som_->sizex + frameWidth(),
          sy = (qreal)h / som_->sizey + frameWidth();

    for (size_t y=0; y<som_->sizey; ++y)
    for (size_t x=0; x<som_->sizex; ++x)
    {
        p.setBrush(QBrush(colors_.get(paint_mult_ * som_->umap[y*som_->sizex+x])));

        p.drawRect( (qreal)x / som_->sizex * w + frameWidth(),
                    (qreal)y / som_->sizey * h + frameWidth(),
                    sx, sy );

    }

}




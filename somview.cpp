#include "somview.h"

#include "core/log.h"
#include "core/som.h"

#include <QPainter>


SomView::SomView(QWidget * parent)
    :   QFrame  (parent),
        som_    (0),
        pmode_  (PM_Band),
        band_sel_(0)
{
    SOM_DEBUG("SomView::SomView()");

    setFrameStyle(QFrame::Panel | QFrame::Sunken);
    setLineWidth(1);

    setFixedSize(386,386);
}

void SomView::paintBand(size_t band_nr)
{
    band_sel_ = band_nr;
    pmode_ = PM_Band;
}

void SomView::paintEvent(QPaintEvent * event)
{
    QFrame::paintEvent(event);

    if (!som_) return;

    switch (pmode_)
    {
        case PM_Band: paint_band_(); break;
        case PM_UMap: paint_umap_(); break;
    }
}


void SomView::paint_band_()
{
    //SOM_DEBUG("SomView::paint_band()");

    // safety-check for band_sel_
    if (!som_ || !som_->dim || band_sel_ >= som_->dim)
    {
        SOM_ERROR("SomView::paint_band_() called with bad settings");
    }

    QPainter p(this);
    p.setPen(Qt::NoPen);

    // draw each grain in each band as rectangle

    int w = width()-2,
        h = height()-2;
    const
    qreal sx = (qreal)w / som_->sizex + 1,
          sy = (qreal)h / som_->sizey + 1;

    for (size_t y=0; y<som_->sizey; ++y)
    for (size_t x=0; x<som_->sizex; ++x)
    {
        p.setBrush(QBrush(colors_.get(som_->data[y*som_->sizex+x][0])));

        p.drawRect( (qreal)x / som_->sizex * w + 1,
                    (qreal)y / som_->sizey * h + 1,
                    sx, sy );

    }

}

void SomView::paint_umap_()
{
    //SOM_DEBUG("SomView::paint_umap()");

    QPainter p(this);
    p.setPen(Qt::NoPen);

    // draw each grain in each band as rectangle

    int w = width()-2,
        h = height()-2;
    const
    qreal sx = (qreal)w / som_->sizex + 1,
          sy = (qreal)h / som_->sizey + 1;

    for (size_t y=0; y<som_->sizey; ++y)
    for (size_t x=0; x<som_->sizex; ++x)
    {
        p.setBrush(QBrush(colors_.get(som_->umap[y*som_->sizex+x])));

        p.drawRect( (qreal)x / som_->sizex * w + 1,
                    (qreal)y / som_->sizey * h + 1,
                    sx, sy );

    }

}



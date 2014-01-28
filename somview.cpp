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
#include "somview.h"

#include "core/log.h"
#include "core/som.h"

#include <QPainter>
#include <QMouseEvent>

SomView::SomView(QWidget * parent)
    :   QFrame      (parent),
        som_        (0),
        pmode_      (PM_Band),
        band_sel_   (0),
        paint_mult_ (1.f)
{
    SOM_DEBUG("SomView::SomView()");

    setFrameStyle(QFrame::Panel | QFrame::Sunken);
    setLineWidth(2);

    setFixedSize(386,386);
}

void SomView::paintBandNr(size_t band_nr)
{
    band_sel_ = band_nr;
    pmode_ = PM_Band;
    update();
}

void SomView::paintEvent(QPaintEvent * event)
{
    QFrame::paintEvent(event);

    if (!som_ || !som_->dim)
    {
        SOM_DEBUG("SomView::paintEvent called without SOM");
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
        case PM_IMap: paint_imap_(); break;
    }
}

void SomView::mousePressEvent(QMouseEvent *event)
{
//    SOM_DEBUG("click " << event->x() << ", " << event->y());

    if (!som_) return;

    int x = event->x() * som_->sizex / (width() - frameWidth()*2);
    int y = event->y() * som_->sizey / (height() - frameWidth()*2);

    int index = y * som_->sizex + x;
    if (index>=0 && (size_t)index < som_->size)
    {
        map_clicked(index);
    }
}


void SomView::paint_band_()
{
    SOM_DEBUGN(1, "SomView::paint_band()");

    QPainter p(this);
    p.setPen(Qt::NoPen);

    int w = width() - frameWidth()*2,
        h = height() - frameWidth()*2;
    const
    qreal sx = (qreal)w / som_->sizex + 1,
          sy = (qreal)h / som_->sizey + 1;

    for (size_t y=0; y<som_->sizey; ++y)
    for (size_t x=0; x<som_->sizex; ++x)
    {
        // spectral color from single band
        p.setBrush(QBrush(colors_.get(som_->map[y*som_->sizex+x][band_sel_])));

        p.drawRect( (qreal)x / som_->sizex * w + frameWidth(),
                    (qreal)y / som_->sizey * h + frameWidth(),
                    sx, sy );

    }
}

void SomView::paint_multi_band_()
{
    SOM_DEBUGN(1, "SomView::paint_multi_band()");

    QPainter p(this);
    p.setPen(Qt::NoPen);

    int w = width() - frameWidth() * 2,
        h = height() - frameWidth() * 2;
    const
    qreal sx = (qreal)w / som_->sizex + 1,
          sy = (qreal)h / som_->sizey + 1;

    for (size_t y=0; y<som_->sizey; ++y)
    for (size_t x=0; x<som_->sizex; ++x)
    {
        // get spectral color from data vector
        p.setBrush(QBrush(
            colors_.get_spectral(&som_->map[y*som_->sizex+x][0], som_->dim, paint_mult_)
                   ));

        p.drawRect( (qreal)x / som_->sizex * w + frameWidth(),
                    (qreal)y / som_->sizey * h + frameWidth(),
                    sx, sy );

    }
}

void SomView::paint_umap_()
{
    SOM_DEBUGN(1, "SomView::paint_umap()");

    QPainter p(this);
    p.setPen(Qt::NoPen);

    int w = width() - frameWidth() * 2,
        h = height() - frameWidth() * 2;
    const
    qreal sx = (qreal)w / som_->sizex + 1,
          sy = (qreal)h / som_->sizey + 1;

    for (size_t y=0; y<som_->sizey; ++y)
    for (size_t x=0; x<som_->sizex; ++x)
    {
        // spectral color from umap
        p.setBrush(QBrush(colors_.get(paint_mult_ * som_->umap[y*som_->sizex+x])));

        p.drawRect( (qreal)x / som_->sizex * w + frameWidth(),
                    (qreal)y / som_->sizey * h + frameWidth(),
                    sx, sy );

    }

}


void SomView::paint_imap_()
{
    SOM_DEBUGN(1, "SomView::paint_imap()");

    if (som_->data.empty()) return;

    QPainter p(this);
    p.setPen(Qt::NoPen);

    int w = width() - frameWidth() * 2,
        h = height() - frameWidth() * 2;
    const
    qreal sx = (qreal)w / som_->sizex + 1,
          sy = (qreal)h / som_->sizey + 1;

    for (size_t y=0; y<som_->sizey; ++y)
    for (size_t x=0; x<som_->sizex; ++x)
    {
        // spectral color from imap
        const float f = (float)som_->imap[y*som_->sizex+x] / som_->data.size();
        if (f<0)
            p.setBrush(QBrush(QColor(0,0,0)));
        else
            p.setBrush(QBrush(colors_.get(paint_mult_ * f)));

        p.drawRect( (qreal)x / som_->sizex * w + frameWidth(),
                    (qreal)y / som_->sizey * h + frameWidth(),
                    sx, sy );

    }

}


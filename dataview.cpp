/***************************************************************************

Copyright (C) 2014  stefan.berke @ modular-audio-graphics.com

This source is free software; you can redistribute it and/or
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

****************************************************************************/

#include "dataview.h"

#include "core/data.h"
#include "core/log.h"

#include <QPainter>

DataView::DataView(QWidget *parent)
    :   QFrame  (parent),
        data_    (0),
        objIndex_(0)
{
    setFrameStyle(QFrame::Panel | QFrame::Sunken);
    setLineWidth(2);

    setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Preferred);
}

void DataView::paintEvent(QPaintEvent * event)
{
    QFrame::paintEvent(event);

    if (!data_ || !data_->numObjects()) return;

    paint_data_curve();
}

void DataView::paint_data_curve()
{
    SOM_DEBUGN(1, "DataView::paint_data_curve()");

    QPainter p(this);

    p.setPen(QColor(255,255,255));

    size_t index = std::min(objIndex_, data_->numObjects());

    // simply draw enough points from the sample data
    // with a bit oversampling
    size_t w = width() * 3,
           w1 = width() - frameWidth() * 2,
           h1 = (height() - frameWidth() * 2) * 1;
    qreal x0=0,y0=0;
    for (size_t i=0; i<w; ++i)
    {
        qreal
            val = data_->getObjectData(index)[
                std::min((size_t)((float)i/w * data_->numDataPoints()), data_->numDataPoints())],
            x = frameWidth() + (float)i / w * w1,
            y = height() - frameWidth() - val * h1;

        if (i>0) p.drawLine(x0,y0,x,y);

        x0 = x;
        y0 = y;
    }
}

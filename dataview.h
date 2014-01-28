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

#ifndef DATAVIEW_H
#define DATAVIEW_H

#include <QFrame>

#include "colorscale.h"

class Data;


class DataView : public QFrame
{
    Q_OBJECT
public:
    explicit DataView(QWidget *parent = 0);

    virtual QSize minSizeHint() const { return QSize(256,100); }
    virtual QSize sizeHint() const { return QSize(1024,300); }

    // ---- Data link ----------

    /** Sets the Data to draw on the next paintEvent.
        Make sure that the data does not reallocate while it
        is connected to this View. Set to NULL to disconnect. */
    void setData(const Data * data) { data_ = data; }

    // ------- properties ------

    /** set the object in Data to draw */
    void draw_object(size_t index);
    size_t draw_object() const { return objIndex_; }

signals:

public slots:

protected:
    virtual void paintEvent(QPaintEvent *);

    void paint_data_curve();

    const Data * data_;

    ColorScale colors_;

    size_t objIndex_;

    int csheight_;
};

#endif // DATAVIEW_H

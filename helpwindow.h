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
    @brief Help Window for rsom

    @version 2013/12/23 init

    copyright 2013 stefan.berke @ modular-audio-graphics.com
*/
#ifndef HELPWINDOW_H
#define HELPWINDOW_H

#include <QWidget>

class ProjectView;
class Properties;

class QTextBrowser;
class QTabWidget;

class HelpWindow : public QWidget
{
    Q_OBJECT
public:
    explicit HelpWindow(const ProjectView & view, QWidget *parent = 0);

    QSize sizeHint() const { return QSize(800,500); }

signals:

public slots:

protected:

    const Properties& props_;

    QTabWidget * tab_;
    QTextBrowser
        * tdoc_,
        * tlicense_,
        * tabout_;

    void load_();
};

#endif // HELPWINDOW_H

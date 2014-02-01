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
#include "mainwindow.h"

#include "core/project.h"
#include "projectview.h"

#include <QLayout>

MainWindow::MainWindow(QWidget *parent) :
    QWidget(parent)
{
    project_ = new RSOM::Project();

    auto l0 = new QVBoxLayout(this);

        view_ = new ProjectView(project_, this);
        l0->addWidget(view_);
}

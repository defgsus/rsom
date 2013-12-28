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
    @brief rsom's MainWindow

    @version 2013/12/18 init

    copyright 2013 stefan.berke @ modular-audio-graphics.com
*/
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QWidget>

class Project;
class ProjectView;

class MainWindow : public QWidget
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = 0);

    const Project *     project()     const { return project_; }
    const ProjectView * projectView() const { return view_; }

signals:

public slots:

protected:
    Project * project_;
    ProjectView * view_;
};

#endif // MAINWINDOW_H

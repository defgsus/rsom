/** @file
    @brief rsom's MainWindow

    @version 2013/12/18 init

    copyright 2013 stefan.berke @ modular-audio-graphics.com

    This program is coverd by the GNU General Public License
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

#include "mainwindow.h"

#include "core/project.h"
#include "projectview.h"

#include <QLayout>

MainWindow::MainWindow(QWidget *parent) :
    QWidget(parent)
{
    project_ = new Project();

    auto l0 = new QVBoxLayout(this);

        view_ = new ProjectView(project_, this);
        l0->addWidget(view_);
}

#include <QApplication>

#include "mainwindow.h"
#include "helpwindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    MainWindow win;
    win.show();

    HelpWindow help;
    help.show();

    return a.exec();
}

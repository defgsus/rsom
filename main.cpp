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

#include <time.h>

/** get system time in seconds */
double sysTime()
{
    timespec cls;
    clock_gettime(CLOCK_MONOTONIC, &cls);
    // second + nanoseconds
    return cls.tv_sec + 0.000000001 * cls.tv_nsec;
}

/** class to messure passed time */
class Messure
{
    public:

    Messure() { start(); }

    /** restart counter */
    void start() { last_time = sysTime(); }

    /** return elapsed time */
    double elapsed() { return sysTime() - last_time; }

    private:

    double last_time;
};

#if (0)
#include <QApplication>

#include "mainwindow.h"
#include "helpwindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    MainWindow win;
    win.show();

#ifdef NDEBUG
    HelpWindow help(*win.projectView());
    help.show();
#endif

    return a.exec();
}
#else

#include "iostream"

#include "core/data.h"
#include "core/som.h"
#include "core/project.h"
#include "testcuda.h"
#include "testsom.h"

int main(int , char **)
{
    return testCuda();
    //return testSom();
    //Data dat; dat.addCsvFile("/home/defgsus/prog/DATA/golstat.txt"); return 0;


    RSOM::Project project;

    project.data().createRandomData(1000, 256);
    project.set_som(31,31, 1);
    std::cout << project.som().info_str() << "\n";

    project.startSomThread();
    while (true)
    {

    }
}

#endif

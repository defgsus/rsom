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
#include "helpwindow.h"

#include "core/log.h"
#include "properties.h"
#include "projectview.h"

#include <QFile>
#include <QTextStream>
#include <QLayout>
#include <QTabWidget>
#include <QTextBrowser>


HelpWindow::HelpWindow(const ProjectView & view, QWidget *parent) :
    QWidget(parent),
    props_  (view.properties())
{   
    auto l0 = new QVBoxLayout(this);
    setLayout(l0);

        tab_ = new QTabWidget(this);
        l0->addWidget(tab_);

            tdoc_ = new QTextBrowser();
            tab_->addTab(tdoc_, "documentation");

            tlicense_ = new QTextBrowser();
            tab_->addTab(tlicense_, "license");

            tabout_ = new QTextBrowser();
            tab_->addTab(tabout_, "about");

    load_();
}


void HelpWindow::load_()
{
    // --- license ---
    {
        QFile file(":/rsom/LICENSE");
        if (!file.open(QFile::ReadOnly | QFile::Text))
        {
            SOM_ERROR("Can't open help resource");
            return;
        }

        QTextStream in(&file);
        tlicense_->setPlainText(in.readAll());
    }

    // --- about ---

    tabout_->setHtml(
                    tr("<h2>Welcome to r-som</h2>"
                       "<p>r-som generates <i>Self-Organizing Maps</i> from audio files and "
                       "exports them to the <i>NI Reaktor</i> table format.</p>"
                       "<p>This is free software. Free as in freedom. See the license for details.</p>"
                       "<pre>version 0.1 build %1 %2</pre>"
                       "<p>authors:</p>"
                       "<pre>stefan.berke@modular-audio-graphics.com, 2013</pre>"
                       ).arg(__DATE__).arg(__TIME__)
                    );

    // --- documentation ---

    QString s;
    {
        QFile file(":/rsom/help.html");
        if (!file.open(QFile::ReadOnly | QFile::Text))
        {
            SOM_ERROR("Can't open help resource");
            return;
        }

        QTextStream in(&file);
        s = in.readAll() + "\n";
    }

    // --- parameters ---

    QTextStream str(&s);

    str << "<a name=\"properties\"></a><h2>properties</h2>";

    // extract each Property::help
    for (auto i=props_.property.begin(); i!=props_.property.end(); ++i)
    {
        str << "<a name=\"" << (*i)->id << "\"></a>"
            << "<h3><b>" << (*i)->name << "</b></h3>"
            << "<p>" << (*i)->help << "</p>";
    }

    tdoc_->setHtml( s );
}

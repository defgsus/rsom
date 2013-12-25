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

    QFile file(":/rsom/LICENSE");
    if (!file.open(QFile::ReadOnly | QFile::Text))
    {
        SOM_ERROR("Can't open help resource");
        return;
    }

    QTextStream in(&file);
    tlicense_->setPlainText(in.readAll());

    // --- about ---

    tabout_->setHtml(
                    tr("<h2>Welcome to r-som</h2>"
                       "<p>This program generates <i>Self-Organizing Maps</i> from audio files and "
                       "exports them to the <i>NI Reaktor</i> table format.</p>"
                       "<p>This software is free as in freedom. See the license for details.</p>"
                       "<pre>build %1 %2</pre>"
                       "<p>authors:</p>"
                       "<pre>stefan.berke@modular-audio-graphics.com</pre>"
                       ).arg(__DATE__).arg(__TIME__)
                    );

    // --- documentation ---

    QString s;
    QTextStream str(&s);


    // property help

    str << "<h2>properties</h2>";

    // extract each Property::help
    for (auto i=props_.property.begin(); i!=props_.property.end(); ++i)
    {
        str << "<a name=\"" << (*i)->id << "\"></a>"
            << "<h3><b>" << (*i)->name << "</b></h3>"
            << "<p>" << (*i)->help << "</p>";
    }

    tdoc_->setHtml( s );
}

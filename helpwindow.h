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

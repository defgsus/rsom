/** @file
    @brief View around core/project.cpp

    @version 2013/12/18 init

    copyright 2013 stefan.berke @ modular-audio-graphics.com

    This program is coverd by the GNU General Public License
*/
#ifndef PROJECTVIEW_H
#define PROJECTVIEW_H

#include <QFrame>


class Project;
class Property;
class WaveView;
class SomView;

class ProjectView : public QFrame
{
    Q_OBJECT
public:
    /** constructs widgets from given project */
    explicit ProjectView(Project * project, QWidget *parent = 0);

    bool loadWave();

signals:

public slots:

protected:
    Project * project_;
    WaveView * waveview_;
    SomView * somview_;

    // properties
    Property
        *wave_bands_,
        *wave_minf_,
        *wave_maxf_;
};

#endif // PROJECTVIEW_H

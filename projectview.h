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
    @brief View around core/project.cpp

    @version 2013/12/18 init

    copyright 2013 stefan.berke @ modular-audio-graphics.com
*/
#ifndef PROJECTVIEW_H
#define PROJECTVIEW_H

#include <QFrame>


namespace RSOM { class Project; }
class Property;
class Properties;
class DataView;
class SomView;

class QLabel;
class QTextBrowser;

class ProjectView : public QFrame
{
    Q_OBJECT
public:
    /** constructs widgets from given project */
    explicit ProjectView(RSOM::Project * project, QWidget *parent = 0);
    virtual ~ProjectView();

    bool loadData();
    bool exportTable();

    const Properties & properties() const { return *props_; }

signals:
    void start_som_signal();
    void som_update_signal();
    void start_training_signal();
    void log_signal(const QString& text);
    void error_signal(const QString& text);

public slots:
    void start_som() { set_som_(); }
    void som_update();

    void log(const QString& text);
    void error(const QString& text);

    void startTraining();
    void stopTraining();

protected:
    virtual void keyPressEvent(QKeyEvent *);

    /** check sanity of Property widgets. */
    void checkWidgets_();

    /** (re-)set the wave parameters using Properties */
    void set_som_();

    bool save_som_();
    bool load_som_();

    /** update the needed maps immidiately */
    void calc_maps_();

    /** update SomView */
    void setSomPaintMode_();

    RSOM::Project * project_;

    DataView * dataview_;
    SomView * somview_;

    QLabel * sominfo_;
    QTextBrowser * log_box_;

    QString data_dir_, export_dir_,
            som_dir_;

    // properties
    Properties * props_;
    Property
        *som_run_,
        *som_size_,
        *som_size_use_f_,
        *som_sizef_,
        *som_seed_,
        *som_alpha_,
        *som_radius_,
        *som_sradius_,
        *som_non_dupl_,
        *som_wrap_,

        *somd_dmode_,
        *somd_band_nr_,
        *somd_mult_,
        *somd_calc_imap_;
};

#endif // PROJECTVIEW_H

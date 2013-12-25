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
class Properties;
class WaveView;
class SomView;

class QLabel;

class ProjectView : public QFrame
{
    Q_OBJECT
public:
    /** constructs widgets from given project */
    explicit ProjectView(Project * project, QWidget *parent = 0);
    virtual ~ProjectView();

    bool loadWave();

    const Properties & properties() const { return *props_; }

signals:
    void start_som_signal();
    void som_update_signal();
    void start_training_signal();

public slots:
    void start_som() { set_som_(); }
    void som_update();

    void startTraining();
    void stopTraining();

protected:
    /** check sanity of Property widgets. */
    void checkWidgets_();

    /** (re-)set the wave parameters using Properties */
    void set_wave_();
    /** (re-)set the wave parameters using Properties */
    void set_som_();

    /** update the needed maps immidiately */
    void calc_maps_();

    /** update SomView */
    void setSomPaintMode_();

    Project * project_;
    WaveView * waveview_;
    SomView * somview_;
    QLabel * sominfo_;

    // properties
    Properties * props_;
    Property
        *wave_bands_,
        *wave_freq_,
        *wave_grain_size_,
        *wave_window_,
        *wave_band_norm_,
        *wave_band_amp_,
        *wave_band_exp_,

        *waved_waveform_,
        *waved_spec_colors_,

        *som_run_,
        *som_size_,
        *som_size_use_f_,
        *som_sizef_,
        *som_size_to_g_,
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

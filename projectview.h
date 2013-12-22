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
    /** (re-)set the wave parameters using Properties */
    void set_wave_();
    /** (re-)set the wave parameters using Properties */
    void set_som_();

    /** update SomView */
    void setSomPaintMode_();

    /** determine if umap must be calculated */
    bool need_umap_();
    /** determine if imap must be calculated */
    bool need_imap_();

    Project * project_;
    WaveView * waveview_;
    SomView * somview_;

    // properties
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

        *som_size_,
        *som_size_use_f_,
        *som_sizef_,
        *som_seed_,
        *som_alpha_,
        *som_radius_,
        *som_sradius_,

        *somd_dmode_,
        *somd_band_nr_,
        *somd_mult_;
};

#endif // PROJECTVIEW_H

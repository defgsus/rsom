#include "projectview.h"

#include "core/log.h"
#include "core/project.h"
#include "property.h"
#include "waveview.h"
#include "somview.h"

#include <QLayout>
#include <QCheckBox>
#include <QSpinBox>
#include <QPushButton>
#include <QFrame>
#include <QLabel>
#include <QFileDialog>

enum SomDrawMode
{
    SDM_SINGLE_BAND,
    SDM_MULTI_BAND,
    SDM_IMAP,
    /* let this be the first of all modes that need the umap. (or change ProjectView::need_umap_) */
    SDM_UMAP
};

ProjectView::ProjectView(Project * p, QWidget *parent) :
    QFrame(parent),
    project_    (p)
{
    // some signals for threadsafety

    connect(this, SIGNAL(start_som_signal()), this, SLOT(start_som()) );
    connect(this, SIGNAL(som_update_signal()), this, SLOT(som_update()) );


    // --- properties ---

    SOM_DEBUG("ProjectView::ProjectView:: creating properties");

    wave_bands_ = new Property("wave_bands", "number bands", "...");
    wave_bands_->init(1, 100000, project_->num_bands());

    wave_freq_ = new Property("wave_freq", "frequency range", "...");
    wave_freq_->init(0.0001f, 100000.f, project_->wave().min_freq, project_->wave().max_freq);

    wave_grain_size_ = new Property("wave_grain_size", "grain size", "...");
    wave_grain_size_->init(1, 2<<16, project_->wave().grain_size);

    wave_window_ = new Property("wave_window_width", "window width", "...");
    wave_window_->init(1, 2<<16, project_->wave().window_width);

    wave_band_norm_ = new Property("wave_band_norm", "normalize\nbands", "...");
    wave_band_norm_->init(false);

    wave_band_amp_ = new Property("wave_band_amp", "band amp", "...");
    wave_band_amp_->init(0.0001f, 1000.f, project_->band_amp());

    wave_band_exp_ = new Property("wave_band_exp", "band exp", "...");
    wave_band_exp_->init(0.0001f, 100.f, project_->band_exponent());

    waved_waveform_ = new Property("waved_waveform", "draw waveform", "...");
    waved_waveform_->init(true);

    waved_spec_colors_ = new Property("waved_spec_colors", "draw spec.\ncolors", "...");
    waved_spec_colors_->init(false);

    // --- som properties

    som_size_ = new Property("som_size", "som size", "...");
    som_size_->init(2, 2<<16, project_->som_sizex(), project_->som_sizey());

    som_size_use_f_ = new Property("som_size_use_f", "som size\nfrom grains", "...");
    som_size_use_f_->init(true);

    som_sizef_ = new Property("som_sizef", "factor", "...");
    som_sizef_->init(0.001f, 10.f, 1.1f);

    som_size_to_g_ = new Property("som_size_to_grains", "som size\nto num. grains", "...");
    som_size_to_g_->init(false);

    som_seed_ = new Property("som_seed", "random seed", "...");
    som_seed_->init(-(2<<16), 2<<16, project_->som_seed());

    som_alpha_ = new Property("som_alpha", "alpha", "...");
    som_alpha_->init(0.f, 1.f, project_->som_alpha());

    som_radius_ = new Property("som_radius", "radius", "...");
    som_radius_->init(0.f, 1.f, project_->som_radius());

    som_sradius_ = new Property("som_search_radius", "local search radius", "...");
    som_sradius_->init(0.f, 2.f, project_->som_search_radius());

    somd_dmode_ = new Property("somd_mode", "draw mode", "...");
    somd_dmode_->init(
            { SDM_SINGLE_BAND, SDM_MULTI_BAND, SDM_UMAP, SDM_IMAP },
            { "single_band", "multi_band", "umap", "imap" },
            { "single band", "spectral color", "neighbour distance", "grain index" },
            SDM_SINGLE_BAND
                );

    somd_band_nr_ = new Property("somd_band_nr", "band index", "...");
    somd_band_nr_->init(0, 0, 0);

    somd_mult_ = new Property("somd_color_scale", "color scale", "...");
    somd_mult_->init(0.0001f, 1000.f, 1.f);

    somd_calc_imap_ = new Property("somd_calc_imap", "calculate imap", "...");
    somd_calc_imap_->init(false);


    SOM_DEBUG("ProjectView::ProjectView:: building widgets");

    setFrameStyle(QFrame::Panel | QFrame::Raised);
    setLineWidth(2);

    auto l0 = new QVBoxLayout(this);

        // -- project file --

        auto l1 = new QHBoxLayout(0);
        l0->addLayout(l1);

            // load button
            auto b = new QPushButton(this);
            l1->addWidget(b);
            b->setText("load project");

            // save button
            auto b1 = new QPushButton(this);
            l1->addWidget(b1);
            b1->setText("save project");

            // label
            auto lab = new QLabel(this);
            l1->addWidget(lab);
            lab->setText(QString::fromStdString(project_->filename()));

            connect(b, &QPushButton::clicked, [=]()
            {
                lab->setText("file");
            });

            l1->addStretch(2);

        // -- project file --

        l1 = new QHBoxLayout(0);
        l0->addLayout(l1);

            // load button
            b = new QPushButton(this);
            l1->addWidget(b);
            b->setText("load audio");

            // label
            lab = new QLabel(this);
            l1->addWidget(lab);
            lab->setText(QString::fromStdString(project_->wavename()));

            connect(b, &QPushButton::clicked, [=]()
            {
                loadWave();

                lab->setText(QString::fromStdString(project_->wavename()));
            });

            l1->addStretch(2);


        // ---- WAVE VIEW ----

        waveview_ = new WaveView(this);

        l1 = new QHBoxLayout(0);
        l0->addLayout(l1);
        {
            // -- view properties --

            auto l2 = new QVBoxLayout(0);
            l1->addLayout(l2);
            {
                // create analyzer widgets
                const Property::LayoutType lt = Property::V_LABEL_WIDGET;
                waved_waveform_->    createWidget(this, l2, lt);
                waved_spec_colors_-> createWidget(this, l2, lt);
            }

            l1->addSpacing(20);

            l1->addWidget( waveview_ );

            l2 = new QVBoxLayout(0);
            l1->addLayout(l2);
            {
                // -- analyze properties --

                const Property::LayoutType lt = Property::H_WIDGET_LABEL;
                wave_bands_->       createWidget(this, l2, lt);
                wave_freq_->        createWidget(this, l2, lt);
                wave_grain_size_->  createWidget(this, l2, lt);
                wave_window_->      createWidget(this, l2, lt);
                l2->addSpacing(10);
                wave_band_norm_->   createWidget(this, l2, lt);
                wave_band_amp_->    createWidget(this, l2, lt);
                wave_band_exp_->    createWidget(this, l2, lt);

                l2->addStretch(2);

            }
        }

        // --- connect wave properties ---

        waved_waveform_->   cb_value_changed( [=](){ waveview_->draw_waveform(waved_waveform_->v_bool[0]); });
        waved_spec_colors_->cb_value_changed( [=](){ waveview_->draw_spec_colors(waved_spec_colors_->v_bool[0]); });

        wave_bands_->       cb_value_changed( std::bind(&ProjectView::set_wave_, this) );
        wave_freq_->        cb_value_changed( std::bind(&ProjectView::set_wave_, this) );
        wave_grain_size_->  cb_value_changed( std::bind(&ProjectView::set_wave_, this) );
        wave_window_->      cb_value_changed( std::bind(&ProjectView::set_wave_, this) );
        wave_band_amp_->    cb_value_changed( std::bind(&ProjectView::set_wave_, this) );
        wave_band_exp_->    cb_value_changed( std::bind(&ProjectView::set_wave_, this) );
        wave_band_norm_->   cb_value_changed( [=]() { checkWidgets_(); set_wave_(); } );


        // ---- SOM view ----

        l1 = new QHBoxLayout(0);
        l0->addLayout(l1);
        {
            somview_ = new SomView(this);

            // --- SOM view parameters ---

            auto l2 = new QVBoxLayout(0);
            l1->addLayout(l2);
            {
                const Property::LayoutType lt = Property::H_WIDGET_LABEL;
                somd_dmode_->    createWidget(this, l2, lt);
                somd_band_nr_->  createWidget(this, l2, lt);
                somd_mult_->     createWidget(this, l2, lt);
                somd_calc_imap_->createWidget(this, l2, lt);

                l2->addStretch(2);
            }

            l1->addSpacing(20);

            l1->addWidget(somview_);

            // --- SOM parameters ----

            l2 = new QVBoxLayout(0);
            l1->addLayout(l2);
            {
                const Property::LayoutType lt = Property::H_WIDGET_LABEL;
                som_size_use_f_-> createWidget(this, l2, lt);
                som_sizef_->      createWidget(this, l2, lt);
                som_size_->       createWidget(this, l2, lt);
                som_size_to_g_->  createWidget(this, l2, lt);
                l2->addSpacing(20);
                som_seed_->       createWidget(this, l2, lt);
                l2->addSpacing(10);
                som_alpha_->      createWidget(this, l2, lt);
                som_radius_->     createWidget(this, l2, lt);
                som_sradius_->    createWidget(this, l2, lt);

                l2->addStretch(2);
            }
        }

        // -- connect Properties --

        somd_dmode_->     cb_value_changed( [&]() { setSomPaintMode_(); });
        somd_band_nr_->   cb_value_changed( [&]() { somview_->paintBandNr(somd_band_nr_->v_int[0]); });
        somd_mult_->      cb_value_changed( [&]() { somview_->paintMultiplier(somd_mult_->v_float[0]); });
        somd_calc_imap_-> cb_value_changed( [&]() { checkWidgets_(); });

        som_size_use_f_-> cb_value_changed( [&]() { checkWidgets_(); set_som_(); } );
        som_sizef_->      cb_value_changed( std::bind(&ProjectView::set_som_, this) );
        som_size_->       cb_value_changed( [&]() { if (som_size_to_g_->v_bool[0]) set_wave_(); else set_som_(); });
        som_size_to_g_->  cb_value_changed( [&]() { checkWidgets_(); set_wave_(); } );

        som_seed_->       cb_value_changed( std::bind(&ProjectView::set_som_, this));
        som_alpha_->      cb_value_changed( [&]() { project_->set_som_alpha(som_alpha_->v_float[0]); } );
        som_radius_->     cb_value_changed( [&]() { project_->set_som_radius(som_radius_->v_float[0]); } );
        som_sradius_->    cb_value_changed( [&]() { project_->set_som_search_radius(som_sradius_->v_float[0]); } );



        // ----- Project callbacks -----

        // connect wave update
        project_->cb_wave_loaded( [&]()
        {
            //waveview_->setWave(&project_->wave());
            waveview_->update();
        } );

        // connect band calc update
        project_->cb_bands(       [&]()
        {
            waveview_->setWave(&project_->wave());
            waveview_->update();
        } );


        // when bands are finished, the som can be (re-)created
        project_->cb_bands_finished( [&]()
        {
            // do not let the wave thread interfere with gui
            start_som_signal();
        } );

        // when SOM is allocated
        project_->cb_som_ready( [&]()
        {
            // connect the view
            somview_->setSom(&project_->som());
            setSomPaintMode_();
        } );

        // when SOM is updated
        project_->cb_som(       [&]()
        {
            som_update_signal();
        } );

    checkWidgets_();
}

void ProjectView::checkWidgets_()
{
    const bool
        tograin = som_size_to_g_->v_bool[0],
        fromgrain = som_size_use_f_->v_bool[0],

        bandnorm = wave_band_norm_->v_bool[0];

    // --- set activity ---

    wave_grain_size_->setActive(  !tograin );
    wave_band_amp_->setActive(    !bandnorm );

    som_size_use_f_->setActive(   !tograin );
    som_sizef_->setActive(        fromgrain && !tograin );
    som_size_->setActive(         !fromgrain || tograin );

    somd_band_nr_->setActive(     somd_dmode_->v_int[0] == SDM_SINGLE_BAND);
    somd_calc_imap_->setActive(   somd_dmode_->v_int[0] == SDM_IMAP);
}


bool ProjectView::loadWave(/*const std::string& fn*/)
{
    QString fn =
    #if (1)
            // "/home/defgsus/prog/C/matrixoptimizer/data/audio/SAT/rausch/radioscan_05_4.5.wav"
            // "/home/defgsus/prog/C/matrixoptimizer/data/audio/SAT/gong/metalfx01.wav"
             "/home/defgsus/prog/C/matrixoptimizer/data/audio/SAT/ldmvoice/danaykroyd.wav";
    #else
        QFileDialog::getOpenFileName(this,
            "Open Sound",
            "/home/defgsus/prog/C/matrixoptimizer/data/audio/SAT",
            "Sound Files (*.wav *.riff *.voc);;All (*)"
            );
    #endif

    if (fn.isNull()) return false;

    // disconnect views
    waveview_->setWave(0);
    somview_->setSom(0);

    if (!project_->load_wave(
                fn.toStdString()
            )) return false;

    return true;
}

void ProjectView::set_wave_()
{
    SOM_DEBUG("ProjectView::set_wave_()");

    // disconnect views
    waveview_->setWave(0);
    // disconnect som view (it will change anyway)
    somview_->setSom(0);

    const bool
        norm = wave_band_norm_->v_bool[0],
        tograin = som_size_to_g_->v_bool[0];

    // set grainsize as function of som size
    if (tograin)
    {
        const size_t soms = std::max(1, som_size_->v_int[0] * som_size_->v_int[1]);
        size_t grains = std::max(1ul, project_->wave().info.frames / soms );
        wave_grain_size_->v_int[0] = grains;
        wave_grain_size_->updateWidget(false);
    }

    project_->set(
            wave_bands_->v_int[0],
            wave_freq_->v_float[0],
            wave_freq_->v_float[1],
            wave_grain_size_->v_int[0],
            wave_window_->v_int[0],
            norm? 0.f : wave_band_amp_->v_float[0],
            wave_band_exp_->v_float[0]
            );

}

void ProjectView::set_som_()
{
    SOM_DEBUG("ProjectView::set_som_()");

    // disconnect view
    somview_->setSom(0);

    // determine size from nr_grains
    // and change widget values
    if (som_size_use_f_->v_bool[0]
    && !som_size_to_g_->v_bool[0])
    {
        // set size as a function of number of grains
        const size_t s = std::max(1.f, ceilf(
            sqrtf(project_->num_grains() * som_sizef_->v_float[0]) ));

        // update size widgets
        som_size_->v_int[0] = som_size_->v_int[1] = s;
        som_size_->updateWidget(false);

    }

    // set sizex * sizey
    project_->set_som(
            som_size_->v_int[0],
            som_size_->v_int[1],
            som_seed_->v_int[0]
        );
}


void ProjectView::setSomPaintMode_()
{
    const int pmode = somd_dmode_->v_int[0];

    SOM_DEBUG("ProjectView::setSomPaintMode_(" << pmode << ")");

    // limit value in 'draw band number'
    if (pmode == SDM_SINGLE_BAND)
        somd_band_nr_->setMax((int)project_->som().dim - 1);

    switch (pmode)
    {
        case SDM_SINGLE_BAND:   somview_->paintBandNr(somd_band_nr_->v_int[0]);
                                somview_->paintMode(SomView::PM_Band); break;
        case SDM_MULTI_BAND:    somview_->paintMode(SomView::PM_MultiBand); break;

        case SDM_IMAP:          somview_->paintMode(SomView::PM_IMap); break;

        default:                somview_->paintMode(SomView::PM_UMap); break;
    }

}

void ProjectView::som_update()
{
    //label->setText(QString::fromStdString(project_->som().info_str()));

    /** @note The umap/imap calculation as well as the
        displaying by SomView happens asynchronously to
        the som training. */

    // calc other info maps if needed
    if (somd_dmode_->v_int[0] == SDM_IMAP)
    {
        if (somd_calc_imap_->v_bool[0])
            project_->som().calc_imap();
    }

    if (somd_dmode_->v_int[0] == SDM_UMAP)
        project_->som().calc_umap();

    somview_->update();
}

bool ProjectView::need_umap_()
{
    return
        somd_dmode_->v_int[0] >= SDM_UMAP;
}

bool ProjectView::need_imap_()
{
    return
        somd_dmode_->v_int[0] == SDM_IMAP;
}

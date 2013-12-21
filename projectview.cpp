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

enum SomDrawMode
{
    SDM_SINGLE_BAND,
    SDM_MULTI_BAND,
    /* let this be the first of all modes that need the umap. (or change ProjectView::need_umap_) */
    SDM_UMAP
};

ProjectView::ProjectView(Project * p, QWidget *parent) :
    QFrame(parent),
    project_    (p)
{
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

    wave_spec_colors_ = new Property("wave_draw_spec_colors", "draw spec.\ncolors", "...");
    wave_spec_colors_->init(false);

    // --- som properties

    som_size_ = new Property("som_size", "som size", "...");
    som_size_->init(1, 2<<16, project_->som_sizex(), project_->som_sizey());

    som_size_use_f_ = new Property("som_size_use_f", "som size\nfrom grains", "...");
    som_size_use_f_->init(false);

    som_sizef_ = new Property("som_sizef", "factor", "...");
    som_sizef_->init(0.001f, 10.f, 1.1f);
    som_sizef_->setActive(false);

    som_seed_ = new Property("som_seed", "random seed", "...");
    som_seed_->init(-(2<<16), 2<<16, project_->som_seed());

    som_alpha_ = new Property("som_alpha", "alpha", "...");
    som_alpha_->init(0.f, 1.f, project_->som_alpha());

    som_radius_ = new Property("som_radius", "radius", "...");
    som_radius_->init(0.f, 1.f, project_->som_radius());

    som_dmode_ = new Property("som_draw_mode", "draw mode", "...");
    som_dmode_->init(
            { SDM_SINGLE_BAND, SDM_MULTI_BAND, SDM_UMAP },
            { "single_band", "multi_band", "umap" },
            { "single band", "spectral color", "neighbour distance" },
            SDM_SINGLE_BAND
                );

    som_band_nr_ = new Property("som_band_nr", "band index", "...");
    som_band_nr_->init(0, 0, 0);

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
                wave_spec_colors_-> createWidget(this, l2, lt);
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

        wave_bands_->     cb_value_changed(std::bind(&ProjectView::set_wave_, this));
        wave_freq_->      cb_value_changed(std::bind(&ProjectView::set_wave_, this));
        wave_grain_size_->cb_value_changed(std::bind(&ProjectView::set_wave_, this));
        wave_window_->    cb_value_changed(std::bind(&ProjectView::set_wave_, this));
        wave_band_amp_->  cb_value_changed(std::bind(&ProjectView::set_wave_, this));
        wave_band_exp_->  cb_value_changed(std::bind(&ProjectView::set_wave_, this));
        wave_band_norm_-> cb_value_changed([=]()
        {
            // disable amp when normalized
            wave_band_amp_->setActive(!wave_band_norm_->v_bool[0]);
            set_wave_();
        });

        wave_spec_colors_->cb_value_changed( [=](){ waveview_->draw_spec_colors(wave_spec_colors_->v_bool[0]); });

        // -- callbacks --

        // connect wave update
        project_->cb_wave_loaded( [&]()
        {
            waveview_->setWave(&project_->wave());
            waveview_->update();
        } );
        // connect band calc update
        project_->cb_bands(       [&](){ waveview_->update(); } );
        //project_->cb_bands_finished( [&](){ set_som_(); });



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
                som_dmode_->    createWidget(this, l2, lt);
                som_band_nr_->  createWidget(this, l2, lt);

                l2->addStretch(2);
            }

            l1->addSpacing(20);

            l1->addWidget(somview_);

            // --- SOM parameters ----

            l2 = new QVBoxLayout(0);
            l1->addLayout(l2);
            {
                const Property::LayoutType lt = Property::H_WIDGET_LABEL;
                som_size_->       createWidget(this, l2, lt);
                som_size_use_f_-> createWidget(this, l2, lt);
                som_sizef_->      createWidget(this, l2, lt);
                l2->addSpacing(20);
                som_seed_->       createWidget(this, l2, lt);
                som_alpha_->      createWidget(this, l2, lt);
                som_radius_->     createWidget(this, l2, lt);

                l2->addStretch(2);
            }
        }

        // -- connect Properties --

        som_size_->       cb_value_changed(std::bind(&ProjectView::set_som_, this));
        som_size_use_f_-> cb_value_changed( [=]() { som_sizef_->setActive(som_size_use_f_->v_bool[0]); set_som_(); } );
        som_sizef_->      cb_value_changed( std::bind(&ProjectView::set_som_, this) );
        som_seed_->       cb_value_changed(std::bind(&ProjectView::set_som_, this));
        som_alpha_->      cb_value_changed( [=]() { project_->set_som_alpha(som_alpha_->v_float[0]); } );
        som_radius_->     cb_value_changed( [=]() { project_->set_som_radius(som_radius_->v_float[0]); } );

        som_dmode_->      cb_value_changed( [&]() { setSomPaintMode_(); });
        som_band_nr_->    cb_value_changed( [=]() { somview_->paintBandNr(som_band_nr_->v_int[0]); });

        // -- callbacks --

        // when SOM is allocated
        project_->cb_som_ready( [&]()
        {
            somview_->setSom(&project_->som());
            // update band_nr_
            if (som_dmode_->v_int[0] == SDM_SINGLE_BAND)
                setSomPaintMode_();
        } );
        // when SOM is updated
        project_->cb_som(       [&]()
        {
            //label->setText(QString::fromStdString(project_->som().info_str()));
            if (need_umap_())
                project_->som().calc_umap();
            somview_->update();
        } );

}

void ProjectView::setSomPaintMode_()
{
    const int pmode = som_dmode_->v_int[0];

    SOM_DEBUG("ProjectView::setSomPaintMode_(" << pmode << ")");

    if (pmode == SDM_SINGLE_BAND)
    {
        som_band_nr_->setMax((int)project_->som().dim - 1);
        som_band_nr_->setActive(true);
    }
    else
        som_band_nr_->setActive(false);

    switch (pmode)
    {
        case SDM_SINGLE_BAND:   somview_->paintBandNr(som_band_nr_->v_int[0]);
                                somview_->paintMode(SomView::PM_Band); break;
        case SDM_MULTI_BAND:    somview_->paintMode(SomView::PM_MultiBand); break;

        default:                somview_->paintMode(SomView::PM_UMap); break;
    }

}

bool ProjectView::need_umap_()
{
    return
        som_dmode_->v_int[0] >= SDM_UMAP;
}

bool ProjectView::loadWave(/*const std::string& fn*/)
{
    // disconnect views
    waveview_->setWave(0);
    somview_->setSom(0);

    if (!project_->load_wave(
            "/home/defgsus/prog/C/matrixoptimizer/data/audio/SAT/ldmvoice/danaykroyd.wav"
            )) return false;

    return true;
}

void ProjectView::set_wave_()
{
    SOM_DEBUG("ProjectView::set_wave_()");

    // disconnect som view (it will change anyway)
    somview_->setSom(0);

    bool n = wave_band_norm_->v_bool[0];

    project_->set(
            wave_bands_->v_int[0],
            wave_freq_->v_float[0],
            wave_freq_->v_float[1],
            wave_grain_size_->v_int[0],
            wave_window_->v_int[0],
            n? 0.f : wave_band_amp_->v_float[0],
            wave_band_exp_->v_float[0]
            );

    set_som_();
}

void ProjectView::set_som_()
{
    SOM_DEBUG("ProjectView::set_som_()");

    // disconnect view
    somview_->setSom(0);

    if (som_size_use_f_->v_bool[0])
    {
        // set size as a function of number of grains
        const size_t s = std::max(1.f, ceilf(
            sqrtf(project_->num_grains() * som_sizef_->v_float[0]) ));

        project_->set_som(
                s,
                s,
                som_seed_->v_int[0]
            );
    }
    else
        // sizex * sizey
        project_->set_som(
                som_size_->v_int[0],
                som_size_->v_int[1],
                som_seed_->v_int[0]
            );

}


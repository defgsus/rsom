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

    wave_bands_ = new Property("wave_bands", "number bands");
    wave_bands_->init(1, 100000, project_->num_bands());
    wave_bands_->help =
            "<b>number of frequency bands</b>"
            "<p>The number of frequency bands that should be calculated "
            "for training the <i>som</i>.</p>"
            "<p>Generally, more bands = more accurate = slower. "
            "In most cases about 16 to 32 bands can be just enough, "
            "about 80 are very accurate.</p>";

    wave_freq_ = new Property("wave_freq", "frequency range");
    wave_freq_->init(0.0001f, 100000.f, project_->wave().min_freq, project_->wave().max_freq);
    wave_freq_->help =
            "<b>low and high frequency range (hz)</b>"
            "<p>Each spectral band will be between, including, the low "
            "and high value.</p>"
            "<p>Try to narrow-in on the range you are interested in to raise accuracy.</p>";

    wave_grain_size_ = new Property("wave_grain_size", "grain size");
    wave_grain_size_->init(1, 2<<16, project_->wave().grain_size);
    wave_grain_size_->help =
            "<b>grain size (samples)</b>"
            "<p>The whole sample will be split into <i>sample-length</i>/<i>grain-size</i> grains. "
            "The lower this number, the higher the number of grains.</p>"
            "<p>Please be careful, a low value might allocate a lot of memory.</p>"
            "<p>The 'grains' can be thought of as indices into the wave file with "
            "associated spectral vectors each.</p>";

    wave_window_ = new Property("wave_window_width", "window width");
    wave_window_->init(1, 2<<16, project_->wave().window_width);
    wave_window_->help =
            "<b>fourier transform window width (samples)</b>"
            "<p>The size of the window determines the length of the sample "
            "that is taken into account for the spectral calculation of each grain.</p>";

    wave_band_norm_ = new Property("wave_band_norm", "normalize\nbands");
    wave_band_norm_->init(false);
    wave_band_norm_->help =
            "<b>normalize band data after calculation</b>"
            "<p>Once the spectral data is analyzed, it can be normalized to nicely "
            "fit between 0 and 1.</p>";

    wave_band_amp_ = new Property("wave_band_amp", "band amp");
    wave_band_amp_->init(0.0001f, 1000.f, project_->band_amp());
    wave_band_amp_->help =
            "<b>spectral data amplitude</b>"
            "<p>Multiplier for the spectral data, used to scale the data into a "
            "user-defined range.</p>";

    wave_band_exp_ = new Property("wave_band_exp", "band exp");
    wave_band_exp_->init(0.0001f, 100.f, project_->band_exponent());
    wave_band_exp_->help =
            "<b>spectral data exponent</b>"
            "<p>Exponent for the spectral data, used the shape the gradient.</p>"
            "<p>Each data point is calculated by "
            "<pre>clamp(value * amplitude) ^ exponent</pre> "
            "Where clamp() means the value is clamped between 0 and 1.</p>";

    waved_waveform_ = new Property("waved_waveform", "draw waveform");
    waved_waveform_->init(true);
    waved_waveform_->help =
            "<b>switch display of waveform on/off</b>";

    waved_spec_colors_ = new Property("waved_spec_colors", "draw spec.\ncolors");
    waved_spec_colors_->init(false);
    waved_spec_colors_->help =
            "<b>switch between <i>band-color</i> and <i>spectral-color</i></b>."
            "<p>The <i>band-color</i> is a color scale defined by amplitude of "
            "each spectral band's grain. The <i>spectral-color</i> is a (arbitrary) "
            "color that tries to be unique for a unique grain, using the whole "
            "band data as source.</p><p>The latter mode is implemented to compare with "
            "colors from the <i>som</i> display. ";

    // --- som properties

    som_size_ = new Property("som_size", "som size");
    som_size_->init(2, 2<<16, project_->som_sizex(), project_->som_sizey());
    som_size_->help =
            "<b><i>som</i> x/y size in pixels/cells</b>"
            "<p>The size of the self-organizing map. It is only editable, "
            "when <b>som size from grains</b> is not selected.</p>";

    som_size_use_f_ = new Property("som_size_use_f", "som size\nfrom grains");
    som_size_use_f_->init(true);
    som_size_use_f_->help =
            "<b>determine <i>som</i> size from number of grains</b>"
            "<p>In this mode, the size of the som will be quadratic with a "
            "side-length choosen so that the number of grains in the wave file "
            "will match the number of cells.</p>";

    som_sizef_ = new Property("som_sizef", "factor");
    som_sizef_->init(0.001f, 10.f, 1.1f);
    som_sizef_->help =
            "<b>number of grains to <i>som</i> size ratio</b>"
            "<p>The number of grains will be multiplied with this number to "
            "define the number of <i>som</i> cells (when <b>som size from grains</b> is "
            "selected).</p>"
            "<p>If you want a tight packing of indices into the map, select a factor close "
            "to 1. To create valeys of undefined indices between clusters, select a value "
            "larger than 1.</p>";

    som_size_to_g_ = new Property("som_size_to_grains", "som size\nto num. grains");
    som_size_to_g_->init(false);
    som_size_to_g_->help =
            "<b>determine number of grains from size of <i>som</i></b>"
            "<p>When selected, the number of grains in the wave data is "
            "set to the number of cells in the <i>som</i>. This is usefull if "
            "you want to calculate a map of a certain size and where the number "
            "of grains should fill the map.</p>";

    som_seed_ = new Property("som_seed", "random seed");
    som_seed_->init(-(2<<16), 2<<16, project_->som_seed());
    som_seed_->help =
            "<b>random seed for initializing <i>som</i></b>"
            "<p>The <i>self-organizing map</i> will be initialized with random "
            "samples from the wave data. The initial random seed is determined "
            "with this number. Keeping the see and nmt changing any other settings "
            "would recreate the same map again and again.</p>";

    som_alpha_ = new Property("som_alpha", "alpha");
    som_alpha_->init(0.f, 1.f, project_->som_alpha());
    som_alpha_->help =
            "<b>opacity of data insertions [0,1].</b>"
            "<p>Whenever data is inserted into the <i>som</i> and when "
            "a position is determined, a circle is drawn with transparency "
            "set by this value.</p>"
            "<p>There is enough paperwork on it's own dedicated to this value. "
            "Probably in most cases it's best to keep it below 0.1.</p> ";

    som_radius_ = new Property("som_radius", "radius");
    som_radius_->init(0.f, 1.f, project_->som_radius());
    som_radius_->help =
            "<b>radius of data insertions [0,1].</b>"
            "<p>The radius of the neighbourhood that is adjusted on "
            "data insertions, messured as factor of half the smallest side-length "
            "of the <i>som</i>.</p>"
            "<p>Hard to describe, easy to experiment with...</p>";

    som_sradius_ = new Property("som_search_radius", "local search radius");
    som_sradius_->init(0.f, 2.f, project_->som_search_radius());
    som_sradius_->help =
            "<b>radius of local search [0,2]</b>"
            "<p>The radius of the local search for data-matching, messured "
            "as factor if half the smallest side-length of the <i>som</i>.</p>"
            "<i>Local search</i> means, once a data-point is inserted into the "
            "map, it will not move more than this radius per following training "
            "step. This does significantly speed-up the training, while keeping "
            "the result close to what <i>global search</i> does.</p>"
            "<p>To switch off <i>local search</i> set a value of 2.</p>"
            "<p>Note that this value has particularily more significance "
            "when <b>ignore vacant cells</b> is activated.</p>";

    som_non_dupl_ = new Property("som_no_duplicates", "ignore vacant cells");
    som_non_dupl_->init(false);
    som_non_dupl_->help =
            "<b>ignore vacant cells for data matching</b>"
            "<p>When a data sample (grain) is inserted into the map, "
            "the best matching position is searched before. But it is not "
            "likely that every cell in the map is a unique best-match "
            "for each data sample. More often, do several data samples "
            "best match with the same cell in the map, while other cells "
            "aren't matched by any sample.</p>"
            "<p>To completely fill the map (given the cell/grains ratio is 1) "
            "use this mode. A cell is only considered as a match for a data "
            "sample when no other data sample was inserted before.</p>";

    som_wrap_ = new Property("som_wrap_edge", "wrap on edges");
    som_wrap_->init(true);
    som_wrap_->help =
            "<b>wrap map operations on edges</b>"
            "<p>Operations like adjusting the neighbourhood on data inserts "
            "that would normally be clipped on edges can be wrapped around.</p>";

    somd_dmode_ = new Property("somd_mode", "draw mode");
    somd_dmode_->init(
            { SDM_SINGLE_BAND, SDM_MULTI_BAND, SDM_UMAP, SDM_IMAP },
            { "single_band", "multi_band", "umap", "imap" },
            { "single band", "spectral color", "neighbour distance", "grain index" },
            SDM_SINGLE_BAND
                );

    somd_band_nr_ = new Property("somd_band_nr", "band index");
    somd_band_nr_->init(0, 0, 0);

    somd_mult_ = new Property("somd_color_scale", "color scale");
    somd_mult_->init(0.0001f, 1000.f, 1.f);

    somd_calc_imap_ = new Property("somd_calc_imap", "calculate imap");
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
                som_non_dupl_->   createWidget(this, l2, lt);
                som_wrap_->       createWidget(this, l2, lt);

                l2->addStretch(2);
            }
        }

        // -- connect Properties --

        somd_dmode_->     cb_value_changed( [&]() { checkWidgets_(); setSomPaintMode_(); });
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

        som_non_dupl_->   cb_value_changed( [&]() { project_->som().do_non_duplicate = som_non_dupl_->v_bool[0]; } );
        som_wrap_->       cb_value_changed( [&]() { project_->som().do_wrap = som_wrap_->v_bool[0]; } );



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
            /** @note The umap/imap calculation must
                not happen asynchronously to the som training.
                This callback however comes from the som thread! */
            calc_maps_();
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
    #if (0)
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

void ProjectView::calc_maps_()
{
    // assume Som::map[] is just ready

    // calc other info maps as needed

    if (somd_dmode_->v_int[0] == SDM_IMAP)
    {
        if (somd_calc_imap_->v_bool[0])
            project_->som().calc_imap();
    }

    if (somd_dmode_->v_int[0] == SDM_UMAP)
        project_->som().calc_umap();
}

void ProjectView::som_update()
{
    //label->setText(QString::fromStdString(project_->som().info_str()));

    somview_->update();
}


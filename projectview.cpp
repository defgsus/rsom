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
#include "projectview.h"

#include "core/log.h"
#include "core/project.h"
//#include "core/write_ntf.h"
#include "core/data.h"

#include "property.h"
#include "properties.h"
#include "somview.h"
#include "helpwindow.h"
#include "dataview.h"

#include <QLayout>
#include <QCheckBox>
#include <QSpinBox>
#include <QPushButton>
#include <QFrame>
#include <QLabel>
#include <QFileDialog>
#include <QTextBrowser>
#include <QKeyEvent>

enum SomDrawMode
{
    SDM_SINGLE_BAND,
    SDM_MULTI_BAND,
    SDM_IMAP,
    /* let this be the first of all modes that need the umap. (or change ProjectView::need_umap_) */
    SDM_UMAP
};

ProjectView::ProjectView(RSOM::Project * p, QWidget *parent) :
    QFrame(parent),
    project_    (p),
    data_dir_   ("."),
    export_dir_ ("."),
    som_dir_   ("."),
    props_      (new Properties)
{
    // some signals for threadsafety
    // we use this for passing events from the Project back to the GUI thread
    connect(this, SIGNAL(start_som_signal()), this, SLOT(start_som()) );
    connect(this, SIGNAL(som_update_signal()), this, SLOT(som_update()) );
    connect(this, SIGNAL(start_training_signal()), this, SLOT(startTraining()) );
    connect(this, SIGNAL(log_signal(const QString&)), this, SLOT(log(const QString&)) );
    connect(this, SIGNAL(error_signal(const QString&)), this, SLOT(error(const QString&)) );

    // --- properties ---

    SOM_DEBUG("ProjectView::ProjectView:: creating properties");

    // little helper macro
    // we collect all properties also in props_ to have them uniformly accessible
    // (e.g. for disk i/o which needs to be done some day)
    #define SOM_NEW_PROPERTY(var__, id__, name__) \
        var__ = new Property(id__, name__); \
        props_->add(var__);

    // note that the values used to initialize the properties
    // need to match the settings in the classes (Project, Som, Wave)
    // Not all properties are checked on start of program but
    // only when user-events force new settings.
/*
    SOM_NEW_PROPERTY(wave_bands_, "wave_bands", "number bands");
    wave_bands_->init(1, 100000, project_->num_bands());
    wave_bands_->help =
            "<b>number of frequency bands</b>"
            "<p>The number of frequency bands that should be calculated "
            "for training the <i>som</i>.</p>"
            "<p>Generally, more bands = more accurate = slower. "
            "In most cases about 16 to 32 bands can be just enough, "
            "about 80 are quite accurate.</p>";

    SOM_NEW_PROPERTY(wave_freq_, "wave_freq", "frequency range");
    wave_freq_->init(0.0001f, 100000.f, project_->wave().min_freq, project_->wave().max_freq);
    wave_freq_->help =
            "<b>low and high frequency range (hz)</b>"
            "<p>Each spectral band will be between, including, the low "
            "and high value.</p>"
            "<p>Try to narrow-in on the range you are interested in to raise accuracy.</p>";

    SOM_NEW_PROPERTY(wave_grain_size_, "wave_grain_size", "grain size");
    wave_grain_size_->init(1, 2<<16, project_->wave().grain_size);
    wave_grain_size_->help =
            "<b>grain size (samples)</b>"
            "<p>The whole sample will be split into <i>sample-length</i>/<i>grain-size</i> grains. "
            "The lower this number, the higher the number of grains.</p>"
            "<p>Please be careful, a low value might allocate a lot of memory.</p>"
            "<p>The 'grains' can be thought of as indices into the wave file with "
            "associated spectral vectors each.</p>";

    SOM_NEW_PROPERTY(wave_window_, "wave_window_width", "window width");
    wave_window_->init(1, 2<<16, project_->wave().window_width);
    wave_window_->help =
            "<b>fourier transform window width (samples)</b>"
            "<p>The size of the window determines the length of the sample "
            "that is taken into account for the spectral calculation of each grain.</p>";

    SOM_NEW_PROPERTY(wave_band_norm_, "wave_band_norm", "normalize\nbands");
    wave_band_norm_->init(false);
    wave_band_norm_->help =
            "<b>normalize band data after calculation</b>"
            "<p>Once the spectral data is analyzed, it can be normalized to nicely "
            "fit between 0 and 1.</p>";

    SOM_NEW_PROPERTY(wave_band_amp_, "wave_band_amp", "band amp");
    wave_band_amp_->init(0.0001f, 1000.f, project_->band_amp());
    wave_band_amp_->help =
            "<b>spectral data amplitude</b>"
            "<p>Multiplier for the spectral data, used to scale the data into a "
            "user-defined range.</p>";

    SOM_NEW_PROPERTY(wave_band_exp_, "wave_band_exp", "band exp");
    wave_band_exp_->init(0.0001f, 100.f, project_->band_exponent());
    wave_band_exp_->help =
            "<b>spectral data exponent</b>"
            "<p>Exponent for the spectral data, used the shape the gradient.</p>"
            "<p>Each data point is calculated by "
            "<pre>clamp(value * amplitude) ^ exponent</pre> "
            "Where clamp() means the value is clamped between 0 and 1.</p>";

    SOM_NEW_PROPERTY(waved_waveform_, "waved_waveform", "draw waveform");
    waved_waveform_->init(true);
    waved_waveform_->help =
            "<b>switch display of waveform on/off</b>";

    SOM_NEW_PROPERTY(waved_spec_colors_, "waved_spec_colors", "draw spec.\ncolors");
    waved_spec_colors_->init(false);
    waved_spec_colors_->help =
            "<b>switch between <i>band-color</i> and <i>spectral-color</i></b>."
            "<p>The <i>band-color</i> is a color scale defined by amplitude of "
            "each spectral band's grain. The <i>spectral-color</i> is a (arbitrary) "
            "color that tries to be unique for a unique grain, using the whole "
            "band data as source.</p><p>The latter mode is implemented to compare with "
            "colors from the <i>som</i> display. ";
*/
    // --- som properties

    SOM_NEW_PROPERTY(som_run_, "", "TRAINING");
    som_run_->init(true);
    som_run_->help =
            "<b>start and stop <i>som</i> training</b>";

    SOM_NEW_PROPERTY(som_size_, "som_size", "som size");
    som_size_->init(2, 2<<16, project_->som_sizex(), project_->som_sizey());
    som_size_->help =
            "<b><i>som</i> x/y size in pixels/cells</b>"
            "<p>The size of the self-organizing map. It is only editable, "
            "when <b>som size from grains</b> is not selected.</p>";

    SOM_NEW_PROPERTY(som_size_use_f_, "som_size_use_f", "som size\nfrom grains");
    som_size_use_f_->init(true);
    som_size_use_f_->help =
            "<b>determine <i>som</i> size from number of grains</b>"
            "<p>In this mode, the size of the som will be quadratic with a "
            "side-length choosen so that the number of grains in the wave file "
            "will match the number of cells.</p>";

    SOM_NEW_PROPERTY(som_sizef_, "som_sizef", "factor");
    som_sizef_->init(0.001f, 10.f, 1.1f);
    som_sizef_->help =
            "<b>number of grains to <i>som</i> size ratio</b>"
            "<p>The number of grains will be multiplied with this number to "
            "define the number of <i>som</i> cells (when <b>som size from grains</b> is "
            "selected).</p>"
            "<p>If you want a tight packing of indices into the map, select a factor close "
            "to 1. To create valeys of undefined indices between clusters, select a value "
            "larger than 1.</p>";

    SOM_NEW_PROPERTY(som_seed_, "som_seed", "random seed");
    som_seed_->init(-(2<<16), 2<<16, project_->som_seed());
    som_seed_->help =
            "<b>random seed for initializing <i>som</i></b>"
            "<p>The <i>self-organizing map</i> will be initialized with random "
            "samples from the wave data. The initial random seed is determined "
            "with this number. Keeping the see and nmt changing any other settings "
            "would recreate the same map again and again.</p>";

    SOM_NEW_PROPERTY(som_alpha_, "som_alpha", "alpha");
    som_alpha_->init(0.f, 1.f, project_->som_alpha());
    som_alpha_->help =
            "<b>opacity of data insertions [0,1].</b>"
            "<p>Whenever data is inserted into the <i>som</i> and when "
            "a position is determined, a circle is drawn with transparency "
            "set by this value.</p>"
            "<p>There is enough paperwork on it's own dedicated to this value. "
            "Probably in most cases it's best to keep it below 0.1.</p> ";

    SOM_NEW_PROPERTY(som_radius_, "som_radius", "radius");
    som_radius_->init(0.f, 1.f, project_->som_radius());
    som_radius_->help =
            "<b>radius of data insertions [0,1].</b>"
            "<p>The radius of the neighbourhood that is adjusted on "
            "data insertions, messured as factor of half the smallest side-length "
            "of the <i>som</i>.</p>"
            "<p>Hard to describe, easy to experiment with...</p>";

    SOM_NEW_PROPERTY(som_sradius_, "som_search_radius", "local search radius");
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
            "when <b>one grain per cell</b> is activated.</p>";

    SOM_NEW_PROPERTY(som_non_dupl_, "som_no_duplicates", "one grain per cell");
    som_non_dupl_->init(true);
    som_non_dupl_->help =
            "<b>only assign one grain per cell</b>"
            "<p>When a data sample (grain) is inserted into the map, "
            "the best matching position is searched for it. But it is not "
            "likely that every cell in the map is a unique best-match "
            "for each data sample. More often a cell will match more "
            "than one data sample, while other cells aren't matched by any sample at all.</p>"
            "<p>To completely fill the map (given the cell/grains ratio is 1) "
            "use this mode, which is also the default for exporting Reaktor maps. "
            "A cell is only considered as a match for a data "
            "sample when no other data sample was inserted before. "
            "(When samples move to another cell they clear the previous cell.)</p>";

    SOM_NEW_PROPERTY(som_wrap_, "som_wrap_edge", "wrap on edges");
    som_wrap_->init(true);
    som_wrap_->help =
            "<b>wrap map operations on edges</b>"
            "<p>Operations like adjusting the neighbourhood on data inserts "
            "that would normally be clipped on edges can be wrapped around.</p>";
    som_wrap_->setActive(false);

    SOM_NEW_PROPERTY(somd_dmode_, "somd_mode", "draw mode");
    somd_dmode_->init(
            { SDM_SINGLE_BAND, SDM_MULTI_BAND, SDM_UMAP, SDM_IMAP },
            { "single_band", "multi_band", "umap", "imap" },
            { "single band", "spectral color", "neighbour distance", "grain index" },
            SDM_SINGLE_BAND
                );
    somd_dmode_->help =
            "<b>display/calculation mode for <i>som view</i></b>"
            "<p>This list lets you choose the contents that are displayed "
            "in the <i>som view</i> as well as the data that is exported "
            "to a reaktor table on <b>export</b>.</p>"
            "<p><b>single band</b>: Each cell's color is determined by a single "
            "band of the spectral data. The band is selected by <b>band index</b>. "
            "This mode lets you study the distribution of certain frequencies across the map.</p>"
            "<p><b>spectral color</b>: Each cell's color is a function of the whole "
            "band data. The representation is completely arbitrary but similiar grains "
            "are colored similiarily.</p>"
            "<p><b>neighbour distance</b>: Classical <i>som</i> coloring where the color "
            "represents the difference of each cell to it's neighbours. Darker areas in the "
            "map represent clusters of similiar grains, while brighter colors represent "
            "the edges between those clusters.</p>"
            "<p><b>grain index</b>: The color represents the position-in-wave of the grain "
            "that is associated to each cell. This is the important data for Reaktor to "
            "build grain synthesizers. Black cells are not associated to grains.</p>";

    SOM_NEW_PROPERTY(somd_band_nr_, "somd_band_nr", "band index");
    somd_band_nr_->init(0, 0, 0);
    somd_band_nr_->help =
            "<b>spectral band to display in single band mode</b>"
            "<p>In <b>single band mode</b>, this value selects "
            "the spectral band to display in the <i>som view</i>, "
            "starting at index 0.</p>";

    SOM_NEW_PROPERTY(somd_mult_, "somd_color_scale", "color scale");
    somd_mult_->init(0.0001f, 1000.f, 1.f);
    somd_mult_->help =
            "<b>amplitude of color in <i>som view</i></b>"
            "<p>A simple multiplier for the colors displayed "
            "in the <i>som view</i>. Especially useful for <b>spectral color</b> "
            "display mode.</p>";

    SOM_NEW_PROPERTY(somd_calc_imap_, "somd_calc_imap", "calculate imap");
    somd_calc_imap_->init(false);
    somd_calc_imap_->help =
            "<b>fully re-calculate the <i>index map</i> on each <i>som view</i> udpate</b>"
            "<p>This feature is *experimental* right now.</p>";

    #undef SOM_NEW_PROPERTY

    const QString dataview_help =
            "<b>data curve display</b>";

    const QString somview_help =
            "<b>self-organizing map display</b>"
            "<p>This window shows aspects of the <i>som</i> data, as defined with "
            "the display parameters on the left. The map data is actually three-dimensional, "
            "so different representation can be choosen to display it in two dimensions.</p>"
            "<p>Be aware that changes to the <i>som</i> parameters on the right will <b>restart</b> the "
            "<i>som</i> in most cases (except <b>alpha</b>, <b>radius</b>, <b>local radius</b>, "
            "<b>one grain per cell</b> and <b>wrap</b>).</p>";


    SOM_DEBUG("ProjectView::ProjectView:: building widgets");

    setFrameStyle(QFrame::Panel | QFrame::Raised);
    setLineWidth(2);

    auto l0 = new QVBoxLayout(this);

        // -- project file --
/** @todo Projects can basically be stored/restored to/from disk by
    means of the Property class. */
/*
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
*/

        // -- data file --

        auto l1 = new QHBoxLayout(0);
        l0->addLayout(l1);

            // load button
            auto b = new QPushButton(this);
            l1->addWidget(b);
            b->setText("load data");

            // label
            auto lab = new QLabel(this);
            l1->addWidget(lab);
            lab->setText(QString::fromStdString(project_->data().filepath()));

            connect(b, &QPushButton::clicked, [=]()
            {
                loadData();

                lab->setText(QString::fromStdString(project_->data().filepath()));
            });

            l1->addStretch(2);

        // ---- DATA VIEW ----

        dataview_ = new DataView(this);
        dataview_->setToolTip(dataview_help);

        l0->addWidget(dataview_);

        /*
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
*/

        // ---- SOM view ----

        QPushButton * b_som_save, * b_som_load;

        l1 = new QHBoxLayout(0);
        l0->addLayout(l1);
        {
            somview_ = new SomView(this);
            somview_->setToolTip(somview_help);

            // --- SOM view parameters ---

            auto l2 = new QVBoxLayout(0);
            l1->addLayout(l2);
            {
                const Property::LayoutType lt = Property::H_WIDGET_LABEL;
                somd_dmode_->    createWidget(this, l2, lt);
                somd_band_nr_->  createWidget(this, l2, lt);
                somd_mult_->     createWidget(this, l2, lt);
                /// @todo calc imap not working yet
                //somd_calc_imap_->createWidget(this, l2, lt);

                l2->addSpacing(10);

                // --- export button ---
                auto but = new QPushButton("Export selected data to Reaktor Table", this);
                l2->addWidget(but);

                connect(but, &QPushButton::pressed, [this]()
                {
                    // stop som if running
                    if (som_run_->v_bool[0])
                    {
                        som_run_->v_bool[0] = false;
                        som_run_->updateWidget(false);
                        stopTraining();
                    }
                    exportTable();
                });

                // -- info label --
                l2->addSpacing(10);
                sominfo_ = new QLabel(this);
                l2->addWidget(sominfo_);

                // -- error/log space --
                l2->addSpacing(10);
                log_box_ = new QTextBrowser(this);
                l2->addWidget(log_box_);
                QPalette pal(log_box_->palette());
                pal.setColor(QPalette::Base, QColor(70,70,70));
                log_box_->setPalette(pal);

                //l2->addStretch(2);
            }

            l1->addSpacing(20);

            l1->addWidget(somview_);

            // --- SOM parameters ----

            l2 = new QVBoxLayout(0);
            l1->addLayout(l2);
            {
                const Property::LayoutType lt = Property::H_WIDGET_LABEL;
                som_run_->        createWidget(this, l2, lt);
                som_size_use_f_-> createWidget(this, l2, lt);
                som_sizef_->      createWidget(this, l2, lt);
                som_size_->       createWidget(this, l2, lt);
                l2->addSpacing(20);
                som_seed_->       createWidget(this, l2, lt);
                l2->addSpacing(10);
                som_alpha_->      createWidget(this, l2, lt);
                som_radius_->     createWidget(this, l2, lt);
                som_sradius_->    createWidget(this, l2, lt);
                som_non_dupl_->   createWidget(this, l2, lt);
                som_wrap_->       createWidget(this, l2, lt);

                b_som_save = new QPushButton(this);
                b_som_save->setText("save");
                l2->addWidget(b_som_save);
                b_som_load = new QPushButton(this);
                b_som_load->setText("load");
                l2->addWidget(b_som_load);

                l2->addStretch(2);
            }
        }

        // -- connect Properties --

        somd_dmode_->     cb_value_changed( [&]() { checkWidgets_(); setSomPaintMode_(); });
        somd_band_nr_->   cb_value_changed( [&]() { somview_->paintBandNr(somd_band_nr_->v_int[0]); });
        somd_mult_->      cb_value_changed( [&]() { somview_->paintMultiplier(somd_mult_->v_float[0]); });
        somd_calc_imap_-> cb_value_changed( [&]() { checkWidgets_(); });

        som_run_->        cb_value_changed( [&]() { if (som_run_->v_bool[0]) startTraining(); else stopTraining(); } );
        som_size_use_f_-> cb_value_changed( [&]() { checkWidgets_(); set_som_(); } );
        som_sizef_->      cb_value_changed( std::bind(&ProjectView::set_som_, this) );
        som_size_->       cb_value_changed( [&]() { set_som_(); });

        som_seed_->       cb_value_changed( std::bind(&ProjectView::set_som_, this));
        som_alpha_->      cb_value_changed( [&]() { project_->set_som_alpha(som_alpha_->v_float[0]); } );
        som_radius_->     cb_value_changed( [&]() { project_->set_som_radius(som_radius_->v_float[0]); } );
        som_sradius_->    cb_value_changed( [&]() { project_->set_som_search_radius(som_sradius_->v_float[0]); } );

        som_non_dupl_->   cb_value_changed( [&]() { project_->som().do_non_duplicate(som_non_dupl_->v_bool[0]); } );
        //som_wrap_->       cb_value_changed( [&]() { project_->som().do_wrap = som_wrap_->v_bool[0]; } );



        // ----- Project callbacks -----
/*
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
            // note: do not let the wave thread interfere with gui
            start_som_signal();
        } );
*/
        // when SOMView is clicked

        connect(somview_, &SomView::map_clicked, [=](size_t index)
        {
            dataview_->draw_object( project_->som().getIMap()[index] );
        });

        // save

        connect(b_som_save, &QPushButton::clicked, [=]() { save_som_(); } );
        connect(b_som_load, &QPushButton::clicked, [=]() { load_som_(); } );


        // when SOM is allocated
        project_->cb_som_ready( [&]()
        {
            // connect the view
            somview_->setSom(&project_->som());
            setSomPaintMode_();
            start_training_signal();
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

        // connect to SOM_ERROR macro
        SomLog::error_func = [this](const std::string& text) { error_signal(QString::fromStdString(text)); };
        // connect to SOM_LOG macro
        SomLog::log_func = [this](const std::string& text) { log_signal(QString::fromStdString(text)); };


    checkWidgets_();
}

ProjectView::~ProjectView()
{
    delete props_;
}

void ProjectView::log(const QString& text)
{
    QString t(text);
    t.replace("\n", "<br/>");
    log_box_->append("<font color=\"#8f8\">" + t + "</font>");
}

void ProjectView::error(const QString& text)
{
    QString t(text);
    t.replace("\n", "<br/>");
    log_box_->append("<font color=\"#f88\">" + t + "</font>");
}

void ProjectView::keyPressEvent(QKeyEvent * event)
{
    if (event->key() == Qt::Key_F1)
    {
        HelpWindow * win = new HelpWindow(*this, 0);
        win->show();
    }
    else
    QWidget::keyPressEvent(event);
}

void ProjectView::checkWidgets_()
{
    const bool
        fromgrain = som_size_use_f_->v_bool[0];

    // --- set activity ---

    som_sizef_->setActive(        fromgrain );
    som_size_->setActive(         !fromgrain );

    somd_band_nr_->setActive(     somd_dmode_->v_int[0] == SDM_SINGLE_BAND);
    somd_calc_imap_->setActive(   somd_dmode_->v_int[0] == SDM_IMAP);
}


bool ProjectView::loadData(/*const std::string& fn*/)
{
#if (0)
    project_->data().createRandomData(100000, 12);
    dataview_->setData(&project_->data());
    somview_->setSom(&project_->som());
    if (som_run_->v_bool[0])
        start_som();
    return true;
#endif

    QString fn =
    #if (1)
            // "/home/defgsus/prog/C/matrixoptimizer/data/audio/SAT/rausch/radioscan_05_4.5.wav"
            // "/home/defgsus/prog/C/matrixoptimizer/data/audio/SAT/gong/metalfx01.wav"
             //"/home/defgsus/prog/starmaps/hdeltagalaxy/";
             "/home/defgsus/prog/starmaps/vieles/";
    #else
        QFileDialog::getOpenFileName(this,
            "Open Sound",
#ifdef NDEBUG
            wave_dir_,
#else
            "/home/defgsus/prog/C/matrixoptimizer/data/audio/SAT",
#endif
            "Sound Files (*.wav *.riff *.voc);;All (*)"
            );
    #endif

    if (fn.isNull()) return false;

    // store this directory
    QDir dir(fn);
    data_dir_ = dir.absolutePath();

    // disconnect views
    dataview_->setData(0);
    somview_->setSom(0);

    //project_->data().maxObjects(7000);
    //if (!project_->data().addCsvFile("/home/defgsus/prog/DATA/golstat.txt")) return false;
    // [for starmaps]
    if (!project_->data().loadAsciiDir( fn.toStdString() )) return false;

    project_->data().clamp(0, 60);
    project_->data().normalize();


    // connect data view
    dataview_->setData(&project_->data());

    if (som_run_->v_bool[0])
        start_som();

    return true;
}



bool ProjectView::exportTable()
{
    /*
    // select dialog window caption
    QString dialog_name;
    switch (somd_dmode_->v_int[0])
    {
        case SDM_SINGLE_BAND: dialog_name = "Single Band data"; break;
        case SDM_UMAP: dialog_name = "Neighbour Difference data"; break;
        case SDM_IMAP: dialog_name = "Seconds-Into-Audio data"; break;
        case SDM_MULTI_BAND:
            SOM_ERROR("Multi-Band can not be exported");
            return false;
        default:
            SOM_ERROR("Unsuported export mode");
            return false;
    }

    // get filename
    QString fn =
            QFileDialog::getSaveFileName(this,
                "Export " + dialog_name + " to Reaktor Table",
                export_dir_,
                "*.ntf"
                );

    if (fn.isNull()) return false;

    // store directory
    QDir dir(fn);
    export_dir_ = dir.absolutePath();

    // generate the data to export

    const size_t
            sizex = project_->som().sizex,
            sizey = project_->som().sizey,
            size = sizex * sizey;
    std::vector<float> data(size);

    float min_val = 0.f, max_val = 1.f;

    switch (somd_dmode_->v_int[0])
    {
        case SDM_SINGLE_BAND:
            for (size_t i=0; i<size; ++i)
                data[i] = project_->som().map[i][somd_band_nr_->v_int[0]];
            break;
        case SDM_UMAP:
            for (size_t i=0; i<size; ++i)
                data[i] = project_->som().umap[i];
            break;
        case SDM_IMAP:
            // convert data indices to seconds-into-wave
            max_val = project_->wave().length_in_secs;
            for (size_t i=0; i<size; ++i)
            {
                int idx = project_->som().imap[i];
                if (idx>=0)
                    data[i] = (float)idx / project_->som().data.size() * max_val;
                else
                    /// @todo non-indexed cells should take a user-defined value on export
                    data[i] = 0.f;
            }
            break;
    }

    if (! save_ntf(fn.toStdString(),
             min_val, max_val,
             sizex, sizey,
             &data[0] ))
    {
        SOM_ERROR("error writing table '" << fn.toStdString() << "'");
        return false;
    }

    SOM_LOG("exported " << dialog_name.toStdString() << " to " << fn.toStdString());
    */
    return true;
}



void ProjectView::set_som_()
{
    SOM_DEBUG("ProjectView::set_som_()");

    // disconnect view
    somview_->setSom(0);

    // determine size from numObjects
    // and change widget values
    if (som_size_use_f_->v_bool[0])
    {
        // set size as a function of number of objects
        const size_t s = std::max(1.f, ceilf(
            sqrtf(project_->data().numObjects() * som_sizef_->v_float[0]) ));

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

bool ProjectView::save_som_()
{
    QString fn =
        QFileDialog::getSaveFileName(this,
            "Save SOM Data",
            som_dir_,
            "SOM Files (*.txt *.som *.*)"
            );

    if (fn.isNull()) return false;

    // store this directory
    QDir dir(fn);
    som_dir_ = dir.absolutePath();

    return project_->som().saveMap(fn.toStdString());
}

bool ProjectView::load_som_()
{
    QString fn =
        QFileDialog::getOpenFileName(this,
            "Load SOM Data",
            som_dir_,
            "SOM Files (*.txt *.som *.*)"
            );

    if (fn.isNull()) return false;

    // store this directory
    QDir dir(fn);
    som_dir_ = dir.absolutePath();

    stopTraining();
    somview_->setSom(0);

    bool r = project_->som().loadMap(fn.toStdString());

    if (!r)
    {
        SOM_ERROR("unable to load map " << fn.toStdString());
    }

    somview_->setSom(&project_->som());
    if (r)
    {
        som_size_use_f_->v_bool[0] = false;
        som_size_use_f_->updateWidget(false);
        som_size_->v_int[0] = project_->som().sizex();
        som_size_->v_int[1] = project_->som().sizey();
        som_size_->updateWidget(false);

    //    startTraining();
    }
    return r;
}


void ProjectView::setSomPaintMode_()
{
    const int pmode = somd_dmode_->v_int[0];

    SOM_DEBUG("ProjectView::setSomPaintMode_(" << pmode << ")");

    // limit value in 'draw band number'
    if (pmode == SDM_SINGLE_BAND)
        somd_band_nr_->setMax((int)project_->som().dim() - 1);

    switch (pmode)
    {
        case SDM_SINGLE_BAND:   somview_->paintBandNr(somd_band_nr_->v_int[0]);
                                somview_->paintMode(SomView::PM_Band); break;
        case SDM_MULTI_BAND:    somview_->paintMode(SomView::PM_MultiBand); break;

        case SDM_IMAP:          somview_->paintMode(SomView::PM_IMap); break;

        case SDM_UMAP:          somview_->paintMode(SomView::PM_UMap); break;
    }

    project_->need_map(pmode == SDM_SINGLE_BAND || pmode == SDM_MULTI_BAND);
    project_->need_imap(pmode == SDM_IMAP);
    project_->need_umap(pmode == SDM_UMAP);

    if (!project_->running())
        somview_->update();
}

void ProjectView::startTraining()
{
    if (project_->som_ready() && som_run_->v_bool[0])
        project_->startSomThread();
}

void ProjectView::stopTraining()
{
    project_->stopSomThread();
}

void ProjectView::calc_maps_()
{
    // assume Som::map[] is just ready

    // calc other info maps as needed

    if (somd_dmode_->v_int[0] == SDM_IMAP)
    {
        //if (somd_calc_imap_->v_bool[0])
            //project_->som().calc_imap();
    }

    //if (somd_dmode_->v_int[0] == SDM_UMAP)
        //project_->som().calc_umap();
}

void ProjectView::som_update()
{
    sominfo_->setText(
        QString::fromStdString(project_->info_str()));

    somview_->update();
}


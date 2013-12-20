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

    // --- som properties

    som_size_ = new Property("som_size", "som size", "...");
    som_size_->init(1, 2<<16, project_->som_sizex(), project_->som_sizey());

    som_seed_ = new Property("som_seed", "random seed", "...");
    som_seed_->init(-(2<<16), 2<<16, project_->som_seed());

    som_alpha_ = new Property("som_alpha", "alpha", "...");
    som_alpha_->init(0.f, 1.f, project_->som_alpha());

    som_radius_ = new Property("som_radius", "radius", "...");
    som_radius_->init(0.f, 1.f, project_->som_radius());


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

        // ---- analyze settings ----

        l1 = new QHBoxLayout(0);
        l0->addLayout(l1);
        {
            // create analyzer widgets
            const Property::LayoutType lt = Property::V_LABEL_WIDGET;
            wave_bands_->     createWidget(this, l1, lt);
            wave_freq_->      createWidget(this, l1, lt);
            wave_grain_size_->createWidget(this, l1, lt);
            wave_window_->    createWidget(this, l1, lt);
            l1->addSpacing(10);
            wave_band_norm_-> createWidget(this, l1, lt);
            wave_band_amp_->  createWidget(this, l1, lt);
            wave_band_exp_->  createWidget(this, l1, lt);

            // connect them
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

            l1->addStretch(2);
        }
        // ---- wave view ----

        waveview_ = new WaveView(this);
        l0->addWidget(waveview_);

        // connect wave update
        project_->cb_wave_loaded( [&]()
        {
            waveview_->setWave(&project_->wave());
            waveview_->update();
        } );
        // connect band calc update
        project_->cb_bands(       [&](){ waveview_->update(); } );

        // ---- SOM view ----

        l1 = new QHBoxLayout(0);
        l0->addLayout(l1);
        {
            somview_ = new SomView(this);
            l1->addWidget(somview_);

            // --- SOM parameters ---

            auto l2 = new QVBoxLayout(0);
            l1->addLayout(l2);

                // create som parameter widgets
                const Property::LayoutType lt = Property::H_WIDGET_LABEL;
                som_size_->     createWidget(this, l2, lt);
                som_seed_->     createWidget(this, l2, lt);
                som_alpha_->    createWidget(this, l2, lt);
                som_radius_->   createWidget(this, l2, lt);

                // connect
                som_size_->   cb_value_changed(std::bind(&ProjectView::set_som_, this));
                som_seed_->   cb_value_changed(std::bind(&ProjectView::set_som_, this));
                som_alpha_->  cb_value_changed( [=]() { project_->set_som_alpha(som_alpha_->v_float[0]); } );
                som_radius_-> cb_value_changed( [=]() { project_->set_som_radius(som_radius_->v_float[0]); } );
                /*
                // labelled spinbox
                #define SOM_SPIN(spin_, class_, name_str_, v_, min_, max_) \
                class_ * spin_; \
                { \
                    auto l3 = new QHBoxLayout(0); \
                    l2->addLayout(l3); \
                        spin_ = new class_(this); \
                        l3->addWidget(spin_); \
                        spin_->setMinimum(min_); \
                        spin_->setMaximum(max_); \
                        spin_->setValue(v_); \
                        auto label = new QLabel(this); \
                        label->setText(name_str_); \
                        l3->addWidget(label); \
                }

                SOM_SPIN(spin_x,      QSpinBox,       "size x",       project_->som_sizex(),   1, 2<<16);
                SOM_SPIN(spin_y,      QSpinBox,       "size y",       project_->som_sizey(),   1, 2<<16);
                SOM_SPIN(spin_seed,   QSpinBox,       "random seed",  project_->som_seed(),    -(2<<16), (2<<16));
                SOM_SPIN(spin_alpha,  QDoubleSpinBox, "alpha",        project_->som_alpha(),   0, 100);
                SOM_SPIN(spin_radius, QDoubleSpinBox, "radius",       project_->som_radius(),  0, 100);

                #undef SOM_SPIN

                // reinit som
                const auto func_setsom( [=](int)
                {
                    somview_->setSom(0);

                    project_->set_som(
                        spin_x->value(),
                        spin_y->value(),
                        spin_seed->value()
                        );

                } );

                connect(spin_x,      static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), func_setsom);
                connect(spin_y,      static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), func_setsom);
                connect(spin_seed,   static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), func_setsom);
                connect(spin_alpha,  static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), [=]()
                                    { project_->set_som_alpha(spin_alpha->value()); } );
                connect(spin_radius,  static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), [=]()
                                    { project_->set_som_radius(spin_radius->value()); } );

                auto label = new QLabel();
                */
                l2->addStretch(2);

                // when SOM is allocated
                project_->cb_som_ready( [&]()
                {
                    somview_->setSom(&project_->som());
                } );
                // when SOM is updated
                project_->cb_som(       [&]()
                {
                    //label->setText(QString::fromStdString(project_->som().info_str()));
                    somview_->update();
                } );

        }
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

}

void ProjectView::set_som_()
{
    SOM_DEBUG("ProjectView::set_som_()");

    // disconnect view
    somview_->setSom(0);

    project_->set_som(
            som_size_->v_int[0],
            som_size_->v_int[1],
            som_seed_->v_int[0]
        );

}


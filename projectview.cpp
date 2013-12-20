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
    // properties
    wave_bands_ = new Property("wave_bands", "number spec. bands", "...");
    wave_bands_->init(1, 100000, project_->num_bands());


    SOM_DEBUG("ProjectView:: building widgets");

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


            #define SOM_SPIN(spin_, class_, name_str_, v_, min_, max_) \
            class_ * spin_; \
            { \
                auto l2 = new QVBoxLayout(0); \
                l1->addLayout(l2); \
                    auto label = new QLabel(this); \
                    label->setText(name_str_); \
                    l2->addWidget(label); \
                    spin_ = new class_(this); \
                    l2->addWidget(spin_); \
                    if (auto dbl = dynamic_cast<QDoubleSpinBox*>(spin_)) \
                        dbl->setDecimals(4); \
                    spin_->setMinimum(min_); \
                    spin_->setMaximum(max_); \
                    spin_->setValue(v_); \
            }

            SOM_SPIN(spin_bands,  QSpinBox, "number bands",   project_->wave().nr_bands,     1, 2<<16);
            SOM_SPIN(spin_minf,   QSpinBox, "low frequency",  project_->wave().min_freq,     1, 2<<16);
            SOM_SPIN(spin_maxf,   QSpinBox, "high frequency", project_->wave().max_freq,     1, 2<<16);
            SOM_SPIN(spin_grains, QSpinBox, "grain size",     project_->wave().grain_size,   1, 2<<16);
            SOM_SPIN(spin_window, QSpinBox, "window width",   project_->wave().window_width, 1, 2<<16);
            l1->addSpacing(10);
            SOM_SPIN(spin_band_amp,  QDoubleSpinBox, "amp",   project_->band_amp(),          0.0001, 100);
            SOM_SPIN(spin_band_exp,  QDoubleSpinBox, "amp exponent",
                                                              project_->band_exponent(),     0.0001, 100);
            #undef SOM_SPIN

            wave_bands_->createWidget(this, l1, Property::V_LABEL_WIDGET);

            // on-change function
            const auto func( [=](int)
            {
                project_->set(spin_bands->value(),
                              spin_minf->value(),
                              spin_maxf->value(),
                              spin_grains->value(),
                              spin_window->value(),
                              spin_band_amp->value(),
                              spin_band_exp->value()
                              );
            } );

            connect(spin_bands,  static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), func);
            connect(spin_band_amp,  static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), func);
            connect(spin_band_exp,  static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), func);
            connect(spin_minf,   static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), func);
            connect(spin_maxf,   static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), func);
            connect(spin_grains, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), func);
            connect(spin_window, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), func);

            l1->addStretch(2);
        }
        // ---- wave view ----

        waveview_ = new WaveView(this);
        l0->addWidget(waveview_);

        // connect wave update
        project_->cb_wave_loaded( [=]()
        {
            waveview_->setWave(&project_->wave());
            waveview_->update();
        } );
        project_->cb_bands(       [=](){ waveview_->update(); } );


        // ---- SOM view ----

        l1 = new QHBoxLayout(0);
        l0->addLayout(l1);
        {
            somview_ = new SomView(this);
            l1->addWidget(somview_);

            // --- SOM parameters ---

            auto l2 = new QVBoxLayout(0);
            l1->addLayout(l2);

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

                l2->addStretch(2);

                // when SOM is allocated
                project_->cb_som_ready( [=]()
                {
                    somview_->setSom(&project_->som());
                } );
                // when SOM is updated
                project_->cb_som(       [=]()
                {
                    label->setText(QString::fromStdString(project_->som().info_str()));
                    somview_->update();
                } );
        }
}


bool ProjectView::loadWave(/*const std::string& fn*/)
{
    waveview_->setWave(0);
    somview_->setSom(0);

    if (!project_->load_wave(
            "/home/defgsus/prog/C/matrixoptimizer/data/audio/SAT/ldmvoice/danaykroyd.wav"
            )) return false;

    return true;
}

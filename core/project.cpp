#include "project.h"

#include "log.h"

#include <functional>

#include <QElapsedTimer>

Project::Project()
    :
      filename_     ("new.rsom"),

      som_alpha_    (0.1),
      som_radius_   (0.1),

      wave_         (new Wave),
      som_          (new Som),
      thread_       (0),

      run_          (false),
      restart_      (false),
      wave_changed_ (false),

      cb_wave_loaded_(0),
      cb_bands_      (0)
{
    SOM_DEBUG("Project::Project()");

    // important to init with sane settings!
    set(8, 25, 8000, 2048, 2048, 0.1f, 1.f);
    set_som(48, 48, 0);
}


Project::~Project()
{
    SOM_DEBUG("Project::~Project()");
    stop_worker();
    if (wave_) delete wave_;
}



// --- access ---------
/*
void Project::lock()
{
    mutex_.lock();
}

void Project::unlock()
{
    mutex_.unlock();
}
*/

void Project::name(const std::string& project_name)
{
    SOM_DEBUG("Project::name(" << project_name << ")");
    name_ = project_name;
}


void Project::set(size_t nr_bands, float min_freq, float max_freq, size_t grain_size, size_t window_width,
                  float band_amp, float band_exp)
{
    SOM_DEBUG("Project::set(" << nr_bands << ", " << min_freq << ", " << max_freq << ", " << grain_size << ", "
              << window_width << ", " << band_amp << ", " << band_exp << ")" );

    stop_worker();

    wave_->set(nr_bands, min_freq, max_freq, grain_size, window_width);
    wave_changed_ = true;
    band_amp_ = band_amp;
    band_exp_ = band_exp;

    if (wave_->ok())
        start_worker();

}

void Project::set_som(size_t sizex, size_t sizey, int rand_seed)
{
    SOM_DEBUG("Project::set_som(" << sizex << ", " << sizey << ", " << rand_seed << ")" );

    stop_worker();

    som_sizex_ = sizex;
    som_sizey_ = sizey;
    som_seed_ = rand_seed;

    if (wave_ && wave_->ok())
    {
        start_worker();
    }
}



bool Project::load_wave(const std::string& soundfile_name)
{
    SOM_DEBUG("Project::load_wave(" << soundfile_name << ")");

    stop_worker();
    //std::lock_guard<std::mutex> lock(mutex_);

    if (!wave_->open(soundfile_name))
    {
        SOM_DEBUG("Project::load_wave:: failed");
        wave_->filename = "*error*";
        return false;
    }

    SOM_DEBUG("Project::load_wave::wave_->update()");

    wave_->update();
    wave_changed_ = true;

    // callback
    if (cb_wave_loaded_) cb_wave_loaded_();

    start_worker();
    return true;
}

bool Project::start_worker()
{
    SOM_DEBUG("Project::start_worker()");

    if (thread_ || run_)
    {
        SOM_DEBUG("Project::start_worker:: already running");
        return true;
    }

    thread_ = new std::thread( std::bind(&Project::work_loop_, this));

    return true;
}

void Project::stop_worker()
{
    SOM_DEBUG("Project::stop_worker()");

    if (!thread_)
    {
        SOM_DEBUG("Project::stop_worker:: no thread_");
        return;
    }

    if (run_)
    {
        run_ = false;
        SOM_DEBUG("Project::stop_worker:: joining thread_");
        thread_->join();
    }

    delete thread_;
    thread_ = 0;
    SOM_DEBUG("Project::stop_worker:: thread_ killed");
}


void Project::work_loop_()
{
    SOM_DEBUG("Project::work_loop_()");

    if (!wave_)
    {
        SOM_ERROR("Project::work_loop_:: no wave_");
        return;
    }

    run_ = true;

    // ------- analyze wave ---------

    if (wave_changed_)
    {
        SOM_LOG("analyzing bands"
                << "\nbands      " << wave_->nr_bands
                << "\nmin freq   " << wave_->min_freq
                << "\nmax freq   " << wave_->max_freq
                << "\namp        " << band_amp_ << " ^ " << band_exp_
                << "\ngrains     " << wave_->nr_grains
                << "\ngrain size " << wave_->grain_size
                << "\nwindow     " << wave_->window_width
                );

        // normalize or scale?
        const float amp = band_amp_>0 ? band_amp_ : 1;

        QElapsedTimer timer;
        timer.start();
        float max_value = 0.000001;

        // get each slice's band
        for (size_t x=0; x < num_grains(); ++x)
        {
            if (!run_)
            {
                SOM_DEBUG("break in spectral analyzis");
                return;
            }

            // get band data and max-value
            max_value = std::max(
                    max_value,
                    wave_->get_bands(x, 1, amp)
                );

            // callback
            if (timer.elapsed() > 200)
            {
                if (cb_bands_) cb_bands_();
                timer.start();
            }

        }

        // normalize or re-scale

        if (band_amp_ <= 0)
            wave_->normalize(max_value, band_exp_);
        else
            wave_->shape(1.f, band_exp_);

        // final update
        if (cb_bands_) cb_bands_();

        // wave is prepared
        wave_changed_ = false;
    }

    // ------- calculate som -------

    SOM_DEBUG("Project::work_loop_:: som init");
    som_->create(som_sizex_, som_sizey_, num_bands(), som_seed_);
    som_->init(*wave_);
    // callback
    if (cb_som_ready_) cb_som_ready_();

    SOM_DEBUG("Project::work_loop_:: starting loop");

    QElapsedTimer timer;
    timer.start();
    while (run_)
    {
        // select grain to train :)
        size_t grain_nr = rand() % num_grains();

        // set training parameters
        som_->alpha = som_alpha_;
        som_->radius = std::max(som_->sizex, som_->sizey) * som_radius_;

        // feed to map
        som_->insert(&wave_->band[grain_nr][0]);

        // callback after period
        if (timer.elapsed() > 200)
        {
            if (cb_som_) cb_som_();
            timer.start();
        }

        //usleep(1000*1);
    }

    run_ = false;
}

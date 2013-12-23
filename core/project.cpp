#include "project.h"

#include "log.h"

#include <functional>
#include <future>

#include <QElapsedTimer>


#define SOM_CALLBACK(cb_name__) \
{ \
    SOM_DEBUGN(0, "Project:: callback " #cb_name__ ); \
    if (cb_name__##_) cb_name__##_(); \
}



Project::Project()
    :
      filename_          ("new.rsom"),

      som_alpha_         (0.05),
      som_radius_        (0.07),
      som_search_radius_ (0.07),

      wave_              (new Wave),
      som_               (new Som),
      wave_thread_       (0),
      som_thread_        (0),

      run_wave_          (false),
      run_som_           (false),
      som_ready_         (false),

      cb_wave_loaded_    (0),
      cb_bands_          (0),
      cb_bands_finished_ (0),
      cb_som_ready_      (0),
      cb_som_            (0)
{
    SOM_DEBUG("Project::Project()");

    // important to init with sane settings!
    set(8, 25, 8000, 2048, 2048, 0.1f, 1.f);
    set_som(48, 48, 0);
}


Project::~Project()
{
    SOM_DEBUG("Project::~Project()");

    if (wave_thread_) stopWaveThread();
    if (som_thread_) stopSomThread();

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

    if (wave_thread_) stopWaveThread();
    if (som_thread_) stopSomThread();

    wave_->set(nr_bands, min_freq, max_freq, grain_size, window_width);
    band_amp_ = band_amp;
    band_exp_ = band_exp;

    if (wave_->ok()) startWaveThread();

}

void Project::set_som(size_t sizex, size_t sizey, int rand_seed)
{
    SOM_DEBUG("Project::set_som(" << sizex << ", " << sizey << ", " << rand_seed << ")" );

    if (som_thread_) stopSomThread();

    som_ready_ = false;

    som_sizex_ = sizex;
    som_sizey_ = sizey;
    som_seed_ = rand_seed;

    // ---- initialize som ------

    SOM_DEBUG("Project::work_loop_:: som init");

    som_->create(som_sizex_, som_sizey_, num_bands(), som_seed_);
    som_->insertWave(*wave_);
    som_->initMap();

    som_ready_ = true;

    // callback
    SOM_CALLBACK(cb_som_ready);

/*
    if (wave_ && wave_->ok())
    {
        startSomThread();
    }
*/
}


bool Project::load_wave(const std::string& soundfile_name)
{
    SOM_DEBUG("Project::load_wave(" << soundfile_name << ")");

    // stop wave thread first
    if (wave_thread_) stopWaveThread();
    // also stop som thread, som will probably reallocate
    if (som_thread_) stopSomThread();

    som_ready_ = false;

    if (!wave_->open(soundfile_name))
    {
        SOM_DEBUG("Project::load_wave:: failed");
        wave_->filename = "*error*";
        return false;
    }

    SOM_DEBUG("Project::load_wave::wave_->update()");

    // resize wave band data
    wave_->update();

    // callback
    SOM_CALLBACK(cb_wave_loaded);

    startWaveThread();

    return true;
}



void Project::startWaveThread()
{
    SOM_DEBUG("startWaveThread()");

    // one piece of work
    auto spec_loop = [this](size_t xstart, size_t xend)
    {
        SOM_DEBUG("Project::startWaveThread::spec_loop(" << xstart << ", " << xend << ", " << run_wave_ << ")");

        // timer for update callbacks
        QElapsedTimer timer;
        timer.start();

        // normalize or scale?
        const float amp = band_amp_>0 ? band_amp_ : 1;

        float max_value = 0.000001;

        // get each slice's band
        for (size_t x=xstart; x < xend; ++x)
        {
            if (!run_wave_)
            {
                SOM_DEBUG("break in spectral analysis");
                return 0.f;
            }

            // get band data and max-value
            max_value = std::max(
                    max_value,
                    wave_->get_bands(x, 1, amp)
                );

            // callback (from first thread)
            if (xstart == 0 && timer.elapsed() > 200)
            {
                SOM_CALLBACK(cb_bands);
                timer.start();
            }

        }

        return max_value;
    };


    // main wave/band calc thread

    auto wave_thread = [this, spec_loop]()
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

        run_wave_ = true;

        /// @todo support less then num_cpu grains!!

        // sub-threads
        std::vector<std::future<float>> tasks;

        const size_t num_threads = 8;

        // create num_cpu slices of work
        const size_t xstep = num_grains() / num_threads + 1;
        size_t xstart = 0, xend = xstep;
        for (size_t i = 0; i < num_threads; ++i)
        {
            tasks.push_back( std::async(
                std::launch::async,
                spec_loop, xstart, xend
            ));
            xstart += xstep;
            xend = std::min(num_grains(), xend + xstep);
        }

        SOM_DEBUGN(0, "waiting for wave tasks");

        // wait for results
        float max_value = 0.000001;
        for (size_t i = 0; i<tasks.size(); ++i)
        {
            tasks[i].wait();
            max_value = std::max( max_value, tasks[i].get() );
        }

        // normalize or re-scale

        if (band_amp_ <= 0)
            wave_->normalize(max_value, band_exp_);
        else
            wave_->shape(1.f, band_exp_);

        // final update
        SOM_CALLBACK(cb_bands);
        SOM_CALLBACK(cb_bands_finished);

        // wave and band-data is prepared now
    };

    // ----- EXECUTE ------

    // stop previous thread
    if (wave_thread_) stopWaveThread();

    // run thread
    wave_thread_ = new std::thread( wave_thread );
}

void Project::stopWaveThread()
{
    SOM_DEBUG("Project::stopWaveThread()");

    if (!wave_thread_)
    {
        SOM_DEBUG("Project::stopWaveThread:: no thread");
        return;
    }

    if (run_wave_)
    {
        run_wave_ = false;
        SOM_DEBUG("Project::stopWaveThread:: joining thread");
        wave_thread_->join();
    }

    delete wave_thread_;
    wave_thread_ = 0;

    SOM_DEBUG("Project::stopWaveThread:: thread killed");
}





bool Project::startSomThread()
{
    SOM_DEBUG("Project::startSomThread()");

    if (som_thread_ || run_som_)
    {
        SOM_DEBUG("Project::startSomThread:: already running");
        return true;
    }

    som_thread_ = new std::thread( std::bind(&Project::work_loop_, this));

    return true;
}

void Project::stopSomThread()
{
    SOM_DEBUG("Project::stopSomThread()");

    if (!som_thread_)
    {
        SOM_DEBUG("Project::stopSomThread:: no thread");
        return;
    }

    if (run_som_)
    {
        run_som_ = false;
        SOM_DEBUG("Project::stopSomThread:: joining thread");
        som_thread_->join();
    }

    delete som_thread_;
    som_thread_ = 0;
    SOM_DEBUG("Project::stopSomThread:: thread killed");
}





/** The whole worker thread.
 *  Only one, currently.
 *
 *      If the SOM init parameters are changed through set_som(),
 *  restart_som_ will be TRUE and the worker thread jumps back to the som init code.
*/
void Project::work_loop_()
{
    SOM_DEBUG("Project::work_loop_()");

    if (!wave_)
    {
        SOM_ERROR("Project::work_loop_:: no wave_");
        return;
    }

    run_som_ = true;

    // ------- calculate som -------

    SOM_DEBUG("Project::work_loop_:: starting training loop");

    QElapsedTimer timer;
    timer.start();
    while (run_som_)
    {
        // set training parameters
        som_->alpha = som_alpha_;
        som_->radius = std::max(som_->sizex, som_->sizey) * som_radius_;
        som_->local_search_radius = std::max(som_->sizex, som_->sizey) * som_search_radius_;

        // feed to map
        som_->insert();

        // callback after period
        if (timer.elapsed() > 200)
        {
            SOM_CALLBACK(cb_som);
            timer.start();
        }

        //usleep(1000*1);
    }

}

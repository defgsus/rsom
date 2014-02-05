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
#include "project.h"

#include "time.h"
#include "data.h"
#include "log.h"

#include <functional>
#include <future>
#include <sstream>



#define SOM_CALLBACK(cb_name__) \
{ \
    SOM_DEBUGN(0, "Project:: callback " #cb_name__ ); \
    if (cb_name__##_) cb_name__##_(); \
}


namespace RSOM
{

Project::Project()
    :
      name_          ("new.rsom"),

      som_alpha_         (0.05),
      som_radius_        (0.2),
      som_search_radius_ (2.0),

      data_              (new Data),
      som_               (new Som(Som::CUDA)),
      som_thread_        (0),

      run_som_           (false),
      som_ready_         (false),

      need_map_          (false),
      need_imap_         (false),
      need_umap_         (false),

      cb_som_ready_      (0),
      cb_som_            (0)
{
    SOM_DEBUG("Project::Project()");

    // only to init Project's som parameters
    set_som(48, 48, 0);
}


Project::~Project()
{
    SOM_DEBUG("Project::~Project()");

    if (som_thread_) stopSomThread();

    if (som_) delete som_;
    if (data_) delete data_;
}

const std::string Project::info_str() const
{
    std::stringstream s;
    s << som_->info_str()
      << "\nsamples per second " << inserts_per_second_;
    return s.str();
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


void Project::set_som(size_t sizex, size_t sizey, int rand_seed)
{
    SOM_DEBUG("Project::set_som(" << sizex << ", " << sizey << ", " << rand_seed << ")" );

    if (som_thread_) stopSomThread();

    som_ready_ = false;

    som_sizex_ = sizex;
    som_sizey_ = sizey;
    som_seed_ = rand_seed;

    // avoid unnescessary som preparation
    if (!data_->numObjects()) return;

    // ---- initialize som ------

    SOM_DEBUG("Project::set_som:: som init");

    som_->create(som_sizex_, som_sizey_, data_->numDataPoints(), som_seed_);

    som_->setData(data_);
    som_->initMap();

    som_ready_ = true;

    // callback
    SOM_CALLBACK(cb_som_ready);

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
 *  run_som_ will be FALSE and the worker thread exits.
*/
void Project::work_loop_()
{
    SOM_DEBUG("Project::work_loop_()");

    if (!data_)
    {
        SOM_ERROR("Project::work_loop_:: no data_");
        return;
    }

    run_som_ = true;

    // ------- calculate som -------

    SOM_DEBUG("Project::work_loop_:: starting training loop");

    last_generation_ = som_->generation();

    Messure timer;
    timer.start();
    while (run_som_)
    {
        // set training parameters
        som_->alpha( som_alpha_ );
        som_->radius( std::max(som_->sizex(), som_->sizey()) * som_radius_ );
        som_->local_search_radius( std::max(som_->sizex(), som_->sizey()) * som_search_radius_ );

        // feed to map
        som_->insert();

        // callback after period
        if (timer.elapsed() > 0.25)
        {
            timer.start();

            // messure speed
            inserts_per_second_ =
                    (som_->generation() - last_generation_) / 0.2;
            last_generation_ = som_->generation();

            // update maps
            if (need_map_) som_->updateMap();
            if (need_imap_) som_->updateIMap();
            if (need_umap_) som_->updateUMap(true);

            SOM_CALLBACK(cb_som);
        }

        //usleep(1000*1);
    }

    SOM_DEBUG("Project::work_loop_:: exit");
}


} // namespace RSOM

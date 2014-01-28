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
/** @file
    @brief debug and logging output

    @version 2013/12/18 init

    copyright 2013 stefan.berke @ modular-audio-graphics.com
*/
#ifndef LOG_H
#define LOG_H

#include <iostream>
#include <sstream>
#include <thread>
#include <functional>

/** this class contains static functors which can be
    used to retrieve the log/error messages. */
class SomLog
{
public:

    static std::function<void(const std::string&)> error_func;
    static std::function<void(const std::string&)> log_func;
};

/** macro for printing a std::ostream compatible stream to console */
#define SOM_PRINT(stream__, stream_arg__) \
{ stream__ << std::hex << std::this_thread::get_id() << std::dec << ": " \
           << stream_arg__ << "\n"; }

#ifndef NDEBUG
#   define SOM_DEBUG(stream_arg__) \
        SOM_PRINT(std::cerr, stream_arg__)
#else
#   define SOM_DEBUG(unused__) { }
#endif


/** SOM_DEBUGN(level, arg) - noisy debug with level */
#if (0)
#   define SOM_DEBUGN(level__, stream_arg__) \
        if (level__<=SOM_DEBUGN_LEVEL) SOM_DEBUG(stream_arg__)
#else
#   define SOM_DEBUGN(level__, stream_arg__) { }
#endif

#define SOM_DEBUGN_LEVEL 2


#define SOM_ERROR(stream_arg__) \
{ \
    std::stringstream stream__; \
    stream__ << stream_arg__; \
    SOM_PRINT(std::cerr, "*error* " << stream__.str()); \
    if (SomLog::error_func) SomLog::error_func(stream__.str()); \
}

#define SOM_LOG(stream_arg__) \
{ \
    std::stringstream stream__; \
    stream__ << stream_arg__; \
    SOM_PRINT(std::cerr, "log: " << stream__.str()); \
    if (SomLog::log_func) SomLog::log_func(stream__.str()); \
}




/*
#define SOM_LOG(stream_arg__) \
    Logger::singleInstance() << stream_arg__ << "\n";

class Logger
{
    public:

    Logger() : instance_(0) { }

    static Logger & singleInstance();

    template <class T>
    friend Logger& operator << (Logger& l, const T& arg)
    {
        l.stream_ << arg;
        return l;
    }

private:
    Logger * instance_;

    std::streambuf stream_;
};



Logger & Logger::singleInstance()
{
    if (!instance_) instance_ = new Logger();
    return *instance_;
}
*/


#endif // LOG_H

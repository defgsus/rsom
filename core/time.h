/***************************************************************************

Copyright (C) 2014  stefan.berke @ modular-audio-graphics.com

This source is free software; you can redistribute it and/or
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

****************************************************************************/
/** @file
    @brief function for messuring time

    @version 2014/02/03 started

    copyright 2014 stefan.berke @ modular-audio-graphics.com
*/
#ifndef TIME_H
#define TIME_H

/** Gets system time in seconds */
double sysTime();

/** Class to messure passed time */
class Messure
{
    public:

    Messure() { start(); }

    /** restart counter */
    void start() { last_time_ = sysTime(); }

    /** return elapsed time */
    double elapsed() { return sysTime() - last_time_; }

    private:

    double last_time_;
};

#endif // TIME_H

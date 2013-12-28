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
    @brief Collection of Property classes

    @version 2013/12/23 init

    copyright 2013 stefan.berke @ modular-audio-graphics.com
*/
#ifndef PROPERTIES_H
#define PROPERTIES_H

#include "property.h"

#include <vector>
#include <memory>

class Properties
{
public:
    Properties();

    void add(Property * p);

    std::vector<Property*> property;

private:
    std::vector<std::shared_ptr<Property>> auto_remove_;
};

#endif // PROPERTIES_H

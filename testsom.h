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

#ifndef TESTSOM_H
#define TESTSOM_H

#include "core/som.h"
#include "core/data.h"

int testSom()
{
    using namespace RSOM;

    const Index dim = 64;

    Data data;
    data.createRandomData(1000, dim);

    Som som(Som::CUDA);
    som.create(32, 32, dim, 1);
    som.setData(&data);
    som.initMap();

    std::cout << som.info_str() << "\n";

    for (int i=0; i<10000; ++i)
        som.insert();

    som.printMap(som.getMap(), som.sizex(), som.sizey(), som.dim(), 0.5, 32,32);

    std::cout << som.info_str() << "\n";

    return 0;
}

#endif // TESTSOM_H

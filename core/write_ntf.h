/** @file
    @brief Native Table Format writer for use with reaktor_som

    @author def.gsus- (berke@cymatrix.org)
    @version 2012/07/11

    copyright 2012 Stefan Berke,
    this program is coverd by the GNU General Public License
*/
#ifndef WRITE_NTF_H_INCLUDED
#define WRITE_NTF_H_INCLUDED

#include "som.h"

// store the som data as 'native table format'
// actually, only the 'umap' data is saved which at this point
// should contain the grain-positions for each som-node
bool save_ntf(const std::string& filename, const Som& som);


#endif // WRITE_NTF_H_INCLUDED

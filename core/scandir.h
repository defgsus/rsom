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

/**	@file

    @brief directory wildcard file scanner

    @author def.gsus-
    @version 2013/03/11 grabed from Merkel
*/

#ifndef SCANDIR_H
#define SCANDIR_H

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <string>
#include <string.h>
#include <algorithm>
#include <vector>
#include "log.h"

/** usage:

    @code
    ScanDir sd("/path/", ".zip");
    // or..
    ScanDir sd;
    if (sd.scandir("/path/", ".zip")) exit(EXIT_FAILURE);

    cout << sd.files[0]; // "/path/1st-found-file.zip"
    @endcode
*/
class ScanDir
{
    public:

    std::vector<std::string> files;

    ScanDir() { }
    ScanDir(const std::string& dir, const std::string& ext="", bool recursive = false)
        { scandir(dir,ext,recursive); }

    /** input: <br>
        'dir' : a string in the form of "/home/mine/" <br>
        'ext' : can be "" to gather all files or anything like ".zip" or "zip" <br>
        'recursive' : on TRUE all subdirectories will be checked too. <br>
        returns: <br>
        errorcode of first opendir() statement or 0 if ok <br>
        fills ScanDir::files vector with filenames in the form of "/home/mine/filename.zip"
        */
    int scandir (const std::string& dir, const std::string& ext="", bool recursive = false)
    {
        DIR *dp;
        struct dirent *dirp;
        if((dp = opendir(dir.c_str())) == NULL)
        {
            SOM_ERROR( "error #" << errno << " on reading directory " << dir );
            return errno;
        }

        struct stat statbuf;
        bool checkExt = (ext!="");
        std::string fn;

        while ((dirp = readdir(dp)) != NULL)
        {
            fn = dir + dirp->d_name;

            // check for directory
            stat(fn.c_str(), &statbuf);
            if (S_ISDIR( statbuf.st_mode ))
            {
                if (!strcmp(dirp->d_name,".") ||
                    !strcmp(dirp->d_name,"..") ||
                        !recursive) continue;
                // recurse in other directories
                scandir(fn + "/", ext, true );
                continue;
            }

            // check for correct extension
            if (checkExt && !isExtension(dirp->d_name, ext)) continue;

            files.push_back( fn );
        }
        closedir(dp);
        return 0;
    }

    /** sort entries alphabetically */
    void sort()
    {
        std::sort(files.begin(), files.end());
    }

    static bool isExtension(char *str, const std::string& ext)
    {
        size_t len = strlen(str);
        // need at least ext+ 1 char for match
        if ( !len || len <= ext.size()) return false;

        char *s = str + len - 1;
        len = ext.size() - 1;
        while (len)
        {
            if (*s != ext[len]) return false;
            len--; s--;
        }
        return true;
    }

    /** remove anything up to the last '/' */
    static std::string stripPath(const std::string& filename)
    {
        std::string::size_type i = filename.find_last_of('/', filename.npos);
        if (i == filename.npos) return filename;

        return filename.substr(i+1, filename.npos);
    }

};


#endif // SCANDIR_H

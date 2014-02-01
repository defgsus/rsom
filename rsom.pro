#-------------------------------------------------
#
# Project created by QtCreator 2013-12-18T17:19:05
#
#-------------------------------------------------

# -- qt stuff --

TARGET    = rsom

QT       += core gui widgets

CONFIG   += console

TEMPLATE  = app

# -- flags --

QMAKE_CXXFLAGS += -std=c++0x
QMAKE_CXXFLAGS_RELEASE += -O2 -DNDEBUG

# -- libs --

windows: LIBS += -lsndfile-1
else:    LIBS += -lsndfile -lcudart


# -- files --

SOURCES += main.cpp \
    core/write_ntf.cpp \
    core/project.cpp \
    core/wavefile.cpp \
    core/som.cpp \
    core/log.cpp \
    core/data.cpp \
    mainwindow.cpp \
    projectview.cpp \
#    waveview.cpp \
    somview.cpp \
    property.cpp \
    colorscale.cpp \
    helpwindow.cpp \
    properties.cpp \
    dataview.cpp \
    core/cudabackend.cpp

HEADERS += \
    mainwindow.h \
    projectview.h \
#    waveview.h \
    somview.h \
    colorscale.h \
    property.h \
    helpwindow.h \
    properties.h \
    core/som.h \
    core/wavefile.h \
    core/write_ntf.h \
    core/project.h \
    core/log.h \
    core/data.h \
    core/scandir.h \
    dataview.h \
    core/som_types.h \
    core/cudabackend.h \
    testcuda.h \
    core/cuda_util.h

RESOURCES += \
    resources.qrc

OTHER_FILES += \
    help.html \
    README

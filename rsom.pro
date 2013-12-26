#-------------------------------------------------
#
# Project created by QtCreator 2013-12-18T17:19:05
#
#-------------------------------------------------

TARGET = rsom

# -- flags --

QMAKE_CXXFLAGS += -std=c++0x
CONFIG(Release) { QMAKE_CXXFLAGS += -DNDEBUG }

# -- libs --

LIBS += -lsndfile

# -- qt stuff --

QT       += core gui widgets

CONFIG   += console

TEMPLATE = app

# -- files --

SOURCES += main.cpp \
    mainwindow.cpp \
    core/write_ntf.cpp \
    core/project.cpp \
    core/wavefile.cpp \
    projectview.cpp \
    waveview.cpp \
    core/som.cpp \
    somview.cpp \
    property.cpp \
    colorscale.cpp \
    helpwindow.cpp \
    properties.cpp

HEADERS += \
    mainwindow.h \
    core/som.h \
    core/wavefile.h \
    core/write_ntf.h \
    core/project.h \
    projectview.h \
    waveview.h \
    core/log.h \
    somview.h \
    colorscale.h \
    property.h \
    helpwindow.h \
    properties.h

RESOURCES += \
    resources.qrc

OTHER_FILES += \
    help.html \
    README

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
else:    LIBS += -lsndfile

CUDA_LIBS = -lcuda -lcudart -lcublas

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
    core/cudabackend.cpp \
    core/backend.cpp \
    core/cpubackend.cpp

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
    core/cuda_util.h \
    core/backend.h \
    core/cpubackend.h \
    testsom.h

CUDA_SOURCES = \
    core/cudasom.cu

RESOURCES += \
    resources.qrc

OTHER_FILES += \
    help.html \
    README

OTHER_FILES += $$CUDA_SOURCES




####### cuda setup ########

# http://stackoverflow.com/questions/16053038/cuda-with-qt-in-qt-creator-on-ubuntu-12-04

NVCC_OPTIONS = --use_fast_math


CUDA_SDK = "/usr/lib/nvidia-cuda-toolkit/"   # Path to cuda SDK install
CUDA_DIR = "/usr/lib/nvidia-cuda-toolkit/"   # Path to cuda toolkit install
#CUDA_SDK = "/usr/local/cuda/"   # Path to cuda SDK install
#CUDA_DIR = "/usr/local/cuda/"   # Path to cuda toolkit install

SYSTEM_NAME = unix          # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_20           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'

# add cuda include paths
INCLUDEPATH += $$CUDA_DIR/include

# add cuda library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib/

CUDA_OBJECTS_DIR = ./

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so')
LIBS += $$CUDA_LIBS

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc -DNDEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}

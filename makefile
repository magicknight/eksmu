# Makefile for EKS program
# Last change: Dec 21, 2011 by Zhihua Liang
#----------------------------------------------------------------------
.SUFFIXES:  .o .c .cxx .f .l  .s  .sh  .h  .a
#======================================================================
#Final executable

TARGET=eksmu
all: $(TARGET)

#  Root 
RPATH = $(ROOTSYS)
RLIB =  -L$(RPATH)/lib -lCore -lCint -lHist -lGraf -lGraf3d -lGpad -lTree -lRint \
	-lPostscript -lMatrix -lPhysics -lm -ldl -lpthread -rdynamic
#####  GNU c++ compiler
# Fortran and C compilers
FC = gfortran 
CC = g++
NC = nvcc

# Set DEBUG, LHAPDF, or CERNROOT to 'yes' to compile a debugging version,
#   The directory where the include files are.
#   By default it was installed into your EKSMU home directory.
vpath %.cc src
vpath %.cu src
vpath %.h  include
vpath %.f  fortran

# link to the LHAPDF library or CERN ROOT libraries
DEBUG=yes
LHAPDF=yes
CERNROOT=yes

# End of user parameters ===============


# -ansi disables language extensions that conflict with the ISO C++ standard
# -W enables extra warning messages for code that may be in error
# -Wall enables warnings about questionable programming practices
ifeq ($(DEBUG), yes)
  CCFLAGS = -g -ansi -W -Wall
#  CUFLAGS = -g -G -ccbin ~/work/gcc-4.4/
  CUFLAGS = -g -G
  FFLAGS = -g -fno-automatic	
else
  CCFLAGS = -O3 -ansi -W -Wall
  FFLAGS = -O3 -fno-automatic
endif

HEADER_PATH = -I./include

ifeq ($(CERNROOT), yes)
   ROOTDIR = $(shell root-config --prefix)
   ROOTLIBS := $(shell root-config --prefix=$(ROOTDIR)  --libs)
   ROOTINCLUDE := -I $(shell root-config --incdir)
   ROOTLIB2 =	-L $(shell root-config --libdir) -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread
endif 

# jetsub5b.o is the multi-thread version of eks
ifeq ($(LHAPDF), yes)
  OBJS = $(TARGET).o controlbox.o card.o recorder.o \
	engine.o eks.o container.o jetsub5b.o pdf.o
  LHALIB = -L$(shell lhapdf-config --libdir) -lLHAPDF
else
# else obj = 
endif

#--------------------cuba--------------------
libs = -L /users/zliang/lib/ -lcuba
headers = -I /users/zliang/include/
#libs = -L /users -lcuba
#headers = -I /usr/local/include/
# Compile all c++ files
#======================================================================
#
$(TARGET): $(OBJS) 
	$(NC) -o $(TARGET) $(CUFLAGS) $(HEADER_PATH) $(OBJS) $(libs) $(ROOTLIB2)  -lm -ldl $(LHALIB) 

#----------------------------------------------------------------------
line_count:
	(cat src/*.cc include/*.h | wc -l )
#==============================================================================

# rule that says how to make a .o object file from a .cc source file
# $< and $@ are macros defined by make
#     $< refers to the file being processed (i.e., compiled or linked)
#     $@ refers to the generated file
%.o: %.f
	$(FC) -c $(FFLAGS) $<

%.o: %.cc
	$(CC) $(CCFLAGS) $(HEADER_PATH)  $(headers) $(ROOTINCLUDE) -c $< -o $@

%.o: %.cu
	$(NC) $(CUFLAGS) $(HEADER_PATH)  $(headers) $(ROOTINCLUDE) -c $< -o $@
clean: 
	rm -f $(TARGET) *.o *.a *~ *.exe lmap *.out *.dvi *.aux *.log
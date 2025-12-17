# Makefile site specific declarations.
SITE := frontier

ROOTDIR := $(PWD)
BUILD_DIR := $(ROOTDIR)/build
OBJECT_DIR := $(ROOTDIR)/object
SOURCE_DIR := $(ROOTDIR)/src

# Directories _must_ be specified before this inclusion.
include $(BUILD_DIR)/site/Makefile.$(SITE)

no_target: 
	@echo "No target given!"

libhyburn_base: builddir
	$(CC) -c $(SOURCE_DIR)/core/init.c -o $(OBJECT_DIR)/init.o $(CFLAGS) -fPIC
	ar rvs $(OBJECT_DIR)/libhyburn.a $(OBJECT_DIR)/init.o
	$(CC) -c $(SOURCE_DIR)/core/store.c -o $(OBJECT_DIR)/store.o $(CFLAGS) -fPIC
	ar rvs $(OBJECT_DIR)/libhyburn.a $(OBJECT_DIR)/store.o
	$(CC) -c $(SOURCE_DIR)/core/read_values.c -o $(OBJECT_DIR)/read_values.o \
		$(CFLAGS) -fPIC
	ar rvs $(OBJECT_DIR)/libhyburn.a $(OBJECT_DIR)/read_values.o
	$(CC) -c $(SOURCE_DIR)/parse-data/parser.c -o $(OBJECT_DIR)/parser.o \
		$(CFLAGS) -fPIC
	ar rvs $(OBJECT_DIR)/libhyburn.a $(OBJECT_DIR)/parser.o
	$(CC) -c $(SOURCE_DIR)/core/kill.c -o $(OBJECT_DIR)/kill.o $(CFLAGS) -fPIC
	ar rvs $(OBJECT_DIR)/libhyburn.a $(OBJECT_DIR)/kill.o

libhyburn_serial: libhyburn_base
	$(CC) -c $(SOURCE_DIR)/core/bn_burner.c -o $(OBJECT_DIR)/bn_burner.o \
		$(CFLAGS) -fPIC
	ar rvs $(OBJECT_DIR)/libhyburn.a $(OBJECT_DIR)/bn_burner.o
	
libhyburn_simd: libhyburn_base
	$(CC) -c $(SOURCE_DIR)/core/bn_burner_SIMD.c -o $(OBJECT_DIR)/bn_burner.o $(CFLAGS)
	ar rvs $(OBJECT_DIR)/libhyburn.a $(OBJECT_DIR)/bn_burner.o
	
libhyburn_pec: libhyburn_base
	$(CC) -c $(SOURCE_DIR)/core/bn_burner_pec.c -o $(OBJECT_DIR)/bn_burner.o $(CFLAGS_SIMD)
	ar rvs $(OBJECT_DIR)/libhyburn.a $(OBJECT_DIR)/bn_burner.o
	
libhyburn_printinfo: clean builddir
	$(CC) -c $(SOURCE_DIR)/core/bn_burner_printinfo.c -o \
		$(OBJECT_DIR)/bn_burner.o $(CFLAGS)
	ar rvs $(OBJECT_DIR)/libhyburn.a $(OBJECT_DIR)/bn_burner.o

clean: 
	rm -rf $(OBJECT_DIR)/*

builddir: 
	mkdir -p $(OBJECT_DIR)

include $(ROOTDIR)/hypy/Makefile
include $(ROOTDIR)/tests/Makefile

include $(BUILD_DIR)/devel/Makefile.test
include $(BUILD_DIR)/devel/Makefile.gpu

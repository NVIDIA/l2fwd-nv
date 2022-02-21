# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(l2fwdnv LANGUAGES CXX C CUDA)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
ENABLE_TESTING()

# FIXME: enable build for a generic target
ADD_COMPILE_DEFINITIONS("ALLOW_EXPERIMENTAL_API")

########################################### OPTIONS ##################################################

OPTION(BUILD_DOCS "Generate Doxygen documentation" OFF)
OPTION(GTEST_POPULATE_CMAKE_TESTS "Populate CMake tests with GTest" ON)
OPTION(CUDA_ENABLED "Whether to enable CUDA for L2FWDNV" ON) #FIXME We probably always need CUDA
OPTION(IPV4_UDP_MODE "Encapsulate eCPRI packets in IPv4/UDP" OFF)
OPTION(PROFILE_LOGS "Periodically dump profiling and monitoring statistics" OFF)
option(ENABLE_NVTX  "Enable use of NVTX" OFF)

############################################ CUDA ###################################################

FIND_PACKAGE(CUDA 11 REQUIRED)
FIND_LIBRARY(CUDA_DRIVER_LIBRARY
             NAMES cuda_driver cuda
             HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                   ENV CUDA_PATH
             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
IF (NOT CUDA_DRIVER_LIBRARY)
    FIND_LIBRARY(CUDA_DRIVER_LIBRARY
                 NAMES cuda_driver cuda
                 HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                       ENV CUDA_PATH
                 PATH_SUFFIXES lib64/stubs lib/x64/stubs lib/stubs stubs compat)
ENDIF ()
MARK_AS_ADVANCED(CUDA_DRIVER_LIBRARY)

############################################ DPDK ###################################################

SET(DPDK_TARGET dpdk_target)
SET(DPDK_TARGET_ARCH x86_64-native-linuxapp-gcc)
# Build on ARM
# SET(DPDK_TARGET_ARCH aarch64-build-gcc)
SET(DPDK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/dpdk)
message( ${DPDK_PATH})
SET(DPDK_TARGET_PATH ${DPDK_PATH}/${DPDK_TARGET_ARCH})
message( ${DPDK_TARGET_PATH} )
SET(DPDK_TARGET_INSTALL_PATH ${DPDK_TARGET_PATH}/install)
STRING(CONCAT DPDK_DISABLED_DRIVERS
        "baseband/*,"
        "bus/ifpga/*,"
        "common/cpt,"
        "common/dpaax,"
        "common/iavf,"
        "common/octeontx,"
        "common/octeontx2,"
        "crypto/nitrox,"
        "net/ark,"
        "net/atlantic,"
        "net/avp,"
        "net/axgbe,"
        "net/bnx2x,"
        "net/bnxt,"
        "net/cxgbe,"
        "net/e1000,"
        "net/ena,"
        "net/enic,"
        "net/fm10k,"
        "net/hinic,"
        "net/hns3,"
        "net/i40e,"
        "net/ixgbe,"
        "vdpa/ifc,"
        "net/igc,"
        "net/liquidio,"
        "net/netvsc,"
        "net/nfp,"
        "net/qede,"
        "net/sfc,"
        "net/thunderx,"
        "net/vdev_netvsc,"
        "net/vmxnet3,"
        "regex/octeontx2,"
)

SET(ENV{CFLAGS} "${CFLAGS} -I${CUDA_INCLUDE_DIRS} -I${CMAKE_CURRENT_SOURCE_DIR}/external/gdrcopy/include")
EXECUTE_PROCESS(WORKING_DIRECTORY ${DPDK_PATH}
        COMMAND meson ${DPDK_TARGET_ARCH}
                --prefix=${DPDK_TARGET_INSTALL_PATH}
                -Dtests=false
                -Ddisable_drivers=${DPDK_DISABLED_DRIVERS}
                # -Dc_args="-I${CUDA_INCLUDE_DIRS} -I${CMAKE_CURRENT_SOURCE_DIR}/external/gdrcopy/include"
                # --cross-file config/arm/arm64_armv8_linux_gcc
                # -Dbuildtype=debug
        )

# Set RTE_PKTMBUF_HEADROOM to 0 for GPU memory best performance
EXECUTE_PROCESS(WORKING_DIRECTORY ${DPDK_PATH}
        COMMAND sed -ri "s,(RTE_PKTMBUF_HEADROOM ).*,\\10," config/rte_config.h
)

ADD_CUSTOM_TARGET(${DPDK_TARGET} ALL
        WORKING_DIRECTORY ${DPDK_TARGET_PATH}
        COMMENT Building DPDK
        COMMAND ninja install
        VERBATIM)

SET(ENV{PKG_CONFIG_PATH} "${DPDK_TARGET_PATH}/meson-private")

EXECUTE_PROCESS(OUTPUT_VARIABLE DPDK_BUILD_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${DPDK_TARGET_PATH}/meson-private
        COMMAND pkg-config --cflags libdpdk.pc)

EXECUTE_PROCESS(OUTPUT_VARIABLE DPDK_LINK_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${DPDK_TARGET_PATH}/meson-private
        COMMAND pkg-config --static --libs libdpdk.pc)

SET(DPDK_LIBS "-Wl,--no-whole-archive -lmlx5 -libverbs -pthread -lnuma -ldl ${DPDK_LINK_FLAGS}")

############################################ GDRCopy ###################################################

SET(GDRC_TARGET gdrcopy_lib)
add_custom_target(
   ${GDRC_TARGET}
   COMMAND make
   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/gdrcopy
)
# gdrcopy makefile doesn't support make with multiple jobs
# dorce cmake to build DPDK first
ADD_DEPENDENCIES(${GDRC_TARGET} ${DPDK_TARGET})

######################################### L2FWDNV ################################################

SET(TARGET l2fwdnv)

SET(L2FWDNV_SRCS
        src/kernel.cu
        src/main.cpp
        src/utils.cpp
        src/pipeline.cpp
)

ADD_EXECUTABLE(${TARGET} ${L2FWDNV_SRCS})
# Force cmake to build libgdrapi.so
ADD_DEPENDENCIES(${TARGET} ${GDRC_TARGET})

# Link with nvToolsExt
if (ENABLE_NVTX)
    set(NVTX_LIB nvToolsExt)
    add_definitions(-DUSE_NVTX=1)
endif (ENABLE_NVTX)

TARGET_INCLUDE_DIRECTORIES(${TARGET} PUBLIC include ${CUDA_INCLUDE_DIRS})

TARGET_LINK_LIBRARIES(${TARGET} ${DPDK_LIBS} ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY} ${NVTX_LIB})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE -O3 -std=c++11 ${CMAKE_COMMON_FLAGS} -g3 ${DPDK_BUILD_FLAGS}")
SET(CMAKE_C_FLAGS "${CMAKE_COMMON_FLAGS} ${DPDK_BUILD_FLAGS}")

# Enable to debug CUDA kernels
string(STRIP ${DPDK_BUILD_FLAGS} DPDK_STRIPPED_FLAGS)
string(REGEX
        REPLACE
        "\-std\="
        "\-Xcompiler \-std\="
        DPDK_STRIPPED_FLAGS
        ${DPDK_STRIPPED_FLAGS}
)

string(REGEX
        REPLACE
        "\-march\="
        "\-Xcompiler \-march\="
        DPDK_STRIPPED_FLAGS
        ${DPDK_STRIPPED_FLAGS}
)

string(REGEX
        REPLACE
        "\-moutline\-atomics"
        "\-Xcompiler \-moutline\-atomics"
        DPDK_STRIPPED_FLAGS
        ${DPDK_STRIPPED_FLAGS}
)


SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_COMMON_FLAGS} ${DPDK_STRIPPED_FLAGS}") # -G -lineinfo")

message(STATUS "CMAKE_COMMON_FLAGS ${CMAKE_COMMON_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}")
message(STATUS "DPDK_BUILD_FLAGS ${DPDK_BUILD_FLAGS}")

ADD_DEPENDENCIES(${TARGET} ${DPDK_TARGET})
INSTALL(TARGETS ${TARGET} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include/l2fwdnv)


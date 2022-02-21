/* 
 * Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved. 
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUDA_RELATED_H
#define CUDA_RELATED_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#ifdef USE_NVTX
  #include <nvToolsExt.h>

  #define COMM_COL 1
  #define SM_COL   2
  #define SML_COL  3
  #define OP_COL   4
  #define COMP_COL 5
  #define SOLVE_COL 6
  #define WARMUP_COL 7
  #define EXEC_COL 8

  #define SEND_COL 9
  #define WAIT_COL 10
  #define KERNEL_COL 11

  #define PUSH_RANGE(name,cid)                                            \
        do {                                                                  \
          const uint32_t colors[] = {                                         \
                  0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff, 0xff000000, 0xff0000ff, 0x55ff3300, 0xff660000, 0x66330000  \
          };                                                                  \
          const int num_colors = sizeof(colors)/sizeof(colors[0]);            \
          int color_id = cid%num_colors;                                  \
          nvtxEventAttributes_t eventAttrib = {0};                        \
          eventAttrib.version = NVTX_VERSION;                             \
          eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;               \
          eventAttrib.colorType = NVTX_COLOR_ARGB;                        \
          eventAttrib.color = colors[color_id];                           \
          eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;              \
          eventAttrib.message.ascii = name;                               \
          nvtxRangePushEx(&eventAttrib);                                  \
        } while(0)

  #define PUSH_RANGE_STR(cid, FMT, ARGS...)       \
        do {                                          \
          char str[128];                              \
          snprintf(str, sizeof(str), FMT, ## ARGS);   \
          PUSH_RANGE(str, cid);                       \
        } while(0)

  #define POP_RANGE do { nvtxRangePop(); } while(0)
#else
  #define PUSH_RANGE(name,cid)
  #define POP_RANGE
#endif

#define CUDA_CHECK(stmt)                                            \
    do {                                                            \
        cudaError_t result = (stmt);                                \
        if(cudaSuccess != result)                                   \
        {                                                           \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n",       \
                   __FILE__, __LINE__, cudaGetErrorString(result)); \
        }                                                           \
        assert(cudaSuccess == result);                              \
    } while(0)

#define CU_CHECK(stmt)                                      \
    do {                                                    \
        CUresult result = (stmt);                           \
        if(CUDA_SUCCESS != result)                          \
        {                                                   \
            fprintf(stderr, "[%s:%d] cu failed with %d \n", \
                   __FILE__, __LINE__, result);             \
        }                                                   \
        assert(CUDA_SUCCESS == result);                     \
    } while(0)

#endif

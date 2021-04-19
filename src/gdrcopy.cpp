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


#include "headers.h"

int gdrcopy_alloc_pin(gdr_t*     pgdr,
                gdr_mh_t*       pgdr_handle,
                uintptr_t*      pdev_addr,
                uintptr_t*      phost_ptr,
                uintptr_t*      free_address,
                size_t*         palloc_size,
                size_t          input_size)
{
    gdr_t g = *pgdr;
    gdr_mh_t mh;
    gdr_info_t info;
    CUdeviceptr  dev_addr = 0;
    void *host_ptr  = NULL;
    const unsigned int FLAG = 1;
    size_t pin_size, alloc_size, rounded_size;
    
    if((NULL == g) || (NULL == pdev_addr) || (NULL == phost_ptr) || (NULL == palloc_size) || (0 == input_size))
    {
        fprintf(stderr, "alloc_pin_gdrcopy: erroneous input parameters, g=%p, pdev_addr=%p, phost_ptr=%p, palloc_size=%p, input_size=%zd\n",
                g, pdev_addr, phost_ptr, palloc_size, input_size);
        return 1;
    }

    /*----------------------------------------------------------------*
    * Setting sizes                                                   */
    if(input_size < NV_MIN_PIN_SIZE)
        input_size = NV_MIN_PIN_SIZE;

    rounded_size = (input_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    pin_size = rounded_size;
    alloc_size = rounded_size + input_size;

    /*----------------------------------------------------------------*
     * Allocate device memory.                                        */
    CUresult e = cuMemAlloc(&dev_addr, alloc_size);
    if(CUDA_SUCCESS != e)
    {
        fprintf(stderr, "cuMemAlloc\n");
        return 1;
    }

    *free_address = (uintptr_t)dev_addr;
    //GDRDRV needs a 64kB aligned address. No more guaranteed with recent cuMemAlloc/cudaMalloc
    if(dev_addr % GPU_PAGE_SIZE)
    {
        pin_size = input_size;
        dev_addr += (GPU_PAGE_SIZE - (dev_addr % GPU_PAGE_SIZE));
    }

    /*----------------------------------------------------------------*
     * Set attributes for the allocated device memory.                */
    if(CUDA_SUCCESS != cuPointerSetAttribute(&FLAG, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dev_addr))
    {
        fprintf(stderr, "cuPointerSetAttribute\n");
        cuMemFree(dev_addr);
        gdr_close(g);
        return 1;
    }
    /*----------------------------------------------------------------*
     * Pin the device buffer                                          */
    if(0 != gdr_pin_buffer(g, dev_addr, pin_size, 0, 0, &mh))
    {
        fprintf(stderr, "gdr_pin_buffer\n");
        cuMemFree(dev_addr);
        gdr_close(g);
        return 1;
    }
    /*----------------------------------------------------------------*
     * Map the buffer to user space                                   */
    if(0!= gdr_map(g, mh, &host_ptr, pin_size))
    {
        fprintf(stderr, "gdr_map\n");
        gdr_unpin_buffer(g, mh);
        cuMemFree(dev_addr);
        gdr_close(g);
        return 1;
    }
    /*----------------------------------------------------------------*
     * Retrieve info about the mapping                                */
    if(0 != gdr_get_info(g, mh, &info))
    {
        fprintf(stderr, "gdr_get_info\n");
        gdr_unmap(g, mh, host_ptr, pin_size);
        gdr_unpin_buffer(g, mh);
        cuMemFree(dev_addr);
        gdr_close(g);
        return 1;        
    }
    /*----------------------------------------------------------------*
     * Success - set up return values                                 */

    pgdr[0]           = g;
    pdev_addr[0]      = (uintptr_t)dev_addr;
    pgdr_handle[0]    = mh;
    phost_ptr[0]      = (uintptr_t)host_ptr;
    //*pmmap_offset = dev_addr - info.va;
    palloc_size[0]    = pin_size;

    return 0;
}

void gdrcopy_cleanup(gdr_t g, CUdeviceptr free_dev_addr, gdr_mh_t gdr_handle, void* host_ptr, size_t alloc_size)
{
    if(NULL != host_ptr)
    {
        gdr_unmap(g, gdr_handle, host_ptr, alloc_size);
    }

	gdr_unpin_buffer(g, gdr_handle);

    if(free_dev_addr)
    {
        cuMemFree(free_dev_addr);
    }
}
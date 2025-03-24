// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/utility/get_shift.hpp"

namespace ck {

// clang-format off
// Assume:
//  1) mean_value, var_value and count is the input data in vgpr from each thread
//  2) mean_value, var_value and count is the over-written reduced output in vgpr for each thread
//  3) Merge mean and M from ThreadwiseWelford
// clang-format on
template <typename T,
          index_t BlockSize,
          typename ThreadClusterLengths_M_K,
          typename ThreadClusterArrangeOrder,
          bool GetActualVariance = true>
struct BlockwiseWelford
{
    static_assert(BlockSize == ThreadClusterLengths_M_K::At(0) * ThreadClusterLengths_M_K::At(1),
                  "The product of cluster lengths should be same as BlockSize!");

    static constexpr auto BufferLength_M = ThreadClusterLengths_M_K::At(0);
    static constexpr auto BufferLength_K = ThreadClusterLengths_M_K::At(1);

    static constexpr auto block_buf_desc_m_k = make_naive_tensor_descriptor_packed(
        make_tuple(Number<BufferLength_M>{}, Number<BufferLength_K>{}));

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    template <typename CountDataType>
    __device__ static inline void
    Merge(T& mean_a, T& var_a, CountDataType& count_a, T mean_b, T var_b, CountDataType count_b)
    {
        CountDataType count  = count_a + count_b;
        T count_b_over_count = count == 0
                                   ? type_convert<T>(0)
                                   : type_convert<T>(count_b) * math::rcp(type_convert<T>(count));
        T delta              = mean_b - mean_a;
        mean_a += delta * count_b_over_count;
        var_a += var_b + delta * delta * count_a * count_b_over_count;
        count_a = count;
    }

    template <typename CountDataType>
    __device__ static void Run(T& mean_value, T& var_value, CountDataType& count)
    {
        __shared__ T mean_block_buf[BlockSize];
        __shared__ T var_block_buf[BlockSize];
        __shared__ CountDataType count_block_buf[BlockSize];

        constexpr auto cluster_len_shift = get_shift<BufferLength_K>();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));

        const auto thread_m_cluster_id = thread_cluster_idx[Number<0>{}];
        const auto thread_k_cluster_id = thread_cluster_idx[Number<1>{}];

        index_t offset1 = block_buf_desc_m_k.CalculateOffset(thread_cluster_idx);

        mean_block_buf[offset1]  = mean_value;
        var_block_buf[offset1]   = var_value;
        count_block_buf[offset1] = count;

        block_sync_lds();

        static_for<0, cluster_len_shift, 1>{}([&](auto I) {
            constexpr index_t indOffset = 1 << (cluster_len_shift - 1 - I());

            if(thread_k_cluster_id < indOffset)
            {
                index_t offset2 = block_buf_desc_m_k.CalculateOffset(thread_cluster_idx +
                                                                     make_tuple(0, indOffset));

                T mean2              = mean_block_buf[offset2];
                T var2               = var_block_buf[offset2];
                CountDataType count2 = count_block_buf[offset2];

                Merge(mean_value, var_value, count, mean2, var2, count2);

                mean_block_buf[offset1]  = mean_value;
                var_block_buf[offset1]   = var_value;
                count_block_buf[offset1] = count;
            }

            block_sync_lds();
        });

        index_t offset = block_buf_desc_m_k.CalculateOffset(make_tuple(thread_m_cluster_id, 0));

        count      = count_block_buf[offset];
        mean_value = mean_block_buf[offset];

        if constexpr(GetActualVariance)
            var_value = var_block_buf[offset] * math::rcp(type_convert<T>(count));
        else
            var_value = var_block_buf[offset];
    };
};
} // namespace ck

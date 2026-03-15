#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/cuda/cuda_stream.h"

namespace infini_train::kernels::cuda {
__device__ __forceinline__ float WarpReduceSum(float val) {
    for (int lane_mask = 16; lane_mask > 0; lane_mask /= 2) { val += __shfl_xor_sync(0xffffffff, val, lane_mask); }
    return val;
}

__device__ __forceinline__ int64_t Offset3D(int b, int h, int t, int H, int Tlen) {
    return ((int64_t)b * H + h) * Tlen + t;
}

/**
 * FlashAttention2-style forward kernel (framework integration version)
 *
 * Layout:
 *   q:   [B, Hq, Tq, D]
 *   k:   [B, Hkv, Tk, D]
 *   v:   [B, Hkv, Tk, D]
 *   o:   [B, Hq, Tq, D]
 *   lse: [B, Hq, Tq]    (float32)
 *
 * Work partition:
 *   - one block handles one (b, hq, q_tile)
 *   - one warp handles one query row
 *   - K/V are tiled through shared memory
 *   - online softmax, no explicit S/P materialization
 */
template <typename T, int BLOCK_R, int BLOCK_C, int HEAD_DIM>
__global__ void ScaledDotProductAttentionForwardKernel(const T *query, const T *key, const T *value, const T *attn_mask,
                                                       T *out, float *softmax_lse, int64_t B, int64_t Hq, int64_t Tq,
                                                       int64_t Hkv, int64_t Tkv, bool is_causal, float scale) {

    constexpr int WARP_SIZE = 32;
    constexpr int FRAG = HEAD_DIM / WARP_SIZE; // assuming HEAD_DIM is divisible by WARP_SIZE
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be a multiple of WARP_SIZE");

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    const int q_tile_start = blockIdx.x * BLOCK_R;
    const int hq = blockIdx.y;
    const int b = blockIdx.z;

    const int q_row = q_tile_start + warp_id;

    // if (warp_id >= BLOCK_R) {
    //     return; // out of query tile
    // }

    // if (q_row >= Tq || hq >= Hq || b >= B) {
    //     return; // out of bounds
    // }

    bool valid_warp = (warp_id < BLOCK_R) && (q_row < Tq) && (hq < Hq) && (b < B);

    const int n_rep = Hq / Hkv;
    const int hkv = hq / n_rep; // corresponding K/V head for this Q head

    extern __shared__ unsigned char shared_mem[];
    T *k_smem = reinterpret_cast<T *>(shared_mem); // [BLOCK_C, HEAD_DIM]
    T *v_smem = k_smem + BLOCK_C * HEAD_DIM;       // [BLOCK_C, HEAD_DIM]

    // ------------------------------------------------------------
    // 1. base pointers
    // ------------------------------------------------------------
    const int64_t q_row_base = valid_warp ? Offset3D(b, hq, q_row, Hq, Tq) * HEAD_DIM : 0;
    const T *q_row_ptr = valid_warp ? (query + q_row_base) : nullptr;

    const int64_t kv_head_base = Offset3D(b, hkv, 0, Hkv, Tkv) * HEAD_DIM;
    const T *k_head_ptr = key + kv_head_base;
    const T *v_head_ptr = value + kv_head_base;

    const int64_t o_row_base = valid_warp ? Offset3D(b, hq, q_row, Hq, Tq) * HEAD_DIM : 0;
    T *o_row_ptr = valid_warp ? out + o_row_base : nullptr;

    // ------------------------------------------------------------
    // 2. load Q row to registers
    // ------------------------------------------------------------
    float q_frag[FRAG];
#pragma unroll
    for (int i = 0; i < FRAG; ++i) { q_frag[i] = 0.0f; }

    if (valid_warp) {
#pragma unroll
        for (int i = 0; i < FRAG; ++i) {
            const int idx = lane_id + i * WARP_SIZE;
            q_frag[i] = common::cuda::Cast<float>(q_row_ptr[idx]) * scale; // apply scaling here
        }
    }

    // ------------------------------------------------------------
    // 3. online softmax states
    // ------------------------------------------------------------
    float m_i = -1e30f; // max for numerical stability
    float l_i = 0.0f;   // sum of exp for normalization

    float o_frag[FRAG] = {0}; // output fragment in registers
#pragma unroll
    for (int i = 0; i < FRAG; ++i) { o_frag[i] = 0.0f; }

    // ------------------------------------------------------------
    // 4. sweep KV tiles
    // ------------------------------------------------------------
    for (int kv_tile_start = 0; kv_tile_start < Tkv; kv_tile_start += BLOCK_C) {
        const int remaining_kv = static_cast<int>(Tkv - kv_tile_start);
        const int kv_tile_size = remaining_kv < BLOCK_C ? remaining_kv : BLOCK_C; // actual tile size for this iteration

        const T *k_tile_ptr = k_head_ptr + (int64_t)kv_tile_start * HEAD_DIM;
        const T *v_tile_ptr = v_head_ptr + (int64_t)kv_tile_start * HEAD_DIM;

        // whole block cooperatively loads K/V tile to shared memory
        const int linear_tid = threadIdx.x;
        const int block_threads = blockDim.x;
        const int num_kv_tile_elements = kv_tile_size * HEAD_DIM;

        for (int idx = linear_tid; idx < num_kv_tile_elements; idx += block_threads) { //?
            const int j = idx / HEAD_DIM;                                              // tile row
            const int d = idx - j * HEAD_DIM;                                          // tile col   d = idx % HEAD_DIM

            const T *k_row_ptr = k_tile_ptr + (int64_t)j * HEAD_DIM;
            const T *v_row_ptr = v_tile_ptr + (int64_t)j * HEAD_DIM;

            k_smem[j * HEAD_DIM + d] = k_row_ptr[d];
            v_smem[j * HEAD_DIM + d] = v_row_ptr[d];
        }
        __syncthreads();

        // only valid warps (those that have a query row to process) participate in the computation
        if (valid_warp) {
#pragma unroll 1
            // current warp processes one query row through this KV tile
            for (int j = 0; j < kv_tile_size; ++j) {
                // compute Q-K dot product for this KV row
                const int k_col = kv_tile_start + j; // global K/V row index

                if (is_causal && k_col > q_row) {
                    continue; // skip masked out positions for causal attention
                }

                const T *k_smem_row_ptr = k_smem + j * HEAD_DIM;
                const T *v_smem_row_ptr = v_smem + j * HEAD_DIM;

                float partial = 0.0f;

#pragma unroll
                for (int i = 0; i < FRAG; ++i) {
                    const int d = lane_id + i * WARP_SIZE;
                    const float k_val = common::cuda::Cast<float>(k_smem_row_ptr[d]);
                    partial = common::cuda::Fma(q_frag[i], k_val, partial);
                }
                // NOTE: attn_mask is not applied in this placeholder kernel, but should be applied here in the future
                float score = WarpReduceSum(partial);

                // online softmax update
                const float m_new = common::cuda::Max(m_i, score);
                const float alpha = common::cuda::Exp(m_i - m_new);  // rescaling factor for old values in the sum
                const float beta = common::cuda::Exp(score - m_new); // contribution of the new score to the sum
                const float l_new = l_i * alpha + beta;              // new normalizer value

                const float old_scale
                    = l_new > 0.0f ? (l_i * alpha / l_new) : 0.0f; // rescaling factor for old values in the sum
                const float new_scale
                    = l_new > 0.0f ? (beta / l_new) : 0.0f; // scaling factor for the new value in the sum

#pragma unroll
                for (int i = 0; i < FRAG; ++i) {
                    const int d = lane_id + i * WARP_SIZE;
                    const float v_val = common::cuda::Cast<float>(v_smem_row_ptr[d]);
                    o_frag[i] = common::cuda::Fma(
                        v_val, new_scale,
                        o_frag[i] * old_scale); // o_frag[i] = o_frag[i] * old_scale + v_val * new_scale
                }

                m_i = m_new;
                l_i = l_new;
            }
        }
        // All threads in block must hit this barrier.
        __syncthreads();
    }

    // ------------------------------------------------------------
    // 5. store O
    // ------------------------------------------------------------
    if (valid_warp) {
#pragma unroll
        for (int i = 0; i < FRAG; ++i) {
            const int d = lane_id + i * WARP_SIZE;
            o_row_ptr[d] = common::cuda::Cast<T>(o_frag[i]);
        }
    }

    // ------------------------------------------------------------
    // 6. store LSE for backward
    // ------------------------------------------------------------
    if (valid_warp && lane_id == 0) {
        const int64_t lse_idx = Offset3D(b, hq, q_row, Hq, Tq);
        softmax_lse[lse_idx] = m_i + common::cuda::Log(l_i); // log(sum) + max for numerical stability
    }
}

template <typename T>
void LaunchScaledDotProductAttentionForward(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &lse,
                                            const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                                            const std::shared_ptr<Tensor> &value,
                                            const std::shared_ptr<Tensor> &attn_mask, double dropout_p, bool is_causal,
                                            double scale) {
    const auto &q_dims = query->Dims();
    const auto &k_dims = key->Dims();
    const auto &v_dims = value->Dims();
    const auto &o_dims = output->Dims();
    const auto &lse_dims = lse->Dims();

    const int64_t B = q_dims[0];
    const int64_t Hq = q_dims[1];
    const int64_t Tq = q_dims[2];
    const int64_t D = q_dims[3];

    const int64_t Hkv = k_dims[1];
    const int64_t Tkv = k_dims[2];

    T *output_ptr = static_cast<T *>(output->DataPtr());
    float *lse_ptr = static_cast<float *>(lse->DataPtr());
    const T *query_ptr = static_cast<const T *>(query->DataPtr());
    const T *key_ptr = static_cast<const T *>(key->DataPtr());
    const T *value_ptr = static_cast<const T *>(value->DataPtr());
    // NOTE: attn_mask is not used in this placeholder kernel, but should be passed to the kernel in the future
    // const T *attn_mask_ptr = attn_mask != nullptr ? static_cast<const T *>(attn_mask->DataPtr()) : nullptr;

    // Kernel launch parameters
    constexpr int BLOCK_R = 32;                 // number of query rows per block (tunable)
    constexpr int BLOCK_C = 32;                 // number of KV rows per tile (tunable)
    const int THREADS_PER_BLOCK = BLOCK_R * 32; // one warp per query row
    const int num_q_tiles = (Tq + BLOCK_R - 1) / BLOCK_R;

    if (THREADS_PER_BLOCK > 1024) {
        LOG_LOC(FATAL,
                "CUDA attention forward: 'THREADS_PER_BLOCK used is larger than the max number of thread per block'");
    }

    const dim3 block_dims(THREADS_PER_BLOCK);
    const dim3 grid_dims(num_q_tiles, Hq, B);

    const auto device = output->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    if (D == 64) {
        constexpr int HEAD_DIM = 64;
        const size_t smem_bytes = 2 * BLOCK_C * HEAD_DIM * sizeof(T); // shared memory size for K and V tiles
        ScaledDotProductAttentionForwardKernel<T, BLOCK_R, BLOCK_C, HEAD_DIM>
            <<<grid_dims, block_dims, smem_bytes, cuda_stream>>>(query_ptr, key_ptr, value_ptr, nullptr, output_ptr,
                                                                 lse_ptr, B, Hq, Tq, Hkv, Tkv, is_causal,
                                                                 static_cast<float>(scale));
    } else if (D == 128) {
        constexpr int HEAD_DIM = 128;
        const size_t smem_bytes = 2 * BLOCK_C * HEAD_DIM * sizeof(T); // shared memory size for K and V tiles
        ScaledDotProductAttentionForwardKernel<T, BLOCK_R, BLOCK_C, HEAD_DIM>
            <<<grid_dims, block_dims, smem_bytes, cuda_stream>>>(query_ptr, key_ptr, value_ptr, nullptr, output_ptr,
                                                                 lse_ptr, B, Hq, Tq, Hkv, Tkv, is_causal,
                                                                 static_cast<float>(scale));
    } else {
        LOG_LOC(FATAL, "CUDA attention forward: 'Unsupported head dimension D. Only D=64 and D=128 are supported in "
                       "this placeholder kernel.'");
    }
}

template <int THREADS> __device__ __forceinline__ float BlockReduceSum(float val) {
    static_assert(THREADS % 32 == 0, "THREADS must be a multiple of warp size");
    constexpr int N_WARPS = THREADS / 32;
    __shared__ float warp_sums[N_WARPS];
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    val = WarpReduceSum(val);
    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (lane_id < N_WARPS) ? warp_sums[lane_id] : 0.0f;
        block_sum = WarpReduceSum(block_sum);
        if (lane_id == 0) {
            warp_sums[0] = block_sum;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

template <typename T, int HEAD_DIM>
__global__ void AttentionBackwardPreprocessDeltaKernel(const T *grad_out, const T *out, float *delta, int64_t B,
                                                       int64_t Hq, int64_t Tq) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = B * Hq * Tq;
    if (idx >= total) {
        return;
    }
    const int64_t t = idx % Tq;
    const int64_t tmp = idx / Tq;
    const int64_t h = tmp % Hq;
    const int64_t b = tmp / Hq;
    const int64_t row_base = Offset3D(static_cast<int>(b), static_cast<int>(h), static_cast<int>(t), Hq, Tq) * HEAD_DIM;
    float acc = 0.0f;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        const float go = common::cuda::Cast<float>(grad_out[row_base + d]);
        const float o = common::cuda::Cast<float>(out[row_base + d]);
        acc = common::cuda::Fma(go, o, acc);
    }
    delta[idx] = acc;
}

template <typename T, int HEAD_DIM, int THREADS>
__global__ void AttentionBackwardDqKernel(const T *grad_out, const T *query, const T *key, const T *value,
                                          const float *softmax_lse, const float *delta, T *grad_query, int64_t B,
                                          int64_t Hq, int64_t Tq, int64_t Hkv, int64_t Tkv, bool is_causal,
                                          float scale) {
    const int q_row = blockIdx.x;
    const int hq = blockIdx.y;
    const int b = blockIdx.z;
    if (q_row >= Tq || hq >= Hq || b >= B) {
        return;
    }

    const int tid = threadIdx.x;
    const int64_t q_base = Offset3D(b, hq, q_row, Hq, Tq) * HEAD_DIM;
    const int n_rep = static_cast<int>(Hq / Hkv);
    const int hkv = hq / n_rep;
    const float lse_i = softmax_lse[Offset3D(b, hq, q_row, Hq, Tq)];
    const float delta_i = delta[Offset3D(b, hq, q_row, Hq, Tq)];

    float dq_local = 0.0f;
    __shared__ float shared_dS;

    for (int k_col = 0; k_col < Tkv; ++k_col) {
        if (is_causal && k_col > q_row) {
            continue;
        }

        const int64_t kv_base = Offset3D(b, hkv, k_col, Hkv, Tkv) * HEAD_DIM;
        float score_partial = 0.0f;
        float dp_partial = 0.0f;
        for (int d = tid; d < HEAD_DIM; d += THREADS) {
            const float qv = common::cuda::Cast<float>(query[q_base + d]);
            const float kv = common::cuda::Cast<float>(key[kv_base + d]);
            const float gov = common::cuda::Cast<float>(grad_out[q_base + d]);
            const float vv = common::cuda::Cast<float>(value[kv_base + d]);
            score_partial = common::cuda::Fma(qv * scale, kv, score_partial);
            dp_partial = common::cuda::Fma(gov, vv, dp_partial);
        }
        const float score = BlockReduceSum<THREADS>(score_partial);
        const float dP = BlockReduceSum<THREADS>(dp_partial);
        if (tid == 0) {
            const float p = common::cuda::Exp(score - lse_i);
            shared_dS = p * (dP - delta_i);
        }
        __syncthreads();

        for (int d = tid; d < HEAD_DIM; d += THREADS) {
            const float kv = common::cuda::Cast<float>(key[kv_base + d]);
            dq_local = common::cuda::Fma(shared_dS * scale, kv, dq_local);
        }
    }
    for (int d = tid; d < HEAD_DIM; d += THREADS) { grad_query[q_base + d] = common::cuda::Cast<T>(dq_local); }
}

template <typename T, int HEAD_DIM, int THREADS>
__global__ void AttentionBackwardDkDvKernel(const T *grad_out, const T *query, const T *key, const T *value,
                                            const float *softmax_lse, const float *delta, T *grad_key, T *grad_value,
                                            int64_t B, int64_t Hq, int64_t Tq, int64_t Hkv, int64_t Tkv, bool is_causal,
                                            float scale) {
    const int k_col = blockIdx.x;
    const int hkv = blockIdx.y;
    const int b = blockIdx.z;
    if (k_col >= Tkv || hkv >= Hkv || b >= B) {
        return;
    }

    const int tid = threadIdx.x;
    const int n_rep = static_cast<int>(Hq / Hkv);
    const int hq_begin = hkv * n_rep;
    const int hq_end = hq_begin + n_rep;
    const int64_t kv_base = Offset3D(b, hkv, k_col, Hkv, Tkv) * HEAD_DIM;
    float dk_local = 0.0f;
    float dv_local = 0.0f;
    __shared__ float shared_p;
    __shared__ float shared_dS;

    for (int hq = hq_begin; hq < hq_end; ++hq) {
        for (int q_row = 0; q_row < Tq; ++q_row) {
            if (is_causal && k_col > q_row) {
                continue;
            }
            const int64_t q_base = Offset3D(b, hq, q_row, Hq, Tq) * HEAD_DIM;
            const float lse_i = softmax_lse[Offset3D(b, hq, q_row, Hq, Tq)];
            const float delta_i = delta[Offset3D(b, hq, q_row, Hq, Tq)];

            float score_partial = 0.0f;
            float dp_partial = 0.0f;
            for (int d = tid; d < HEAD_DIM; d += THREADS) {
                const float qv = common::cuda::Cast<float>(query[q_base + d]);
                const float kv = common::cuda::Cast<float>(key[kv_base + d]);
                const float gov = common::cuda::Cast<float>(grad_out[q_base + d]);
                const float vv = common::cuda::Cast<float>(value[kv_base + d]);
                score_partial = common::cuda::Fma(qv * scale, kv, score_partial);
                dp_partial = common::cuda::Fma(gov, vv, dp_partial);
            }

            const float score = BlockReduceSum<THREADS>(score_partial);
            const float dP = BlockReduceSum<THREADS>(dp_partial);
            if (tid == 0) {
                shared_p = common::cuda::Exp(score - lse_i);
                shared_dS = shared_p * (dP - delta_i);
            }
            __syncthreads();

            for (int d = tid; d < HEAD_DIM; d += THREADS) {
                const float qv = common::cuda::Cast<float>(query[q_base + d]);
                const float gov = common::cuda::Cast<float>(grad_out[q_base + d]);
                dk_local = common::cuda::Fma(shared_dS * scale, qv, dk_local);
                dv_local = common::cuda::Fma(shared_p, gov, dv_local);
            }
        }
    }

    for (int d = tid; d < HEAD_DIM; d += THREADS) {
        grad_key[kv_base + d] = common::cuda::Cast<T>(dk_local);
        grad_value[kv_base + d] = common::cuda::Cast<T>(dv_local);
    }
}

template <typename T>
void LaunchScaledDotProductAttentionBackward(const std::shared_ptr<Tensor> &grad_query,
                                             const std::shared_ptr<Tensor> &grad_key,
                                             const std::shared_ptr<Tensor> &grad_value,
                                             const std::shared_ptr<Tensor> &grad_out,
                                             const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                                             const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &out,
                                             const std::shared_ptr<Tensor> &softmax_lse, bool is_causal, double scale) {
    const auto &q_dims = query->Dims();
    const auto &k_dims = key->Dims();

    const int64_t B = q_dims[0];
    const int64_t Hq = q_dims[1];
    const int64_t Tq = q_dims[2];
    const int64_t D = q_dims[3];
    const int64_t Hkv = k_dims[1];
    const int64_t Tkv = k_dims[2];

    const auto device = grad_out->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    grad_query->Fill<T>(0);
    grad_key->Fill<T>(0);
    grad_value->Fill<T>(0);

    auto delta = std::make_shared<Tensor>(std::vector<int64_t>{B, Hq, Tq}, DataType::kFLOAT32, device);
    constexpr int DELTA_THREADS = 256;
    const int64_t delta_rows = B * Hq * Tq;
    const int64_t delta_blocks = (delta_rows + DELTA_THREADS - 1) / DELTA_THREADS;

    if (D == 64) {
        constexpr int HEAD_DIM = 64;
        constexpr int THREADS = 64;
        AttentionBackwardPreprocessDeltaKernel<T, HEAD_DIM>
            <<<static_cast<int>(delta_blocks), DELTA_THREADS, 0, cuda_stream>>>(
                static_cast<const T *>(grad_out->DataPtr()), static_cast<const T *>(out->DataPtr()),
                static_cast<float *>(delta->DataPtr()), B, Hq, Tq);
        const dim3 dq_grid(static_cast<unsigned int>(Tq), static_cast<unsigned int>(Hq), static_cast<unsigned int>(B));
        AttentionBackwardDqKernel<T, HEAD_DIM, THREADS><<<dq_grid, THREADS, 0, cuda_stream>>>(
            static_cast<const T *>(grad_out->DataPtr()), static_cast<const T *>(query->DataPtr()),
            static_cast<const T *>(key->DataPtr()), static_cast<const T *>(value->DataPtr()),
            static_cast<const float *>(softmax_lse->DataPtr()), static_cast<const float *>(delta->DataPtr()),
            static_cast<T *>(grad_query->DataPtr()), B, Hq, Tq, Hkv, Tkv, is_causal, static_cast<float>(scale));
        const dim3 dkv_grid(static_cast<unsigned int>(Tkv), static_cast<unsigned int>(Hkv),
                            static_cast<unsigned int>(B));
        AttentionBackwardDkDvKernel<T, HEAD_DIM, THREADS><<<dkv_grid, THREADS, 0, cuda_stream>>>(
            static_cast<const T *>(grad_out->DataPtr()), static_cast<const T *>(query->DataPtr()),
            static_cast<const T *>(key->DataPtr()), static_cast<const T *>(value->DataPtr()),
            static_cast<const float *>(softmax_lse->DataPtr()), static_cast<const float *>(delta->DataPtr()),
            static_cast<T *>(grad_key->DataPtr()), static_cast<T *>(grad_value->DataPtr()), B, Hq, Tq, Hkv, Tkv,
            is_causal, static_cast<float>(scale));
    } else if (D == 128) {
        constexpr int HEAD_DIM = 128;
        constexpr int THREADS = 128;
        AttentionBackwardPreprocessDeltaKernel<T, HEAD_DIM>
            <<<static_cast<int>(delta_blocks), DELTA_THREADS, 0, cuda_stream>>>(
                static_cast<const T *>(grad_out->DataPtr()), static_cast<const T *>(out->DataPtr()),
                static_cast<float *>(delta->DataPtr()), B, Hq, Tq);
        const dim3 dq_grid(static_cast<unsigned int>(Tq), static_cast<unsigned int>(Hq), static_cast<unsigned int>(B));
        AttentionBackwardDqKernel<T, HEAD_DIM, THREADS><<<dq_grid, THREADS, 0, cuda_stream>>>(
            static_cast<const T *>(grad_out->DataPtr()), static_cast<const T *>(query->DataPtr()),
            static_cast<const T *>(key->DataPtr()), static_cast<const T *>(value->DataPtr()),
            static_cast<const float *>(softmax_lse->DataPtr()), static_cast<const float *>(delta->DataPtr()),
            static_cast<T *>(grad_query->DataPtr()), B, Hq, Tq, Hkv, Tkv, is_causal, static_cast<float>(scale));
        const dim3 dkv_grid(static_cast<unsigned int>(Tkv), static_cast<unsigned int>(Hkv),
                            static_cast<unsigned int>(B));
        AttentionBackwardDkDvKernel<T, HEAD_DIM, THREADS><<<dkv_grid, THREADS, 0, cuda_stream>>>(
            static_cast<const T *>(grad_out->DataPtr()), static_cast<const T *>(query->DataPtr()),
            static_cast<const T *>(key->DataPtr()), static_cast<const T *>(value->DataPtr()),
            static_cast<const float *>(softmax_lse->DataPtr()), static_cast<const float *>(delta->DataPtr()),
            static_cast<T *>(grad_key->DataPtr()), static_cast<T *>(grad_value->DataPtr()), B, Hq, Tq, Hkv, Tkv,
            is_causal, static_cast<float>(scale));
    } else {
        LOG_LOC(FATAL, "CUDA attention backward: unsupported D, only 64/128 are supported");
    }

    CUDA_CHECK(cudaGetLastError());
}

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttentionForward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                                 const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask,
                                 double dropout_p, bool is_causal, double scale, bool enable_gqa) {
    LOG(INFO) << "flash attention path";
    const auto &q_dims = query->Dims();
    const auto &k_dims = key->Dims();
    const auto &v_dims = value->Dims();

    const int64_t B = q_dims[0];
    const int64_t Hq = q_dims[1];
    const int64_t Tq = q_dims[2];
    const int64_t D = q_dims[3];

    const int64_t Hkv = k_dims[1];
    const int64_t Tkv = k_dims[2];

    // Allocate output tensors
    const auto dtype = query->Dtype();
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{B, Hq, Tq, D}, dtype, query->GetDevice());
    auto softmax_lse
        = std::make_shared<Tensor>(std::vector<int64_t>{B, Hq, Tq}, DataType::kFLOAT32, query->GetDevice());

    switch (dtype) {
        DISPATCH_CASE(WRAP(LaunchScaledDotProductAttentionForward<float>(output, softmax_lse, query, key, value,
                                                                         attn_mask, dropout_p, is_causal, scale);),
                      DataType::kFLOAT32)

        DISPATCH_CASE(WRAP(LaunchScaledDotProductAttentionForward<nv_bfloat16>(
                               output, softmax_lse, query, key, value, attn_mask, dropout_p, is_causal, scale);),
                      DataType::kBFLOAT16)

    default:
        LOG_LOC(FATAL, "CUDA flash attention forward: 'Unsupported data type'");
    }
    return {output, softmax_lse};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
ScaledDotProductAttentionBackward(const std::shared_ptr<Tensor> &grad_out, const std::shared_ptr<Tensor> &query,
                                  const std::shared_ptr<Tensor> &key, const std::shared_ptr<Tensor> &value,
                                  const std::shared_ptr<Tensor> &attn_mask, const std::shared_ptr<Tensor> &out,
                                  const std::shared_ptr<Tensor> &softmax_lse, double dropout_p, bool is_causal,
                                  double actual_scale, bool enable_gqa) {
    CHECK(grad_out != nullptr) << "grad_out must not be nullptr";
    CHECK(query != nullptr) << "query must not be nullptr";
    CHECK(key != nullptr) << "key must not be nullptr";
    CHECK(value != nullptr) << "value must not be nullptr";
    CHECK(out != nullptr) << "out must not be nullptr";
    CHECK(softmax_lse != nullptr) << "softmax_lse must not be nullptr";

    CHECK_EQ(dropout_p, 0.0) << "CUDA attention backward: dropout is not supported yet";
    CHECK(attn_mask == nullptr) << "CUDA attention backward: attn_mask is not supported yet";
    CHECK(!enable_gqa || (query->Dims()[1] % key->Dims()[1] == 0))
        << "CUDA attention backward: invalid GQA heads, Hq must be divisible by Hkv";

    const auto dtype = grad_out->Dtype();
    auto grad_query = std::make_shared<Tensor>(query->Dims(), dtype, query->GetDevice());
    auto grad_key = std::make_shared<Tensor>(key->Dims(), dtype, key->GetDevice());
    auto grad_value = std::make_shared<Tensor>(value->Dims(), dtype, value->GetDevice());

    switch (dtype) {
        DISPATCH_CASE(
            WRAP(LaunchScaledDotProductAttentionBackward<float>(grad_query, grad_key, grad_value, grad_out, query, key,
                                                                value, out, softmax_lse, is_causal, actual_scale);),
            DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(LaunchScaledDotProductAttentionBackward<nv_bfloat16>(grad_query, grad_key, grad_value,
                                                                                grad_out, query, key, value, out,
                                                                                softmax_lse, is_causal, actual_scale);),
                      DataType::kBFLOAT16)
    default:
        LOG_LOC(FATAL, "CUDA attention backward: unsupported dtype");
    }

    // grad_attn_mask is not supported yet.
    return {grad_query, grad_key, grad_value, nullptr};
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ATTENTION_KERNEL(kernel_name)                                                                    \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ATTENTION_KERNEL(ScaledDotProductAttentionForward)
REGISTER_CUDA_ATTENTION_KERNEL(ScaledDotProductAttentionBackward)

#undef REGISTER_CUDA_ATTENTION_KERNEL

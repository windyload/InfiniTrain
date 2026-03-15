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

    if (warp_id >= BLOCK_R) {
        return; // out of query tile
    }

    const int q_row = q_tile_start + warp_id;
    if (q_row >= Tq || hq >= Hq || b >= B) {
        return; // out of bounds
    }

    const int n_rep = Hq / Hkv;
    const int hkv = hq / n_rep; // corresponding K/V head for this Q head

    extern __shared__ unsigned char shared_mem[];
    T *k_smem = reinterpret_cast<T *>(shared_mem); // [BLOCK_C, HEAD_DIM]
    T *v_smem = k_smem + BLOCK_C * HEAD_DIM;       // [BLOCK_C, HEAD_DIM]

    // ------------------------------------------------------------
    // 1. base pointers
    // ------------------------------------------------------------
    const int64_t q_row_base = Offset3D(b, hq, q_row, Hq, Tq) * HEAD_DIM;
    const T *q_row_ptr = query + q_row_base;

    const int64_t kv_head_base = Offset3D(b, hkv, 0, Hkv, Tkv) * HEAD_DIM;
    const T *k_head_ptr = key + kv_head_base;
    const T *v_head_ptr = value + kv_head_base;

    const int64_t o_row_base = Offset3D(b, hq, q_row, Hq, Tq) * HEAD_DIM;
    T *o_row_ptr = out + o_row_base;

    // ------------------------------------------------------------
    // 2. load Q row to registers
    // ------------------------------------------------------------
    float q_frag[FRAG];
#pragma unroll
    for (int i = 0; i < FRAG; ++i) {
        const int idx = lane_id + i * WARP_SIZE;
        q_frag[i] = common::cuda::Cast<float>(q_row_ptr[idx]) * scale; // apply scaling here
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

        // load K/V tile to shared memory
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

        // current warp processes one query row through this KV tile
#pragma unroll 1
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
            float score = WarpReduceSum(partial); // + attn_mask[b, hq, q_row, k_col] if attn_mask is not nullptr

            // online softmax update
            const float m_new = common::cuda::Max(m_i, score);
            const float alpha = common::cuda::Exp(m_i - m_new);  // rescaling factor for old values in the sum
            const float beta = common::cuda::Exp(score - m_new); // contribution of the new score to the sum
            const float l_new = l_i * alpha + beta;              // new normalizer value

            const float old_scale
                = l_new > 0.0f ? (l_i * alpha / l_new) : 0.0f;            // rescaling factor for old values in the sum
            const float new_scale = l_new > 0.0f ? (beta / l_new) : 0.0f; // scaling factor for the new value in the sum

#pragma unroll
            for (int i = 0; i < FRAG; ++i) {
                const int d = lane_id + i * WARP_SIZE;
                const float v_val = common::cuda::Cast<float>(v_smem_row_ptr[d]);
                o_frag[i]
                    = common::cuda::Fma(v_val, new_scale,
                                        o_frag[i] * old_scale); // o_frag[i] = o_frag[i] * old_scale + v_val * new_scale
                                                                // ;accumulate the new value with proper rescaling
            }

            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    // ------------------------------------------------------------
    // 5. store O
    // ------------------------------------------------------------
#pragma unroll
    for (int i = 0; i < FRAG; ++i) {
        const int d = lane_id + i * WARP_SIZE;
        o_row_ptr[d] = common::cuda::Cast<T>(o_frag[i]);
    }

    // ------------------------------------------------------------
    // 6. store LSE for backward
    // ------------------------------------------------------------
    if (lane_id == 0) {
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
    constexpr int BLOCK_R = 4;                  // number of query rows per block (tunable)
    constexpr int BLOCK_C = 64;                 // number of KV rows per tile (tunable)
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

    LOG(INFO) << "Reached CUDA ScaledDotProductAttentionForward"
              << ", B=" << B << ", Hq=" << Hq << ", Tq=" << Tq << ", Hkv=" << Hkv << ", Tkv=" << Tkv << ", D=" << D
              << ", is_causal=" << is_causal << ", scale=" << scale;

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
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

    LOG(INFO) << "FlashAttention forward launch finished successfully";
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

    LOG(FATAL) << "CUDA ScaledDotProductAttentionBackward is not implemented yet";
    return {nullptr, nullptr, nullptr, nullptr};
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ATTENTION_KERNEL(kernel_name)                                                                    \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ATTENTION_KERNEL(ScaledDotProductAttentionForward)
REGISTER_CUDA_ATTENTION_KERNEL(ScaledDotProductAttentionBackward)

#undef REGISTER_CUDA_ATTENTION_KERNEL

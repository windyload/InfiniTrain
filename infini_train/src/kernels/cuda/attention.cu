#include <memory>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttentionForward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                                 const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask,
                                 double dropout_p, bool is_causal, double scale, bool enable_gqa) {
    LOG(FATAL) << "Reached CUDA ScaledDotProductAttentionForward";

    CHECK(query != nullptr) << "query must not be nullptr";
    CHECK(key != nullptr) << "key must not be nullptr";
    CHECK(value != nullptr) << "value must not be nullptr";

    const auto &q_dims = query->Dims();
    const auto &k_dims = key->Dims();
    const auto &v_dims = value->Dims();

    CHECK_EQ(q_dims.size(), 4) << "query must be 4D: (B, Tq, Hq, D)";
    CHECK_EQ(k_dims.size(), 4) << "key must be 4D: (B, Tk, Hk, D)";
    CHECK_EQ(v_dims.size(), 4) << "value must be 4D: (B, Tk, Hk, D)";

    const auto B = q_dims[0];
    const auto Tq = q_dims[1];
    const auto Hq = q_dims[2];
    const auto D = q_dims[3];

    CHECK_EQ(k_dims[0], B) << "key batch size must match query";
    CHECK_EQ(v_dims[0], B) << "value batch size must match query";
    CHECK_EQ(k_dims[1], v_dims[1]) << "key/value sequence length must match";
    CHECK_EQ(k_dims[2], Hq) << "current placeholder requires Hk == Hq";
    CHECK_EQ(v_dims[2], Hq) << "current placeholder requires Hv == Hq";
    CHECK_EQ(k_dims[3], D) << "key head_dim must match query";
    CHECK_EQ(v_dims[3], D) << "value head_dim must match query";

    CHECK(query->GetDevice().type() == key->GetDevice().type()) << "query and key must be on the same device";
    CHECK(query->GetDevice().type() == value->GetDevice().type()) << "query and value must be on the same device";

    if (attn_mask != nullptr) {
        CHECK(query->GetDevice().type() == attn_mask->GetDevice().type())
            << "attn_mask must be on the same device as query";
    }

    CHECK_GE(dropout_p, 0.0) << "dropout_p must be >= 0";
    CHECK_LT(dropout_p, 1.0) << "dropout_p must be < 1";
    CHECK_EQ(dropout_p, 0.0) << "placeholder kernel does not support dropout yet";
    CHECK(!enable_gqa) << "placeholder kernel does not support GQA yet";
    CHECK_GT(scale, 0.0) << "scale must be > 0";

    const auto dtype = query->Dtype();
    const auto device = query->GetDevice();

    // Placeholder output:
    // out shape: (B, Tq, Hq, D)
    auto out = std::make_shared<Tensor>(q_dims, dtype, device);

    // Placeholder cached tensor for backward:
    // softmax_lse shape: (B, Hq, Tq)
    auto softmax_lse = std::make_shared<Tensor>(std::vector<int64_t>{B, Hq, Tq}, DataType::kFLOAT32, device);

    // NOTE:
    // This is only a stub for framework integration.
    // It does NOT compute real attention.
    //
    // The returned tensors only satisfy:
    // - valid shapes
    // - valid dtypes
    // - valid devices
    //
    // If your Tensor constructor does not initialize memory, that is okay for now
    // as long as the call chain can be validated.

    return {out, softmax_lse};
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

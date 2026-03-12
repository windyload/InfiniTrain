#include "infini_train/include/autograd/attention.h"

#include <cmath>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

constexpr char ScaledDotProductAttention::kType[];

double ScaledDotProductAttention::ResolveScale(const std::shared_ptr<Tensor> &query) const {
    CHECK(query != nullptr) << "query must not be nullptr";

    if (scale_.has_value()) {
        return *scale_;
    }

    const auto &q_dims = query->Dims();
    CHECK(!q_dims.empty()) << "query dims must not be empty";

    const auto head_dim = q_dims.back();
    CHECK_GT(head_dim, 0) << "query last dim (head_dim) must be > 0";

    return 1.0 / std::sqrt(static_cast<double>(head_dim));
}

std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttention::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK(input_tensors.size() == 3 || input_tensors.size() == 4)
        << "ScaledDotProductAttention expects 3 or 4 input tensors: "
        << "[query, key, value] or [query, key, value, attn_mask]";

    const auto &query = input_tensors[0];
    const auto &key = input_tensors[1];
    const auto &value = input_tensors[2];

    CHECK(query != nullptr) << "query must not be nullptr";
    CHECK(key != nullptr) << "key must not be nullptr";
    CHECK(value != nullptr) << "value must not be nullptr";

    std::shared_ptr<Tensor> attn_mask = nullptr;
    if (input_tensors.size() == 4) {
        attn_mask = input_tensors[3];
    }

    // device checks
    CHECK_EQ(query->GetDevice().type(), key->GetDevice().type())
        << "query and key must be on the same device";
    CHECK_EQ(query->GetDevice().type(), value->GetDevice().type())
        << "query and value must be on the same device";
    if (attn_mask != nullptr) {
        CHECK_EQ(query->GetDevice().type(), attn_mask->GetDevice().type())
            << "query and attn_mask must be on the same device";
    }

    // basic parameter checks
    CHECK_GE(dropout_p_, 0.0) << "dropout_p must be >= 0";
    CHECK_LT(dropout_p_, 1.0) << "dropout_p must be < 1";

    // If not implemented in backend yet, fail fast rather than silently ignore.
    CHECK_EQ(dropout_p_, 0.0)
        << "ScaledDotProductAttention currently does not support dropout yet";
    CHECK(!enable_gqa_)
        << "ScaledDotProductAttention currently does not support GQA yet";

    // shape checks for expected layout:
    // query: (B, Tq, Hq, D)
    // key:   (B, Tk, Hk, D)
    // value: (B, Tk, Hk, D)
    const auto &q_dims = query->Dims();
    const auto &k_dims = key->Dims();
    const auto &v_dims = value->Dims();

    CHECK_EQ(q_dims.size(), 4) << "query must be a 4D tensor of shape (B, Tq, Hq, D)";
    CHECK_EQ(k_dims.size(), 4) << "key must be a 4D tensor of shape (B, Tk, Hk, D)";
    CHECK_EQ(v_dims.size(), 4) << "value must be a 4D tensor of shape (B, Tk, Hk, D)";

    const auto Bq = q_dims[0];
    const auto Tq = q_dims[1];
    const auto Hq = q_dims[2];
    const auto Dq = q_dims[3];

    const auto Bk = k_dims[0];
    const auto Tk = k_dims[1];
    const auto Hk = k_dims[2];
    const auto Dk = k_dims[3];

    const auto Bv = v_dims[0];
    const auto Tv = v_dims[1];
    const auto Hv = v_dims[2];
    const auto Dv = v_dims[3];

    CHECK_EQ(Bq, Bk) << "query and key batch size must match";
    CHECK_EQ(Bq, Bv) << "query and value batch size must match";

    CHECK_EQ(Tk, Tv) << "key and value sequence length must match";
    CHECK_EQ(Hk, Hv) << "key and value head count must match";

    CHECK_EQ(Dq, Dk) << "query and key head_dim must match";
    CHECK_EQ(Dq, Dv) << "query and value head_dim must match in current implementation";

    CHECK_EQ(Hq, Hk)
        << "query and key head count must match in current implementation (GQA not enabled)";

    CHECK_GT(Bq, 0) << "batch size must be > 0";
    CHECK_GT(Tq, 0) << "query sequence length must be > 0";
    CHECK_GT(Tk, 0) << "key/value sequence length must be > 0";
    CHECK_GT(Hq, 0) << "head count must be > 0";
    CHECK_GT(Dq, 0) << "head_dim must be > 0";

    const double actual_scale = ResolveScale(query);
    const auto device = query->GetDevice().type();

    // Dispatcher contract for FlashAttention forward:
    // inputs:
    //   query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    //
    // returns:
    //   outputs[0] = out
    //   outputs[1] = softmax_lse
    auto outputs = Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {device, "ScaledDotProductAttentionForward"},
        query,
        key,
        value,
        attn_mask,
        dropout_p_,
        is_causal_,
        actual_scale,
        enable_gqa_);

    CHECK_GE(outputs.size(), 2)
        << "ScaledDotProductAttentionForward should return {out, softmax_lse}";
    CHECK(outputs[0] != nullptr)
        << "ScaledDotProductAttentionForward output[0] (out) must not be nullptr";
    CHECK(outputs[1] != nullptr)
        << "ScaledDotProductAttentionForward output[1] (softmax_lse) must not be nullptr";

    forward_softmax_lse_ = outputs[1];

    // Expose only the real attention output to autograd graph.
    return {outputs[0]};
}

void ScaledDotProductAttention::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors,
    const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    CHECK(input_tensors.size() == 3 || input_tensors.size() == 4)
        << "ScaledDotProductAttention expects 3 or 4 input tensors";
    CHECK_EQ(output_tensors.size(), 1)
        << "ScaledDotProductAttention forward should expose exactly 1 output tensor";

    const auto &query = input_tensors[0];
    const auto &key = input_tensors[1];
    const auto &value = input_tensors[2];

    CHECK(query != nullptr) << "query must not be nullptr";
    CHECK(key != nullptr) << "key must not be nullptr";
    CHECK(value != nullptr) << "value must not be nullptr";

    std::shared_ptr<Tensor> attn_mask = nullptr;
    if (input_tensors.size() == 4) {
        attn_mask = input_tensors[3];
    }

    const auto &out = output_tensors[0];
    CHECK(out != nullptr) << "attention output must not be nullptr";

    const auto &softmax_lse = forward_softmax_lse_;
    CHECK(softmax_lse != nullptr)
        << "forward_softmax_lse_ must not be nullptr after forward";

    has_attn_mask_ = (attn_mask != nullptr);
    has_softmax_lse_ = true;
    actual_scale_ = ResolveScale(query);

    // Fixed saved tensor layout:
    // [0] query
    // [1] key
    // [2] value
    // [3] attn_mask   (may be nullptr)
    // [4] out
    // [5] softmax_lse
    saved_tensors_ = {query, key, value, attn_mask, out, softmax_lse};
}

std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttention::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1)
        << "ScaledDotProductAttention backward expects 1 grad output";

    const auto &grad_out = grad_outputs[0];
    CHECK(grad_out != nullptr) << "grad_output must not be nullptr";

    CHECK_EQ(saved_tensors_.size(), 6)
        << "ScaledDotProductAttention saved_tensors_ size must be 6";

    const auto &query = saved_tensors_[0];
    const auto &key = saved_tensors_[1];
    const auto &value = saved_tensors_[2];
    const auto &attn_mask = saved_tensors_[3];
    const auto &out = saved_tensors_[4];
    const auto &softmax_lse = saved_tensors_[5];

    CHECK(query != nullptr) << "saved query must not be nullptr";
    CHECK(key != nullptr) << "saved key must not be nullptr";
    CHECK(value != nullptr) << "saved value must not be nullptr";
    CHECK(out != nullptr) << "saved out must not be nullptr";

    CHECK(has_softmax_lse_)
        << "FlashAttention backward expects softmax_lse saved from forward";
    CHECK(softmax_lse != nullptr)
        << "saved softmax_lse must not be nullptr";

    const auto device = query->GetDevice().type();

    // Dispatcher contract for FlashAttention backward:
    // inputs:
    //   grad_out, query, key, value, attn_mask, out, softmax_lse,
    //   dropout_p, is_causal, actual_scale, enable_gqa
    //
    // returns:
    //   grad_query, grad_key, grad_value, grad_attn_mask
    auto [grad_query, grad_key, grad_value, grad_attn_mask] =
        Dispatcher::Instance().Call<
            std::tuple<std::shared_ptr<Tensor>,
                       std::shared_ptr<Tensor>,
                       std::shared_ptr<Tensor>,
                       std::shared_ptr<Tensor>>>(
            {device, "ScaledDotProductAttentionBackward"},
            grad_out,
            query,
            key,
            value,
            attn_mask,
            out,
            softmax_lse,
            dropout_p_,
            is_causal_,
            actual_scale_,
            enable_gqa_);

    CHECK(grad_query != nullptr) << "grad_query must not be nullptr";
    CHECK(grad_key != nullptr) << "grad_key must not be nullptr";
    CHECK(grad_value != nullptr) << "grad_value must not be nullptr";

    if (has_attn_mask_) {
        return {grad_query, grad_key, grad_value, grad_attn_mask};
    }
    return {grad_query, grad_key, grad_value};
}

} // namespace infini_train::autograd
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

namespace {

struct DiffStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float rmse = 0.0f;
    float ref_max_abs = 0.0f;
};

struct RefResult {
    std::vector<float> out;
    std::vector<float> lse;
};

struct TestCase {
    std::string name;
    int64_t B;
    int64_t Hq;
    int64_t Hkv;
    int64_t T;
    int64_t D;
    bool is_causal;
    DataType dtype;
    bool enable_gqa;
    float out_atol;
    float out_rtol;
    float lse_atol;
    float lse_rtol;
};

struct BenchCase {
    std::string name;
    int64_t B;
    int64_t H;
    int64_t T;
    int64_t D;
    DataType dtype;
    bool is_causal;
    int iters;
    int warmup;
};

struct GradCase {
    std::string name;
    int64_t B;
    int64_t H;
    int64_t T;
    int64_t D;
    bool is_causal;
    DataType dtype;
    float dq_atol;
    float dq_rtol;
    float dk_atol;
    float dk_rtol;
    float dv_atol;
    float dv_rtol;
};

std::vector<float> RandomHostData(size_t numel, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> data(numel);
    for (size_t i = 0; i < numel; ++i) { data[i] = dis(gen); }
    return data;
}

std::vector<float> TensorToHostFloat(const std::shared_ptr<Tensor> &tensor) {
    Tensor fp32 = tensor->To(DataType::kFLOAT32);
    Tensor cpu = fp32.To(Device());

    if (fp32.GetDevice().IsCUDA()) {
        auto *impl = core::GetDeviceGuardImpl(fp32.GetDevice().type());
        impl->SynchronizeDevice(fp32.GetDevice());
    }

    std::vector<float> host(cpu.NumElements());
    std::memcpy(host.data(), cpu.DataPtr(), cpu.SizeInBytes());
    return host;
}

std::shared_ptr<Tensor> MakeTensorFromHost(const std::vector<float> &host, const std::vector<int64_t> &dims,
                                           const Device &dev, DataType dtype) {
    auto fp32 = std::make_shared<Tensor>(host.data(), dims, DataType::kFLOAT32, dev);
    if (dtype == DataType::kFLOAT32) {
        return fp32;
    }
    return std::make_shared<Tensor>(fp32->To(dtype));
}

std::shared_ptr<Tensor> BuildCausalMask(int64_t T, const Device &dev, DataType dtype) {
    std::vector<float> mask_host(static_cast<size_t>(T * T), 0.0f);
    for (int64_t i = 0; i < T; ++i) {
        for (int64_t j = i + 1; j < T; ++j) { mask_host[static_cast<size_t>(i * T + j)] = 1.0f; }
    }
    auto mask = std::make_shared<Tensor>(mask_host.data(), std::vector<int64_t>{1, 1, T, T}, DataType::kFLOAT32, dev);
    if (dtype == DataType::kFLOAT32) {
        return mask;
    }
    return std::make_shared<Tensor>(mask->To(dtype));
}

RefResult ReferenceAttentionCpu(const std::vector<float> &q, const std::vector<float> &k, const std::vector<float> &v,
                                int64_t B, int64_t Hq, int64_t Hkv, int64_t T, int64_t D, bool is_causal) {
    CHECK_EQ(Hq % Hkv, 0) << "Hq must be divisible by Hkv for GQA mapping";
    const int64_t n_rep = Hq / Hkv;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    RefResult ref;
    ref.out.resize(static_cast<size_t>(B * Hq * T * D), 0.0f);
    ref.lse.resize(static_cast<size_t>(B * Hq * T), 0.0f);

    auto q_idx = [Hq, T, D](int64_t b, int64_t h, int64_t t, int64_t d) {
        return static_cast<size_t>(((b * Hq + h) * T + t) * D + d);
    };
    auto kv_idx = [Hkv, T, D](int64_t b, int64_t h, int64_t t, int64_t d) {
        return static_cast<size_t>(((b * Hkv + h) * T + t) * D + d);
    };
    auto lse_idx = [Hq, T](int64_t b, int64_t h, int64_t t) { return static_cast<size_t>((b * Hq + h) * T + t); };

    std::vector<float> scores(static_cast<size_t>(T), 0.0f);

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hq = 0; hq < Hq; ++hq) {
            const int64_t hkv = hq / n_rep;
            for (int64_t tq = 0; tq < T; ++tq) {
                float max_score = -std::numeric_limits<float>::infinity();
                for (int64_t tk = 0; tk < T; ++tk) {
                    if (is_causal && tk > tq) {
                        scores[static_cast<size_t>(tk)] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    float dot = 0.0f;
                    for (int64_t d = 0; d < D; ++d) { dot += q[q_idx(b, hq, tq, d)] * k[kv_idx(b, hkv, tk, d)]; }
                    const float s = dot * scale;
                    scores[static_cast<size_t>(tk)] = s;
                    max_score = std::max(max_score, s);
                }

                float sum_exp = 0.0f;
                for (int64_t tk = 0; tk < T; ++tk) {
                    const float e = std::exp(scores[static_cast<size_t>(tk)] - max_score);
                    scores[static_cast<size_t>(tk)] = e;
                    sum_exp += e;
                }
                ref.lse[lse_idx(b, hq, tq)] = max_score + std::log(sum_exp);

                for (int64_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int64_t tk = 0; tk < T; ++tk) {
                        const float p = scores[static_cast<size_t>(tk)] / sum_exp;
                        acc += p * v[kv_idx(b, hkv, tk, d)];
                    }
                    ref.out[q_idx(b, hq, tq, d)] = acc;
                }
            }
        }
    }

    return ref;
}

DiffStats ComputeDiffStats(const std::vector<float> &test, const std::vector<float> &ref) {
    CHECK_EQ(test.size(), ref.size());
    DiffStats s;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < test.size(); ++i) {
        const float abs_diff = std::fabs(test[i] - ref[i]);
        s.max_abs = std::max(s.max_abs, abs_diff);
        s.ref_max_abs = std::max(s.ref_max_abs, std::fabs(ref[i]));
        sum_abs += static_cast<double>(abs_diff);
        sum_sq += static_cast<double>(abs_diff) * static_cast<double>(abs_diff);
    }
    s.mean_abs = static_cast<float>(sum_abs / static_cast<double>(test.size()));
    s.rmse = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(test.size())));
    return s;
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
CallFusedForwardRaw(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                    const std::shared_ptr<Tensor> &v, bool is_causal, double scale, bool enable_gqa) {
    auto outputs = Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {q->GetDevice().type(), "ScaledDotProductAttentionForward"}, q, k, v,
        /*attn_mask=*/std::shared_ptr<Tensor>(nullptr),
        /*dropout_p=*/0.0, is_causal, scale, enable_gqa);
    CHECK_GE(outputs.size(), 2);
    return {outputs[0], outputs[1]};
}

bool RunAccuracyCase(const TestCase &tc, uint32_t seed) {
    const Device cuda_dev(Device::DeviceType::kCUDA, 0);

    const std::vector<int64_t> q_dims{tc.B, tc.Hq, tc.T, tc.D};
    const std::vector<int64_t> kv_dims{tc.B, tc.Hkv, tc.T, tc.D};

    const size_t q_numel = static_cast<size_t>(tc.B * tc.Hq * tc.T * tc.D);
    const size_t kv_numel = static_cast<size_t>(tc.B * tc.Hkv * tc.T * tc.D);

    auto q_host = RandomHostData(q_numel, seed);
    auto k_host = RandomHostData(kv_numel, seed + 17);
    auto v_host = RandomHostData(kv_numel, seed + 23);

    auto q = MakeTensorFromHost(q_host, q_dims, cuda_dev, tc.dtype);
    auto k = MakeTensorFromHost(k_host, kv_dims, cuda_dev, tc.dtype);
    auto v = MakeTensorFromHost(v_host, kv_dims, cuda_dev, tc.dtype);

    core::GetDeviceGuardImpl(cuda_dev.type())->SynchronizeDevice(cuda_dev);

    const double scale = 1.0 / std::sqrt(static_cast<double>(tc.D));
    auto [out_raw, lse_raw] = CallFusedForwardRaw(q, k, v, tc.is_causal, scale, tc.enable_gqa);

    auto out_host = TensorToHostFloat(out_raw);
    auto lse_host = TensorToHostFloat(lse_raw);

    auto ref = ReferenceAttentionCpu(q_host, k_host, v_host, tc.B, tc.Hq, tc.Hkv, tc.T, tc.D, tc.is_causal);

    const auto out_stats = ComputeDiffStats(out_host, ref.out);
    const auto lse_stats = ComputeDiffStats(lse_host, ref.lse);

    const float out_threshold = tc.out_atol + tc.out_rtol * out_stats.ref_max_abs;
    const float lse_threshold = tc.lse_atol + tc.lse_rtol * lse_stats.ref_max_abs;

    const bool out_pass = out_stats.max_abs <= out_threshold;
    const bool lse_pass = lse_stats.max_abs <= lse_threshold;

    std::cout << std::fixed << std::setprecision(6) << "[case] " << tc.name
              << " | dtype=" << (tc.dtype == DataType::kBFLOAT16 ? "bf16" : "fp32") << " B=" << tc.B << " Hq=" << tc.Hq
              << " Hkv=" << tc.Hkv << " T=" << tc.T << " D=" << tc.D << " causal=" << (tc.is_causal ? "true" : "false")
              << " gqa=" << (tc.enable_gqa ? "true" : "false") << " | out.max_abs=" << out_stats.max_abs
              << " out.th=" << out_threshold << " | lse.max_abs=" << lse_stats.max_abs << " lse.th=" << lse_threshold
              << " => " << ((out_pass && lse_pass) ? "PASS" : "FAIL") << std::endl;

    return out_pass && lse_pass;
}

std::shared_ptr<Tensor> RunNaiveForward(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                        const std::shared_ptr<Tensor> &v, bool is_causal) {
    const int64_t D = q->Dims()[3];
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    auto att = q->Matmul(k->Transpose(-2, -1)) * scale;
    if (is_causal) {
        auto mask = BuildCausalMask(q->Dims()[2], q->GetDevice(), att->Dtype());
        att = att->MaskedFill(mask, std::numeric_limits<float>::lowest());
    }
    att = nn::function::Softmax(att, -1);
    return att->Matmul(v);
}

double BenchmarkLatencyMs(int iters, int warmup, int64_t B, int64_t H, int64_t T, int64_t D, DataType dtype,
                          bool is_causal) {
    const Device cuda_dev(Device::DeviceType::kCUDA, 0);
    const std::vector<int64_t> dims{B, H, T, D};
    const size_t numel = static_cast<size_t>(B * H * T * D);

    auto q_host = RandomHostData(numel, 7);
    auto k_host = RandomHostData(numel, 17);
    auto v_host = RandomHostData(numel, 23);

    auto q = MakeTensorFromHost(q_host, dims, cuda_dev, dtype);
    auto k = MakeTensorFromHost(k_host, dims, cuda_dev, dtype);
    auto v = MakeTensorFromHost(v_host, dims, cuda_dev, dtype);

    core::GetDeviceGuardImpl(cuda_dev.type())->SynchronizeDevice(cuda_dev);

    const double scale = 1.0 / std::sqrt(static_cast<double>(D));

    for (int i = 0; i < warmup; ++i) {
        auto [out, lse] = CallFusedForwardRaw(q, k, v, is_causal, scale, false);
        (void)out;
        (void)lse;
    }
    core::GetDeviceGuardImpl(cuda_dev.type())->SynchronizeDevice(cuda_dev);

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        auto [out, lse] = CallFusedForwardRaw(q, k, v, is_causal, scale, false);
        (void)out;
        (void)lse;
    }
    core::GetDeviceGuardImpl(cuda_dev.type())->SynchronizeDevice(cuda_dev);
    const auto t1 = std::chrono::high_resolution_clock::now();

    const auto total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return total_ms / static_cast<double>(iters);
}

double BenchmarkLatencyBaselineMs(int iters, int warmup, int64_t B, int64_t H, int64_t T, int64_t D, DataType dtype,
                                  bool is_causal) {
    const Device cuda_dev(Device::DeviceType::kCUDA, 0);
    const std::vector<int64_t> dims{B, H, T, D};
    const size_t numel = static_cast<size_t>(B * H * T * D);

    auto q_host = RandomHostData(numel, 37);
    auto k_host = RandomHostData(numel, 47);
    auto v_host = RandomHostData(numel, 53);

    auto q = MakeTensorFromHost(q_host, dims, cuda_dev, dtype);
    auto k = MakeTensorFromHost(k_host, dims, cuda_dev, dtype);
    auto v = MakeTensorFromHost(v_host, dims, cuda_dev, dtype);

    core::GetDeviceGuardImpl(cuda_dev.type())->SynchronizeDevice(cuda_dev);

    for (int i = 0; i < warmup; ++i) {
        auto out = RunNaiveForward(q, k, v, is_causal);
        (void)out;
    }
    core::GetDeviceGuardImpl(cuda_dev.type())->SynchronizeDevice(cuda_dev);

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        auto out = RunNaiveForward(q, k, v, is_causal);
        (void)out;
    }
    core::GetDeviceGuardImpl(cuda_dev.type())->SynchronizeDevice(cuda_dev);
    const auto t1 = std::chrono::high_resolution_clock::now();

    const auto total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return total_ms / static_cast<double>(iters);
}

bool RunBackwardCase(const GradCase &tc, uint32_t seed) {
    const Device cuda_dev(Device::DeviceType::kCUDA, 0);
    const std::vector<int64_t> dims{tc.B, tc.H, tc.T, tc.D};
    const size_t numel = static_cast<size_t>(tc.B * tc.H * tc.T * tc.D);

    auto q_host = RandomHostData(numel, seed);
    auto k_host = RandomHostData(numel, seed + 17);
    auto v_host = RandomHostData(numel, seed + 23);
    auto grad_out_host = RandomHostData(numel, seed + 29);

    auto q_fused = MakeTensorFromHost(q_host, dims, cuda_dev, tc.dtype)->RequiresGrad();
    auto k_fused = MakeTensorFromHost(k_host, dims, cuda_dev, tc.dtype)->RequiresGrad();
    auto v_fused = MakeTensorFromHost(v_host, dims, cuda_dev, tc.dtype)->RequiresGrad();
    auto grad_out_fused = MakeTensorFromHost(grad_out_host, dims, cuda_dev, tc.dtype);

    auto out_fused = nn::function::ScaledDotProductAttention(q_fused, k_fused, v_fused,
                                                             /*attn_mask=*/nullptr,
                                                             /*dropout_p=*/0.0, tc.is_causal, std::nullopt,
                                                             /*enable_gqa=*/false);
    out_fused->Backward(grad_out_fused);

    auto dq_fused = TensorToHostFloat(q_fused->grad());
    auto dk_fused = TensorToHostFloat(k_fused->grad());
    auto dv_fused = TensorToHostFloat(v_fused->grad());

    auto q_base = MakeTensorFromHost(q_host, dims, cuda_dev, tc.dtype)->RequiresGrad();
    auto k_base = MakeTensorFromHost(k_host, dims, cuda_dev, tc.dtype)->RequiresGrad();
    auto v_base = MakeTensorFromHost(v_host, dims, cuda_dev, tc.dtype)->RequiresGrad();
    auto grad_out_base = MakeTensorFromHost(grad_out_host, dims, cuda_dev, tc.dtype);

    auto out_base = RunNaiveForward(q_base, k_base, v_base, tc.is_causal);
    out_base->Backward(grad_out_base);

    auto dq_base = TensorToHostFloat(q_base->grad());
    auto dk_base = TensorToHostFloat(k_base->grad());
    auto dv_base = TensorToHostFloat(v_base->grad());

    const auto dq_stats = ComputeDiffStats(dq_fused, dq_base);
    const auto dk_stats = ComputeDiffStats(dk_fused, dk_base);
    const auto dv_stats = ComputeDiffStats(dv_fused, dv_base);

    const float dq_th = tc.dq_atol + tc.dq_rtol * dq_stats.ref_max_abs;
    const float dk_th = tc.dk_atol + tc.dk_rtol * dk_stats.ref_max_abs;
    const float dv_th = tc.dv_atol + tc.dv_rtol * dv_stats.ref_max_abs;

    const bool dq_pass = dq_stats.max_abs <= dq_th;
    const bool dk_pass = dk_stats.max_abs <= dk_th;
    const bool dv_pass = dv_stats.max_abs <= dv_th;
    const bool pass = dq_pass && dk_pass && dv_pass;

    std::cout << std::fixed << std::setprecision(6) << "[grad] " << tc.name
              << " | dtype=" << (tc.dtype == DataType::kBFLOAT16 ? "bf16" : "fp32") << " B=" << tc.B << " H=" << tc.H
              << " T=" << tc.T << " D=" << tc.D << " causal=" << (tc.is_causal ? "true" : "false")
              << " | dQ.max_abs=" << dq_stats.max_abs << " dQ.th=" << dq_th << " | dK.max_abs=" << dk_stats.max_abs
              << " dK.th=" << dk_th << " | dV.max_abs=" << dv_stats.max_abs << " dV.th=" << dv_th << " => "
              << (pass ? "PASS" : "FAIL") << std::endl;

    return pass;
}
} // namespace

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

#ifndef USE_CUDA
    LOG(FATAL) << "This test requires CUDA build. Rebuild with -DUSE_CUDA=ON.";
    return 1;
#else
    nn::parallel::global::InitAllEnv(1, 1, false, 1, 1);

    const std::vector<TestCase> cases = {
        {"fp32_noncausal_t16_h4", 2, 4, 4, 16, 64, false, DataType::kFLOAT32, false, 3e-4f, 3e-4f, 3e-4f, 3e-4f},
        {"fp32_causal_t16_h4", 2, 4, 4, 16, 64, true, DataType::kFLOAT32, false, 5e-4f, 5e-4f, 5e-4f, 5e-4f},
        {"fp32_noncausal_t33_h4", 2, 4, 4, 33, 64, false, DataType::kFLOAT32, false, 5e-4f, 5e-4f, 5e-4f, 5e-4f},
        {"fp32_causal_t33_h4", 2, 4, 4, 33, 64, true, DataType::kFLOAT32, false, 7e-4f, 7e-4f, 7e-4f, 7e-4f},
        {"fp32_noncausal_t64_h8", 1, 8, 8, 64, 64, false, DataType::kFLOAT32, false, 8e-4f, 8e-4f, 8e-4f, 8e-4f},
        {"fp32_causal_t64_h8", 1, 8, 8, 64, 64, true, DataType::kFLOAT32, false, 1e-3f, 1e-3f, 1e-3f, 1e-3f},

        {"bf16_noncausal_t16_h4", 2, 4, 4, 16, 64, false, DataType::kBFLOAT16, false, 4e-2f, 2e-2f, 2e-2f, 2e-2f},
        {"bf16_causal_t16_h4", 2, 4, 4, 16, 64, true, DataType::kBFLOAT16, false, 5e-2f, 2e-2f, 3e-2f, 2e-2f},
        {"bf16_noncausal_t33_h4", 2, 4, 4, 33, 64, false, DataType::kBFLOAT16, false, 6e-2f, 2e-2f, 4e-2f, 2e-2f},
        {"bf16_causal_t33_h4", 2, 4, 4, 33, 64, true, DataType::kBFLOAT16, false, 7e-2f, 2e-2f, 5e-2f, 2e-2f},
        {"bf16_noncausal_t64_h8", 1, 8, 8, 64, 64, false, DataType::kBFLOAT16, false, 8e-2f, 2e-2f, 5e-2f, 2e-2f},
        {"bf16_causal_t64_h8", 1, 8, 8, 64, 64, true, DataType::kBFLOAT16, false, 9e-2f, 2e-2f, 6e-2f, 2e-2f},

        {"gqa_fp32_noncausal_h8_h2_t16", 2, 8, 2, 16, 64, false, DataType::kFLOAT32, true, 5e-4f, 5e-4f, 5e-4f, 5e-4f},
        {"gqa_fp32_causal_h8_h2_t16", 2, 8, 2, 16, 64, true, DataType::kFLOAT32, true, 7e-4f, 7e-4f, 7e-4f, 7e-4f},
        {"gqa_fp32_noncausal_h12_h3_t33", 1, 12, 3, 33, 64, false, DataType::kFLOAT32, true, 8e-4f, 8e-4f, 8e-4f,
         8e-4f},
        {"gqa_fp32_causal_h12_h3_t33", 1, 12, 3, 33, 64, true, DataType::kFLOAT32, true, 1e-3f, 1e-3f, 1e-3f, 1e-3f},
    };

    bool all_pass = true;
    for (size_t i = 0; i < cases.size(); ++i) {
        all_pass = RunAccuracyCase(cases[i], 20260315u + static_cast<uint32_t>(i) * 97u) && all_pass;
    }

    const std::vector<GradCase> grad_cases = {
        {"fp32_noncausal_t16_h4", 2, 4, 16, 64, false, DataType::kFLOAT32, 8e-4f, 8e-4f, 8e-4f, 8e-4f, 8e-4f, 8e-4f},
        {"fp32_causal_t16_h4", 2, 4, 16, 64, true, DataType::kFLOAT32, 1e-3f, 1e-3f, 1e-3f, 1e-3f, 1e-3f, 1e-3f},
        {"fp32_noncausal_t33_h4", 2, 4, 33, 64, false, DataType::kFLOAT32, 1.5e-3f, 1.5e-3f, 1.5e-3f, 1.5e-3f, 1.5e-3f,
         1.5e-3f},
        {"fp32_causal_t33_h4", 2, 4, 33, 64, true, DataType::kFLOAT32, 2e-3f, 2e-3f, 2e-3f, 2e-3f, 2e-3f, 2e-3f},
        {"fp32_noncausal_t64_h8", 1, 8, 64, 64, false, DataType::kFLOAT32, 2e-3f, 2e-3f, 2e-3f, 2e-3f, 2e-3f, 2e-3f},
        {"fp32_causal_t64_h8", 1, 8, 64, 64, true, DataType::kFLOAT32, 3e-3f, 3e-3f, 3e-3f, 3e-3f, 3e-3f, 3e-3f},
    };

    for (size_t i = 0; i < grad_cases.size(); ++i) {
        all_pass = RunBackwardCase(grad_cases[i], 20270315u + static_cast<uint32_t>(i) * 131u) && all_pass;
    }

    const std::vector<BenchCase> bench_cases = {
        {"fp32_noncausal", 4, 8, 128, 64, DataType::kFLOAT32, false, 100, 20},
        {"fp32_causal", 4, 8, 128, 64, DataType::kFLOAT32, true, 100, 20},
        {"bf16_noncausal", 4, 8, 128, 64, DataType::kBFLOAT16, false, 100, 20},
    };

    std::cout << std::fixed << std::setprecision(3);
    for (const auto &bc : bench_cases) {
        const double fused = BenchmarkLatencyMs(bc.iters, bc.warmup, bc.B, bc.H, bc.T, bc.D, bc.dtype, bc.is_causal);
        const double baseline
            = BenchmarkLatencyBaselineMs(bc.iters, bc.warmup, bc.B, bc.H, bc.T, bc.D, bc.dtype, bc.is_causal);
        const double speedup = baseline > 0.0 ? (baseline / fused) : 0.0;
        std::cout << "[bench] " << bc.name << " B=" << bc.B << " H=" << bc.H << " T=" << bc.T << " D=" << bc.D
                  << " | fused=" << fused << " ms/iter"
                  << " baseline=" << baseline << " ms/iter"
                  << " speedup=" << speedup << "x" << std::endl;
    }

    if (!all_pass) {
        std::cerr << "Attention forward/backward alignment FAILED." << std::endl;
        return 2;
    }

    std::cout << "Attention forward/backward alignment PASSED." << std::endl;
    return 0;
#endif
}

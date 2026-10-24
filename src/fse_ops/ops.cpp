#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

torch::Tensor median_filter5x5(const torch::Tensor& x) {
    TORCH_CHECK(x.device().is_cpu(),
                "median_filter5x5: input must be a CPU tensor");
    TORCH_CHECK(x.layout() == at::kStrided,
                "median_filter5x5: input must have strided layout");
    TORCH_CHECK(x.scalar_type() == at::kFloat,
                "median_filter5x5: input must have dtype float32");
    TORCH_CHECK(x.dim() >= 2,
                "median_filter5x5: input must have at least two dimensions");

    const int64_t height = x.size(-2);
    const int64_t width = x.size(-1);
    TORCH_CHECK(height >= 5 && width >= 5,
                "median_filter5x5: H and W must both be at least 5");

    const int64_t out_height = height - 4;
    const int64_t out_width = width - 4;
    std::vector<int64_t> out_sizes = x.sizes().vec();
    out_sizes[out_sizes.size() - 2] = out_height;
    out_sizes.back() = out_width;
    auto output = torch::empty(out_sizes, x.options().requires_grad(false));

    // Empty leading dimensions are valid; no storage needs to be accessed.
    if (output.numel() == 0) {
        return output;
    }

    // Resolve lazy negation and flatten all leading dimensions into images.
    const auto input = x.resolve_neg().contiguous();
    const float* input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    const int64_t image_size = height * width;
    const int64_t row_count = output.numel() / out_width;

    at::parallel_for(0, row_count, 16, [&](int64_t begin, int64_t end) {
        std::array<float, 25> window;
        for (int64_t row = begin; row < end; ++row) {
            const int64_t image = row / out_height;
            const int64_t y = row % out_height;
            const float* input_row = input_data + image * image_size + y * width;
            float* output_row = output_data + row * out_width;

            for (int64_t col = 0; col < out_width; ++col) {
                bool has_nan = false;
                for (int64_t dy = 0; dy < 5; ++dy) {
                    const float* source = input_row + dy * width + col;
                    for (int64_t dx = 0; dx < 5; ++dx) {
                        const float value = source[dx];
                        window[dy * 5 + dx] = value;
                        has_nan = has_nan || std::isnan(value);
                    }
                }

                // Propagate NaNs before using the ordinary float ordering.
                if (has_nan) {
                    output_row[col] = std::numeric_limits<float>::quiet_NaN();
                } else {
                    std::nth_element(window.begin(), window.begin() + 12,
                                     window.end());
                    output_row[col] = window[12];
                }
            }
        }
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("median_filter5x5", &median_filter5x5,
          "5x5 median filter on CPU float32 tensors (stride 1, no padding)");
}

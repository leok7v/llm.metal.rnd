/*
 MIT License

 Copyright (c) 2024 Andrej Karpathy
 Copyright (c) 2024 James Thompson

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#include "llm_cpu.h"
#include "metal_compute.h"
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK_TENSORS 0

const float infinity = (__builtin_inff()); // Use built-in infinity to avoid warnings

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

void logFloats(float *a, size_t len) {
  for (int i = 0; i < len; i++) {
    if (i != len - 1) {
      // Only log to the hundredth place
      printf("%.03f, ", a[i]);
    } else {
      printf("%.03f\n", a[i]);
    }
  }
}

void logStridedFloats(float *a, size_t offset, size_t len, size_t stride) {
  for (size_t i = offset; i < offset + len; i += stride) {
    if (i != len - 1) {
      printf("%.03f, ", a[i]);
    } else {
      printf("%.03f\n", a[i]);
    }
  }
}

void logInts(int *a, size_t len) {
  for (int i = 0; i < len; i++) {
    if (i != len - 1) {
      printf("%d, ", a[i]);
    } else {
      printf("%d\n", a[i]);
    }
  }
}

// poor man's tensor checker
int check_tensor(float *a, float *b, size_t n, char *label) {
  int print_upto = 5;
  int ok = 1;
  printf("%s\n", label);
  for (int i = 0; i < n; i++) {
    if (fabsf(a[i] - b[i]) <= 1e-2 ||
        (a[i] == -infinity && b[i] == -infinity)) {
      if (i < print_upto) {
        printf("OK ");
      }
    } else {
      if (i < print_upto) {
        printf("NOT OK ");
      }
      ok = 0;
    }
    if (i < print_upto) {
      printf("%f %f\n", a[i], b[i]);
    }
  }
  // print the final result
  if (ok) {
    printf("TENSOR OK\n");
  } else {
    printf("TENSOR NOT OK\n");
  }
  return ok;
}

void tensor_stats(float *a, size_t n) {
  size_t count = 0;
  size_t infCount = 0;
  size_t nonZeroCount = 0;
  size_t firstZero = 0;
  float minVal = 100000.0;
  float maxVal = 0.0f;
  double meanVal = 0.0f;
  for (size_t i = 0; i < n; i++) {
    if (isnan(a[i])) {
      count++;
    }
    if (isinf(a[i])) {
      infCount++;
    }
    if (!firstZero && a[i] == 0.0) {
      firstZero = i;
    }
    if (a[i] != 0.0) {
      nonZeroCount++;
    }
    minVal = min(minVal, a[i]);
    maxVal = max(maxVal, a[i]);
    meanVal += a[i] / n;
  }
  printf("NaNs: %zu \n"
         "Non-zero count: %zu\n"
         "First zero idx: %zu\n"
         "Infinities: %zu\n"
         "Min: %f\n"
         "Max: %f\n"
         "Mean: %f\n\n",
         count, nonZeroCount, firstZero, infCount, minVal, maxVal, meanVal);
}

// ----------------------------------------------------------------------------
// kernel launchers
void encoder_forward_kernel2(int grid_size, int block_size, float *out,
                             int *inp, float *wte, float *wpe, int B, int T,
                             int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 7, Buffer, out, Buffer,
               inp, Buffer, wte, Buffer, wpe, Scalar, &B, Scalar, &T, Scalar,
               &C);
}

void mean_kernel(int grid_size, int block_size, int shared_size, float *mean,
                 float *inp, int N, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, shared_size, 4, Buffer,
               mean, Buffer, inp, Scalar, &N, Scalar, &C);
}

void rstd_kernel(int grid_size, int block_size, int shared_size, float *rstd,
                 float *inp, float *mean, int N, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, shared_size, 5, Buffer,
               rstd, Buffer, inp, Buffer, mean, Scalar, &N, Scalar, &C);
}

void normalization_kernel(int grid_size, int block_size, float *out, float *inp,
                          float *mean, float *rstd, float *weight, float *bias,
                          int B, int T, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 9, Buffer, out, Buffer,
               inp, Buffer, mean, Buffer, rstd, Buffer, weight, Buffer, bias,
               Scalar, &B, Scalar, &T, Scalar, &C);
}

void add_bias_kernel(int grid_size, int block_size, float *out, float *bias,
                     int OC) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 3, Buffer, out, Buffer,
               bias, Scalar, &OC);
}

void permute_kernel(int grid_size, int block_size, float *q, float *k, float *v,
                    float *inp, int B, int N, int NH, int d) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 8, Buffer, q, Buffer, k,
               Buffer, v, Buffer, inp, Scalar, &B, Scalar, &N, Scalar, &NH,
               Scalar, &d);
}

void scale_kernel(int grid_size, int block_size, float *preatt, float scale,
                  int B, int NH, int T) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 5, Buffer, preatt,
               Scalar, &scale, Scalar, &B, Scalar, &NH, Scalar, &T);
}

void softmax_forward_kernel1(int grid_size, int block_size, float *att,
                             float *preatt, int N, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 4, Buffer, att, Buffer,
               preatt, Scalar, &N, Scalar, &C);
}

/*
void max_kernel(int grid_size, int block_size, size_t shared_size,
                   float *max_out, float *inp, int N, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, shared_size, 4, 
               Buffer, max_out, Buffer, inp, Scalar, &N, Scalar, &C);
}

void sum_kernel(int grid_size, int block_size, size_t shared_size,
                   float *max_out, float *inp, float *maxs, int N, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, shared_size, 5, 
               Buffer, max_out, Buffer, inp, Buffer, maxs, Scalar, &N, Scalar, &C);
}

void softmax_forward_kernel2(int grid_size, int block_size, size_t shared_size,
                                 float *out, float *inp, float *maxs, float *sums, int N, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, shared_size, 6, 
               Buffer, out, Buffer, inp, Buffer, maxs, Buffer, sums, Scalar, &N, Scalar, &C);
}
*/

void unpermute_kernel(int grid_size, int block_size, float *inp, float *out,
                      int B, int T, int NH, int HS) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 6, Buffer, inp, Buffer,
               out, Scalar, &B, Scalar, &T, Scalar, &NH, Scalar, &HS);
}

void residual_forward_kernel(int grid_size, int block_size, float *out,
                             float *inp1, float *inp2) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 3, Buffer, out, Buffer,
               inp1, Buffer, inp2);
}

void gelu_kernel(int grid_size, int block_size, float *out, float *inp) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 2, Buffer, out, Buffer,
               inp);
}

void crossentropy_forward_kernel1(int grid_size, int block_size, float *losses,
                                  float *probs, int *targets, int B, int T,
                                  int V) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 5, Buffer, losses,
               Buffer, probs, Buffer, targets, Scalar, &T, Scalar, &V);
}

void encoder_forward(float *out, int *inp, float *wte, float *wpe, int B, int T,
                     int C) {
  const int N = B * T * C;
  const int block_size = 512;
  const int grid_size = N;
  encoder_forward_kernel2(grid_size, block_size, out, inp, wte, wpe, B, T, C);
#if CHECK_TENSORS
  metalCommitCommandsAndWait();
  size_t gSz = B * T * C;
  float *g = malloc(sizeof(float) * gSz);
  memcpy(g, out, sizeof(float) * gSz);
  cpu_encoder_forward(out, inp, wte, wpe, B, T, C);
  check_tensor(out, g, gSz, "encoder_forward");
#endif
}

void layernorm_forward(float *out, float *mean, float *rstd, float *inp,
                       float *weight, float *bias, int B, int T, int C) {
  int N = B * T;
  const int block_size = 512;
  // in mean and rstd, threads cooperate within blocks via reductions
  mean_kernel(B * T * C, block_size, block_size * sizeof(float), mean, inp, N,
              C);
  rstd_kernel(B * T * C, block_size, block_size * sizeof(float), rstd, inp,
              mean, N, C);
  // in the normalization, everything just gets flattened out
  const int block_size2 = 128;
  const int grid_size = B * T * C;
  normalization_kernel(grid_size, block_size2, out, inp, mean, rstd, weight,
                       bias, B, T, C);
#if CHECK_TENSORS
  metalCommitCommandsAndWait();
  size_t gSz = B * T * C;
  size_t fSz = sizeof(float);
  float *g = malloc(fSz * gSz);
  memcpy(g, out, fSz * gSz);
  cpu_layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C);
  check_tensor(out, g, gSz, "layer_norm1");
#endif
}

// kernel 1 is the most naive matmul kernel
void matmul_forward(float *out, float *inp, float *weight, float *bias, int B,
                    int T, int C, int OC) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // inp is (B,T,C), weight is (OC, C), bias is (OC)
  // out will be (B,T,OC)
  // [m k] by [k n]
  metalCheck(metalSgemmBatched(false, true, B * T, C, OC, C, inp, weight, out,
                               1, alpha, beta));

  // and now we still have to add the bias... (ew)
  if (bias != NULL) {
    int block_size = 128;
    int grid_size = OC * B * T;
    add_bias_kernel(grid_size, block_size, out, bias, OC);
  }
#if CHECK_TENSORS
  metalCommitCommandsAndWait();
  size_t gSz = B * T * OC;
  size_t fSz = sizeof(float);
  float *g = malloc(fSz * gSz);
  memcpy(g, out, fSz * gSz);
  cpu_matmul_forward(out, inp, weight, bias, B, T, C, OC);
  check_tensor(out, g, gSz, "matmul");
#endif
}

void softmax_forward_1(float *out, float *inp, int B, int T, int V) {
  const int block_size = 128;
  //  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  softmax_forward_kernel1(B * T, block_size, out, inp, B * T, V);
#if CHECK_TENSORS
  metalCommitCommandsAndWait();
  size_t gSz = B * T;
  size_t fSz = sizeof(float);
  float *g = malloc(fSz * gSz);
  memcpy(g, out, fSz * gSz);
  cpu_softmax_forward(out, inp, B, T, V);
  check_tensor(out, g, gSz, "softmax");
#endif
}

/*
void softmax_forward_2(float *out, float *inp, int B, int T, int V) {
  const int block_size = 256;
  int N = B * T;
  float *maxs = NULL;
  float *sums = NULL;
  metalCheck(metalMalloc((void**)&maxs, N * sizeof(float)));
  metalCheck(metalMalloc((void**)&sums, N * sizeof(float)));
  max_kernel(N * V, block_size, block_size * sizeof(float), maxs, inp, N, V);
  sum_kernel(N * V, block_size, block_size * sizeof(float), sums, inp, maxs, N, V);
  softmax_forward_kernel2(N * V, block_size, block_size * sizeof(float), out, inp, maxs, sums, N, V);
  metalFree(maxs);
  metalFree(sums);
}
*/

#define USE_SOFTMAX_LSE
#undef  USE_SOFTMAX_LSE

#ifdef USE_SOFTMAX_LSE
#define softmax_forward softmax_forward_2
#else
#define softmax_forward softmax_forward_1
#endif

void attention_forward(float *out, float *vaccum, float *qkvr, float *preatt,
                       float *att, float *inp, int B, int T, int C, int NH) {
  const int block_size = 128;
  int HS = C / NH; // head size
  // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
  float *q = qkvr + 0 * B * T * C;
  float *k = qkvr + 1 * B * T * C;
  float *v = qkvr + 2 * B * T * C;
  permute_kernel(B * NH * T * HS, block_size, q, k, v, inp, B, T, NH, HS);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // (B, NH, T, HS) • (B, NH, T, HS)^T -> (B, NH, T, T)
  metalCheck(metalSgemmBatched(false, true, T, HS, T, HS, q, k, preatt, B * NH,
                               alpha, beta));
  // multiply all elements of preatt elementwise by scale
  const float scale = 1.0 / sqrtf(HS);
  scale_kernel(B * NH * T * T, block_size, preatt, scale, B, NH, T);
  softmax_forward_1(att, preatt, B * NH, T, T);
//softmax_forward_two_pass(att, preatt, B * NH, T, T);
  // v^T • att^T or (B, NH, T, HS)^T • (B, NH, T, T)^T -> (B, NH, HS, T)
  metalCheck(metalSgemmBatched(true, true, T, HS, T, T, v, att, vaccum, B * NH,
                               alpha, beta));
  // re-assemble all head outputs side by side
  // permute B, NH, HS, T ->  B, T, NH, HS
  unpermute_kernel(B * NH * HS * T, block_size, vaccum, out, B, T, NH, HS);
}

void residual_forward(float *out, float *inp1, float *inp2, int N) {
  const int block_size = 128;
  residual_forward_kernel(N, block_size, out, inp1, inp2);
#if CHECK_TENSORS
  metalCommitCommandsAndWait();
  size_t gSz = N;
  size_t fSz = sizeof(float);
  float *g = malloc(fSz * gSz);
  memcpy(g, out, fSz * gSz);
  cpu_residual_forward(out, inp1, inp2, N);
  check_tensor(out, g, gSz, "residual");
#endif
}

void gelu_forward(float *out, float *inp, int N) {
  const int block_size = 256;
  gelu_kernel(N, block_size, out, inp);
#if CHECK_TENSORS
  metalCommitCommandsAndWait();
  size_t gSz = N;
  size_t fSz = sizeof(float);
  float *g = malloc(fSz * gSz);
  memcpy(g, out, fSz * gSz);
  cpu_gelu_forward(out, inp, N);
  check_tensor(out, g, gSz, "gelu");
#endif
}

#if 0
// Function to compare precision between kernel1 and kernel_2
void compare_softmax_implementations(float *inp, int B, int T, int V) {
  size_t out_size = B * T * V * sizeof(float);
  float *out1, *out2, *copy;
  
  metalCheck(metalMalloc((void**)&out1, out_size));
  metalCheck(metalMalloc((void**)&copy, out_size));
  metalCheck(metalMalloc((void**)&out2, out_size));
  // sentinel values to ensure we can detect memory writes
  for (size_t i = 0; i < B * T * V; i++) {
    out1[i] = -1024.0f;
    out2[i] = -1024.0f;
    copy[i] = -1024.0f;
  }
  printf("Comparing softmax implementations:\n");
  
  softmax_forward_1(out1, inp, B, T, V);
  metalCommitCommandsAndWait();
  size_t written1 = 0;  
  for (size_t i = 0; i < B * T * V; i++) {
    if (out1[i] != -1024.0f) { written1++; } 
  }
  softmax_forward_2(out2, inp, B, T, V);
  metalCommitCommandsAndWait();
  size_t written2 = 0;
  for (size_t i = 0; i < B * T * V; i++) {
    if (out2[i] != -1024.0f) { written2++; } 
  }
  printf("Kernel1 wrote %zu elements, Kernel2 wrote %zu elements\n", written1, written2);  
  // Compare results
  double max_diff = 0.0;
  double avg_diff = 0.0;
  size_t total_elements = B * T * V;
  size_t nan_count = 0;
  size_t inf_count = 0;
  size_t zero_count_1 = 0;
  size_t zero_count_lse = 0;
  printf("Total elements: %zu = B: %u * T: %u * V: %u\n", total_elements, B, T, V);

  for (size_t i = 0; i < total_elements; i++) {
    if (isnan(out2[i])) nan_count++;
    if (isinf(out2[i])) inf_count++;
    if (out1[i] == 0.0f) zero_count_1++;
    if (out2[i] == 0.0f) zero_count_lse++;
    
    double diff = fabs(out1[i] - out2[i]);
    max_diff = fmax(max_diff, diff);
    avg_diff += diff;
  }
  avg_diff /= total_elements;
  
  printf("Max difference: %.2e\n", max_diff);
  printf("Average difference: %.2e\n", avg_diff);
  printf("Relative max difference: %.2e\n", max_diff / (fabs(out1[0]) + 1e-10));
  printf("NaN count in Kernel2: %zu\n", nan_count);
  printf("Inf count in Kernel2: %zu\n", inf_count);
  printf("Zero count kernel1: %zu, kernel2: %zu\n", zero_count_1, zero_count_lse);
  
  // Print first few values for manual inspection
  printf("First 5 values comparison:\n");
  printf("Index | Kernel1        | Kernel2        | Difference\n");
  printf("------|----------------|----------------|------------\n");
  for (int i = 0; i < 5 && i < total_elements; i++) {
    printf("%5d | %14.10f | %14.10f | %.2e\n", i, out1[i], out2[i], fabs(out1[i] - out2[i]));
  }
  int wx[3];
  memset(wx, 0xFF, sizeof(wx)); // Initialize to -1 (invalid index)
  printf("\nWorst %zu differences:\n", sizeof(wx) / sizeof(wx[0]));
  printf("Index    | Kernel1        | Kernel2        | Difference | Row | Col\n");
  printf("---------|----------------|----------------|------------|-----|----\n");
  memcpy(copy, out1, out_size);
  for (int worst = 0; worst < sizeof(wx) / sizeof(wx[0]); worst++) {
    double worst_diff = 0.0;
    size_t worst_idx = 0;
    for (size_t i = 0; i < total_elements; i++) {
      double diff = fabs(copy[i] - out2[i]);
      if (diff > worst_diff) {
        worst_diff = diff;
        worst_idx = i;
      }
    }
    
    if (worst_diff > 0) {
      // Calculate row and column for this index
      size_t row = worst_idx / V;
      size_t col = worst_idx % V;
      printf("%8zu | %14.10f | %14.10f | %.2e   | %3zu | %3zu\n", 
             worst_idx, out1[worst_idx], out2[worst_idx], worst_diff, row, col);
      // Zero out this element so we can find the next worst
      copy[worst_idx] = out2[worst_idx];
      wx[worst] = worst_idx; // Store the index of the worst difference
    }
  }
  for (int j = 0; j < sizeof(wx) / sizeof(wx[0]) && wx[j] >= 0; j++) {
    size_t idx = wx[j];
    size_t row = idx / V;
    size_t col = idx % V;
    printf("Worst %d: Index %zu, Row %zu, Col %zu\nkernel1: %.10f kernel2: %.10f  diff: %.2e\n",
           j + 1, idx, row, col, out1[idx], out2[idx], fabs(out1[idx] - out2[idx]));
    float max_in_row = -infinity;
    printf("row: %zu col: %zu: %.10f, max in row=", row, col, inp[row * V + col]);
    for (int i = 0; i < V; i++) {
      if (inp[row * V + i] > max_in_row) max_in_row = inp[row * V + i];
    }
    printf("%.10f\n", max_in_row);
    float manual_sum = 0.0f;
    for (int i = 0; i < V; i++) {
      manual_sum += exp(inp[row * V + i] - max_in_row);   
    }
    float manual_lse = max_in_row + log(manual_sum);
    printf("Manual softmax calculation for row %zu: max=%.10f, sum=%.10f, kernel2=%.10f\n", 
           row, max_in_row, manual_sum, manual_lse);
    printf("Expected softmax[%zu] = exp(%.10f - %.10f) = %.10f\n\n",
           idx, inp[idx], manual_lse, exp(inp[idx] - manual_lse));   
  }
  metalFree(out1);
  metalFree(out2);
}
#endif

void crossentropy_forward(float *losses, float *probs, int *targets, int B,
                          int T, int V) {
  const int block_size = 256;
  crossentropy_forward_kernel1(B * T, block_size, losses, probs, targets, B, T,
                               V);
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
  float *wte;      // (V, C)
  float *wpe;      // (maxT, C)
  float *ln1w;     // (L, C)
  float *ln1b;     // (L, C)
  float *qkvw;     // (L, 3*C, C)
  float *qkvb;     // (L, 3*C)
  float *attprojw; // (L, C, C)
  float *attprojb; // (L, C)
  float *ln2w;     // (L, C)
  float *ln2b;     // (L, C)
  float *fcw;      // (L, 4*C, C)
  float *fcb;      // (L, 4*C)
  float *fcprojw;  // (L, C, 4*C)
  float *fcprojb;  // (L, C)
  float *lnfw;     // (C)
  float *lnfb;     // (C)
} ParameterTensors;

// allocate memory for the parameters and point the individual tensors to the
// right places
float *malloc_and_point_parameters(ParameterTensors *params,
                                   size_t *param_sizes, int on_device) {
  // on_device: 0 = CPU, 1 = GPU
  // calculate the number of parameters
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += param_sizes[i];
  }
  // malloc all parameters all at once on the device
  float *params_memory = NULL;
  if (on_device) {
    metalCheck(
        metalMalloc((void **)&params_memory, num_parameters * sizeof(float)));
  } else {
    params_memory = (float *)malloc(num_parameters * sizeof(float));
  }
  // assign all the tensors their place in the array
  float **ptrs[] = {
      &params->wte,     &params->wpe,     &params->ln1w,     &params->ln1b,
      &params->qkvw,    &params->qkvb,    &params->attprojw, &params->attprojb,
      &params->ln2w,    &params->ln2b,    &params->fcw,      &params->fcb,
      &params->fcprojw, &params->fcprojb, &params->lnfw,     &params->lnfb};
  float *params_memory_iterator = params_memory;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    *(ptrs[i]) = params_memory_iterator;
    params_memory_iterator += param_sizes[i];
  }
  return params_memory;
}

#define NUM_ACTIVATION_TENSORS 25
typedef struct {
  float *logits;    // (B, T, V)
  float *encoded;   // (B, T, C)
  float *ln1;       // (L, B, T, C)
  float *ln1_mean;  // (L, B, T)
  float *ln1_rstd;  // (L, B, T)
  float *qkv;       // (L, B, T, 3*C)
  float *atty;      // (L, B, T, C)
  float *preatt;    // (L, B, NH, T, T)
  float *att;       // (L, B, NH, T, T)
  float *attproj;   // (L, B, T, C)
  float *residual2; // (L, B, T, C)
  float *ln2;       // (L, B, T, C)
  float *ln2_mean;  // (L, B, T)
  float *ln2_rstd;  // (L, B, T)
  float *fch;       // (L, B, T, 4*C)
  float *fch_gelu;  // (L, B, T, 4*C)
  float *fcproj;    // (L, B, T, C)
  float *residual3; // (L, B, T, C)
  float *lnf;       // (B, T, C)
  float *lnf_mean;  // (B, T)
  float *lnf_rstd;  // (B, T)
  float *probs;     // (B, T, V)
  float *losses;    // (B, T)
  // adding these two compared to the CPU .c code, needed for attention kernel
  // as buffers
  float *qkvr;    // (L, B, T, 3*C)
  float *v_accum; // (L, B, T, C)
} ActivationTensors;

float *malloc_and_point_activations(ActivationTensors *acts,
                                    size_t *act_sizes) {
  size_t num_activations = 0;
  for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    num_activations += act_sizes[i];
  }
  float *acts_memory = NULL;
  metalCheck(
      metalMalloc((void **)&acts_memory, num_activations * sizeof(float)));
  float **ptrs[] = {
      &acts->logits,   &acts->encoded,   &acts->ln1,       &acts->ln1_mean,
      &acts->ln1_rstd, &acts->qkv,       &acts->atty,      &acts->preatt,
      &acts->att,      &acts->attproj,   &acts->residual2, &acts->ln2,
      &acts->ln2_mean, &acts->ln2_rstd,  &acts->fch,       &acts->fch_gelu,
      &acts->fcproj,   &acts->residual3, &acts->lnf,       &acts->lnf_mean,
      &acts->lnf_rstd, &acts->probs,     &acts->losses,    &acts->qkvr,
      &acts->v_accum};
  float *acts_memory_iterator = acts_memory;
  for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    *(ptrs[i]) = acts_memory_iterator;
    acts_memory_iterator += act_sizes[i];
  }
  return acts_memory;
}

typedef struct {
  int max_seq_len; // max sequence length, e.g. 1024
  int vocab_size;  // vocab size, e.g. 50257
  int num_layers;  // number of layers, e.g. 12
  int num_heads;   // number of heads in attention, e.g. 12
  int channels;    // number of channels, e.g. 768
} GPT2Config;

typedef struct {
  GPT2Config config;
  // the weights of the model, and their sizes
  ParameterTensors params;
  size_t param_sizes[NUM_PARAMETER_TENSORS];
  float *params_memory;
  size_t num_parameters;
  // gradients of the weights
  ParameterTensors grads;
  float *grads_memory;
  // buffers for the AdamW optimizer
  float *m_memory;
  float *v_memory;
  // the activations of the model, and their sizes
  ActivationTensors acts;
  size_t act_sizes[NUM_ACTIVATION_TENSORS];
  float *acts_memory;
  size_t num_activations;
  // gradients of the activations
  ActivationTensors grads_acts;
  float *grads_acts_memory;
  // other run state configuration
  int batch_size;  // the batch size (B) of current forward pass
  int seq_len;     // the sequence length (T) of current forward pass
  int *inputs;     // the input tokens for the current forward pass
  int *targets;    // the target tokens for the current forward pass
  float mean_loss; // after a forward pass with targets, will be populated with
                   // the mean loss
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, char *checkpoint_path) {

  // read in model from a checkpoint file
  FILE *model_file = fopen(checkpoint_path, "rb");
  if (model_file == NULL) {
    printf("Error opening model file\n");
    exit(1);
  }
  int model_header[256];
  fread(model_header, sizeof(int), 256, model_file);
  if (model_header[0] != 20240326) {
    printf("Bad magic model file");
    exit(1);
  }
  if (model_header[1] != 1) {
    printf("Bad version in model file");
    exit(1);
  }

  // read in hyperparameters
  int maxT, V, L, NH, C;
  model->config.max_seq_len = maxT = model_header[2];
  model->config.vocab_size = V = model_header[3];
  model->config.num_layers = L = model_header[4];
  model->config.num_heads = NH = model_header[5];
  model->config.channels = C = model_header[6];
  printf("[GPT-2]\n");
  printf("max_seq_len: %d\n", maxT);
  printf("batch_size: %d\n", model->batch_size);
  printf("vocab_size: %d\n", V);
  printf("num_layers: %d\n", L);
  printf("num_heads: %d\n", NH);
  printf("channels: %d\n", C);

  // allocate space for all the parameters and read them in
  model->param_sizes[0] = V * C;
  model->param_sizes[1] = maxT * C;
  model->param_sizes[2] = L * C;
  model->param_sizes[3] = L * C;
  model->param_sizes[4] = L * (3 * C) * C;
  model->param_sizes[5] = L * (3 * C);
  model->param_sizes[6] = L * C * C;
  model->param_sizes[7] = L * C;
  model->param_sizes[8] = L * C;
  model->param_sizes[9] = L * C;
  model->param_sizes[10] = L * (4 * C) * C;
  model->param_sizes[11] = L * (4 * C);
  model->param_sizes[12] = L * C * (4 * C);
  model->param_sizes[13] = L * C;
  model->param_sizes[14] = C;
  model->param_sizes[15] = C;
  // count the number of paramaters
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += model->param_sizes[i];
  }
  printf("num_parameters: %zu\n", num_parameters);
  model->num_parameters = num_parameters;
  // create memory for model parameters on the device
  model->params_memory =
      malloc_and_point_parameters(&model->params, model->param_sizes, 1);

  // read in all the parameters from file and copy them to device
  //  float* params_memory_cpu = (float*)malloc(num_parameters * sizeof(float));
  //  metalCheck(metalMalloc((void**)&model->params_memory, num_parameters *
  //  sizeof(float)));
  fread(model->params_memory, sizeof(float), num_parameters, model_file);
  //  metalCheck(cudaMemcpy(model->params_memory, params_memory_cpu,
  //  num_parameters * sizeof(float), cudaMemcpyHostToDevice));
  //  free(params_memory_cpu);
  fclose(model_file);
  // other inits
  model->acts_memory = NULL;
  model->grads_memory = NULL;
  model->m_memory = NULL;
  model->v_memory = NULL;
  model->grads_acts_memory = NULL;
  model->inputs = NULL;
  model->targets = NULL;
  model->batch_size = 0;
  model->seq_len = 0;
  model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void gpt2_forward(GPT2 *model, int *inputs, int *targets, int B, int T) {
  // targets are optional and could be NULL

  // ensure the model was initialized or error out
  if (model->params_memory == NULL) {
    printf("Error: model was not initialized properly.\n");
    exit(1);
  }

  // convenience parameters
  int V = model->config.vocab_size;
  int L = model->config.num_layers;
  int NH = model->config.num_heads;
  int C = model->config.channels;

  // allocate space for all the activations if needed (done here, lazily)
  if (model->acts_memory == NULL) {
    // record the current B,T as well
    model->batch_size = B;
    model->seq_len = T;
    // and now allocate the space
    size_t act_offset = 0;
    model->act_sizes[act_offset++] = B * T * V;
    model->act_sizes[act_offset++] = B * T * C;
    model->act_sizes[act_offset++] = L * B * T * C;
    model->act_sizes[act_offset++] = L * B * T;
    model->act_sizes[act_offset++] = L * B * T;
    model->act_sizes[act_offset++] = L * B * T * 3 * C;
    model->act_sizes[act_offset++] = L * B * T * C;
    model->act_sizes[act_offset++] = L * B * NH * T * T;
    model->act_sizes[act_offset++] = L * B * NH * T * T;
    model->act_sizes[act_offset++] = L * B * T * C;
    model->act_sizes[act_offset++] = L * B * T * C;
    model->act_sizes[act_offset++] = L * B * T * C;
    model->act_sizes[act_offset++] = L * B * T;
    model->act_sizes[act_offset++] = L * B * T;
    model->act_sizes[act_offset++] = L * B * T * 4 * C;
    model->act_sizes[act_offset++] = L * B * T * 4 * C;
    model->act_sizes[act_offset++] = L * B * T * C;
    model->act_sizes[act_offset++] = L * B * T * C;
    model->act_sizes[act_offset++] = B * T * C;
    model->act_sizes[act_offset++] = B * T;
    model->act_sizes[act_offset++] = B * T;
    model->act_sizes[act_offset++] = B * T * V;
    model->act_sizes[act_offset++] = B * T;
    model->act_sizes[act_offset++] = L * B * T * 3 * C; // qkvr
    model->act_sizes[act_offset++] = L * B * T * C;     // v_accum
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
      num_activations += model->act_sizes[i];
    }
    printf("num_activations: %zu\n", num_activations);
    model->num_activations = num_activations;
    model->acts_memory =
        malloc_and_point_activations(&model->acts, model->act_sizes);
  } else {
    // validate B,T is no larger than what was previously allocated
    // in principle, we could re-allocate a larger chunk of memory, for now we
    // just error out
    if (B > model->batch_size || T > model->seq_len) {
      printf("Error: batch size or sequence length is inadequately large\n");
      printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size,
             model->seq_len, B, T);
      exit(1);
    }
  }
  // forward pass
  ParameterTensors params = model->params; // for brevity
  ActivationTensors acts = model->acts;
  float *residual;

  encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T,
                  C); // encoding goes into residual[0]
  for (int l = 0; l < L; l++) {
    residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;
    // get the pointers of the weights for this layer
    float *l_ln1w = params.ln1w + l * C;
    float *l_ln1b = params.ln1b + l * C;
    float *l_qkvw = params.qkvw + l * 3 * C * C;
    float *l_qkvb = params.qkvb + l * 3 * C;
    float *l_attprojw = params.attprojw + l * C * C;
    float *l_attprojb = params.attprojb + l * C;
    float *l_ln2w = params.ln2w + l * C;
    float *l_ln2b = params.ln2b + l * C;
    float *l_fcw = params.fcw + l * 4 * C * C;
    float *l_fcb = params.fcb + l * 4 * C;
    float *l_fcprojw = params.fcprojw + l * C * 4 * C;
    float *l_fcprojb = params.fcprojb + l * C;

    // get the pointers of the activations for this layer
    float *l_ln1 = acts.ln1 + l * B * T * C;
    float *l_ln1_mean = acts.ln1_mean + l * B * T;
    float *l_ln1_rstd = acts.ln1_rstd + l * B * T;
    float *l_qkv = acts.qkv + l * B * T * 3 * C;
    float *l_qkvr = acts.qkvr + l * B * T * 3 * C;
    float *l_atty = acts.atty + l * B * T * C;
    float *l_preatt = acts.preatt + l * B * NH * T * T;
    float *l_att = acts.att + l * B * NH * T * T;
    float *l_v_accum = acts.v_accum + l * B * T * C;
    float *l_attproj = acts.attproj + l * B * T * C;
    float *l_residual2 = acts.residual2 + l * B * T * C;
    float *l_ln2 = acts.ln2 + l * B * T * C;
    float *l_ln2_mean = acts.ln2_mean + l * B * T;
    float *l_ln2_rstd = acts.ln2_rstd + l * B * T;
    float *l_fch = acts.fch + l * B * T * 4 * C;
    float *l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
    float *l_fcproj = acts.fcproj + l * B * T * C;
    float *l_residual3 = acts.residual3 + l * B * T * C;

    // now do the forward pass
    layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b,
                      B, T, C);
    matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
    attention_forward(l_atty, l_v_accum, l_qkvr, l_preatt, l_att, l_qkv, B, T,
                      C, NH);
    matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
    residual_forward(l_residual2, residual, l_attproj, B * T * C);
    layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w,
                      l_ln2b, B, T, C);
    matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
    gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
    matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
    residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);

    metalCommitCommands();
  }

  residual =
      acts.residual3 + (L - 1) * B * T * C; // last residual is in residual3
  layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual,
                    params.lnfw, params.lnfb, B, T, C);

  matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);
  softmax_forward(acts.probs, acts.logits, B, T, V);
  // also forward the cross-entropy loss function if we have the targets
  if (targets != NULL) {
    crossentropy_forward(acts.losses, acts.probs, targets, B, T, V);
    metalCommitCommandsAndWait();
    float mean_loss = 0.0f;
    for (int i = 0; i < B * T; i++) {
      mean_loss += acts.losses[i];
    }
    mean_loss /= B * T;
    model->mean_loss = mean_loss;

  } else {
    // if we don't have targets, we don't have a loss
    model->mean_loss = -1.0f;
    metalCommitCommandsAndWait();
  }
}

void gpt2_free(GPT2 *model) {
  metalCheck(metalFree(model->params_memory));
  // Not needed until backprop is implemented.
  //  metalCheck(metalFree(model->grads_memory));
  //  metalCheck(metalFree(model->grads_acts_memory));
  //  metalCheck(metalFree(model->m_memory));
  //  metalCheck(metalFree(model->v_memory));
  //  metalCheck(metalFree(model->inputs));
  //  metalCheck(metalFree(model->targets));
  metalCheck(metalFree(model->acts_memory));
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.cu), we'll skip the int main below

// ----------------------------------------------------------------------------
// data loader lite
// returns random batches of data from a file of integers

typedef struct {
  // hyperparameters
  int B;
  int T;
  // input handling and its state
  FILE *tokens_file;
  long file_size;
  long current_position;
  // output memory
  int *batch;
  int *inputs;
  int *targets;
  // convenience variables
  int num_batches;
} DataLoader;

void dataloader_init(DataLoader *loader, char *filename, int B, int T) {
  loader->B = B;
  loader->T = T;

  // open the input file for reading
  loader->tokens_file = fopen(filename, "rb");
  if (loader->tokens_file == NULL) {
    printf("Error opening tokens file\n");
    exit(1);
  }

  // determine the file size
  fseek(loader->tokens_file, 0, SEEK_END);
  loader->file_size = ftell(loader->tokens_file);
  fseek(loader->tokens_file, 0, SEEK_SET);
  if (loader->file_size < (B * T + 1) * sizeof(int)) {
    printf("Error: file size is too small for the batch size and sequence "
           "length\n");
    exit(1);
  }
  loader->current_position = 0; // start at the beginning

  // allocate space for B*T + 1 integers to store the inputs and targets
  metalCheck(metalMalloc((void **)&loader->batch, (B * T + 1) * sizeof(int)));
  //  loader->batch = (int*) malloc((B * T + 1) * sizeof(int));
  loader->inputs = loader->batch;
  loader->targets = loader->batch + 1; // targets are shifted by one
  loader->num_batches = loader->file_size / (B * T * sizeof(int));
}

void dataloader_reset(DataLoader *loader) { loader->current_position = 0; }

void dataloader_next_batch(DataLoader *loader) {
  int B = loader->B;
  int T = loader->T;
  // if we are at the end of the file, loop back to the beginning
  if (loader->current_position + (B * T + 1) * sizeof(int) >
      loader->file_size) {
    loader->current_position = 0;
  }
  // read the B*T+1 integers from the file into batch
  fseek(loader->tokens_file, loader->current_position, SEEK_SET);
  fread(loader->batch, sizeof(int), B * T + 1, loader->tokens_file);
  // advance the current position by B*T integers
  loader->current_position += B * T * sizeof(int);
}

void dataloader_free(DataLoader *loader) {
  fclose(loader->tokens_file);
  metalFree(loader->batch);
}

// ----------------------------------------------------------------------------
// sampler

#define GPT2_EOT 50256

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float *probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}
// ----------------------------------------------------------------------------
// Tokenizer (only supports decoding)

typedef struct {
  uint32_t vocab_size;
  char **token_table;
  int init_ok;
} Tokenizer;

void safe_printf(const char *piece) {
  // the tokens are raw bytes, and we we only want to print the printable ones
  // many bytes can be various control codes, backspace, etc.
  if (piece == NULL) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  // handle individual byte tokens
  // every token is asserted to be at least one byte so doing piece[1] is ok
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // weird byte, don't print it
    }
  }
  printf("%s", piece);
}

void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    // try to be more helpful as we just added this feature, erase later
    printf("---\n");
    printf("WARNING: Failed to open the tokenizer file %s\n", filename);
    printf("The Tokenizer is a new feature added April 14 2024.\n");
    printf("Re-run `python train_gpt2.py` to write it\n");
    printf("---\n");
    tokenizer->init_ok = 0;
    return;
  }
  // read in the header
  uint32_t header[256];
  fread(header, sizeof(uint32_t), 256, file);
  assert(header[0] == 20240328);
  assert(header[1] == 1);
  tokenizer->vocab_size = header[2];
  // read in all the tokens
  unsigned char length;
  tokenizer->token_table =
      (char **)malloc(tokenizer->vocab_size * sizeof(char *));
  for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
    fread(&length, sizeof(unsigned char), 1, file);
    assert(length > 0); // every token should be at least one character
    char *token_bytes = (char *)malloc(length + 1);
    fread(token_bytes, sizeof(char), length, file);
    token_bytes[length] = '\0'; // Add null terminator for printing
    tokenizer->token_table[i] = token_bytes;
  }
  // cleanups
  fclose(file);
  tokenizer->init_ok = 1;
}

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
  if (tokenizer->init_ok == 0) {
    return NULL;
  }
  if (token_id < tokenizer->vocab_size) {
    return tokenizer->token_table[token_id];
  } else {
    printf("invalid token id %d!\n", token_id);
    return NULL;
  }
}

void tokenizer_free(Tokenizer *tokenizer) {
  if (tokenizer->init_ok) {
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
      free(tokenizer->token_table[i]);
    }
    free(tokenizer->token_table);
  }
}

/*
test_sum() and test_max() are used to test the sum and max kernels.
Achieved performance is memory-bound, reaching ~80-100 GB/s bandwidth, 
close to M3's theoretical 100 GB/s unified memory bandwidth. FLOPs not 
relevant; the kernels are limited by memory access, not compute. 
For compute-bound, we'd see closer to 3 TFLOPS. 
*/

void sum_kernel(int grid_size, int block_size, int shared_size, float *out,
                float *inp, int N, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, shared_size, 4, Buffer, out, Buffer,
               inp, Scalar, &N, Scalar, &C);
}

void sum_kernel_test(float *out, float *inp, int B, int T, int V) {
  for(int bt = 0; bt < B * T; bt++) {
    for(int v = 0; v < V; v++) {
      inp[bt * V + v] = (float)(v + bt);
    }
  }
  const int block_size = metalComputePipelineStateMaxTotalThreadsPerThreadgroup();
  size_t shared_mem_size = block_size * sizeof(float);
  sum_kernel((B * T) * block_size, block_size, shared_mem_size, out, inp, B * T, V);
  metalCommitCommandsAndWait();
  for(int i = 0; i < B * T; i++) {
    float exp_sum = 0.0f;
    for(int j = 0; j < V; j++) {
      exp_sum += (float)(j + i);
    }
    if(out[i] != exp_sum) {
      printf("Mismatch at row %d: got %f expected %f\n", i, out[i], exp_sum);
    }
  }
}

void test_sum() {
  int B = 4;
  int T = 64;
//int V = 50257; // vocab size
  int V = 4000;  // 4096 * 4096 <= 16,777,215 which 2^24 - 1 float precision limit
  int N = B * T;
  float *inp, *out;
  metalMalloc((void**)&inp, N * V * sizeof(float));
  metalMalloc((void**)&out, N * sizeof(float));
  sum_kernel_test(out, inp, B, T, V);
  metalCommitCommandsAndWait();
  // Measure performance
  clock_t start = clock();
  const int M = 100000; // number of test iterations
  for(int i = 0; i < M; i++) {
    sum_kernel(N * 1024, 1024, 1024 * sizeof(float), out, inp, N, V);
  }
  metalCommitCommandsAndWait();
  clock_t end = clock();
  double time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / (double)M;
  printf("Average time per sum_kernel call: %f ms\n", time_ms);
  metalFree(inp);
  metalFree(out);
}

// In train_gpt2_metal.c add:
void max_kernel(int grid_size, int block_size, int shared_size, float *out,
                float *inp, int N, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, shared_size, 4, Buffer, out, Buffer,
               inp, Scalar, &N, Scalar, &C);
}

void max_kernel_test(float *out, float *inp, int B, int T, int V) {
//srand(time(NULL));
  for (int bt = 0; bt < B * T; bt++) {
    int btV = bt * V;
    for(int v = 0; v < V; v++) {
      inp[btV + v] = (float)(v + bt);
    }
    for (int i = V-1; i > 0; i--) { // shuffle the last V-1 elements
      int j = rand() % (i + 1);
      float temp = inp[btV + i];
      inp[btV + i] = inp[btV + j];
      inp[btV + j] = temp;
    }
  }
  const int block_size = metalComputePipelineStateMaxTotalThreadsPerThreadgroup();
  size_t shared_mem_size = block_size * sizeof(float);
  max_kernel((B * T) * block_size, block_size, shared_mem_size, out, inp, B * T, V);
  metalCommitCommandsAndWait();
  for(int i = 0; i < B * T; i++) {
    float exp_max = (float)(V - 1 + i);
    if(fabs(out[i] - exp_max) > 0) {
      printf("Mismatch at row %d: got %f expected %f\n", i, out[i], exp_max);
    }
  }
}

void test_max() {
  int B = 4;
  int T = 64;
  int V = 50257; // vocab size
//int V = 4000;
  int N = B * T;
  float *inp, *out;
  metalMalloc((void**)&inp, N * V * sizeof(float));
  metalMalloc((void**)&out, N * sizeof(float));
  max_kernel_test(out, inp, B, T, V);
  metalCommitCommandsAndWait();
  clock_t start = clock();
  const int M = 10000; // number of test iterations
  for(int i = 0; i < M; i++) {
    max_kernel(N * 1024, 1024, 1024 * sizeof(float), out, inp, N, V);
  }
  metalCommitCommandsAndWait();
  clock_t end = clock();
  double time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / (double)M;
  printf("Average time per max_kernel call: %f ms\n", time_ms);
  metalFree(inp);
  metalFree(out);
}

void softmax_kernel_fused(int grid_size, int block_size, int shared_size, float *out,
                          float *inp, int N, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, shared_size, 4, Buffer, out, Buffer,
               inp, Scalar, &N, Scalar, &C);
}

void softmax_kernel_test(float *out, float *inp, int B, int T, int V) {
  float min_val = +infinity;
  float max_val = -infinity;
  for(int bt = 0; bt < B * T; bt++) {
    int btV = bt * V;
    for(int v = 0; v < V; v++) {
      inp[bt * V + v] = (float)(v + bt) / (float)(V + B * T) * 318.0f - 306.0f;
      min_val = fminf(min_val, inp[bt * V + v]);
      max_val = fmaxf(max_val, inp[bt * V + v]);
    }
    for (int i = V-1; i > 0; i--) { // shuffle the last V-1 elements
      int j = rand() % (i + 1);
      float temp = inp[btV + i];
      inp[btV + i] = inp[btV + j];
      inp[btV + j] = temp;
    }
  }
  printf("Softmax test: min_val = %f, max_val = %f\n", min_val, max_val);
  const int block_size = metalComputePipelineStateMaxTotalThreadsPerThreadgroup();
  size_t shared_mem_size = block_size * sizeof(float);
  softmax_kernel_fused((B * T) * block_size, block_size, shared_mem_size, out, inp, B * T, V);
  metalCommitCommandsAndWait();
  float max_diff = 0.0f;
  float max_sum_diff = 0.0f;
  for(int i = 0; i < B * T; i++) {
    float offset = -infinity;
    for(int j = 0; j < V; j++) { offset = fmaxf(offset, inp[i * V + j]); } 
    float sum = 0.0f;
    for(int j = 0; j < V; j++) { sum += (float)exp(inp[i * V + j] - offset); } 
    float exp_sum = 0.0f;
    for(int j = 0; j < V; j++) { 
      float cpu_j = (float)exp(inp[i * V + j] - offset) / sum;
      exp_sum += cpu_j;
      max_diff = fmaxf(max_diff, fabs(cpu_j - out[i * V + j]));
      if (fabs(cpu_j - out[i * V + j]) > 1e-6) {
        printf("Mismatch at row %3d, col %3d: got %f expected %f\n", i, j, out[i * V + j],
               cpu_j);
      }
    } 
    float softmax_sum = 0.0f;
    for(int j = 0; j < V; j++) {
      softmax_sum += out[i * V + j];
    }
    max_sum_diff = fmaxf(max_sum_diff, fabs(softmax_sum - exp_sum));
    if(fabs(softmax_sum - 1.0f) > 1e-5) {
      printf("Mismatch at row %3d: got %f expected %f\n", i, softmax_sum, 1.0f);
    }
  }
  printf("Max difference in softmax output: %.1e\n", max_diff);
  printf("Max difference in softmax sum   : %.1e\n", max_sum_diff);
}

void test_softmax() {
  int B = 4;
  int T = 64;
  int V = 50257; // vocab size
  int N = B * T;
  float *inp, *out;
  metalMalloc((void**)&inp, N * V * sizeof(float));
  metalMalloc((void**)&out, N * V * sizeof(float));
  softmax_kernel_test(out, inp, B, T, V);
  clock_t start = clock();
  const int M = 10000;
  for(int i = 0; i < M; i++) {
    softmax_kernel_fused(N * 1024, 1024, 1024 * sizeof(float), out, inp, N, V);
  }
  metalCommitCommandsAndWait();
  clock_t end = clock();
  double time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / (double)M;
  printf("Average time per softmax_kernel_fused call: %f ms\n", time_ms);
  metalFree(inp);
  metalFree(out);
}

// ----------------------------------------------------------------------------
// main training loop
int main(void) {
  GPT2 model;
  gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

  initKernels("encoder_forward_kernel2", "mean_kernel", "rstd_kernel",
              "normalization_kernel", "add_bias_kernel", "permute_kernel",
              "unpermute_kernel",
              "residual_forward_kernel", "gelu_kernel",
              "crossentropy_forward_kernel1", "scale_kernel",
              "softmax_forward_kernel1", 
              "sum_kernel", "max_kernel", "softmax_kernel_fused",
              NULL);

  // build the DataLoaders from tokens files. for now use tiny_shakespeare if
  // available, else tiny_stories
  char *tiny_stories_train = "data/TinyStories_train.bin";
  char *tiny_stories_val = "data/TinyStories_val.bin";
  char *tiny_shakespeare_train = "data/tiny_shakespeare_train.bin";
  char *tiny_shakespeare_val = "data/tiny_shakespeare_val.bin";
  char *train_tokens = access(tiny_shakespeare_train, F_OK) != -1
                           ? tiny_shakespeare_train
                           : tiny_stories_train;
  char *val_tokens = access(tiny_shakespeare_val, F_OK) != -1
                         ? tiny_shakespeare_val
                         : tiny_stories_val;
  int B = 4;
  int T = 64;
  DataLoader train_loader;
  dataloader_init(&train_loader, train_tokens, B, T);
  printf("train dataset num_batches: %d\n", train_loader.num_batches);
  DataLoader val_loader;
  dataloader_init(&val_loader, val_tokens, B, T);
  printf("val dataset num_batches: %d\n", val_loader.num_batches);
  int val_num_batches = 5;
  test_softmax();
  test_sum();
  test_max();
  // See how logits behave:
  gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
  metalCommitCommandsAndWait();
  int V = model.config.vocab_size;
  float* logits = model.acts.logits; // B, T, V 
  float min_val = +infinity;
  float max_val = -infinity;
  for (int i = 0; i < B * T; i++) {
    for (int j = 0; j < V; j++) {
      min_val = fminf(min_val, logits[i * V + j]);
      max_val = fmaxf(max_val, logits[i * V + j]);
    }
  }
  printf("logits min: %f, max: %f\n", min_val, max_val);
  exit(0); // exit after the test
#if 0
  // Test and compare softmax implementations on a small batch
  printf("\n=== Testing Softmax Implementations ===\n");
  dataloader_next_batch(&val_loader);
  gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
  metalCommitCommandsAndWait();
  
  // Compare on the final logits (which get fed to softmax)
  int V = model.config.vocab_size;
  compare_softmax_implementations(model.acts.logits, B, T, V);
  printf("=== End Softmax Test ===\n\n");
#endif
  // build the Tokenizer
  Tokenizer tokenizer;
  tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

  printf("batch size: %d\n", B);
  printf("sequence length: %d\n", T);
  printf("val_num_batches: %d\n", val_num_batches);

  // some memory for generating samples from the model
  unsigned long long rng_state = 1337;
  const int genT = 64;
  int *gen_tokens = NULL;
  metalMalloc((void **)&gen_tokens, sizeof(int) * B * T);

  // train
  struct timespec start, end;
  for (int step = 0; step <= 40; step++) {
    // once in a while estimate the validation loss
    if (step % 10 == 0) {
      float val_loss = 0.0f;
      dataloader_reset(&val_loader);
      for (int i = 0; i < val_num_batches; i++) {
        dataloader_next_batch(&val_loader);
        gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
        val_loss += model.mean_loss;
      }
      val_loss /= val_num_batches;
      printf("val loss %f\n", val_loss);
    }

    // once in a while do model inference to print generated text
    if (step > 0 && step % 20 == 0) {
      for (int i = 0; i < B * T; ++i) {
        gen_tokens[i] = GPT2_EOT;
      }
      printf("generating:\n---\n");
      for (int t = 1; t < genT; t++) {
        // note that inference is wasteful here because
        // for each t, we re-compute all activations between 0 and t
        // leaving this alone because you want separate code for inference
        // anyway the inference here is just for sanity checking purposes
        gpt2_forward(&model, gen_tokens, NULL, B, T);
        float *probs = model.acts.probs + (t - 1) * model.config.vocab_size;
        float coin = random_f32(&rng_state);
        // move probs back to CPU and sample
        int next_token = sample_mult(probs, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;
        if (tokenizer.init_ok) {
          const char *token_str = tokenizer_decode(&tokenizer, next_token);
          safe_printf(token_str);
        } else {
          // fall back to printing the token id
          printf("%d ", next_token);
        }
        fflush(stdout);
      }
      printf("\n---\n");
    }

    // do a training step
    clock_gettime(CLOCK_MONOTONIC, &start);
    dataloader_next_batch(&train_loader);
    gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
    // these are still TODO
    // gpt2_zero_grad(&model);
    // gpt2_backward(&model);
    // gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss,
           time_elapsed_s * 1000);
  }

  // free
  dataloader_free(&train_loader);
  dataloader_free(&val_loader);
  tokenizer_free(&tokenizer);
  gpt2_free(&model);
  metalFree(gen_tokens);
  return 0;
}
#endif

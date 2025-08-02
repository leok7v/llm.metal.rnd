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

#include <metal_stdlib>
using namespace metal;

#define MAX_RANK 5
#define SIMD_GROUP_SIZE 32 // TODO: Shouldn't hardcode this. Can get it from host API (threadExecutionWidth).
#define M_PI 3.14159265358979323846264338327950288

inline void flat_to_nd(int index,
                       int inShape[MAX_RANK], // The shape of the tensor
                       int inStrides[MAX_RANK], // The strides for the tensor
                       int outCoords[MAX_RANK]) { // The resulting output.
  int flatIdx = index;
  // Start at the most significant rank and work our way down.
  for (int i = 0; i < MAX_RANK; i++) {
    int coord = flatIdx / inStrides[i];
    outCoords[i] = min(coord, inShape[i] - 1);
    // Take what's left over to the next rank...
    flatIdx -= coord * inStrides[i];
  }
}

inline int nd_to_flat(int inCoords[MAX_RANK],
                      int inStrides[MAX_RANK]) {
  int flatIdx = 0;
  for (int i = 0; i < MAX_RANK; i++) {
    flatIdx += inCoords[i] * inStrides[i];
  }
  return flatIdx;
}

inline void calc_strides(const int srcShape[MAX_RANK],
                         int outStrides[MAX_RANK]) {
  for (int i = 0; i < MAX_RANK; i++) {
    int prod = 1;
    for (int j = i + 1; j < MAX_RANK; j++) {
      prod *= srcShape[j];
    }
    outStrides[i] = prod;
  }
}

inline int calc_perm_idx(const int srcShape[MAX_RANK],
                         const int permOrder[MAX_RANK],
                         const uint index) {
  int strides[MAX_RANK];
  calc_strides(srcShape, strides);

  int permuted_shape[MAX_RANK];
  int permuted_strides[MAX_RANK];
  for (int i = 0; i < MAX_RANK; i++) {
    int axis = permOrder[i];
    permuted_strides[i] = strides[axis];
    permuted_shape[i] = srcShape[axis];
  }

  int strides_T[MAX_RANK];
  calc_strides(permuted_shape, strides_T);
  int coords[MAX_RANK];
  flat_to_nd(index, permuted_shape, strides_T, coords);
  int permIdx = nd_to_flat(coords, permuted_strides);

  return permIdx;
}

template<typename T>
static inline T simdGroupReduceMax(T val, uint simd_size) {
  for (int offset = simd_size / 2; offset > 0; offset >>= 1) {
    T other = simd_shuffle_down(val, offset);
    val = max(val, other);
  }
  return val;
}

template<typename T>
static inline T simdGroupReduceSum(T val, uint simd_size) {
  for (int offset = simd_size / 2; offset > 0; offset >>= 1) {
    T other = simd_shuffle_down(val, offset);
    val += other;
  }
  return val;
}

kernel void encoder_forward_kernel2(device float* out [[buffer(0)]],
                                    device int* inp [[buffer(1)]],
                                    device float* wte [[buffer(2)]],
                                    device float* wpe [[buffer(3)]],
                                    constant uint& B [[ buffer(4) ]],
                                    constant uint& T [[ buffer(5) ]],
                                    constant uint& C [[ buffer(6) ]],
                                    uint tid [[thread_position_in_grid]]) {
  uint N = B * T * C;

  if (tid < N) {
    int bt = tid / C;
    int b = bt / T;
    int t = bt % T;
    int c = tid % C;

    int ix = inp[b * T + t];
    device float* out_btc = out + b * T * C + t * C + c;
    device float* wte_ix = wte + ix * C + c;
    device float* wpe_tc = wpe + t * C + c;
    *out_btc = *wte_ix + *wpe_tc;
  }
}

kernel void mean_kernel(device float* mean [[buffer(0)]],
                        device float* inp [[buffer(1)]],
                        constant int& N [[buffer(2)]],
                        constant int& C [[buffer(3)]],
                        uint block_size [[threads_per_threadgroup]],
                        uint idx [[threadgroup_position_in_grid]],
                        uint tgid [[thread_position_in_threadgroup]],
                        threadgroup float* shared [[threadgroup(0)]]) {
  int index = idx; // range [0, B*T)
  int thread_id = tgid; // range [0, block_size)
  device float* x = inp + index * C;
  // thread coarsening
  float sum = 0.0f;
  for (int i = thread_id; i < C; i += block_size) {
    sum += x[i];
  }
  shared[thread_id] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // reductions
  for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id < stride) {
      shared[thread_id] += shared[thread_id + stride];
    }
  }
  // write the final result (at thread 0) to global memory
  if (thread_id == 0) {
    mean[index] = shared[0] / C;
  }
}

kernel void rstd_kernel(device float* rstd [[buffer(0)]],
                        device float* inp [[buffer(1)]],
                        device float* mean [[buffer(2)]],
                        constant uint& N [[ buffer(3) ]],
                        constant uint& C [[ buffer(4) ]],
                        uint idx [[threadgroup_position_in_grid]],
                        uint tgid [[thread_position_in_threadgroup]],
                        uint bsize [[threads_per_threadgroup]],
                        threadgroup float* shared [[threadgroup(0)]]) {
//  if (idx >= N) return; // Guard against out-of-bounds work items

  device float* x = inp + idx * C;
  float m = mean[idx];
  // thread coarsening
  float sum = 0.0f;
  for (uint i = tgid; i < C; i += bsize) {
    float diff = x[i] - m;
    sum += diff * diff;
  }
  shared[tgid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // reductions
  for (uint stride = bsize / 2; stride >= 1; stride /= 2) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tgid < stride) {
      shared[tgid] += shared[tgid + stride];
    }
  }

  // write the final result (at thread 0) to global memory
  if (tgid == 0) {
    rstd[idx] = 1.0f / precise::sqrt(shared[0] / C + 1e-5f);
  }
}

kernel void normalization_kernel(device float* out [[buffer(0)]],
                                 device float* inp [[buffer(1)]],
                                 device float* mean [[buffer(2)]],
                                 device float* rstd [[buffer(3)]],
                                 device float* weight [[buffer(4)]],
                                 device float* bias [[buffer(5)]],
                                 constant uint& B [[ buffer(6) ]],
                                 constant uint& T [[ buffer(7) ]],
                                 constant uint& C [[ buffer(8) ]],
                                 uint tid [[thread_position_in_grid]]) {
  uint bt = tid / C;
  uint c = tid % C;
  float m = mean[bt];
  float s = rstd[bt];
  float xi = inp[tid];
  float n = s * (xi - m);
  float o = bias[c] + n * weight[c];
  out[tid] = o;
}

kernel void permute_kernel(device float* q [[ buffer(0) ]],
                           device float* k [[ buffer(1) ]],
                           device float* v [[ buffer(2) ]],
                           const device float* inp [[ buffer(3) ]],
                           constant uint& B [[ buffer(4) ]],
                           constant uint& N [[ buffer(5) ]],
                           constant uint& NH [[ buffer(6) ]],
                           constant uint& d [[ buffer(7) ]],
                           uint tid [[ thread_position_in_grid ]]) {
  if (tid < B * NH * N * d) {
    uint b = tid / (NH * N * d);
    uint rest = tid % (NH * N * d);
    uint nh_ = rest / (N * d);
    rest = rest % (N * d);
    uint n = rest / d;
    uint d_ = rest % d;

    uint inp_idx = \
    (b * N * 3 * NH * d)
    +   (n * 3 * NH * d)
    +       (0 * NH * d)
    +          (nh_ * d)
    +                d_;

    q[tid] = inp[inp_idx];
    k[tid] = inp[inp_idx + NH * d];
    v[tid] = inp[inp_idx + 2 * NH * d];
  }
}

kernel void unpermute_kernel(const device float* inp [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& B [[buffer(2)]],
                             constant uint& T [[buffer(3)]],
                             constant uint& NH [[buffer(4)]],
                             constant uint& HS [[buffer(5)]],
                             uint tid [[thread_position_in_grid]]) {

  // B, NH, HS, T ->  B, T, NH, HS
  // 0   1   2  3     0  3  1   2
  const int src_shape[5] = {(int)B, (int)NH, (int)HS, (int)T, 1};
  const int perm_order[5] = {0, 3, 1, 2, 4};
  int permIdx = calc_perm_idx(src_shape, perm_order, tid);
  out[tid] = inp[permIdx];
}

kernel void add_bias_kernel(device float* out [[ buffer(0) ]],
                            device float* bias [[ buffer(1) ]],
                            constant uint& OC [[ buffer(2) ]],
                            uint tid [[ thread_position_in_grid ]]) {
  out[tid] = out[tid] + bias[tid % OC];
}

kernel void scale_kernel(device float* inout [[buffer(0)]],
                         constant float& scale [[buffer(1)]],
                         constant uint& B [[buffer(2)]],
                         constant uint& NH [[buffer(3)]],
                         constant uint& T [[ buffer(4) ]],
                         uint tid [[thread_position_in_grid]])
{
  int rest = tid % (NH * T * T);
  rest = rest % (T * T);
  int t2 = rest / T;
  int t = rest % T;
  if (t > t2) {
    inout[tid] = -INFINITY;
  } else {
    inout[tid] *= scale;
  }
}

kernel void softmax_forward_kernel1(device float* out [[buffer(0)]],
                                    device float* inp [[buffer(1)]],
                                    constant int& N,
                                    constant int& C,
                                    uint tid [[thread_position_in_grid]]) {
  device float* inp_row = inp + tid * C;
  device float* out_row = out + tid * C;

  float maxval = -INFINITY;
  for (int j = 0; j < C; j++) {
    if (inp_row[j] > maxval) {
      maxval = inp_row[j];
    }
  }

  float sum = 0.0f;
  for (int j = 0; j < C; j++) {
    out_row[j] = exp(inp_row[j] - maxval);
    sum += out_row[j];
  }
  for (int j = 0; j < C; j++) {
    out_row[j] /= sum;
  }
}

// LogSumExp state for associative parallel reduction
struct LogSumExpState {
    float max_val;
    float sum_exp;
};

// Associative combine operation for LogSumExp states
inline LogSumExpState combine_lse_states(LogSumExpState a, LogSumExpState b) {
    // Handle empty states
    if (a.sum_exp == 0.0f) return b;
    if (b.sum_exp == 0.0f) return a;
    
    float new_max = max(a.max_val, b.max_val);
    float correction_a = precise::exp(a.max_val - new_max);
    float correction_b = precise::exp(b.max_val - new_max);
    return {new_max, a.sum_exp * correction_a + b.sum_exp * correction_b};
}

// Kernel 1: Compute LogSumExp reduction for each row
kernel void logsumexp_kernel(device float* logsumexp_out [[buffer(0)]],
                             device float* inp [[buffer(1)]],
                             constant int& N [[buffer(2)]],
                             constant int& C [[buffer(3)]],
                             uint idx [[threadgroup_position_in_grid]],
                             uint tgid [[thread_position_in_threadgroup]],
                             uint block_size [[threads_per_threadgroup]],
                             threadgroup LogSumExpState* shared [[threadgroup(0)]]) {
    
    // Each threadgroup processes one row
    if (idx >= (uint)N) return;
    
    device float* inp_row = inp + idx * C;
    
    // Initialize local state
    LogSumExpState local_state;
    local_state.max_val = -INFINITY;
    local_state.sum_exp = 0.0f;
    
    // Each thread processes multiple elements with stride = block_size
    for (int i = tgid; i < C; i += block_size) {
        float x = inp_row[i];
        if (local_state.sum_exp == 0.0f) {
            // First element for this thread
            local_state.max_val = x;
            local_state.sum_exp = 1.0f;
        } else {
            // Subsequent elements - use combine function
            LogSumExpState new_state;
            new_state.max_val = x;
            new_state.sum_exp = 1.0f;
            local_state = combine_lse_states(local_state, new_state);
        }
    }
    
    // Store local results in shared memory
    shared[tgid] = local_state;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction in shared memory
    for (uint stride = block_size / 2; stride > 0; stride /= 2) {
        if (tgid < stride) {
            shared[tgid] = combine_lse_states(shared[tgid], shared[tgid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Thread 0 writes the final LogSumExp result
    if (tgid == 0) {
        LogSumExpState final_state = shared[0];
        logsumexp_out[idx] = final_state.max_val + precise::log(final_state.sum_exp);
    }
}

// Kernel 2: Compute softmax values using precomputed LogSumExp
kernel void softmax_from_lse_kernel(device float* out [[buffer(0)]],
                                    device float* inp [[buffer(1)]],
                                    device float* logsumexp [[buffer(2)]],
                                    constant int& N [[buffer(3)]],
                                    constant int& C [[buffer(4)]],
                                    uint tid [[thread_position_in_grid]]) {
    if (tid >= (uint)(N * C)) return;    
    uint row = tid / C;
    float z = inp[tid] - logsumexp[row];
    out[tid] = precise::exp(z);
}

kernel void softmax_forward_kernel4(device float* out [[buffer(0)]],
                                    device float* inp [[buffer(1)]],
                                    constant uint& N [[ buffer(2) ]],
                                    constant uint& C [[ buffer(3) ]],
                                    uint simdSize [[ thread_execution_width ]],
                                    uint laneID [[ thread_index_in_simdgroup ]],
                                    uint tgIdx [[thread_position_in_threadgroup]],
                                    uint tid [[thread_position_in_grid]],
                                    uint idx [[threadgroup_position_in_grid]],
                                    uint bsize [[threads_per_grid]],
                                    uint simdGroupID [[simdgroup_index_in_threadgroup]],
                                    uint simdGroupsPerBlock [[simdgroups_per_threadgroup]],
                                    threadgroup float* shared [[threadgroup(0)]]) {
  // out is (N, C) just like inp. Each row of inp will get softmaxed.
  // each row of C elements is handled by bsize threads
  // furthermore, each bsize threads get executed in simd groups of SIMD_GROUP_SIZE threads

  // shared[] must be allocated to have 2 * simdGroupsPerBlock elements
  // first half for max values, the second half for sum values
  threadgroup float* maxvals = &shared[0];
  threadgroup float* sumvals = &shared[simdGroupsPerBlock];

  // one row of inp, i.e. inp[idx, :] of shape (C,)
  device float* x = inp + idx * C;

  // first, thread coarsening by directly accessing global memory in series
  float maxval = -INFINITY;
  for (uint i = tgIdx; i < C; i += bsize) {
    maxval = fmax(maxval, x[i]);
  }
  // now within-simd group reductions for maxval
  maxval = simdGroupReduceMax(maxval, simdSize);

  // the 0th thread of each simd group writes the maxval of that simd group to shared memory
  if (laneID == 0) maxvals[simdGroupID] = maxval;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // now the 0th thread reduces the maxvals in shared memory, i.e. across simd groups
  if (tgIdx == 0) {
    float val = maxvals[0];
    for (uint i = 1; i < simdGroupsPerBlock; i++) {
      val = fmax(val, maxvals[i]);
    }
    // store the final max in the first position
    maxvals[0] = val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // broadcast the max to all threads
  float offset = maxvals[0];

  // compute exp and write the result to global memory
  for (uint i = tgIdx; i < C; i += bsize) {
    out[idx * C + i] = exp(x[i] - offset);
  }

  // okay now we calculated exp(x - max(x))
  // step 2: sum all the values and divide by the sum

  // thread coarsening for sum
  x = out + idx * C;
  float sumval = 0.0f;
  for (uint i = tgIdx; i < C; i += bsize) {
    sumval += x[i];
  }
  // within-simd group reduction for sumval
  sumval = simdGroupReduceSum(sumval, simdSize);

  // write sumval to shared memory
  if (laneID == 0) sumvals[simdGroupID] = sumval;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // inter-thread reduction of sum
  if (tgIdx == 0) {
    float val = sumvals[0];
    for (uint i = 1; i < simdGroupsPerBlock; ++i) {
      val += sumvals[i];
    }
    sumvals[0] = val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // broadcast the sum to all threads
  float sum = sumvals[0];

  // divide the whole row by the sum
  for (uint i = tgIdx; i < C; i += bsize) {
    out[idx * C + i] = x[i] / sum;
  }
}

kernel void residual_forward_kernel(device float* out [[ buffer(0) ]],
                                    device float* inp1 [[ buffer(1) ]],
                                    device float* inp2 [[ buffer(2) ]],
                                    uint tid [[ thread_position_in_grid ]]) {
  out[tid] = inp1[tid] + inp2[tid];
}

kernel void gelu_kernel(device float* out [[ buffer(0) ]],
                        device float* inp [[ buffer(1) ]],
                        uint tid [[ thread_position_in_grid ]]) {
  float xi = inp[tid];
  float s = sqrt(2.0f / M_PI);
  float cube = 0.044715f * xi * xi * xi;
  // Use precise variant for tanh since fast-math mode is on.
  out[tid] = 0.5f * xi * (1.0f + precise::tanh(s * (xi + cube)));
}

kernel void crossentropy_forward_kernel1(device float* losses [[ buffer(0) ]],
                                         device float* probs [[ buffer(1) ]],
                                         device int* targets [[ buffer(2) ]],
                                         constant uint& T [[ buffer(3) ]],
                                         constant uint& V [[ buffer(4) ]],
                                         uint tid [[ thread_position_in_grid ]]) {
  uint b = tid / T;
  uint t = tid % T;
  device float* probs_bt = probs + b * T * V + t * V;
  int ix = targets[b * T + t];
  losses[b * T + t] = -log(probs_bt[ix]);
}

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
#include "math.h"

void cpu_encoder_forward(float *out, int *inp, float *wte, float *wpe, int B,
                         int T, int C) {
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to the output position in out[b,t,:]
      float *out_bt = out + b * T * C + t * C;
      // get the index of the token at inp[b, t]
      int ix = inp[b * T + t];
      // seek to the position in wte corresponding to the token
      float *wte_ix = wte + ix * C;
      // seek to the position in wpe corresponding to the position
      float *wpe_t = wpe + t * C;
      // add the two vectors and store the result in out[b,t,:]
      for (int i = 0; i < C; i++) {
        out_bt[i] = wte_ix[i] + wpe_t[i];
      }
    }
  }
}

void cpu_encoder_backward(float *dwte, float *dwpe, float *dout, int *inp,
                          int B, int T, int C) {
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *dout_bt = dout + b * T * C + t * C;
      int ix = inp[b * T + t];
      float *dwte_ix = dwte + ix * C;
      float *dwpe_t = dwpe + t * C;
      for (int i = 0; i < C; i++) {
        float d = dout_bt[i];
        dwte_ix[i] += d;
        dwpe_t[i] += d;
      }
    }
  }
}

void cpu_layernorm_forward(float *out, float *mean, float *rstd, float *inp,
                           float *weight, float *bias, int B, int T, int C) {
  float eps = 1e-5f;
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to the input position inp[b,t,:]
      float *x = inp + b * T * C + t * C;
      // calculate the mean
      float m = 0.0f;
      for (int i = 0; i < C; i++) {
        m += x[i];
      }
      m = m / C;
      // calculate the variance (without any bias correction)
      float v = 0.0f;
      for (int i = 0; i < C; i++) {
        float xshift = x[i] - m;
        v += xshift * xshift;
      }
      v = v / C;
      // calculate the rstd
      float s = 1.0f / sqrtf(v + eps);
      // seek to the output position in out[b,t,:]
      float *out_bt = out + b * T * C + t * C;
      for (int i = 0; i < C; i++) {
        float n = (s * (x[i] - m));        // normalized output
        float o = n * weight[i] + bias[i]; // scale and shift it
        out_bt[i] = o;                     // write
      }
      // cache the mean and rstd for the backward pass later
      mean[b * T + t] = m;
      rstd[b * T + t] = s;
    }
  }
}

void cpu_layernorm_backward(float *dinp, float *dweight, float *dbias,
                            float *dout, float *inp, float *weight, float *mean,
                            float *rstd, int B, int T, int C) {
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *dout_bt = dout + b * T * C + t * C;
      float *inp_bt = inp + b * T * C + t * C;
      float *dinp_bt = dinp + b * T * C + t * C;
      float mean_bt = mean[b * T + t];
      float rstd_bt = rstd[b * T + t];

      // first: two reduce operations
      float dnorm_mean = 0.0f;
      float dnorm_norm_mean = 0.0f;
      for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
      }
      dnorm_mean = dnorm_mean / C;
      dnorm_norm_mean = dnorm_norm_mean / C;

      // now iterate again and accumulate all the gradients
      for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // gradient contribution to bias
        dbias[i] += dout_bt[i];
        // gradient contribution to weight
        dweight[i] += norm_bti * dout_bt[i];
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i;                    // term 1
        dval -= dnorm_mean;                 // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt;                    // final scale
        dinp_bt[i] += dval;
      }
    }
  }
}

void cpu_matmul_forward(float *out, float *inp, float *weight, float *bias,
                        int B, int T, int C, int OC) {
  // most of the running time is spent here and in matmul_backward
  // OC is short for "output channels"
  // inp is (B,T,C), weight is (OC, C), bias is (OC)
  // out will be (B,T,OC)
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *out_bt = out + b * T * OC + t * OC;
      float *inp_bt = inp + b * T * C + t * C;
      for (int o = 0; o < OC; o++) {
        float val = (bias != NULL) ? bias[o] : 0.0f;
        float *wrow = weight + o * C;
        for (int i = 0; i < C; i++) {
          val += inp_bt[i] * wrow[i];
        }
        out_bt[o] = val;
      }
    }
  }
}

void cpu_matmul_backward(float *dinp, float *dweight, float *dbias, float *dout,
                         float *inp, float *weight, int B, int T, int C,
                         int OC) {
  // most of the running time is spent here and in matmul_forward
  // this backward could be done in a single "round" of loops
  // but that doesn't afford an efficient parallelization strategy

  // backward into inp first, parallelize over B,T
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *dout_bt = dout + b * T * OC + t * OC;
      float *dinp_bt = dinp + b * T * C + t * C;
      for (int o = 0; o < OC; o++) {
        float *wrow = weight + o * C;
        float d = dout_bt[o];
        for (int i = 0; i < C; i++) {
          dinp_bt[i] += wrow[i] * d;
        }
      }
    }
  }
  // backward into weight/bias, parallelize over output channels OC
#pragma omp parallel for
  for (int o = 0; o < OC; o++) {
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        float *dout_bt = dout + b * T * OC + t * OC;
        float *inp_bt = inp + b * T * C + t * C;
        float *dwrow = dweight + o * C;
        float d = dout_bt[o];
        if (dbias != NULL) {
          dbias[o] += d;
        }
        for (int i = 0; i < C; i++) {
          dwrow[i] += inp_bt[i] * d;
        }
      }
    }
  }
}

void cpu_attention_forward(float *out, float *preatt, float *att, float *inp,
                           int B, int T, int C, int NH) {
  // input is (B, T, 3C) Q,K,V
  // preatt, att are (B, NH, T, T)
  // output is (B, T, C)
  int C3 = C * 3;
  int hs = C / NH; // head size
  float scale = 1.0 / sqrtf(hs);

#pragma omp parallel for collapse(3)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      for (int h = 0; h < NH; h++) {
        float *query_t = inp + b * T * C3 + t * C3 + h * hs;
        float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
        float *att_bth = att + b * NH * T * T + h * T * T + t * T;

        // pass 1: calculate query dot key and maxval
        float maxval = -10000.0f; // TODO something better
        for (int t2 = 0; t2 <= t; t2++) {
          float *key_t2 =
              inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

          // (query_t) dot (key_t2)
          float val = 0.0f;
          for (int i = 0; i < hs; i++) {
            val += query_t[i] * key_t2[i];
          }
          val *= scale;
          if (val > maxval) {
            maxval = val;
          }

          preatt_bth[t2] = val;
        }

        // pass 2: calculate the exp and keep track of sum
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
          float expv = expf(preatt_bth[t2] - maxval);
          expsum += expv;
          att_bth[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // pass 3: normalize to get the softmax
        for (int t2 = 0; t2 < T; t2++) {
          if (t2 <= t) {
            att_bth[t2] *= expsum_inv;
          } else {
            // causal attention mask. not strictly necessary to set to zero here
            // only doing this explicitly for debugging and checking to PyTorch
            att_bth[t2] = 0.0f;
          }
        }

        // pass 4: accumulate weighted values into the output of attention
        float *out_bth = out + b * T * C + t * C + h * hs;
        for (int i = 0; i < hs; i++) {
          out_bth[i] = 0.0f;
        }
        for (int t2 = 0; t2 <= t; t2++) {
          float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs +
                            C * 2; // +C*2 because it's value
          float att_btht2 = att_bth[t2];
          for (int i = 0; i < hs; i++) {
            out_bth[i] += att_btht2 * value_t2[i];
          }
        }
      }
    }
  }
}

void cpu_attention_backward(float *dinp, float *dpreatt, float *datt,
                            float *dout, float *inp, float *att, int B, int T,
                            int C, int NH) {
  // inp/dinp are (B, T, 3C) Q,K,V
  // att/datt/dpreatt are (B, NH, T, T)
  // dout is (B, T, C)
  int C3 = C * 3;
  int hs = C / NH; // head size
  float scale = 1.0 / sqrtf(hs);

  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      for (int h = 0; h < NH; h++) {
        float *att_bth = att + b * NH * T * T + h * T * T + t * T;
        float *datt_bth = datt + b * NH * T * T + h * T * T + t * T;
        float *dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
        float *dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
        float *query_t = inp + b * T * C3 + t * C3 + h * hs;

        // backward pass 4, through the value accumulation
        float *dout_bth = dout + b * T * C + t * C + h * hs;
        for (int t2 = 0; t2 <= t; t2++) {
          float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs +
                            C * 2; // +C*2 because it's value
          float *dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2;
          for (int i = 0; i < hs; i++) {
            // in the forward pass this was:
            // out_bth[i] += att_bth[t2] * value_t2[i];
            // so now we have:
            datt_bth[t2] += value_t2[i] * dout_bth[i];
            dvalue_t2[i] += att_bth[t2] * dout_bth[i];
          }
        }

        // backward pass 2 & 3, the softmax
        // note that softmax (like e.g. tanh) doesn't need the input (preatt) to
        // backward
        for (int t2 = 0; t2 <= t; t2++) {
          for (int t3 = 0; t3 <= t; t3++) {
            float indicator = t2 == t3 ? 1.0f : 0.0f;
            float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
            dpreatt_bth[t3] += local_derivative * datt_bth[t2];
          }
        }

        // backward pass 1, the query @ key matmul
        for (int t2 = 0; t2 <= t; t2++) {
          float *key_t2 =
              inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
          float *dkey_t2 =
              dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
          for (int i = 0; i < hs; i++) {
            // in the forward pass this was:
            // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
            // so now we have:
            dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
            dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
          }
        }
      }
    }
  }
}

void cpu_gelu_forward(float *out, float *inp, int N) {
  float s = sqrtf(2.0f / M_PI);
  for (int i = 0; i < N; i++) {
    float x = inp[i];
    float cube = 0.044715f * x * x * x;
    out[i] = 0.5f * x * (1.0f + tanhf(s * (x + cube)));
  }
}

void cpu_gelu_backward(float *dinp, float *inp, float *dout, int N) {
  float s = sqrtf(2.0f / M_PI);
  for (int i = 0; i < N; i++) {
    float x = inp[i];
    float cube = 0.044715f * x * x * x;
    float tanh_arg = s * (x + cube);
    float tanh_out = tanhf(tanh_arg);
    float coshf_out = coshf(tanh_arg);
    float sech_out = 1.0f / (coshf_out * coshf_out);
    float local_grad =
        0.5f * (1.0f + tanh_out) +
        x * 0.5f * sech_out * s * (1.0f + 3.0f * 0.044715f * x * x);
    dinp[i] += local_grad * dout[i];
  }
}

void cpu_residual_forward(float *out, float *inp1, float *inp2, int N) {
  for (int i = 0; i < N; i++) {
    out[i] = inp1[i] + inp2[i];
  }
}

void cpu_residual_backward(float *dinp1, float *dinp2, float *dout, int N) {
  for (int i = 0; i < N; i++) {
    dinp1[i] += dout[i];
    dinp2[i] += dout[i];
  }
}

void cpu_softmax_forward(float *probs, float *logits, int B, int T, int V) {
  // output: probs are (B,T,V) of the probabilities
  // input: logits is (B,T,V) of the unnormalized log probabilities
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // probs <- softmax(logits)
      float *logits_bt = logits + b * T * V + t * V;
      float *probs_bt = probs + b * T * V + t * V;

      float maxval = -10000.0f; // TODO something better
      for (int i = 0; i < V; i++) {
        if (logits_bt[i] > maxval) {
          maxval = logits_bt[i];
        }
      }
      float sum = 0.0f;
      for (int i = 0; i < V; i++) {
        probs_bt[i] = expf(logits_bt[i] - maxval);
        sum += probs_bt[i];
      }
      for (int i = 0; i < V; i++) {
        probs_bt[i] /= sum;
      }
    }
  }
}

void cpu_crossentropy_forward(float *losses, float *probs, int *targets, int B,
                              int T, int V) {
  // output: losses is (B,T) of the individual losses at each position
  // input: probs are (B,T,V) of the probabilities
  // input: targets is (B,T) of integers giving the correct index in logits
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // loss = -log(probs[target])
      float *probs_bt = probs + b * T * V + t * V;
      int ix = targets[b * T + t];
      losses[b * T + t] = -logf(probs_bt[ix]);
    }
  }
}

void cpu_crossentropy_softmax_backward(float *dlogits, float *dlosses,
                                       float *probs, int *targets, int B, int T,
                                       int V) {
  // backwards through both softmax and crossentropy
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *dlogits_bt = dlogits + b * T * V + t * V;
      float *probs_bt = probs + b * T * V + t * V;
      float dloss = dlosses[b * T + t];
      int ix = targets[b * T + t];
      for (int i = 0; i < V; i++) {
        float p = probs_bt[i];
        float indicator = i == ix ? 1.0f : 0.0f;
        dlogits_bt[i] += (p - indicator) * dloss;
      }
    }
  }
}

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

#ifndef llm_cpu_h
#define llm_cpu_h

#include <stdio.h>

#endif /* llm_cpu_h */
void cpu_encoder_forward(float *out, int *inp, float *wte, float *wpe, int B,
                         int T, int C);
void cpu_encoder_backward(float *dwte, float *dwpe, float *dout, int *inp,
                          int B, int T, int C);
void cpu_layernorm_forward(float *out, float *mean, float *rstd, float *inp,
                           float *weight, float *bias, int B, int T, int C);
void cpu_layernorm_backward(float *dinp, float *dweight, float *dbias,
                            float *dout, float *inp, float *weight, float *mean,
                            float *rstd, int B, int T, int C);
void cpu_matmul_forward(float *out, float *inp, float *weight, float *bias,
                        int B, int T, int C, int OC);
void cpu_matmul_backward(float *dinp, float *dweight, float *dbias, float *dout,
                         float *inp, float *weight, int B, int T, int C,
                         int OC);
void cpu_attention_forward(float *out, float *preatt, float *att, float *inp,
                           int B, int T, int C, int NH);
void cpu_attention_backward(float *dinp, float *dpreatt, float *datt,
                            float *dout, float *inp, float *att, int B, int T,
                            int C, int NH);
void cpu_gelu_forward(float *out, float *inp, int N);
void cpu_gelu_backward(float *dinp, float *inp, float *dout, int N);
void cpu_residual_forward(float *out, float *inp1, float *inp2, int N);
void cpu_residual_backward(float *dinp1, float *dinp2, float *dout, int N);
void cpu_softmax_forward(float *probs, float *logits, int B, int T, int V);
void cpu_crossentropy_forward(float *losses, float *probs, int *targets, int B,
                              int T, int V);
void cpu_crossentropy_softmax_backward(float *dlogits, float *dlosses,
                                       float *probs, int *targets, int B, int T,
                                       int V);

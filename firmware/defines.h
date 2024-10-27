#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 182
#define N_INPUT_2_1 180
#define N_INPUT_1_1 182
#define N_INPUT_2_1 180
#define ln_feature_out_9 182
#define ln_seq_out_9 180
#define att_feature_out_10 182
#define att_seq_out_10 180
#define att_feature_out_10 182
#define att_seq_out_10 180
#define att_feature_out_10 182
#define att_seq_out_10 180
#define ln_feature_out_12 182
#define ln_seq_out_12 180
#define ffn_feature_out_13 182
#define ffn_seq_out_13 180
#define ffn_feature_out_13 182
#define ffn_seq_out_13 180
#define ffn_feature_out_13 182
#define ffn_seq_out_13 180
#define ln_feature_out_15 182
#define ln_seq_out_15 180
#define att_feature_out_16 182
#define att_seq_out_16 180
#define att_feature_out_16 182
#define att_seq_out_16 180
#define att_feature_out_16 182
#define att_seq_out_16 180
#define ln_feature_out_18 182
#define ln_seq_out_18 180
#define ffn_feature_out_19 182
#define ffn_seq_out_19 180
#define ffn_feature_out_19 182
#define ffn_seq_out_19 180
#define ffn_feature_out_19 182
#define ffn_seq_out_19 180
#define ln_feature_out_21 182
#define ln_seq_out_21 180
#define att_feature_out_22 182
#define att_seq_out_22 180
#define att_feature_out_22 182
#define att_seq_out_22 180
#define att_feature_out_22 182
#define att_seq_out_22 180
#define ln_feature_out_24 182
#define ln_seq_out_24 180
#define ffn_feature_out_25 182
#define ffn_seq_out_25 180
#define ffn_feature_out_25 182
#define ffn_seq_out_25 180
#define ffn_feature_out_25 182
#define ffn_seq_out_25 180
#define ln_feature_out_27 182
#define ln_seq_out_27 180
#define att_feature_out_28 182
#define att_seq_out_28 180
#define att_feature_out_28 182
#define att_seq_out_28 180
#define att_feature_out_28 182
#define att_seq_out_28 180
#define ln_feature_out_30 182
#define ln_seq_out_30 180
#define ffn_feature_out_31 182
#define ffn_seq_out_31 180
#define ffn_feature_out_31 182
#define ffn_seq_out_31 180
#define ln_feature_out_4 182
#define ln_seq_out_4 180

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0>, 1*1> input_t;
typedef ap_fixed<80,32> layers_0_norm1_accum_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_0_norm1_scale_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_0_norm1_bias_t;
typedef nnet::array<ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0>, 1*1> layer9_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_0_norm1_sum_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_0_norm1_sum_sqr_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_0_norm1_mean_t;
typedef ap_ufixed<18,2,AP_RND_CONV,AP_SAT,0> layers_0_norm1_var_table_t;
typedef ap_uint<1> layer9_index;
typedef ap_fixed<80,32> layers_0_self_attn_accum_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_in_proj_bias_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_out_proj_weight_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask10_t;
typedef nnet::array<ap_fixed<18,9,AP_RND_CONV,AP_WRAP,0>, 1*1> layer10_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_0_self_attn_exp_table_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_0_self_attn_inv_table_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_scale_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_in_proj_out_t;
typedef ap_ufixed<18,6,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_row_sum_t;
typedef ap_fixed<18,8,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_out_proj_in_t;
typedef ap_uint<1> layer10_index;
typedef nnet::array<ap_fixed<18,9,AP_RND_CONV,AP_WRAP,0>, 1*1> layer11_t;
typedef ap_fixed<80,32> layers_0_norm2_accum_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_0_norm2_scale_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_0_norm2_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer12_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_0_norm2_sum_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_0_norm2_sum_sqr_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_0_norm2_mean_t;
typedef ap_ufixed<18,-2,AP_RND_CONV,AP_SAT,0> layers_0_norm2_var_table_t;
typedef ap_uint<1> layer12_index;
typedef ap_fixed<80,32> layers_0_ffn_accum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0> layers_0_ffn_in_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_0_ffn_in_proj_bias_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0> layers_0_ffn_out_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_0_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,9,AP_RND_CONV,AP_WRAP,0>, 1*1> layer13_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0> layers_0_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_0_ffn_cdf_table_t;
typedef ap_uint<1> layer13_index;
typedef nnet::array<ap_fixed<18,10,AP_RND_CONV,AP_WRAP,0>, 1*1> layer14_t;
typedef ap_fixed<80,32> layers_1_norm1_accum_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_1_norm1_scale_t;
typedef ap_fixed<18,-2,AP_RND_CONV,AP_WRAP,0> layers_1_norm1_bias_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer15_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_1_norm1_sum_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_1_norm1_sum_sqr_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_1_norm1_mean_t;
typedef ap_ufixed<18,-2,AP_RND_CONV,AP_SAT,0> layers_1_norm1_var_table_t;
typedef ap_uint<1> layer15_index;
typedef ap_fixed<80,32> layers_1_self_attn_accum_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_in_proj_bias_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_out_proj_weight_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask16_t;
typedef nnet::array<ap_fixed<18,10,AP_RND_CONV,AP_WRAP,0>, 1*1> layer16_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_1_self_attn_exp_table_t;
typedef ap_ufixed<18,-2,AP_RND_CONV,AP_SAT,0> layers_1_self_attn_inv_table_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_scale_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_in_proj_out_t;
typedef ap_ufixed<18,6,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_row_sum_t;
typedef ap_fixed<18,10,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_out_proj_in_t;
typedef ap_uint<1> layer16_index;
typedef nnet::array<ap_fixed<18,10,AP_RND_CONV,AP_WRAP,0>, 1*1> layer17_t;
typedef ap_fixed<80,32> layers_1_norm2_accum_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_1_norm2_scale_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_1_norm2_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer18_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_1_norm2_sum_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_1_norm2_sum_sqr_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_1_norm2_mean_t;
typedef ap_ufixed<18,-5,AP_RND_CONV,AP_SAT,0> layers_1_norm2_var_table_t;
typedef ap_uint<1> layer18_index;
typedef ap_fixed<80,32> layers_1_ffn_accum_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_1_ffn_in_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_1_ffn_in_proj_bias_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0> layers_1_ffn_out_proj_weight_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_1_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,11,AP_RND_CONV,AP_WRAP,0>, 1*1> layer19_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_1_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_1_ffn_cdf_table_t;
typedef ap_uint<1> layer19_index;
typedef nnet::array<ap_fixed<18,11,AP_RND_CONV,AP_WRAP,0>, 1*1> layer20_t;
typedef ap_fixed<80,32> layers_2_norm1_accum_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_2_norm1_scale_t;
typedef ap_fixed<18,-2,AP_RND_CONV,AP_WRAP,0> layers_2_norm1_bias_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer21_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_2_norm1_sum_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_2_norm1_sum_sqr_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_2_norm1_mean_t;
typedef ap_ufixed<18,-5,AP_RND_CONV,AP_SAT,0> layers_2_norm1_var_table_t;
typedef ap_uint<1> layer21_index;
typedef ap_fixed<80,32> layers_2_self_attn_accum_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_in_proj_bias_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_out_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask22_t;
typedef nnet::array<ap_fixed<18,10,AP_RND_CONV,AP_WRAP,0>, 1*1> layer22_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_2_self_attn_exp_table_t;
typedef ap_ufixed<18,-2,AP_RND_CONV,AP_SAT,0> layers_2_self_attn_inv_table_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_scale_t;
typedef ap_fixed<18,6,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_in_proj_out_t;
typedef ap_ufixed<18,6,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_row_sum_t;
typedef ap_fixed<18,11,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_out_proj_in_t;
typedef ap_uint<1> layer22_index;
typedef nnet::array<ap_fixed<18,11,AP_RND_CONV,AP_WRAP,0>, 1*1> layer23_t;
typedef ap_fixed<80,32> layers_2_norm2_accum_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_2_norm2_scale_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_2_norm2_bias_t;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer24_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_2_norm2_sum_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_2_norm2_sum_sqr_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_2_norm2_mean_t;
typedef ap_ufixed<18,-6,AP_RND_CONV,AP_SAT,0> layers_2_norm2_var_table_t;
typedef ap_uint<1> layer24_index;
typedef ap_fixed<80,32> layers_2_ffn_accum_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_2_ffn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_2_ffn_in_proj_bias_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0> layers_2_ffn_out_proj_weight_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0> layers_2_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,10,AP_RND_CONV,AP_WRAP,0>, 1*1> layer25_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_2_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_2_ffn_cdf_table_t;
typedef ap_uint<1> layer25_index;
typedef nnet::array<ap_fixed<18,12,AP_RND_CONV,AP_WRAP,0>, 1*1> layer26_t;
typedef ap_fixed<80,32> layers_3_norm1_accum_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_3_norm1_scale_t;
typedef ap_fixed<18,-1,AP_RND_CONV,AP_WRAP,0> layers_3_norm1_bias_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer27_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_3_norm1_sum_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_3_norm1_sum_sqr_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_3_norm1_mean_t;
typedef ap_ufixed<18,-6,AP_RND_CONV,AP_SAT,0> layers_3_norm1_var_table_t;
typedef ap_uint<1> layer27_index;
typedef ap_fixed<80,32> layers_3_self_attn_accum_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_in_proj_bias_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_out_proj_weight_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask28_t;
typedef nnet::array<ap_fixed<18,11,AP_RND_CONV,AP_WRAP,0>, 1*1> layer28_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_3_self_attn_exp_table_t;
typedef ap_ufixed<18,-2,AP_RND_CONV,AP_SAT,0> layers_3_self_attn_inv_table_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_scale_t;
typedef ap_fixed<18,6,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_in_proj_out_t;
typedef ap_ufixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_row_sum_t;
typedef ap_fixed<18,11,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_out_proj_in_t;
typedef ap_uint<1> layer28_index;
typedef nnet::array<ap_fixed<18,12,AP_RND_CONV,AP_WRAP,0>, 1*1> layer29_t;
typedef ap_fixed<80,32> layers_3_norm2_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_3_norm2_scale_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_3_norm2_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer30_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_3_norm2_sum_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_3_norm2_sum_sqr_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0> layers_3_norm2_mean_t;
typedef ap_ufixed<18,-7,AP_RND_CONV,AP_SAT,0> layers_3_norm2_var_table_t;
typedef ap_uint<1> layer30_index;
typedef ap_fixed<80,32> layers_3_ffn_accum_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_3_ffn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_3_ffn_in_proj_bias_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0> layers_3_ffn_out_proj_weight_t;
typedef ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0> layers_3_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,10,AP_RND_CONV,AP_WRAP,0>, 1*1> layer31_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> layers_3_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_3_ffn_cdf_table_t;
typedef ap_uint<1> layer31_index;
typedef nnet::array<ap_fixed<18,12,AP_RND_CONV,AP_WRAP,0>, 1*1> layer32_t;
typedef ap_fixed<80,32> norm_accum_t;
typedef ap_fixed<18,-2,AP_RND_CONV,AP_WRAP,0> norm_scale_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> norm_bias_t;
typedef nnet::array<ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0>, 1*1> result_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> norm_sum_t;
typedef ap_fixed<18,7,AP_RND_CONV,AP_WRAP,0> norm_sum_sqr_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0> norm_mean_t;
typedef ap_ufixed<18,-8,AP_RND_CONV,AP_SAT,0> norm_var_table_t;
typedef ap_uint<1> layer4_index;

#endif

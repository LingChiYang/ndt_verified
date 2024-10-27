#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_feedforwardnetwork_stream.h"
#include "nnet_utils/nnet_layernorm_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_multiheadattention_stream.h"
#include "nnet_utils/nnet_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/scale9.h"
#include "weights/bias9.h"
#include "weights/in_proj_weight10.h"
#include "weights/in_proj_bias10.h"
#include "weights/out_proj_weight10.h"
#include "weights/out_proj_bias10.h"
#include "weights/mask10.h"
#include "weights/scale12.h"
#include "weights/bias12.h"
#include "weights/in_proj_weight13.h"
#include "weights/in_proj_bias13.h"
#include "weights/out_proj_weight13.h"
#include "weights/out_proj_bias13.h"
#include "weights/scale15.h"
#include "weights/bias15.h"
#include "weights/in_proj_weight16.h"
#include "weights/in_proj_bias16.h"
#include "weights/out_proj_weight16.h"
#include "weights/out_proj_bias16.h"
#include "weights/mask16.h"
#include "weights/scale18.h"
#include "weights/bias18.h"
#include "weights/in_proj_weight19.h"
#include "weights/in_proj_bias19.h"
#include "weights/out_proj_weight19.h"
#include "weights/out_proj_bias19.h"
#include "weights/scale21.h"
#include "weights/bias21.h"
#include "weights/in_proj_weight22.h"
#include "weights/in_proj_bias22.h"
#include "weights/out_proj_weight22.h"
#include "weights/out_proj_bias22.h"
#include "weights/mask22.h"
#include "weights/scale24.h"
#include "weights/bias24.h"
#include "weights/in_proj_weight25.h"
#include "weights/in_proj_bias25.h"
#include "weights/out_proj_weight25.h"
#include "weights/out_proj_bias25.h"
#include "weights/scale27.h"
#include "weights/bias27.h"
#include "weights/in_proj_weight28.h"
#include "weights/in_proj_bias28.h"
#include "weights/out_proj_weight28.h"
#include "weights/out_proj_bias28.h"
#include "weights/mask28.h"
#include "weights/scale30.h"
#include "weights/bias30.h"
#include "weights/in_proj_weight31.h"
#include "weights/in_proj_bias31.h"
#include "weights/out_proj_weight31.h"
#include "weights/out_proj_bias31.h"
#include "weights/scale4.h"
#include "weights/bias4.h"

// hls-fpga-machine-learning insert layer-config
// layers_0_norm1
struct config9 : nnet::layernorm_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned table_size = 1024;
    static const unsigned table_range = 16;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_0_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_0_norm1_mean_t mean_t;
    typedef layers_0_norm1_sum_t sum_t;   
    typedef layers_0_norm1_bias_t bias_t;
    typedef layers_0_norm1_scale_t scale_t;
    typedef layers_0_norm1_var_table_t var_table_t;
    typedef layers_0_norm1_accum_t accum_t;
};

// layers_0_self_attn
struct config10 : nnet::mha_config {
    static const unsigned n_head = 2;
    static const unsigned head_dim = 91;
    static const unsigned embed_dim = 182;
    static const unsigned seq_len = 180;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 1024;
    static const unsigned exp_table_size = 1024;
    typedef layers_0_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_0_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_0_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_0_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask10_t mask_t;
    typedef layers_0_self_attn_exp_table_t exp_table_t;
    typedef layers_0_self_attn_inv_table_t inv_table_t;
    typedef layers_0_self_attn_scale_t scale_t;
    typedef layers_0_self_attn_accum_t accum_t;
    typedef layers_0_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_0_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_0_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 64;
    static const unsigned exp_range = 8;
    
};

// layers_0_add1
struct config11 : nnet::merge_config {
    static const unsigned n_elem = 182*180*1;
};

// layers_0_norm2
struct config12 : nnet::layernorm_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned table_size = 1024;
    static const unsigned table_range = 512;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_0_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_0_norm2_mean_t mean_t;
    typedef layers_0_norm2_sum_t sum_t;   
    typedef layers_0_norm2_bias_t bias_t;
    typedef layers_0_norm2_scale_t scale_t;
    typedef layers_0_norm2_var_table_t var_table_t;
    typedef layers_0_norm2_accum_t accum_t;
};

// layers_0_ffn
struct config13 : nnet::ffn_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned hidden_dim = 128;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = false;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_0_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_0_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_0_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_0_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_0_ffn_hidden_t hidden_t;
    typedef layers_0_ffn_accum_t accum_t;
    typedef layers_0_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_0_add2
struct config14 : nnet::merge_config {
    static const unsigned n_elem = 182*180*1;
};

// layers_1_norm1
struct config15 : nnet::layernorm_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned table_size = 1024;
    static const unsigned table_range = 2048;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_1_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_1_norm1_mean_t mean_t;
    typedef layers_1_norm1_sum_t sum_t;   
    typedef layers_1_norm1_bias_t bias_t;
    typedef layers_1_norm1_scale_t scale_t;
    typedef layers_1_norm1_var_table_t var_table_t;
    typedef layers_1_norm1_accum_t accum_t;
};

// layers_1_self_attn
struct config16 : nnet::mha_config {
    static const unsigned n_head = 2;
    static const unsigned head_dim = 91;
    static const unsigned embed_dim = 182;
    static const unsigned seq_len = 180;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 1024;
    static const unsigned exp_table_size = 1024;
    typedef layers_1_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_1_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_1_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_1_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask16_t mask_t;
    typedef layers_1_self_attn_exp_table_t exp_table_t;
    typedef layers_1_self_attn_inv_table_t inv_table_t;
    typedef layers_1_self_attn_scale_t scale_t;
    typedef layers_1_self_attn_accum_t accum_t;
    typedef layers_1_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_1_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_1_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 64;
    static const unsigned exp_range = 8;
    
};

// layers_1_add1
struct config17 : nnet::merge_config {
    static const unsigned n_elem = 182*180*1;
};

// layers_1_norm2
struct config18 : nnet::layernorm_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned table_size = 1024;
    static const unsigned table_range = 8192;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_1_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_1_norm2_mean_t mean_t;
    typedef layers_1_norm2_sum_t sum_t;   
    typedef layers_1_norm2_bias_t bias_t;
    typedef layers_1_norm2_scale_t scale_t;
    typedef layers_1_norm2_var_table_t var_table_t;
    typedef layers_1_norm2_accum_t accum_t;
};

// layers_1_ffn
struct config19 : nnet::ffn_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned hidden_dim = 128;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = false;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_1_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_1_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_1_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_1_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_1_ffn_hidden_t hidden_t;
    typedef layers_1_ffn_accum_t accum_t;
    typedef layers_1_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_1_add2
struct config20 : nnet::merge_config {
    static const unsigned n_elem = 182*180*1;
};

// layers_2_norm1
struct config21 : nnet::layernorm_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned table_size = 1024;
    static const unsigned table_range = 16384;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_2_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_2_norm1_mean_t mean_t;
    typedef layers_2_norm1_sum_t sum_t;   
    typedef layers_2_norm1_bias_t bias_t;
    typedef layers_2_norm1_scale_t scale_t;
    typedef layers_2_norm1_var_table_t var_table_t;
    typedef layers_2_norm1_accum_t accum_t;
};

// layers_2_self_attn
struct config22 : nnet::mha_config {
    static const unsigned n_head = 2;
    static const unsigned head_dim = 91;
    static const unsigned embed_dim = 182;
    static const unsigned seq_len = 180;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 1024;
    static const unsigned exp_table_size = 1024;
    typedef layers_2_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_2_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_2_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_2_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask22_t mask_t;
    typedef layers_2_self_attn_exp_table_t exp_table_t;
    typedef layers_2_self_attn_inv_table_t inv_table_t;
    typedef layers_2_self_attn_scale_t scale_t;
    typedef layers_2_self_attn_accum_t accum_t;
    typedef layers_2_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_2_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_2_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 64;
    static const unsigned exp_range = 8;
    
};

// layers_2_add1
struct config23 : nnet::merge_config {
    static const unsigned n_elem = 182*180*1;
};

// layers_2_norm2
struct config24 : nnet::layernorm_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned table_size = 1024;
    static const unsigned table_range = 65536;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_2_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_2_norm2_mean_t mean_t;
    typedef layers_2_norm2_sum_t sum_t;   
    typedef layers_2_norm2_bias_t bias_t;
    typedef layers_2_norm2_scale_t scale_t;
    typedef layers_2_norm2_var_table_t var_table_t;
    typedef layers_2_norm2_accum_t accum_t;
};

// layers_2_ffn
struct config25 : nnet::ffn_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned hidden_dim = 128;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = false;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_2_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_2_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_2_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_2_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_2_ffn_hidden_t hidden_t;
    typedef layers_2_ffn_accum_t accum_t;
    typedef layers_2_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_2_add2
struct config26 : nnet::merge_config {
    static const unsigned n_elem = 182*180*1;
};

// layers_3_norm1
struct config27 : nnet::layernorm_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned table_size = 1024;
    static const unsigned table_range = 65536;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_3_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_3_norm1_mean_t mean_t;
    typedef layers_3_norm1_sum_t sum_t;   
    typedef layers_3_norm1_bias_t bias_t;
    typedef layers_3_norm1_scale_t scale_t;
    typedef layers_3_norm1_var_table_t var_table_t;
    typedef layers_3_norm1_accum_t accum_t;
};

// layers_3_self_attn
struct config28 : nnet::mha_config {
    static const unsigned n_head = 2;
    static const unsigned head_dim = 91;
    static const unsigned embed_dim = 182;
    static const unsigned seq_len = 180;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 1024;
    static const unsigned exp_table_size = 1024;
    typedef layers_3_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_3_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_3_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_3_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask28_t mask_t;
    typedef layers_3_self_attn_exp_table_t exp_table_t;
    typedef layers_3_self_attn_inv_table_t inv_table_t;
    typedef layers_3_self_attn_scale_t scale_t;
    typedef layers_3_self_attn_accum_t accum_t;
    typedef layers_3_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_3_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_3_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 64;
    static const unsigned exp_range = 8;
    
};

// layers_3_add1
struct config29 : nnet::merge_config {
    static const unsigned n_elem = 182*180*1;
};

// layers_3_norm2
struct config30 : nnet::layernorm_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned table_size = 1024;
    static const unsigned table_range = 131072;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_3_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_3_norm2_mean_t mean_t;
    typedef layers_3_norm2_sum_t sum_t;   
    typedef layers_3_norm2_bias_t bias_t;
    typedef layers_3_norm2_scale_t scale_t;
    typedef layers_3_norm2_var_table_t var_table_t;
    typedef layers_3_norm2_accum_t accum_t;
};

// layers_3_ffn
struct config31 : nnet::ffn_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned hidden_dim = 128;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = false;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_3_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_3_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_3_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_3_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_3_ffn_hidden_t hidden_t;
    typedef layers_3_ffn_accum_t accum_t;
    typedef layers_3_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_3_add2
struct config32 : nnet::merge_config {
    static const unsigned n_elem = 182*180*1;
};

// norm
struct config4 : nnet::layernorm_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned table_size = 1024;
    static const unsigned table_range = 262144;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef norm_sum_sqr_t sum_sqr_t;
    typedef norm_mean_t mean_t;
    typedef norm_sum_t sum_t;   
    typedef norm_bias_t bias_t;
    typedef norm_scale_t scale_t;
    typedef norm_var_table_t var_table_t;
    typedef norm_accum_t accum_t;
};


#endif

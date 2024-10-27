#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &src,
    hls::stream<result_t> &layer4_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=src,layer4_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<layers_0_norm1_scale_t, 182>(scale9, "scale9.txt");
        nnet::load_weights_from_txt<layers_0_norm1_bias_t, 182>(bias9, "bias9.txt");
        nnet::load_weights_from_txt<layers_0_self_attn_in_proj_weight_t, 99372>(in_proj_weight10, "in_proj_weight10.txt");
        nnet::load_weights_from_txt<layers_0_self_attn_in_proj_bias_t, 546>(in_proj_bias10, "in_proj_bias10.txt");
        nnet::load_weights_from_txt<layers_0_self_attn_out_proj_weight_t, 33124>(out_proj_weight10, "out_proj_weight10.txt");
        nnet::load_weights_from_txt<layers_0_self_attn_out_proj_bias_t, 182>(out_proj_bias10, "out_proj_bias10.txt");
        nnet::load_weights_from_txt<mask10_t, 32400>(mask10, "mask10.txt");
        nnet::load_weights_from_txt<layers_0_norm2_scale_t, 182>(scale12, "scale12.txt");
        nnet::load_weights_from_txt<layers_0_norm2_bias_t, 182>(bias12, "bias12.txt");
        nnet::load_weights_from_txt<layers_0_ffn_in_proj_weight_t, 23296>(in_proj_weight13, "in_proj_weight13.txt");
        nnet::load_weights_from_txt<layers_0_ffn_in_proj_bias_t, 128>(in_proj_bias13, "in_proj_bias13.txt");
        nnet::load_weights_from_txt<layers_0_ffn_out_proj_weight_t, 23296>(out_proj_weight13, "out_proj_weight13.txt");
        nnet::load_weights_from_txt<layers_0_ffn_out_proj_bias_t, 182>(out_proj_bias13, "out_proj_bias13.txt");
        nnet::load_weights_from_txt<layers_1_norm1_scale_t, 182>(scale15, "scale15.txt");
        nnet::load_weights_from_txt<layers_1_norm1_bias_t, 182>(bias15, "bias15.txt");
        nnet::load_weights_from_txt<layers_1_self_attn_in_proj_weight_t, 99372>(in_proj_weight16, "in_proj_weight16.txt");
        nnet::load_weights_from_txt<layers_1_self_attn_in_proj_bias_t, 546>(in_proj_bias16, "in_proj_bias16.txt");
        nnet::load_weights_from_txt<layers_1_self_attn_out_proj_weight_t, 33124>(out_proj_weight16, "out_proj_weight16.txt");
        nnet::load_weights_from_txt<layers_1_self_attn_out_proj_bias_t, 182>(out_proj_bias16, "out_proj_bias16.txt");
        nnet::load_weights_from_txt<mask16_t, 32400>(mask16, "mask16.txt");
        nnet::load_weights_from_txt<layers_1_norm2_scale_t, 182>(scale18, "scale18.txt");
        nnet::load_weights_from_txt<layers_1_norm2_bias_t, 182>(bias18, "bias18.txt");
        nnet::load_weights_from_txt<layers_1_ffn_in_proj_weight_t, 23296>(in_proj_weight19, "in_proj_weight19.txt");
        nnet::load_weights_from_txt<layers_1_ffn_in_proj_bias_t, 128>(in_proj_bias19, "in_proj_bias19.txt");
        nnet::load_weights_from_txt<layers_1_ffn_out_proj_weight_t, 23296>(out_proj_weight19, "out_proj_weight19.txt");
        nnet::load_weights_from_txt<layers_1_ffn_out_proj_bias_t, 182>(out_proj_bias19, "out_proj_bias19.txt");
        nnet::load_weights_from_txt<layers_2_norm1_scale_t, 182>(scale21, "scale21.txt");
        nnet::load_weights_from_txt<layers_2_norm1_bias_t, 182>(bias21, "bias21.txt");
        nnet::load_weights_from_txt<layers_2_self_attn_in_proj_weight_t, 99372>(in_proj_weight22, "in_proj_weight22.txt");
        nnet::load_weights_from_txt<layers_2_self_attn_in_proj_bias_t, 546>(in_proj_bias22, "in_proj_bias22.txt");
        nnet::load_weights_from_txt<layers_2_self_attn_out_proj_weight_t, 33124>(out_proj_weight22, "out_proj_weight22.txt");
        nnet::load_weights_from_txt<layers_2_self_attn_out_proj_bias_t, 182>(out_proj_bias22, "out_proj_bias22.txt");
        nnet::load_weights_from_txt<mask22_t, 32400>(mask22, "mask22.txt");
        nnet::load_weights_from_txt<layers_2_norm2_scale_t, 182>(scale24, "scale24.txt");
        nnet::load_weights_from_txt<layers_2_norm2_bias_t, 182>(bias24, "bias24.txt");
        nnet::load_weights_from_txt<layers_2_ffn_in_proj_weight_t, 23296>(in_proj_weight25, "in_proj_weight25.txt");
        nnet::load_weights_from_txt<layers_2_ffn_in_proj_bias_t, 128>(in_proj_bias25, "in_proj_bias25.txt");
        nnet::load_weights_from_txt<layers_2_ffn_out_proj_weight_t, 23296>(out_proj_weight25, "out_proj_weight25.txt");
        nnet::load_weights_from_txt<layers_2_ffn_out_proj_bias_t, 182>(out_proj_bias25, "out_proj_bias25.txt");
        nnet::load_weights_from_txt<layers_3_norm1_scale_t, 182>(scale27, "scale27.txt");
        nnet::load_weights_from_txt<layers_3_norm1_bias_t, 182>(bias27, "bias27.txt");
        nnet::load_weights_from_txt<layers_3_self_attn_in_proj_weight_t, 99372>(in_proj_weight28, "in_proj_weight28.txt");
        nnet::load_weights_from_txt<layers_3_self_attn_in_proj_bias_t, 546>(in_proj_bias28, "in_proj_bias28.txt");
        nnet::load_weights_from_txt<layers_3_self_attn_out_proj_weight_t, 33124>(out_proj_weight28, "out_proj_weight28.txt");
        nnet::load_weights_from_txt<layers_3_self_attn_out_proj_bias_t, 182>(out_proj_bias28, "out_proj_bias28.txt");
        nnet::load_weights_from_txt<mask28_t, 32400>(mask28, "mask28.txt");
        nnet::load_weights_from_txt<layers_3_norm2_scale_t, 182>(scale30, "scale30.txt");
        nnet::load_weights_from_txt<layers_3_norm2_bias_t, 182>(bias30, "bias30.txt");
        nnet::load_weights_from_txt<layers_3_ffn_in_proj_weight_t, 23296>(in_proj_weight31, "in_proj_weight31.txt");
        nnet::load_weights_from_txt<layers_3_ffn_in_proj_bias_t, 128>(in_proj_bias31, "in_proj_bias31.txt");
        nnet::load_weights_from_txt<layers_3_ffn_out_proj_weight_t, 23296>(out_proj_weight31, "out_proj_weight31.txt");
        nnet::load_weights_from_txt<layers_3_ffn_out_proj_bias_t, 182>(out_proj_bias31, "out_proj_bias31.txt");
        nnet::load_weights_from_txt<norm_scale_t, 182>(scale4, "scale4.txt");
        nnet::load_weights_from_txt<norm_bias_t, 182>(bias4, "bias4.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<input_t> layer33_cpy1("layer33_cpy1");
    #pragma HLS STREAM variable=layer33_cpy1 depth=32760
    hls::stream<input_t> layer33_cpy2("layer33_cpy2");
    #pragma HLS STREAM variable=layer33_cpy2 depth=32760
    nnet::clone_stream<input_t, input_t, 32760>(src, layer33_cpy1, layer33_cpy2); // clone_src

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=32760
    nnet::LayerNormalize<input_t, layer9_t, config9>(layer33_cpy1, layer9_out, scale9, bias9); // layers_0_norm1

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=32760
    nnet::MultiHeadAttention<layer9_t, layer10_t, config10>(layer9_out, layer10_out, in_proj_weight10, in_proj_bias10, out_proj_weight10, out_proj_bias10, mask10); // layers_0_self_attn

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=32760
    nnet::add<layer10_t, input_t, layer11_t, config11>(layer10_out, layer33_cpy2, layer11_out); // layers_0_add1

    hls::stream<layer11_t> layer34_cpy1("layer34_cpy1");
    #pragma HLS STREAM variable=layer34_cpy1 depth=32760
    hls::stream<layer11_t> layer34_cpy2("layer34_cpy2");
    #pragma HLS STREAM variable=layer34_cpy2 depth=32760
    nnet::clone_stream<layer11_t, layer11_t, 32760>(layer11_out, layer34_cpy1, layer34_cpy2); // clone_layers_0_add1

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=32760
    nnet::LayerNormalize<layer11_t, layer12_t, config12>(layer34_cpy1, layer12_out, scale12, bias12); // layers_0_norm2

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=32760
    nnet::FeedForwardNetwork<layer12_t, layer13_t, config13>(layer12_out, layer13_out, in_proj_weight13, in_proj_bias13, out_proj_weight13, out_proj_bias13); // layers_0_ffn

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=32760
    nnet::add<layer13_t, layer11_t, layer14_t, config14>(layer13_out, layer34_cpy2, layer14_out); // layers_0_add2

    hls::stream<layer14_t> layer35_cpy1("layer35_cpy1");
    #pragma HLS STREAM variable=layer35_cpy1 depth=32760
    hls::stream<layer14_t> layer35_cpy2("layer35_cpy2");
    #pragma HLS STREAM variable=layer35_cpy2 depth=32760
    nnet::clone_stream<layer14_t, layer14_t, 32760>(layer14_out, layer35_cpy1, layer35_cpy2); // clone_layers_0_add2

    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=32760
    nnet::LayerNormalize<layer14_t, layer15_t, config15>(layer35_cpy1, layer15_out, scale15, bias15); // layers_1_norm1

    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=32760
    nnet::MultiHeadAttention<layer15_t, layer16_t, config16>(layer15_out, layer16_out, in_proj_weight16, in_proj_bias16, out_proj_weight16, out_proj_bias16, mask16); // layers_1_self_attn

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=32760
    nnet::add<layer16_t, layer14_t, layer17_t, config17>(layer16_out, layer35_cpy2, layer17_out); // layers_1_add1

    hls::stream<layer17_t> layer36_cpy1("layer36_cpy1");
    #pragma HLS STREAM variable=layer36_cpy1 depth=32760
    hls::stream<layer17_t> layer36_cpy2("layer36_cpy2");
    #pragma HLS STREAM variable=layer36_cpy2 depth=32760
    nnet::clone_stream<layer17_t, layer17_t, 32760>(layer17_out, layer36_cpy1, layer36_cpy2); // clone_layers_1_add1

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=32760
    nnet::LayerNormalize<layer17_t, layer18_t, config18>(layer36_cpy1, layer18_out, scale18, bias18); // layers_1_norm2

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS STREAM variable=layer19_out depth=32760
    nnet::FeedForwardNetwork<layer18_t, layer19_t, config19>(layer18_out, layer19_out, in_proj_weight19, in_proj_bias19, out_proj_weight19, out_proj_bias19); // layers_1_ffn

    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=32760
    nnet::add<layer19_t, layer17_t, layer20_t, config20>(layer19_out, layer36_cpy2, layer20_out); // layers_1_add2

    hls::stream<layer20_t> layer37_cpy1("layer37_cpy1");
    #pragma HLS STREAM variable=layer37_cpy1 depth=32760
    hls::stream<layer20_t> layer37_cpy2("layer37_cpy2");
    #pragma HLS STREAM variable=layer37_cpy2 depth=32760
    nnet::clone_stream<layer20_t, layer20_t, 32760>(layer20_out, layer37_cpy1, layer37_cpy2); // clone_layers_1_add2

    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=32760
    nnet::LayerNormalize<layer20_t, layer21_t, config21>(layer37_cpy1, layer21_out, scale21, bias21); // layers_2_norm1

    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS STREAM variable=layer22_out depth=32760
    nnet::MultiHeadAttention<layer21_t, layer22_t, config22>(layer21_out, layer22_out, in_proj_weight22, in_proj_bias22, out_proj_weight22, out_proj_bias22, mask22); // layers_2_self_attn

    hls::stream<layer23_t> layer23_out("layer23_out");
    #pragma HLS STREAM variable=layer23_out depth=32760
    nnet::add<layer22_t, layer20_t, layer23_t, config23>(layer22_out, layer37_cpy2, layer23_out); // layers_2_add1

    hls::stream<layer23_t> layer38_cpy1("layer38_cpy1");
    #pragma HLS STREAM variable=layer38_cpy1 depth=32760
    hls::stream<layer23_t> layer38_cpy2("layer38_cpy2");
    #pragma HLS STREAM variable=layer38_cpy2 depth=32760
    nnet::clone_stream<layer23_t, layer23_t, 32760>(layer23_out, layer38_cpy1, layer38_cpy2); // clone_layers_2_add1

    hls::stream<layer24_t> layer24_out("layer24_out");
    #pragma HLS STREAM variable=layer24_out depth=32760
    nnet::LayerNormalize<layer23_t, layer24_t, config24>(layer38_cpy1, layer24_out, scale24, bias24); // layers_2_norm2

    hls::stream<layer25_t> layer25_out("layer25_out");
    #pragma HLS STREAM variable=layer25_out depth=32760
    nnet::FeedForwardNetwork<layer24_t, layer25_t, config25>(layer24_out, layer25_out, in_proj_weight25, in_proj_bias25, out_proj_weight25, out_proj_bias25); // layers_2_ffn

    hls::stream<layer26_t> layer26_out("layer26_out");
    #pragma HLS STREAM variable=layer26_out depth=32760
    nnet::add<layer25_t, layer23_t, layer26_t, config26>(layer25_out, layer38_cpy2, layer26_out); // layers_2_add2

    hls::stream<layer26_t> layer39_cpy1("layer39_cpy1");
    #pragma HLS STREAM variable=layer39_cpy1 depth=32760
    hls::stream<layer26_t> layer39_cpy2("layer39_cpy2");
    #pragma HLS STREAM variable=layer39_cpy2 depth=32760
    nnet::clone_stream<layer26_t, layer26_t, 32760>(layer26_out, layer39_cpy1, layer39_cpy2); // clone_layers_2_add2

    hls::stream<layer27_t> layer27_out("layer27_out");
    #pragma HLS STREAM variable=layer27_out depth=32760
    nnet::LayerNormalize<layer26_t, layer27_t, config27>(layer39_cpy1, layer27_out, scale27, bias27); // layers_3_norm1

    hls::stream<layer28_t> layer28_out("layer28_out");
    #pragma HLS STREAM variable=layer28_out depth=32760
    nnet::MultiHeadAttention<layer27_t, layer28_t, config28>(layer27_out, layer28_out, in_proj_weight28, in_proj_bias28, out_proj_weight28, out_proj_bias28, mask28); // layers_3_self_attn

    hls::stream<layer29_t> layer29_out("layer29_out");
    #pragma HLS STREAM variable=layer29_out depth=32760
    nnet::add<layer28_t, layer26_t, layer29_t, config29>(layer28_out, layer39_cpy2, layer29_out); // layers_3_add1

    hls::stream<layer29_t> layer40_cpy1("layer40_cpy1");
    #pragma HLS STREAM variable=layer40_cpy1 depth=32760
    hls::stream<layer29_t> layer40_cpy2("layer40_cpy2");
    #pragma HLS STREAM variable=layer40_cpy2 depth=32760
    nnet::clone_stream<layer29_t, layer29_t, 32760>(layer29_out, layer40_cpy1, layer40_cpy2); // clone_layers_3_add1

    hls::stream<layer30_t> layer30_out("layer30_out");
    #pragma HLS STREAM variable=layer30_out depth=32760
    nnet::LayerNormalize<layer29_t, layer30_t, config30>(layer40_cpy1, layer30_out, scale30, bias30); // layers_3_norm2

    hls::stream<layer31_t> layer31_out("layer31_out");
    #pragma HLS STREAM variable=layer31_out depth=32760
    nnet::FeedForwardNetwork<layer30_t, layer31_t, config31>(layer30_out, layer31_out, in_proj_weight31, in_proj_bias31, out_proj_weight31, out_proj_bias31); // layers_3_ffn

    hls::stream<layer32_t> layer32_out("layer32_out");
    #pragma HLS STREAM variable=layer32_out depth=32760
    nnet::add<layer31_t, layer29_t, layer32_t, config32>(layer31_out, layer40_cpy2, layer32_out); // layers_3_add2

    nnet::LayerNormalize<layer32_t, result_t, config4>(layer32_out, layer4_out, scale4, bias4); // norm

}

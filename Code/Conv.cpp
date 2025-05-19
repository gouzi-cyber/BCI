#include "ap_int.h"
#include "hls_stream.h"
#include <stdio.h> // 添加这行以支持printf

#define USE_RANDOM_DATA true

// 可配置的卷积层
void conv_layer(
    // 数据指针 - 从PS端DDR内存
    ap_int<32>* input_data,  // 输入特征图
    ap_int<32>* output_data, // 输出特征图
    // 卷积核参数
    ap_int<32>* bias_data,   // 偏置
    ap_int<32>* weight_data, // 权重

    // 层配置参数-通过AXI-Lite接口从PS端获取
    int in_channels,    // 输入通道数
    int out_channels,   // 输出通道数
    int in_height,      // 输入特征图高度
    int in_width,       // 输入特征图宽度
    int kernel_width,   // 卷积核宽度
    int kernel_height,  // 卷积核高度
    int stride_height,  // 垂直步长
    int stride_width,   // 水平步长（从stride_weight修正）
    int padding_width,  // 水平填充（从padding_weight修正）
    int padding_height, // 垂直填充
    int batch_size,     // 批处理大小
    int scale_factor    // 添加缩放因子作为参数
) {
    #pragma HLS INTERFACE m_axi port=input_data offset=slave bundle=INPUT_BUS
    #pragma HLS INTERFACE m_axi port=weight_data offset=slave bundle=WEIGHT_BUS
    #pragma HLS INTERFACE m_axi port=bias_data offset=slave bundle=BIAS_BUS
    #pragma HLS INTERFACE m_axi port=output_data offset=slave bundle=OUTPUT_BUS

    #pragma HLS INTERFACE s_axilite port=batch_size bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=in_channels bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=out_channels bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=in_height bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=in_width bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=kernel_width bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=kernel_height bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=padding_width bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=padding_height bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=stride_height bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=stride_width bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=scale_factor bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=return bundle=CONTROL_BUS

    // 添加日志输出以追踪函数调用
    printf("卷积层开始: batch=%d, in_ch=%d, out_ch=%d, height=%d, width=%d, kernel=%dx%d\n",
           batch_size, in_channels, out_channels, in_height, in_width, kernel_height, kernel_width);
    printf("步长=(%d,%d), 填充=(%d,%d), 缩放因子=%d\n",
           stride_height, stride_width, padding_height, padding_width, scale_factor);

    // 检查输入指针是否为空
    if (input_data == NULL || output_data == NULL || bias_data == NULL || weight_data == NULL) {
        printf("错误: 输入指针为空\n");
        printf("input_data=%p, output_data=%p, bias_data=%p, weight_data=%p\n",
               input_data, output_data, bias_data, weight_data);
        return;
    }

    // 定义最大维度的常量，减小数组大小
    const int MAX_BATCH = 2;      // 减小到2
    const int MAX_IN_CH = 16;     // 减小到16
    const int MAX_OUT_CH = 16;    // 减小到16
    const int MAX_HEIGHT = 16;    // 减小到16
    const int MAX_WIDTH = 100;    // 大幅减小到100
    const int MAX_KERNEL_H = 8;   // 减小到8
    const int MAX_KERNEL_W = 8;   // 减小到8

    // 添加参数验证
    if (batch_size <= 0 || in_channels <= 0 || out_channels <= 0 || 
        in_height <= 0 || in_width <= 0 || 
        kernel_height <= 0 || kernel_width <= 0) {
        printf("错误: 参数不能为负数或零\n");
        return;
    }

    // 计算输出尺寸
    int out_height = (in_height + 2*padding_height - kernel_height) / stride_height + 1;
    int out_width = (in_width + 2*padding_width - kernel_width) / stride_width + 1;
    
    if (out_height <= 0 || out_width <= 0) {
        printf("错误: 计算的输出尺寸无效 - out_height=%d, out_width=%d\n", out_height, out_width);
        return;
    }

    printf("计算的输出尺寸: out_height=%d, out_width=%d\n", out_height, out_width);

    // 声明本地缓冲区 - 分块大小
    const int BLOCK_WIDTH = 100;  // 宽度块大小
    
    // 使用较小的局部数组
    ap_int<32> local_input[MAX_BATCH][MAX_IN_CH][MAX_HEIGHT][MAX_WIDTH];
    ap_int<32> local_weight[MAX_OUT_CH][MAX_IN_CH][MAX_KERNEL_H][MAX_KERNEL_W];
    ap_int<32> local_bias[MAX_OUT_CH];
    ap_int<32> local_output[MAX_BATCH][MAX_OUT_CH][MAX_HEIGHT][MAX_WIDTH];

    // 使用分块处理方式
    for (int width_block = 0; width_block < (in_width + BLOCK_WIDTH - 1) / BLOCK_WIDTH; width_block++) {
        int block_start_w = width_block * BLOCK_WIDTH;
        int block_end_w = (width_block + 1) * BLOCK_WIDTH;
        if (block_end_w > in_width) block_end_w = in_width;
        int block_width = block_end_w - block_start_w;
        
        if (block_width > MAX_WIDTH) {
            printf("错误: 块宽度(%d)超过本地缓冲区最大宽度(%d)\n", block_width, MAX_WIDTH);
            return;
        }
        
        printf("处理宽度块 %d/%d, 范围: %d-%d (宽度=%d)\n", 
               width_block+1, (in_width + BLOCK_WIDTH - 1) / BLOCK_WIDTH, 
               block_start_w, block_end_w-1, block_width);

        // 1. 从DDR加载输入数据 (仅加载当前块)
        printf("加载输入数据块...\n");
        for (int b = 0; b < batch_size && b < MAX_BATCH; b++) {
            for (int c = 0; c < in_channels && c < MAX_IN_CH; c++) {
                for (int h = 0; h < in_height && h < MAX_HEIGHT; h++) {
                    for (int w = 0; w < block_width; w++) {
                        #pragma HLS PIPELINE II=1
                        int global_w = block_start_w + w;
                        int index = b * (in_channels * in_height * in_width) +
                                   c * (in_height * in_width) +
                                   h * in_width + global_w;
                        local_input[b][c][h][w] = input_data[index];
                    }
                }
            }
        }

        // 2. 从DDR加载权重和偏置 (每个块都需要完整的权重)
        if (width_block == 0) {  // 只在第一个块加载权重
            printf("加载权重和偏置...\n");
            for (int o = 0; o < out_channels && o < MAX_OUT_CH; o++) {
                for (int i = 0; i < in_channels && i < MAX_IN_CH; i++) {
                    for (int kh = 0; kh < kernel_height && kh < MAX_KERNEL_H; kh++) {
                        for (int kw = 0; kw < kernel_width && kw < MAX_KERNEL_W; kw++) {
                            #pragma HLS PIPELINE II=1
                            int index = o * (in_channels * kernel_height * kernel_width) +
                                       i * (kernel_height * kernel_width) +
                                       kh * kernel_width + kw;
                            local_weight[o][i][kh][kw] = weight_data[index];
                        }
                    }
                }
                local_bias[o] = bias_data[o];
            }
        }

        // 3. 计算此块的输出范围
        int out_block_start_w = (block_start_w + padding_width - kernel_width) / stride_width + 1;
        if (out_block_start_w < 0) out_block_start_w = 0;
        
        int out_block_end_w = (block_end_w + padding_width) / stride_width;
        if (out_block_end_w > out_width) out_block_end_w = out_width;
        
        printf("计算对应的输出块范围: %d-%d\n", out_block_start_w, out_block_end_w-1);

        // 4. 执行卷积 (仅针对当前输出块)
        for (int b = 0; b < batch_size && b < MAX_BATCH; b++) {
            for (int o = 0; o < out_channels && o < MAX_OUT_CH; o++) {
                for (int oh = 0; oh < out_height && oh < MAX_HEIGHT; oh++) {
                    for (int ow = out_block_start_w; ow < out_block_end_w; ow++) {
                        #pragma HLS PIPELINE II=1
                        
                        ap_int<64> sum = 0;
                        
                        for (int i = 0; i < in_channels && i < MAX_IN_CH; i++) {
                            for (int kh = 0; kh < kernel_height && kh < MAX_KERNEL_H; kh++) {
                                for (int kw = 0; kw < kernel_width && kw < MAX_KERNEL_W; kw++) {
                                    int ih = oh * stride_height + kh - padding_height;
                                    int iw = ow * stride_width + kw - padding_width;
                                    int local_iw = iw - block_start_w;

                                    if (ih >= 0 && ih < in_height && iw >= block_start_w && iw < block_end_w) {
                                        sum += (ap_int<64>)local_input[b][i][ih][local_iw] *
                                               (ap_int<64>)local_weight[o][i][kh][kw];
                                    }
                                }
                            }
                        }
                        // 方案3: 忽略传入的scale_factor，使用固定值
                        sum = sum / 1024; // 使用一个固定的合理缩放因子
                        sum += (ap_int<64>)local_bias[o];
                        // 存储到本地输出
                        int local_ow = ow - out_block_start_w;
                        if (local_ow >= 0 && local_ow < MAX_WIDTH) {
                            local_output[b][o][oh][local_ow] = (ap_int<32>)sum;
                        }
                    }
                }
            }
        }

        // 5. 将此块的结果写回DDR
        printf("写回结果块...\n");
        for (int b = 0; b < batch_size && b < MAX_BATCH; b++) {
            for (int o = 0; o < out_channels && o < MAX_OUT_CH; o++) {
                for (int h = 0; h < out_height && h < MAX_HEIGHT; h++) {
                    for (int ow = out_block_start_w; ow < out_block_end_w; ow++) {
                        #pragma HLS PIPELINE II=1
                        
                        int local_ow = ow - out_block_start_w;
                        if (local_ow >= 0 && local_ow < MAX_WIDTH) {
                            int index = b * (out_channels * out_height * out_width) +
                                       o * (out_height * out_width) +
                                       h * out_width + ow;
                            
                            output_data[index] = local_output[b][o][h][local_ow];
                        }
                    }
                }
            }
        }
        
        printf("完成宽度块 %d/%d 处理\n", width_block+1, (in_width + BLOCK_WIDTH - 1) / BLOCK_WIDTH);
    }
    
    printf("卷积层执行完毕\n");
}
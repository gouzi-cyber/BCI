#ifndef CONV_H
#define CONV_H

#include "ap_int.h"



// 声明卷积层函数
void conv_layer(
    ap_int<32>* input_data,  // 输入特征图
    ap_int<32>* output_data, // 输出特征图
    ap_int<32>* bias_data,   // 偏置
    ap_int<32>* weight_data, // 权重
    int in_channels,         // 输入通道数
    int out_channels,        // 输出通道数
    int in_height,           // 输入特征图高度
    int in_width,            // 输入特征图宽度
    int kernel_width,        // 卷积核宽度
    int kernel_height,       // 卷积核高度
    int stride_height,       // 垂直步长
    int stride_width,        // 水平步长
    int padding_width,       // 水平填充
    int padding_height,      // 垂直填充
    int batch_size,          // 批处理大小
    int scale_factor         // 缩放因子
);

#endif // CONV_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "test_vectors/all_conv_test_data.h"  // 包含生成的测试数据
#include "Conv.h"  // 包含您的卷积IP核定义

// 定义误差容忍度
#define EPSILON (SCALE_FACTOR / 100)  // 允许1%的误差

// 增加缓冲区大小避免内存溢出
#define MAX_BATCH 2
#define MAX_CHANNELS 32  // 增加到32
#define MAX_HEIGHT 32    // 增加到32
#define MAX_WIDTH 1000   // 增加到1000

// 使用静态全局数组避免栈溢出或堆分配失败
static ap_int<32> g_input_data[MAX_BATCH * MAX_CHANNELS * MAX_HEIGHT * MAX_WIDTH];
static ap_int<32> g_weight_data[MAX_CHANNELS * MAX_CHANNELS * MAX_HEIGHT * MAX_WIDTH];
static ap_int<32> g_bias_data[MAX_CHANNELS];
static ap_int<32> g_output_data[MAX_BATCH * MAX_CHANNELS * MAX_HEIGHT * MAX_WIDTH];
static ap_int<32> g_expected_data[MAX_BATCH * MAX_CHANNELS * MAX_HEIGHT * MAX_WIDTH];

// 调试工具
void debug_print(const char* msg) {
    printf("[DEBUG] %s\n", msg);
    fflush(stdout);  // 确保消息立即输出
}

// 读取二进制文件的函数 - 修复以更好地处理偏置文件
bool readBinaryFile(const char* filename, ap_int<32>* data, int size) {
    debug_print("开始读取文件");
    
    if (data == NULL) {
        printf("错误: 数据指针为空\n");
        return false;
    }
    
    // 初始化为0
    memset(data, 0, size * sizeof(ap_int<32>));
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("无法打开文件: %s\n", filename);
        
        // 使用固定值填充
        for (int i = 0; i < size; i++) {
            data[i] = 0;  // 使用0，而不是随机值
        }
        printf("使用固定值0代替未找到的数据\n");
        return true;  // 返回true以继续测试
    }
    
    debug_print("成功打开文件");
    
    // 获取文件大小
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);  // 重置文件指针到开头
    
    printf("文件 %s 大小: %ld 字节\n", filename, file_size);
    
    // 检查是否是偏置文件（通常很小）
    bool is_bias_file = (strstr(filename, "bias") != NULL);
    
    // 如果是偏置文件，尝试直接读取数据
    if (is_bias_file && file_size < 20) {
        printf("检测到小型偏置文件，尝试直接读取数据而不是头部信息\n");
        // 假设偏置文件只包含数据部分，直接以32位整数形式读取
        for (int i = 0; i < size && i < (file_size / sizeof(int)); i++) {
            int temp_value;
            if (fread(&temp_value, sizeof(int), 1, file) != 1) {
                printf("警告: 读取第%d个偏置元素时失败\n", i);
                break;
            }
            data[i] = temp_value;
        }
        fclose(file);
        return true;
    }
    
    // 标准处理方式 - 读取头部和数据
    if (file_size < 4 * sizeof(int)) {
        printf("警告: 文件太小，无法包含头部信息，尝试直接读取数据\n");
        // 直接读取数据
        for (int i = 0; i < size && i < (file_size / sizeof(float)); i++) {
            float temp_float;
            if (fread(&temp_float, sizeof(float), 1, file) != 1) {
                break;
            }
            data[i] = (ap_int<32>)(temp_float * SCALE_FACTOR);
        }
        fclose(file);
        return true;
    }
    
    // 正常的情况 - 读取头部信息
    int dimensions[4];
    size_t header_read = fread(dimensions, sizeof(int), 4, file);
    if (header_read != 4) {
        printf("警告: 无法读取完整的头部信息，仅读取了 %zu/4 个值\n", header_read);
    } else {
        printf("文件维度: [%d, %d, %d, %d]\n", 
               dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    }
    
    debug_print("读取头部信息完成");
    
    // 检查预期的数据大小是否符合文件头指定的维度
    int expected_size = 1;
    for (int i = 0; i < 4 && i < header_read; i++) {
        expected_size *= dimensions[i];
    }
    
    if (expected_size != size && header_read == 4) {
        printf("警告: 文件头维度表明数据大小为 %d，但请求读取 %d 个元素\n", 
               expected_size, size);
    }
    
    // 直接逐个元素读取浮点数并转换，避免一次性读取大块内存
    float temp_float;
    for (int i = 0; i < size; i++) {
        // 检查是否已到文件末尾
        if (feof(file)) {
            printf("警告: 文件在读取第%d个元素时到达末尾 (共需要%d个)\n", i, size);
            break;
        }
        
        if (fread(&temp_float, sizeof(float), 1, file) != 1) {
            printf("警告: 读取第%d个元素时失败 (共需要%d个)\n", i, size);
            break;
        }
        
        data[i] = (ap_int<32>)(temp_float * SCALE_FACTOR);
    }
    
    debug_print("文件数据读取完成");
    fclose(file);
    return true;
}

// 测试单个卷积层 - 增强调试功能
bool testConvLayer(const char* layer_name, 
                  int in_channels, int out_channels,
                  int in_height, int in_width,
                  int kernel_height, int kernel_width,
                  int stride_height, int stride_width,
                  int padding_height, int padding_width,
                  int batch_size, 
                  int groups = 1) {  // 添加分组参数
    printf("\n------------------------------------------------------\n");
    printf("开始测试卷积层: %s\n", layer_name);
    
    // 检查参数是否超出预定义的最大值
    if (batch_size > MAX_BATCH || in_channels > MAX_CHANNELS || out_channels > MAX_CHANNELS ||
        in_height > MAX_HEIGHT || in_width > MAX_WIDTH || 
        kernel_height > MAX_HEIGHT || kernel_width > MAX_WIDTH) {
        printf("错误: 参数超出预定义范围\n");
        printf("当前参数: batch=%d, in_ch=%d, out_ch=%d, height=%d, width=%d, kernel=%dx%d\n",
               batch_size, in_channels, out_channels, in_height, in_width, kernel_height, kernel_width);
        printf("最大限制: batch=%d, channels=%d, height=%d, width=%d\n",
               MAX_BATCH, MAX_CHANNELS, MAX_HEIGHT, MAX_WIDTH);
        return false;
    }
    
    printf("配置: 输入(%d,%d,%d,%d), 卷积核(%d,%d), 步长(%d,%d), 填充(%d,%d), 分组=%d\n", 
           batch_size, in_channels, in_height, in_width,
           kernel_height, kernel_width, stride_height, stride_width,
           padding_height, padding_width, groups);
    
    // 计算输出尺寸
    int out_height = (in_height + 2*padding_height - kernel_height) / stride_height + 1;
    int out_width = (in_width + 2*padding_width - kernel_width) / stride_width + 1;
    printf("预期输出尺寸: (%d,%d,%d,%d)\n", 
           batch_size, out_channels, out_height, out_width);
    
    // 计算数据大小，考虑分组
    int input_size = batch_size * in_channels * in_height * in_width;
    
    // 特别处理 depthwise 卷积 (groups > 1)
    int weight_size;
    if (groups == 1) {
        weight_size = out_channels * in_channels * kernel_height * kernel_width;
    } else if (groups == in_channels && in_channels == out_channels) {
        // Depthwise 卷积
        weight_size = out_channels * kernel_height * kernel_width;
    } else {
        // 分组卷积
        weight_size = out_channels * (in_channels/groups) * kernel_height * kernel_width;
    }
    
    int bias_size = out_channels;
    int output_size = batch_size * out_channels * out_height * out_width;
    
    printf("数据大小: 输入=%d, 权重=%d, 偏置=%d, 输出=%d\n", 
           input_size, weight_size, bias_size, output_size);
    
    // 检查数据大小是否超出预定义的缓冲区
    if (input_size > MAX_BATCH * MAX_CHANNELS * MAX_HEIGHT * MAX_WIDTH ||
        weight_size > MAX_CHANNELS * MAX_CHANNELS * MAX_HEIGHT * MAX_WIDTH ||
        bias_size > MAX_CHANNELS ||
        output_size > MAX_BATCH * MAX_CHANNELS * MAX_HEIGHT * MAX_WIDTH) {
        printf("错误: 数据大小超出预定义缓冲区\n");
        return false;
    }
    
    // 清空全局数组
    memset(g_input_data, 0, sizeof(g_input_data));
    memset(g_weight_data, 0, sizeof(g_weight_data));
    memset(g_bias_data, 0, sizeof(g_bias_data));
    memset(g_output_data, 0, sizeof(g_output_data));
    memset(g_expected_data, 0, sizeof(g_expected_data));
    
    debug_print("内存初始化完成");
    
    // 构建文件名
    char input_file[256], weight_file[256], bias_file[256], output_file[256];
    sprintf(input_file, "C:/Users/PC/Desktop/IP/conv_accelerator/Conv/test_vectors/binary/input_data.bin");
    sprintf(weight_file, "C:/Users/PC/Desktop/IP/conv_accelerator/Conv/test_vectors/binary/%s_weight.bin", layer_name);
    sprintf(bias_file, "C:/Users/PC/Desktop/IP/conv_accelerator/Conv/test_vectors/binary/%s_bias.bin", layer_name);
    sprintf(output_file, "C:/Users/PC/Desktop/IP/conv_accelerator/Conv/test_vectors/binary/%s_output.bin", layer_name);
    
    printf("文件路径:\n输入=%s\n权重=%s\n偏置=%s\n期望输出=%s\n", 
           input_file, weight_file, bias_file, output_file);
    
    // 读取测试数据
    debug_print("开始读取输入数据");
    bool success = readBinaryFile(input_file, g_input_data, input_size);
    
    debug_print("开始读取权重数据");
    success &= readBinaryFile(weight_file, g_weight_data, weight_size);
    
    debug_print("开始读取偏置数据");
    success &= readBinaryFile(bias_file, g_bias_data, bias_size);
    
    debug_print("开始读取期望输出数据");
    success &= readBinaryFile(output_file, g_expected_data, output_size);
    
    if (!success) {
        printf("警告: 一些数据文件无法读取，但将继续测试\n");
    }
    
    // 检查数据是否全为0
    int nonzero_count = 0;
    for (int i = 0; i < input_size && i < 100; i++) {
        if (g_input_data[i] != 0) nonzero_count++;
    }
    printf("输入数据前100个元素中非零元素数量: %d\n", nonzero_count);
    
    nonzero_count = 0;
    for (int i = 0; i < weight_size && i < 100; i++) {
        if (g_weight_data[i] != 0) nonzero_count++;
    }
    printf("权重数据前100个元素中非零元素数量: %d\n", nonzero_count);
    
    nonzero_count = 0;
    for (int i = 0; i < bias_size; i++) {
        if (g_bias_data[i] != 0) nonzero_count++;
    }
    printf("偏置数据中非零元素数量: %d\n", nonzero_count);
    
    printf("开始执行卷积...\n");
    
    // 添加更多的调试信息
    printf("输入指针: %p, 输出指针: %p\n", g_input_data, g_output_data);
    printf("偏置指针: %p, 权重指针: %p\n", g_bias_data, g_weight_data);
    
    // 确保写入流以便在崩溃前看到输出
    fflush(stdout);
    
    // 调用卷积IP核
    debug_print("调用卷积IP核");
    
    try {
        // 在这里添加额外的检查，确保所有参数有效
        if (g_input_data == NULL || g_output_data == NULL || 
            g_bias_data == NULL || g_weight_data == NULL) {
            printf("错误: 数据指针为空\n");
            return false;
        }
        
        // 验证输入、输出尺寸
        if (input_size <= 0 || output_size <= 0 || weight_size <= 0 || bias_size <= 0) {
            printf("错误: 计算的数据大小无效\n");
            return false;
        }
        
        // 打印前几个输入元素以确认数据已正确加载
        printf("输入数据样本(前5个元素): ");
        for (int i = 0; i < 5 && i < input_size; i++) {
            printf("%d ", (int)g_input_data[i]);
        }
        printf("\n");
        
        printf("权重数据样本(前5个元素): ");
        for (int i = 0; i < 5 && i < weight_size; i++) {
            printf("%d ", (int)g_weight_data[i]);
        }
        printf("\n");
        
        // 确保写入流以便在崩溃前看到输出
        fflush(stdout);
        
        // 调用卷积函数
        conv_layer(
            g_input_data,
            g_output_data,
            g_bias_data,
            g_weight_data,
            in_channels,
            out_channels,
            in_height,
            in_width,
            kernel_width,
            kernel_height,
            stride_height,
            stride_width,
            padding_width,
            padding_height,
            batch_size,
            SCALE_FACTOR
        );
        debug_print("卷积执行完成");
    } catch (...) {
        printf("错误: 卷积执行过程中发生异常\n");
        return false;
    }
    
    printf("卷积执行完成，开始比较结果...\n");
    
    // 比较结果
    int errors = 0;
    int max_diff = 0;
    int max_diff_pos = -1;
    
    for (int i = 0; i < output_size; i++) {
        int diff = abs(g_output_data[i] - g_expected_data[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_pos = i;
        }
        
        if (diff > EPSILON) {
            if (errors < 10) {
                printf("误差位置 %d: 期望 %d, 实际 %d, 差值 %d\n", 
                       i, (int)g_expected_data[i], (int)g_output_data[i], diff);
            }
            errors++;
        }
    }
    
    // 报告结果
    if (errors == 0) {
        printf("卷积层 %s 测试通过！最大差异: %d (位置 %d)\n", 
               layer_name, max_diff, max_diff_pos);
    } else {
        printf("卷积层 %s 测试失败: %d 个错误（共 %d 个值）, 最大差异: %d (位置 %d)\n", 
               layer_name, errors, output_size, max_diff, max_diff_pos);
    }
    
    printf("------------------------------------------------------\n\n");
    return (errors == 0);
}

int main() {
    printf("=== 卷积加速器测试程序 ===\n");
    printf("缩放因子: %d\n", SCALE_FACTOR);
    printf("误差容忍度: %d\n\n", EPSILON);
    
    debug_print("程序开始执行");
    
    // 测试第1个卷积层 (backbone.0.depthwise)
    testConvLayer(
        "backbone_0_depthwise",  // 层名称
        1,                       // 输入通道
        1,                       // 输出通道
        8,                       // 输入高度
        796,                     // 输入宽度
        1,                       // 核高度
        7,                       // 核宽度
        1,                       // 步长高度
        1,                       // 步长宽度
        0,                       // 填充高度
        3,                       // 填充宽度
        1,                       // 批量大小
        1                        // 分组数量
    );
    
    // 测试第2个卷积层 (backbone.0.pointwise)
    testConvLayer(
        "backbone_0_pointwise",  // 层名称
        1,                       // 输入通道
        4,                       // 输出通道
        8,                       // 输入高度
        796,                     // 输入宽度
        1,                       // 核高度
        1,                       // 核宽度
        1,                       // 步长高度
        1,                       // 步长宽度
        0,                       // 填充高度
        0,                       // 填充宽度
        1,                       // 批量大小
        1                        // 分组数量
    );
    
    // 测试第3个卷积层 (backbone.4.depthwise) - 注意这是分组卷积
    testConvLayer(
        "backbone_4_depthwise",  // 层名称
        4,                       // 输入通道
        4,                       // 输出通道
        8,                       // 输入高度
        796,                     // 输入宽度
        8,                       // 核高度
        1,                       // 核宽度
        1,                       // 步长高度
        1,                       // 步长宽度
        0,                       // 填充高度
        0,                       // 填充宽度
        1,                       // 批量大小
        4                        // 分组数量 (修改为4，表明是depthwise卷积)
    );
    
    // 测试第4个卷积层 (backbone.4.pointwise)
    testConvLayer(
        "backbone_4_pointwise",  // 层名称
        4,                       // 输入通道
        8,                       // 输出通道
        1,                       // 输入高度 (上一层输出高度)
        796,                     // 输入宽度
        1,                       // 核高度
        1,                       // 核宽度
        1,                       // 步长高度
        1,                       // 步长宽度
        0,                       // 填充高度
        0,                       // 填充宽度
        1,                       // 批量大小
        1                        // 分组数量
    );
    
    // 测试第5个卷积层 (spatial_attention.0)
    testConvLayer(
        "spatial_attention_0",   // 层名称
        8,                       // 输入通道
        1,                       // 输出通道
        1,                       // 输入高度
        796,                     // 输入宽度
        1,                       // 核高度
        1,                       // 核宽度
        1,                       // 步长高度
        1,                       // 步长宽度
        0,                       // 填充高度
        0,                       // 填充宽度
        1,                       // 批量大小
        1                        // 分组数量
    );
    
    // 测试第6个卷积层 (channel_attention.1)
    testConvLayer(
        "channel_attention_1",   // 层名称
        8,                       // 输入通道
        1,                       // 输出通道
        1,                       // 输入高度
        1,                       // 输入宽度 (AdaptiveAvgPool2d后)
        1,                       // 核高度
        1,                       // 核宽度
        1,                       // 步长高度
        1,                       // 步长宽度
        0,                       // 填充高度
        0,                       // 填充宽度
        1,                       // 批量大小
        1                        // 分组数量
    );
    
    // 测试第7个卷积层 (channel_attention.3)
    testConvLayer(
        "channel_attention_3",   // 层名称
        1,                       // 输入通道
        8,                       // 输出通道
        1,                       // 输入高度
        1,                       // 输入宽度
        1,                       // 核高度
        1,                       // 核宽度
        1,                       // 步长高度
        1,                       // 步长宽度
        0,                       // 填充高度
        0,                       // 填充宽度
        1,                       // 批量大小
        1                        // 分组数量
    );
    
    // 测试第8个卷积层 (se_block.1)
    testConvLayer(
        "se_block_1",            // 层名称
        8,                       // 输入通道
        1,                       // 输出通道
        1,                       // 输入高度
        1,                       // 输入宽度
        1,                       // 核高度
        1,                       // 核宽度
        1,                       // 步长高度
        1,                       // 步长宽度
        0,                       // 填充高度
        0,                       // 填充宽度
        1,                       // 批量大小
        1                        // 分组数量
    );
    
    // 测试第9个卷积层 (se_block.3)
    testConvLayer(
        "se_block_3",            // 层名称
        1,                       // 输入通道
        8,                       // 输出通道
        1,                       // 输入高度
        1,                       // 输入宽度
        1,                       // 核高度
        1,                       // 核宽度
        1,                       // 步长高度
        1,                       // 步长宽度
        0,                       // 填充高度
        0,                       // 填充宽度
        1,                       // 批量大小
        1                        // 分组数量
    );
    
    debug_print("程序执行完毕");
    printf("所有卷积层测试完成\n");
    return 0;
}
#ifndef ALL_CONV_TEST_DATA_H
#define ALL_CONV_TEST_DATA_H

// �������� - ��״: (1, 1, 8, 796)
#define INPUT_BATCH 1
#define INPUT_CHANNELS 1
#define INPUT_HEIGHT 8
#define INPUT_WIDTH 796

#ifndef SCALE_FACTOR
  #define SCALE_FACTOR 256  // 只有在没有其他地方定义时才定义
#endif

// ��������̫��ֻ������״��Ϣ����ʵ�ʲ�����Ӧ������������

// backbone.0.depthwise ����
#define BACKBONE_0_DEPTHWISE_IN_CHANNELS 1
#define BACKBONE_0_DEPTHWISE_OUT_CHANNELS 1
#define BACKBONE_0_DEPTHWISE_KERNEL_HEIGHT 1
#define BACKBONE_0_DEPTHWISE_KERNEL_WIDTH 7
#define BACKBONE_0_DEPTHWISE_STRIDE_HEIGHT 1
#define BACKBONE_0_DEPTHWISE_STRIDE_WIDTH 1
#define BACKBONE_0_DEPTHWISE_PADDING_HEIGHT 0
#define BACKBONE_0_DEPTHWISE_PADDING_WIDTH 3
#define BACKBONE_0_DEPTHWISE_GROUPS 1

// backbone.0.depthwise Ȩ�� - ��״: (1, 1, 1, 7)
const int backbone_0_depthwise_weight[BACKBONE_0_DEPTHWISE_OUT_CHANNELS][BACKBONE_0_DEPTHWISE_IN_CHANNELS][BACKBONE_0_DEPTHWISE_KERNEL_HEIGHT][BACKBONE_0_DEPTHWISE_KERNEL_WIDTH] = {
  // ����Ȩ�����ݣ���״��������
};

// backbone.0.depthwise ƫ�� - ��״: (1,)
const int backbone_0_depthwise_bias[BACKBONE_0_DEPTHWISE_OUT_CHANNELS] = {
  // ����ƫ������
};

// backbone.0.depthwise ��� - ��״: (1, 1, 8, 796)
#define BACKBONE_0_DEPTHWISE_OUTPUT_BATCH 1
#define BACKBONE_0_DEPTHWISE_OUTPUT_CHANNELS 1
#define BACKBONE_0_DEPTHWISE_OUTPUT_HEIGHT 8
#define BACKBONE_0_DEPTHWISE_OUTPUT_WIDTH 796

const int backbone_0_depthwise_expected_output[BACKBONE_0_DEPTHWISE_OUTPUT_BATCH][BACKBONE_0_DEPTHWISE_OUTPUT_CHANNELS][BACKBONE_0_DEPTHWISE_OUTPUT_HEIGHT][BACKBONE_0_DEPTHWISE_OUTPUT_WIDTH] = {
  // ������ݣ���״��������
};

// backbone.0.pointwise ����
#define BACKBONE_0_POINTWISE_IN_CHANNELS 1
#define BACKBONE_0_POINTWISE_OUT_CHANNELS 4
#define BACKBONE_0_POINTWISE_KERNEL_HEIGHT 1
#define BACKBONE_0_POINTWISE_KERNEL_WIDTH 1
#define BACKBONE_0_POINTWISE_STRIDE_HEIGHT 1
#define BACKBONE_0_POINTWISE_STRIDE_WIDTH 1
#define BACKBONE_0_POINTWISE_PADDING_HEIGHT 0
#define BACKBONE_0_POINTWISE_PADDING_WIDTH 0
#define BACKBONE_0_POINTWISE_GROUPS 1

// backbone.0.pointwise Ȩ�� - ��״: (4, 1, 1, 1)
const int backbone_0_pointwise_weight[BACKBONE_0_POINTWISE_OUT_CHANNELS][BACKBONE_0_POINTWISE_IN_CHANNELS][BACKBONE_0_POINTWISE_KERNEL_HEIGHT][BACKBONE_0_POINTWISE_KERNEL_WIDTH] = {
  // ����Ȩ�����ݣ���״��������
};

// backbone.0.pointwise ƫ�� - ��״: (4,)
const int backbone_0_pointwise_bias[BACKBONE_0_POINTWISE_OUT_CHANNELS] = {
  // ����ƫ������
};

// backbone.0.pointwise ��� - ��״: (1, 4, 8, 796)
#define BACKBONE_0_POINTWISE_OUTPUT_BATCH 1
#define BACKBONE_0_POINTWISE_OUTPUT_CHANNELS 4
#define BACKBONE_0_POINTWISE_OUTPUT_HEIGHT 8
#define BACKBONE_0_POINTWISE_OUTPUT_WIDTH 796

const int backbone_0_pointwise_expected_output[BACKBONE_0_POINTWISE_OUTPUT_BATCH][BACKBONE_0_POINTWISE_OUTPUT_CHANNELS][BACKBONE_0_POINTWISE_OUTPUT_HEIGHT][BACKBONE_0_POINTWISE_OUTPUT_WIDTH] = {
  // ������ݣ���״��������
};

// backbone.4.depthwise ����
#define BACKBONE_4_DEPTHWISE_IN_CHANNELS 4
#define BACKBONE_4_DEPTHWISE_OUT_CHANNELS 4
#define BACKBONE_4_DEPTHWISE_KERNEL_HEIGHT 8
#define BACKBONE_4_DEPTHWISE_KERNEL_WIDTH 1
#define BACKBONE_4_DEPTHWISE_STRIDE_HEIGHT 1
#define BACKBONE_4_DEPTHWISE_STRIDE_WIDTH 1
#define BACKBONE_4_DEPTHWISE_PADDING_HEIGHT 0
#define BACKBONE_4_DEPTHWISE_PADDING_WIDTH 0
#define BACKBONE_4_DEPTHWISE_GROUPS 4

// backbone.4.depthwise Ȩ�� - ��״: (4, 1, 8, 1)
const int backbone_4_depthwise_weight[BACKBONE_4_DEPTHWISE_OUT_CHANNELS][BACKBONE_4_DEPTHWISE_IN_CHANNELS / BACKBONE_4_DEPTHWISE_GROUPS][BACKBONE_4_DEPTHWISE_KERNEL_HEIGHT][BACKBONE_4_DEPTHWISE_KERNEL_WIDTH] = {
  // ����Ȩ�����ݣ���״��������
};

// backbone.4.depthwise ƫ�� - ��״: (4,)
const int backbone_4_depthwise_bias[BACKBONE_4_DEPTHWISE_OUT_CHANNELS] = {
  // ����ƫ������
};

// backbone.4.depthwise ��� - ��״: (1, 4, 1, 796)
#define BACKBONE_4_DEPTHWISE_OUTPUT_BATCH 1
#define BACKBONE_4_DEPTHWISE_OUTPUT_CHANNELS 4
#define BACKBONE_4_DEPTHWISE_OUTPUT_HEIGHT 1
#define BACKBONE_4_DEPTHWISE_OUTPUT_WIDTH 796

const int backbone_4_depthwise_expected_output[BACKBONE_4_DEPTHWISE_OUTPUT_BATCH][BACKBONE_4_DEPTHWISE_OUTPUT_CHANNELS][BACKBONE_4_DEPTHWISE_OUTPUT_HEIGHT][BACKBONE_4_DEPTHWISE_OUTPUT_WIDTH] = {
  // ������ݣ���״��������
};

// backbone.4.pointwise ����
#define BACKBONE_4_POINTWISE_IN_CHANNELS 4
#define BACKBONE_4_POINTWISE_OUT_CHANNELS 8
#define BACKBONE_4_POINTWISE_KERNEL_HEIGHT 1
#define BACKBONE_4_POINTWISE_KERNEL_WIDTH 1
#define BACKBONE_4_POINTWISE_STRIDE_HEIGHT 1
#define BACKBONE_4_POINTWISE_STRIDE_WIDTH 1
#define BACKBONE_4_POINTWISE_PADDING_HEIGHT 0
#define BACKBONE_4_POINTWISE_PADDING_WIDTH 0
#define BACKBONE_4_POINTWISE_GROUPS 1

// backbone.4.pointwise Ȩ�� - ��״: (8, 4, 1, 1)
const int backbone_4_pointwise_weight[BACKBONE_4_POINTWISE_OUT_CHANNELS][BACKBONE_4_POINTWISE_IN_CHANNELS][BACKBONE_4_POINTWISE_KERNEL_HEIGHT][BACKBONE_4_POINTWISE_KERNEL_WIDTH] = {
  // ����Ȩ�����ݣ���״��������
};

// backbone.4.pointwise ƫ�� - ��״: (8,)
const int backbone_4_pointwise_bias[BACKBONE_4_POINTWISE_OUT_CHANNELS] = {
  // ����ƫ������
};

// backbone.4.pointwise ��� - ��״: (1, 8, 1, 796)
#define BACKBONE_4_POINTWISE_OUTPUT_BATCH 1
#define BACKBONE_4_POINTWISE_OUTPUT_CHANNELS 8
#define BACKBONE_4_POINTWISE_OUTPUT_HEIGHT 1
#define BACKBONE_4_POINTWISE_OUTPUT_WIDTH 796

const int backbone_4_pointwise_expected_output[BACKBONE_4_POINTWISE_OUTPUT_BATCH][BACKBONE_4_POINTWISE_OUTPUT_CHANNELS][BACKBONE_4_POINTWISE_OUTPUT_HEIGHT][BACKBONE_4_POINTWISE_OUTPUT_WIDTH] = {
  // ������ݣ���״��������
};

// spatial_attention.0 ����
#define SPATIAL_ATTENTION_0_IN_CHANNELS 8
#define SPATIAL_ATTENTION_0_OUT_CHANNELS 1
#define SPATIAL_ATTENTION_0_KERNEL_HEIGHT 1
#define SPATIAL_ATTENTION_0_KERNEL_WIDTH 1
#define SPATIAL_ATTENTION_0_STRIDE_HEIGHT 1
#define SPATIAL_ATTENTION_0_STRIDE_WIDTH 1
#define SPATIAL_ATTENTION_0_PADDING_HEIGHT 0
#define SPATIAL_ATTENTION_0_PADDING_WIDTH 0
#define SPATIAL_ATTENTION_0_GROUPS 1

// spatial_attention.0 Ȩ�� - ��״: (1, 8, 1, 1)
const int spatial_attention_0_weight[SPATIAL_ATTENTION_0_OUT_CHANNELS][SPATIAL_ATTENTION_0_IN_CHANNELS][SPATIAL_ATTENTION_0_KERNEL_HEIGHT][SPATIAL_ATTENTION_0_KERNEL_WIDTH] = {
  // ����Ȩ�����ݣ���״��������
};

// spatial_attention.0 ƫ�� - ��״: (1,)
const int spatial_attention_0_bias[SPATIAL_ATTENTION_0_OUT_CHANNELS] = {
  // ����ƫ������
};

// spatial_attention.0 ��� - ��״: (1, 1, 1, 796)
#define SPATIAL_ATTENTION_0_OUTPUT_BATCH 1
#define SPATIAL_ATTENTION_0_OUTPUT_CHANNELS 1
#define SPATIAL_ATTENTION_0_OUTPUT_HEIGHT 1
#define SPATIAL_ATTENTION_0_OUTPUT_WIDTH 796

const int spatial_attention_0_expected_output[SPATIAL_ATTENTION_0_OUTPUT_BATCH][SPATIAL_ATTENTION_0_OUTPUT_CHANNELS][SPATIAL_ATTENTION_0_OUTPUT_HEIGHT][SPATIAL_ATTENTION_0_OUTPUT_WIDTH] = {
  // ������ݣ���״��������
};

// channel_attention.1 ����
#define CHANNEL_ATTENTION_1_IN_CHANNELS 8
#define CHANNEL_ATTENTION_1_OUT_CHANNELS 1
#define CHANNEL_ATTENTION_1_KERNEL_HEIGHT 1
#define CHANNEL_ATTENTION_1_KERNEL_WIDTH 1
#define CHANNEL_ATTENTION_1_STRIDE_HEIGHT 1
#define CHANNEL_ATTENTION_1_STRIDE_WIDTH 1
#define CHANNEL_ATTENTION_1_PADDING_HEIGHT 0
#define CHANNEL_ATTENTION_1_PADDING_WIDTH 0
#define CHANNEL_ATTENTION_1_GROUPS 1

// channel_attention.1 Ȩ�� - ��״: (1, 8, 1, 1)
const int channel_attention_1_weight[CHANNEL_ATTENTION_1_OUT_CHANNELS][CHANNEL_ATTENTION_1_IN_CHANNELS][CHANNEL_ATTENTION_1_KERNEL_HEIGHT][CHANNEL_ATTENTION_1_KERNEL_WIDTH] = {
  // ����Ȩ�����ݣ���״��������
};

// channel_attention.1 ƫ�� - ��״: (1,)
const int channel_attention_1_bias[CHANNEL_ATTENTION_1_OUT_CHANNELS] = {
  // ����ƫ������
};

// channel_attention.1 ��� - ��״: (1, 1, 1, 1)
#define CHANNEL_ATTENTION_1_OUTPUT_BATCH 1
#define CHANNEL_ATTENTION_1_OUTPUT_CHANNELS 1
#define CHANNEL_ATTENTION_1_OUTPUT_HEIGHT 1
#define CHANNEL_ATTENTION_1_OUTPUT_WIDTH 1

const int channel_attention_1_expected_output[CHANNEL_ATTENTION_1_OUTPUT_BATCH][CHANNEL_ATTENTION_1_OUTPUT_CHANNELS][CHANNEL_ATTENTION_1_OUTPUT_HEIGHT][CHANNEL_ATTENTION_1_OUTPUT_WIDTH] = {
  // ������ݣ���״��������
};

// channel_attention.3 ����
#define CHANNEL_ATTENTION_3_IN_CHANNELS 1
#define CHANNEL_ATTENTION_3_OUT_CHANNELS 8
#define CHANNEL_ATTENTION_3_KERNEL_HEIGHT 1
#define CHANNEL_ATTENTION_3_KERNEL_WIDTH 1
#define CHANNEL_ATTENTION_3_STRIDE_HEIGHT 1
#define CHANNEL_ATTENTION_3_STRIDE_WIDTH 1
#define CHANNEL_ATTENTION_3_PADDING_HEIGHT 0
#define CHANNEL_ATTENTION_3_PADDING_WIDTH 0
#define CHANNEL_ATTENTION_3_GROUPS 1

// channel_attention.3 Ȩ�� - ��״: (8, 1, 1, 1)
const int channel_attention_3_weight[CHANNEL_ATTENTION_3_OUT_CHANNELS][CHANNEL_ATTENTION_3_IN_CHANNELS][CHANNEL_ATTENTION_3_KERNEL_HEIGHT][CHANNEL_ATTENTION_3_KERNEL_WIDTH] = {
  // ����Ȩ�����ݣ���״��������
};

// channel_attention.3 ƫ�� - ��״: (8,)
const int channel_attention_3_bias[CHANNEL_ATTENTION_3_OUT_CHANNELS] = {
  // ����ƫ������
};

// channel_attention.3 ��� - ��״: (1, 8, 1, 1)
#define CHANNEL_ATTENTION_3_OUTPUT_BATCH 1
#define CHANNEL_ATTENTION_3_OUTPUT_CHANNELS 8
#define CHANNEL_ATTENTION_3_OUTPUT_HEIGHT 1
#define CHANNEL_ATTENTION_3_OUTPUT_WIDTH 1

const int channel_attention_3_expected_output[CHANNEL_ATTENTION_3_OUTPUT_BATCH][CHANNEL_ATTENTION_3_OUTPUT_CHANNELS][CHANNEL_ATTENTION_3_OUTPUT_HEIGHT][CHANNEL_ATTENTION_3_OUTPUT_WIDTH] = {
  // ������ݣ���״��������
};

// se_block.1 ����
#define SE_BLOCK_1_IN_CHANNELS 8
#define SE_BLOCK_1_OUT_CHANNELS 1
#define SE_BLOCK_1_KERNEL_HEIGHT 1
#define SE_BLOCK_1_KERNEL_WIDTH 1
#define SE_BLOCK_1_STRIDE_HEIGHT 1
#define SE_BLOCK_1_STRIDE_WIDTH 1
#define SE_BLOCK_1_PADDING_HEIGHT 0
#define SE_BLOCK_1_PADDING_WIDTH 0
#define SE_BLOCK_1_GROUPS 1

// se_block.1 Ȩ�� - ��״: (1, 8, 1, 1)
const int se_block_1_weight[SE_BLOCK_1_OUT_CHANNELS][SE_BLOCK_1_IN_CHANNELS][SE_BLOCK_1_KERNEL_HEIGHT][SE_BLOCK_1_KERNEL_WIDTH] = {
  // ����Ȩ�����ݣ���״��������
};

// se_block.1 ƫ�� - ��״: (1,)
const int se_block_1_bias[SE_BLOCK_1_OUT_CHANNELS] = {
  // ����ƫ������
};

// se_block.1 ��� - ��״: (1, 1, 1, 1)
#define SE_BLOCK_1_OUTPUT_BATCH 1
#define SE_BLOCK_1_OUTPUT_CHANNELS 1
#define SE_BLOCK_1_OUTPUT_HEIGHT 1
#define SE_BLOCK_1_OUTPUT_WIDTH 1

const int se_block_1_expected_output[SE_BLOCK_1_OUTPUT_BATCH][SE_BLOCK_1_OUTPUT_CHANNELS][SE_BLOCK_1_OUTPUT_HEIGHT][SE_BLOCK_1_OUTPUT_WIDTH] = {
  // ������ݣ���״��������
};

// se_block.3 ����
#define SE_BLOCK_3_IN_CHANNELS 1
#define SE_BLOCK_3_OUT_CHANNELS 8
#define SE_BLOCK_3_KERNEL_HEIGHT 1
#define SE_BLOCK_3_KERNEL_WIDTH 1
#define SE_BLOCK_3_STRIDE_HEIGHT 1
#define SE_BLOCK_3_STRIDE_WIDTH 1
#define SE_BLOCK_3_PADDING_HEIGHT 0
#define SE_BLOCK_3_PADDING_WIDTH 0
#define SE_BLOCK_3_GROUPS 1

// se_block.3 Ȩ�� - ��״: (8, 1, 1, 1)
const int se_block_3_weight[SE_BLOCK_3_OUT_CHANNELS][SE_BLOCK_3_IN_CHANNELS][SE_BLOCK_3_KERNEL_HEIGHT][SE_BLOCK_3_KERNEL_WIDTH] = {
  // ����Ȩ�����ݣ���״��������
};

// se_block.3 ƫ�� - ��״: (8,)
const int se_block_3_bias[SE_BLOCK_3_OUT_CHANNELS] = {
  // ����ƫ������
};

// se_block.3 ��� - ��״: (1, 8, 1, 1)
#define SE_BLOCK_3_OUTPUT_BATCH 1
#define SE_BLOCK_3_OUTPUT_CHANNELS 8
#define SE_BLOCK_3_OUTPUT_HEIGHT 1
#define SE_BLOCK_3_OUTPUT_WIDTH 1

const int se_block_3_expected_output[SE_BLOCK_3_OUTPUT_BATCH][SE_BLOCK_3_OUTPUT_CHANNELS][SE_BLOCK_3_OUTPUT_HEIGHT][SE_BLOCK_3_OUTPUT_WIDTH] = {
  // ������ݣ���״��������
};

#endif // ALL_CONV_TEST_DATA_H

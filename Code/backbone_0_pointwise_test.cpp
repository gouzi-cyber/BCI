#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "Conv.h" //把卷积IP核定义包含进来

//定义缩放因子
#define SCALE_FACTOR 1024

//定义误差容忍度
#define EPSILON (SCALE_FACTOR/100)  //允许1％的误差

//调试工具
void debug_print(const char *msg){
  printf("[DEBUG] %s\n", msg);
  fflush(stdout);//确保消息立即输出
}


//测试单个卷积层-使用手动构建的数据
bool testConvLayerWithManualData(const char*layer_name,
                                 int in_channels,int out_channels,
                                 int in_height,int in_width,
                                 int kernel_height, int kernel_width,
                                 int stride_height,int stride_width,
                                 int padding_height,int padding_width,
                                 int batch_size,
                                 int groups=1){    //添加分组参数
  printf("\n----------------------------------------------\n");
  printf("开始测试卷积层：%s\n",layer_name);
  printf("配置：输入（%d，%d，%d，%d），卷积核（%d，%d），步长（%d，%d），填充（%d，%d），分组=%d\n",
         batch_size,in_channels,in_height,in_width,
         kernel_height,kernel_width,stride_height,stride_width,
         padding_height,padding_width,groups);

  //计算输出尺寸
  int out_height = (in_height+2*padding_height-kernel_height)/stride_height+1;
  int out_width = (in_width+2*padding_width-kernel_width)/stride_width+1;
  printf("预期输出尺寸：（%d，%d，%d，%d）\n",batch_size,out_channels,out_height,out_width);

  //计算数据大小
  int input_size=batch_size*in_channels*in_height*in_width;
  int weight_size=out_channels*in_channels*kernel_height*kernel_width;
  if (groups>1){
    weight_size=out_channels*(in_channels*groups)*kernel_height*kernel_width;
  }
  int bias_size=out_channels;
  int output_size=batch_size*out_channels*out_height*out_width;
  printf("数据大小：输入=%d，权重=%d，偏置=%d，输出=%d\n",
         input_size,weight_size,bias_size,output_size);

  //动态分配内存，避免栈溢出
  ap_int<32>*input_data=new ap_int<32>[input_size];
  ap_int<32>*weight_data=new ap_int<32>[weight_size];
  ap_int<32>*bias_data=new ap_int<32>[bias_size];
  ap_int<32>*output_data=new ap_int[output_size];
  ap_int<32>*expected_output=new ap_int<32>[output_size];

  //检查内存分配
  if (!input_data || !weight_data || !bias_data || !output_data || !expected_output) {
    printf("错误：内存分配失败\n");

    //释放已分配的内存
    if (input_data) delete [] input_data;
    if (weight_data) delete [] weight_data;
    if (bias_data) delete [] bias_data;
    if (output_data) delete [] output_data;
    if (expected_output) delete [] expected_output;
    return false;
  }

  //初始化数据-使用简单模式填充数据
  //1.输入数据：1,2,3。。。
  for(int i=0;i<input_size;i++){
    input_datap[i];=(i%10)+1;//值从1到10循环
  }

  //权重数据：1 for first filter,2 for second filter,etc.
  for (int o=0;o<output_size;o++){
    for (int i=0;i<in_channels/groups;i++){
      for(int kh=0;kh<kernel_height;kh++){
        for (int kw=0;kw<kernel_width;kw++){
          int index=o*(in_channels/groups*kernel_height*kernel_width)+
                    i*(kernel_height*kernel_width)+
                    kh*kernel_width+kw;
          weight_data[index]=o+1;//每个过滤器使用不同的值
        }
      }
    }
  }

  //3.偏置：10,20,30 。。。
  for (int o=0;o<output_size;o++){
    bias_data[o]=(o+1)*10;
  }

  //计算预期输出（简化版）
  memset(expected_output,0,output_size*sizeof(ap_int<32>));

  //这里我们可以手动计算一些简单的预期值，但是对于真实卷积这里很复杂
  //在实际产品中，可以使用参考实现或者经过测试的库来计算预期输出

  //为了简化，我们只为部分输出元素设置预期值
  //在实际开发中，会需要一个完整的参考实现

  //如果是1*1卷积（点卷积）且没有padding和stride为1，计算会简单很多
  if(kernel_height==1 && kernel_width==1 &&
     padding_height==0 && padding_width==0&&
     stride_height==0 && stride_width==1){
     for (int b=0;b<batch_size;b++){
       for (int o=0;o<output_channels;o++){
         for (int h=0;h<out_channels;h++){
           for (w=0;w<out_channels;w++){
             ap_int<64>sum=0;
             for (int i=0;i<in_channels;i++){
               //计算输入和权重的索引
               int input_idx=b*(in_channels*in_height*in_width)+
                             i*(in_height*in_width)+
                             h*in_width+w;

               int weight_idx=o*(in_channels*kernel_height*kernel_width)+
                              i*(kernel_height*kernel_width)+
                              0*kernel_width+0;//kh=0,kw=0 for 1*1 kernel
               sum+=(ap_int<64>)input_data[input_idx]*(ap_int<64>)weight_data[weight_idx];
             }
             //添加偏置并缩放
             sum=sum/SCALE_FACTOR;
             sum+=(ap_int<64>)bias_data[o];

             //保存预期输出
             int output_idx=b*(out_channels*out_height*out_width)+
                            o*(out_height*out_width)+
                            h*out_width+w;
             expected_output[output_idx]=(ap_int<32>)sum;
           }
         }
       }
     }
    }else{
      printf("警告：手动计算完整的预期输出对于非1*1卷积比较复杂，测试将只比较IP核的输出一致性\n");
      //在这种情况下，可以选择使用初始运行的输出作为“黄精标准”
      //或者实现一个完整的参考卷积
    }
    printf("输出数据样本（前10个元素）：");
    for (int i=0;i<10 && i<input_size;i++){
      printf("%d",(int)input_data[i]);
    }
    printf("\n");

    printf("权重数据样本（前10个元素）：");
    for (int i=0;i<10 && i<weight_size;i++){
      printf("%d",(int)weight_data[i]);
    }
    printf("\n");

    printf("偏置数据：")
    for (int i=0;i<bias_size;i++){
      printf("%d",(int)bias_data[i]);
    }
    printf("\n");

    //打印期望输出的一些元素
    if (kernel_height==1 && kernel_width==1 &&
        padding_height==0 && padding_width==0&&
        stride_height==1 && stride_width==1){
        printf("期望输出样本（前10个元素）：");
        for (int i=0;i<10 && i<output_size;i++){
          printf("%d",(int)expected_output[i]);
        }
        printf("\n");
    }
    printf("开始执行卷积...\n");
    fflush(stdout);

    try{
      //调用卷积函数
      conv_layer(
          input_data,
          ouput_data,
          bias_data,
          weight_data,
          in_channels,
          out_channels,
          in_height,in_width,
          kernel_height,kernel_width,
          stride_height,stride_width,
          padding_height,padding_width,
          batch_size,
          SCALE_FACTOR
      );
      debug_print("卷积执行完成");
    }catch(...){
      printf("错误：卷积执行过程中发生异常\n");

      //释放内存
      delete[] input_data;
      delete[] weight_data;
      delete[] bias_data;
      delete[] output_data;
      delete[] expected_output;

      return false;
    }
   printf("卷积执行完成，打印结果...\n");

   //打印输出的一些元素
   printf("实际输出样本（前10个元素）：");
   for (int i=0;i<10 && i<output_size;i++){
     printf("%d",(int)output_data[i]);
   }
   printf("\n");

   //比较结果（如果有预期输出）
   int errors=0;
   int max_diff=0;
   int max_diff_pos=-1;

   if(kernel_height==1 && kernel_width==1 &&
       padding_height==0 && padding_width==0&&
       stride_height==1 && stride_width==1){

       for (int i=0;i<output_size;i++){
         int diff=abs(output_data[i]-expected_output[i]);
         if (diff>max_diff){
           max_diff=diff;
           max_diff_pos=i;
         }
         if (diff>EPSILON){
           if(errors<10){
             printf("误差位置%d：期望%d，实际%d，差值%d\n",
                    i,(int)expected_output[i],(int)output_data[i],diff);
           }
           errors++;
         }
       }
       //报告结果
       if(errors==0){
         printf("卷积层%s 测试通过！最大差异：%d（位置%d）\n",
                layer_name,max_diff,max_diff_pos);
       }else{
         printf("卷积层%s，测试失败：%d个错误（共%d个值），最大差异：%d（位置%d）\n",
                layer_name,errors,output_size,max_diff,max_diff_pos);
       }else{
         //对于非1*1卷积，我们只检查输出是否非全零
         int nonzero_count=0;
         for(int i=0;i<output_size;i++){
           if(output_data[i]!=0) nonzero_count++;
         }
         if (nonzero_count>0){
           printf("卷积层%s 测试成功完成：输出中有%d/%d 个非零元素\n",
                  layer_name,nonzero_count,output_size);
         }else{
           printf("卷积层%s 测试警告：所有输出元素均为零，可能有问题\n",layer_name);
         }
       }
       delete[] input_data;
       delete[] weight_data;
       delete[] bias_data;
       delete[] output_data;

       delete[] expected_output;
       printf("-------------------------------------------------\n\n");
       return true;
       }
     int main(){
       printf("===卷积加速测试程序（手动数据构建）====\n");
       printf("缩放因子：%d\n",SCALE_FACTOR);
       printf("误差容忍度：%d\n\n",EPSILON);

       debug_print("程序开始执行");

     // 测试简单的1x1点卷积案例
     testConvLayerWithManualData(
         "simple_pointwise",        // 层名称
         3,                         // 输入通道 (增加到3通道)
         4,                         // 输出通道
         8,                         // 输入高度 (增加到8x8)
         8,                         // 输入宽度
         1,                         // 核高度 (1x1卷积)
         1,                         // 核宽度
         1,                         // 步长高度
         1,                         // 步长宽度
         0,                         // 填充高度
         0,                         // 填充宽度
         1,                         // 批量大小
         1                          // 分组数量
     );

     // 测试3x3标准卷积
     testConvLayerWithManualData(
         "simple_conv3x3",          // 层名称
         2,                         // 输入通道
         4,                         // 输出通道
         10,                        // 输入高度
         10,                        // 输入宽度
         3,                         // 核高度 (3x3卷积)
         3,                         // 核宽度
         1,                         // 步长高度
         1,                         // 步长宽度
         1,                         // 填充高度
         1,                         // 填充宽度
         1,                         // 批量大小
         1                          // 分组数量
     );

     // 测试带步长的卷积
     testConvLayerWithManualData(
         "strided_conv",            // 层名称
         2,                         // 输入通道
         3,                         // 输出通道
         12,                        // 输入高度
         12,                        // 输入宽度
         3,                         // 核高度
         3,                         // 核宽度
         2,                         // 步长高度 (步长为2)
         2,                         // 步长宽度
         1,                         // 填充高度
         1,                         // 填充宽度
         1,                         // 批量大小
         1                          // 分组数量
     );

     debug_print("程序执行完毕");
     return 0;
   }
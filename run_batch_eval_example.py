"""
批量推理评估脚本 - 使用示例

使用说明:
1. 准备数据:
   - images/ 目录: 存放原始图片
   - masks/ 目录: 存放对应的标注掩码（二值图像，前景为白色，背景为黑色）
   - 文件命名示例:
     * images/img001.jpg  对应  masks/img001_mask.png
     * 或 images/img001.jpg  对应  masks/img001.png

2. 运行脚本:
   python run_batch_eval_example.py

3. 查看结果:
   - output/evaluation_results.json: 详细评估指标
   - output/visualizations/: 可视化对比图
   - output/predictions/: 预测掩码
"""

from batch_inference_eval import batch_inference_and_eval
import os

# 配置参数
CONFIG = {
    # 输入路径
    'image_dir': 'assets/images',           # 图像目录
    'mask_dir': 'assets/masks',             # 标注掩码目录
    
    # 输出路径
    'output_dir': 'output/batch_eval',      # 输出目录
    
    # 推理参数
    'text_prompt': 'person',                # 文本提示词（根据你的数据集修改）
    'checkpoint_path': 'sam3.pt',           # 模型权重路径
    
    # 文件命名
    'mask_suffix': '_mask',                 # 标注文件后缀
}


def main():
    """主函数"""
    
    print("="*70)
    print("SAM3 批量推理和评估脚本")
    print("="*70)
    print()
    
    # 检查路径
    if not os.path.exists(CONFIG['image_dir']):
        print(f"错误: 图像目录不存在: {CONFIG['image_dir']}")
        print("请创建目录并放入图像文件")
        return
    
    if not os.path.exists(CONFIG['mask_dir']):
        print(f"错误: 标注目录不存在: {CONFIG['mask_dir']}")
        print("请创建目录并放入标注掩码文件")
        return
    
    if not os.path.exists(CONFIG['checkpoint_path']):
        print(f"错误: 模型权重文件不存在: {CONFIG['checkpoint_path']}")
        print("请确保 sam3.pt 在当前目录")
        return
    
    # 运行批量推理和评估
    try:
        overall_metrics, results = batch_inference_and_eval(
            image_dir=CONFIG['image_dir'],
            mask_dir=CONFIG['mask_dir'],
            output_dir=CONFIG['output_dir'],
            text_prompt=CONFIG['text_prompt'],
            checkpoint_path=CONFIG['checkpoint_path'],
            mask_suffix=CONFIG['mask_suffix']
        )
        
        print("\n" + "="*70)
        print("处理完成！")
        print("="*70)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

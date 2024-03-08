import cv2
import json

def draw_detections(image_path, detections_json_path, output_image_path):
    # 加载检测结果
    with open(detections_json_path, 'r') as f:
        detections = json.load(f)
    
    # 加载图片
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return
    
    # 遍历所有检测结果
    for det in detections:
        # 获取边界框和类别ID
        x, y, w, h = det['bbox']
        category_id = det['category_id']
        
        # 绘制边界框
        start_point = (int(x), int(y))
        end_point = (int(x + w), int(y + h))
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
        
        # 在边界框旁边显示类别ID
        text = f'ID: {category_id}'
        cv2.putText(image, text, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存修改后的图片
    cv2.imwrite(output_image_path, image)
    print(f"Annotated image saved to {output_image_path}")

# 示例用法
image_path = 'data/val2017/val2017/000000000139.jpg'  # 需要绘制检测结果的图片路径
detections_json_path = 'results.json'  # 检测结果的JSON文件路径
output_image_path = 'output_image.jpg'  # 输出图片的路径

draw_detections(image_path, detections_json_path, output_image_path)

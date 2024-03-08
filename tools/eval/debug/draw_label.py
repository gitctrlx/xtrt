import cv2
import json
import os

def extract_image_id_from_path(image_path):
    # 从图片路径中提取文件名（不包括扩展名）
    filename = os.path.basename(image_path)
    # 假设image_id是文件名的前面部分，去掉前导零
    image_id = int(filename.split('.')[0].lstrip('0'))
    return image_id

def draw_annotations(input_image_path, output_image_path, image_id_to_find, annotations_file):
    # 加载COCO标注文件
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # 加载图片
    image = cv2.imread(input_image_path)
    if image is None:
        print("Error loading image")
        return

    # 遍历标注数据，找到与指定image_id匹配的标注
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id_to_find:
            # 获取边界框和类别ID
            bbox = annotation['bbox']
            category_id = annotation['category_id']
            
            # 计算边界框的坐标
            x, y, width, height = bbox
            start_point = (int(x), int(y))
            end_point = (int(x + width), int(y + height))
            
            # 绘制边界框
            cv2.rectangle(image, start_point, end_point, (255, 0, 0), 2)
            
            # 在边界框旁边显示类别ID
            cv2.putText(image, f'ID: {category_id}', (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # 保存修改后的图片
    cv2.imwrite(output_image_path, image)
    print(f"Output image saved to {output_image_path}")

# 示例用法
input_image_path = 'data/val2017/val2017/000000000139.jpg'  # 输入图片的路径
image_id_to_find = extract_image_id_from_path(input_image_path)  # 指定的image_id
output_image_path = f'{image_id_to_find}_label.jpg'  # 输出图片的路径
annotations_file = 'instances_val2017.json'  # COCO标注文件的路径

draw_annotations(input_image_path, output_image_path, image_id_to_find, annotations_file)

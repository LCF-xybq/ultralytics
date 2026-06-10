import cv2
import numpy as np


def find_and_draw_largest_empty_circle(image_path):
    # 1. 读取图像并确保是单通道灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法找到图像文件: {image_path}")
    
    # 确保图像只有 0 和 255（二值化，这步很重要，防止模糊边缘带来的噪点）
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 2. 图像反转：将 255 变 0（障碍物），0 变 255（寻找圆的空间）
    # 这是因为距离变换通常计算到最近的零点的距离，
    # 我们要找的是离白色线条最远的点。
    inverted_img = cv2.bitwise_not(binary_img)
    
    # 3. 添加边界（重要）：防止最大圆超出图片画幅边缘
    # 在四周 pad 一圈 0（即厚度为 1 的障碍物墙），确保圆边界不超出图片。
    padded_img = cv2.copyMakeBorder(inverted_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    
    # 4. 执行距离变换
    # cv2.DIST_L2: 欧氏距离
    # cv2.DIST_MASK_PRECISE: 使用精确的距离估算（更准确）
    dist_transform = cv2.distanceTransform(padded_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # 5. 在距离图中寻找最大值
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
    
    # 6. 提取结果
    radius = max_val
    # 减去 padding 带来的 1 像素偏移，得到原图上的坐标
    center_x = max_loc[0] - 1
    center_y = max_loc[1] - 1
    
    center = (center_x, center_y)
    print(f"最大圆的圆心坐标 (x, y): {center}")
    print(f"最大圆的半径: {radius:.2f} 像素")

    # 7. **【可视化 - 将圆形画在原图上】**
    # 7.1 将灰度原图转换为彩色 BGR 格式，以便用彩色绘制
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 7.2 绘制圆心（一个小圆点），使用红色 (BGR: 0, 0, 255)
    # 圆心点半径设为 1，厚度为 -1（填充）
    cv2.circle(vis_img, center, 1, (0, 0, 255), -1)
    
    # 7.3 绘制最大圆边界，使用红色 (BGR: 0, 0, 255)
    # 厚度设为 1（或根据图像大小调整）
    cv2.circle(vis_img, center, int(radius), (0, 0, 255), 1)
    
    # 将圆心和半径信息写在图像上（可选）
    info_str = f"Center: ({center_x}, {center_y}) Radius: {radius:.2f}"
    cv2.putText(vis_img, info_str, (5, vis_img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 8. 返回绘制了圆形的结果图像
    return vis_img

# result_img = find_and_draw_largest_empty_circle('../runs/plant_height_yolov8/test_results/gradient/labels_raw_gradient/DJI_20250830090542_0025_V.jpeg')
result_img = find_and_draw_largest_empty_circle('../runs/plant_height_yolov8/test_results/gradient/labels_raw_gradient_cycle/DJI_20260323152451_0002_V.jpeg')
if result_img is not None:
    # 保存结果
    cv2.imwrite('result_with_circle.png', result_img)
    print("结果图像已保存为 'result_with_circle.png'")

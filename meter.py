import os
import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from skimage import measure
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks


class MeterReader:
    def __init__(self):
        # 初始化OCR，使用中英文识别模型，添加额外的配置
        self.reader = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            show_log=False,
            det_db_thresh=0.3,  # 降低检测阈值
            det_db_box_thresh=0.3,  # 降低检测框阈值
            rec_model_dir='ch_PP-OCRv3_rec_infer',  # 使用v3模型
            det_model_dir='ch_PP-OCRv3_det_infer',
            cls_model_dir='ch_ppocr_mobile_v2.0_cls_infer'
        )
        print("OCR初始化完成")

    def preprocess_image(self, image):
        results = []

        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 基础增强
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results.append(enhanced)

        # 去噪处理
        denoised = cv2.fastNlMeansDenoising(enhanced)
        results.append(denoised)

        # 自适应二值化
        binary_adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        results.append(binary_adaptive)

        # Otsu二值化
        _, binary_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(binary_otsu)

        # 边缘增强
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpening)
        results.append(sharpened)

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel)
        results.append(morphed)

        # 对每个结果进行缩放
        scale_factors = [1.5, 2.0]  # 尝试不同的缩放比例
        scaled_results = []
        for img in results:
            height, width = img.shape[:2]
            for scale in scale_factors:
                scaled = cv2.resize(img, (int(width * scale), int(height * scale)))
                scaled_results.append(scaled)

        # 添加分割处理的结果
        split_results = self.process_split_digits(gray)
        if split_results:
            results.extend(split_results)

        # 合并所有结果
        results.extend(scaled_results)

        return results

    def process_split_digits(self, gray_image):
        """增强的分裂数字检测和处理方法"""
        results = []

        # 使用自适应阈值进行二值化
        binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

        # 使用形态学操作增强特征
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        
        # 垂直边缘检测
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vertical)
        vertical = cv2.dilate(vertical, kernel_vertical, iterations=1)

        # 水平边缘检测
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horizontal)
        horizontal = cv2.dilate(horizontal, kernel_horizontal, iterations=1)

        # 寻找连通区域
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 10 and w > 5:  # 过滤太小的区域
                roi = gray_image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                # 分析数字的垂直分布
                vertical_profile = np.sum(roi, axis=1)
                normalized_profile = vertical_profile / np.max(vertical_profile)

                # 检测垂直方向的峰值
                peaks = []
                for i in range(1, len(normalized_profile)-1):
                    if normalized_profile[i] > normalized_profile[i-1] and normalized_profile[i] > normalized_profile[i+1]:
                        if normalized_profile[i] > 0.5:  # 显著的峰值
                            peaks.append(i)

                # 如果检测到多个峰值，可能是分裂数字
                if len(peaks) >= 2:
                    # 计算峰值之间的距离
                    peak_distances = np.diff(peaks)
                    avg_distance = np.mean(peak_distances)
                    
                    # 如果峰值间距合适（在数字高度的合理范围内）
                    if 0.3 * h <= avg_distance <= 0.7 * h:
                        # 分别处理上下两部分
                        for i in range(len(peaks)-1):
                            # 提取每个峰值周围的区域
                            y1 = max(0, peaks[i] - int(avg_distance*0.3))
                            y2 = min(h, peaks[i] + int(avg_distance*0.3))
                            part = roi[y1:y2, :]
                            
                            if part.size > 0:
                                # 增强对比度
                                part = cv2.equalizeHist(part)
                                
                                # 自适应二值化
                                part_binary = cv2.adaptiveThreshold(part, 255,
                                                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                  cv2.THRESH_BINARY, 11, 2)
                                
                                # 添加处理后的部分
                                results.append(part_binary)

                                # 创建放大版本
                                scaled = cv2.resize(part_binary, (int(w*1.5), int(h*0.5)))
                                results.append(scaled)

        return results

    def calculate_position_score(self, box):
        """根据文本框的位置计算分数"""
        if not box or len(box) != 4:
            return 0

        # 计算文本框的中心点
        center_x = sum(point[0] for point in box) / 4
        center_y = sum(point[1] for point in box) / 4

        # 计算文本框的宽度和高度
        width = max(point[0] for point in box) - min(point[0] for point in box)
        height = max(point[1] for point in box) - min(point[1] for point in box)

        # 计算宽高比
        aspect_ratio = width / height if height > 0 else 0

        # 电表读数通常在图像的中下部位置，且宽度适中
        position_score = 1.0

        # 根据垂直位置调整分数（偏好中下部位置）
        if center_y < height:  # 如果在上半部分，降低分数
            position_score *= 0.9

        # 根据宽高比调整分数（电表读数通常是较宽的数字序列）
        if 4.2 <= aspect_ratio <= 6.8:  # 理想的宽高比范围
            position_score *= 2.5
        elif aspect_ratio > 7.0 or aspect_ratio < 3.0:  # 不太可能是电表读数
            position_score *= 0.2

        return position_score

    def is_valid_meter_reading(self, text):
        """检查是否是有效的电表读数格式"""
        # 移除所有非数字字符
        digits = ''.join(c for c in text if c.isdigit())

        # 基本长度检查
        if len(digits) < 4 or len(digits) > 6:  # 放宽长度限制
            return False

        # 检查是否包含过多的非数字字符
        non_digits = sum(1 for c in text if not c.isdigit() and c != '.')
        if non_digits > 2:  # 允许最多两个非数字字符
            return False

        # 检查数值范围（放宽限制）
        try:
            value = float(digits)
            if value < 50:  # 降低最小值限制
                return False
        except ValueError:
            return False

        return True

    def calculate_box_area(self, box):
        """计算文本框的面积"""
        if not box or len(box) != 4:
            return 0
        # 使用多边形面积公式计算
        box = np.array(box)
        x = box[:, 0]
        y = box[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area

    def detect_split_digits(self, image, box):
        """增强的分裂数字检测方法"""
        if not box or len(box) != 4:
            return False

        x_min = min(point[0] for point in box)
        x_max = max(point[0] for point in box)
        y_min = min(point[1] for point in box)
        y_max = max(point[1] for point in box)

        # 扩展检测区域
        height = y_max - y_min
        y_min = max(0, int(y_min - height * 0.2))
        y_max = min(image.shape[0], int(y_max + height * 0.2))

        # 提取ROI
        roi = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        if roi.size == 0:
            return False

        # 转换为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # 计算垂直方向的投影分布
        vertical_projection = np.sum(enhanced, axis=1)
        normalized_projection = vertical_projection / np.max(vertical_projection)

        # 分析投影曲线的特征
        # 1. 检查是否存在明显的双峰
        peaks, _ = find_peaks(normalized_projection, height=0.5, distance=height*0.2)
        if len(peaks) >= 2:
            # 检查峰值之间的距离是否合适
            peak_distances = np.diff(peaks)
            avg_distance = np.mean(peak_distances)
            if 0.2 * height <= avg_distance <= 0.8 * height:
                return True

        # 2. 检查垂直边缘
        edges_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        edges_y = np.absolute(edges_y)
        edges_y = np.uint8(edges_y)

        # 分析边缘的垂直分布
        edge_projection = np.sum(edges_y, axis=1)
        normalized_edge_projection = edge_projection / np.max(edge_projection)

        # 检查是否存在明显的边缘跳变
        edge_peaks, _ = find_peaks(normalized_edge_projection, height=0.3, distance=height*0.2)
        if len(edge_peaks) >= 2:
            edge_distances = np.diff(edge_peaks)
            avg_edge_distance = np.mean(edge_distances)
            if 0.2 * height <= avg_edge_distance <= 0.8 * height:
                return True

        return False

    def is_scale_number(self, digits):
        """增强的刻度值检测"""
        # 检查是否只包含0和1
        if all(d in '01' for d in digits):
            return True

        # 检查是否是常见的刻度值
        common_scales = ['10000', '1000', '100', '10','10010','10001']
        if digits in common_scales:
            return True

        # 检查是否全是0结尾
        if digits.endswith('0' * (len(digits) // 2)):
            return True

        # 检查是否是等差数列
        try:
            if len(digits) >= 2:
                # 检查是否以0结尾
                if digits.endswith('0'):
                    # 去掉末尾的0后检查剩余数字是否都相同
                    base_digits = digits.rstrip('0')
                    if len(base_digits) == 1:
                        return True
        except:
            pass

        return False

    def calculate_number_quality_score(self, digits, box=None, image=None):
        """增强的数字质量评分"""
        score = 1.0

        # 如果是刻度值，显著降低分数
        if self.is_scale_number(digits):
            score *= 0.01
            return score

        # 计算文本框面积并根据面积调整分数
        if box is not None and image is not None:
            area = self.calculate_box_area(box)
            # 假设较大的文本框更可能是主要读数
            if area > 1000:  # 阈值需要根据实际情况调整
                score *= 1.5
            elif area < 100:  # 较小的文本框可能是刻度值
                score *= 0.5

            # 检查最后一位数字区域的颜色特征（红底）
            if len(digits) == 6:
                x_min = min(point[0] for point in box)
                x_max = max(point[0] for point in box)
                y_min = min(point[1] for point in box)
                y_max = max(point[1] for point in box)
                
                # 计算最后一位数字的大致位置
                digit_width = (x_max - x_min) / 6
                last_digit_x = int(x_max - digit_width)
                
                # 提取最后一位数字的区域
                roi = image[int(y_min):int(y_max), last_digit_x:int(x_max)]
                if roi.size > 0:
                    # 转换到HSV颜色空间以更好地检测红色
                    if len(roi.shape) == 3:
                        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        # 定义红色的HSV范围
                        lower_red1 = np.array([0, 50, 50])
                        upper_red1 = np.array([10, 255, 255])
                        lower_red2 = np.array([170, 50, 50])
                        upper_red2 = np.array([180, 255, 255])
                        
                        # 创建红色掩码
                        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                        red_mask = cv2.bitwise_or(mask1, mask2)
                        
                        # 计算红色像素的比例
                        red_ratio = np.sum(red_mask > 0) / red_mask.size
                        
                        # 如果最后一位确实是红底，提高分数
                        if red_ratio > 0.3:  # 阈值可以根据实际情况调整
                            score *= 2.0
                        else:
                            score *= 0.5

            # 检查前五位区域的颜色特征（黑底）
            if len(digits) >= 5:
                first_five_x = int(x_min + (x_max - x_min) * 5/6)
                roi_first_five = image[int(y_min):int(y_max), int(x_min):first_five_x]
                if roi_first_five.size > 0 and len(roi_first_five.shape) == 3:
                    # 转换到灰度图
                    gray = cv2.cvtColor(roi_first_five, cv2.COLOR_BGR2GRAY)
                    # 计算暗色像素的比例（黑底）
                    dark_ratio = np.sum(gray < 50) / gray.size  # 阈值可调整
                    if dark_ratio > 0.4:  # 阈值可以根据实际情况调整
                        score *= 1.5

        # 检查分裂数字
        if image is not None and box is not None:
            if self.detect_split_digits(image, box):
                score *= 2.0  # 显著提高分裂数字的权重

        # 其他现有的评分逻辑
        unique_digits = len(set(digits))
        if unique_digits >= 4:
            score *= 1.8
        elif unique_digits >= 3:
            score *= 1.4

        consecutive_count = 0
        for i in range(len(digits) - 1):
            if abs(int(digits[i]) - int(digits[i + 1])) == 1:
                consecutive_count += 1

        if consecutive_count >= 2:
            score *= 1.6
        elif consecutive_count == 1:
            score *= 1.3

        # 检查重复模式
        has_repeating_pattern = False
        if len(digits) >= 4:
            for i in range(len(digits) - 1):
                if digits[i:i + 2] == digits[i + 2:i + 4]:
                    has_repeating_pattern = True
                    break
        if has_repeating_pattern:
            score *= 0.3

        return score

    def extract_number(self, text, confidence, box=None, image=None):
        """增强的数字提取方法"""
        if not isinstance(text, str):
            return None, 0

        # 移除常见的错误字符
        text = text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
        text = text.replace(' ', '').replace(',', '.')

        final_confidence = confidence

        if box is not None:
            position_score = self.calculate_position_score(box)
            final_confidence *= position_score

        try:
            digits = ''.join(c for c in text if c.isdigit())

            # 优先处理6位数字的情况
            if len(digits) == 6:
                integer_part = digits[:5]
                decimal_part = digits[5:]
                number = float(f"{integer_part}.{decimal_part}")
                
                if 0 <= number <= 35000:
                    # 使用增强的质量评分
                    quality_score = self.calculate_number_quality_score(digits, box, image)
                    final_confidence *= quality_score
                    
                    # 额外的长度奖励
                    final_confidence *= 2.0  # 6位数优先
                    
                    if digits.startswith('0'):
                        final_confidence *= 1.5
                    
                    print(f"6位数字: {number}, 原始置信度: {confidence}, 质量分数: {quality_score}, 最终置信度: {final_confidence}")
                    return number, final_confidence
            
            # 处理5位数字的情况（作为备选）
            elif len(digits) == 5:
                number = float(digits)
                
                if 0 <= number <= 35000:
                    quality_score = self.calculate_number_quality_score(digits, box, image)
                    final_confidence *= quality_score
                    
                    # 5位数降低优先级
                    final_confidence *= 0.8
                    
                    print(f"5位数字: {number}, 原始置信度: {confidence}, 质量分数: {quality_score}, 最终置信度: {final_confidence}")
                    return number, final_confidence

        except (ValueError, IndexError):
            pass

        return None, 0

    def process_ocr_result(self, result, image):
        numbers = []

        if not result or not isinstance(result, list):
            return numbers

        # 处理多层嵌套的结果
        def process_nested_result(item):
            if isinstance(item, list):
                if len(item) == 2 and isinstance(item[0], list) and isinstance(item[1], tuple):
                    # 获取文本框位置信息和文本内容
                    box, (text, confidence) = item
                    # 检查是否是有效的电表读数格式
                    if self.is_valid_meter_reading(text):
                        number, conf = self.extract_number(text, confidence, box, image)
                        if number is not None:
                            numbers.append((number, conf, self.calculate_box_area(box)))
                else:
                    # 递归处理嵌套列表
                    for subitem in item:
                        process_nested_result(subitem)

        # 开始处理结果
        for item in result:
            try:
                process_nested_result(item)
            except Exception as e:
                print(f"处理OCR结果时出错: {str(e)}")
                continue

        return numbers

    def recognize_digits(self, image):
        all_numbers = []  # 存储所有识别到的有效数字

        try:
            # 获取多个预处理版本的图像
            processed_images = self.preprocess_image(image)

            # 对每个预处理版本都进行OCR识别
            for processed_img in processed_images:
                # 使用OCR识别
                result = self.reader.ocr(processed_img, cls=True)
                print(f"OCR结果: {result}")

                numbers = self.process_ocr_result(result, processed_img)
                if numbers:
                    all_numbers.extend(numbers)

            # 如果有识别结果
            if all_numbers:
                # 按质量分数和置信度的综合评分排序
                all_numbers.sort(key=lambda x: (x[1] * x[2]), reverse=True)  # 使用置信度和面积的乘积作为排序依据
                print(f"所有提取的数字及置信度: {all_numbers}")

                # 过滤和验证结果
                valid_numbers = []
                for num, conf, area in all_numbers:
                    # 放宽数值范围的限制
                    if 50 <= num <= 35000:  # 降低最小值限制
                        valid_numbers.append((num, conf, area))

                if valid_numbers:
                    # 如果有多个结果，使用聚类方法选择最可能的结果
                    if len(valid_numbers) > 1:
                        # 将数字按值分组
                        numbers_array = np.array([n[0] for n in valid_numbers])
                        scores_array = np.array([n[1] * n[2] for n in valid_numbers])  # 综合评分
                        
                        # 使用相对误差来判断数字是否相似
                        groups = {}
                        for i, num in enumerate(numbers_array):
                            found_group = False
                            for key in groups:
                                # 使用相对误差而不是绝对差异
                                relative_diff = abs(num - key) / max(key, num)
                                if relative_diff < 0.01:  # 允许1%的相对误差
                                    groups[key].append((num, scores_array[i]))
                                    found_group = True
                                    break
                            if not found_group:
                                groups[num] = [(num, scores_array[i])]

                        # 选择最佳组
                        best_group_score = 0
                        best_number = valid_numbers[0][0]
                        
                        for key, group in groups.items():
                            group_size = len(group)
                            group_score = sum(score for _, score in group)
                            max_score_in_group = max(score for _, score in group)
                            
                            # 综合考虑组大小、总分和最高分
                            total_score = group_score * group_size * max_score_in_group
                            
                            if total_score > best_group_score:
                                best_group_score = total_score
                                # 使用组内最高分对应的数字
                                best_number = group[max(range(len(group)), 
                                                     key=lambda i: group[i][1])][0]

                        return round(best_number, 1)
                    else:
                        return round(valid_numbers[0][0], 1)

            print("没有找到有效的电表读数")
            return -1.0

        except Exception as e:
            print(f"OCR识别出错: {str(e)}")
            return -1.0

    def process_image(self, image_path):
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")

            print(f"\n处理图片: {image_path}")

            # 识别数字
            reading = self.recognize_digits(image)
            print(f"最终结果: {reading}")

            # 如果结果无效，返回错误标记
            if reading < 0:
                return 9999.9  # 使用9999.9表示识别失败

            return reading

        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {str(e)}")
            return 9999.9  # 使用9999.9表示识别失败


def main():
    # 初始化读表器
    meter_reader = MeterReader()

    # 获取数据集目录中的所有图片
    dataset_path = "ML/Dataset"
    results = []

    # 处理每张图片
    for filename in sorted(os.listdir(dataset_path)):
        if filename.endswith('.jpg'):
            image_path = os.path.join(dataset_path, filename)

            # 处理图片
            reading = meter_reader.process_image(image_path)

            # 添加结果（对于错误值使用特殊标记）
            if reading == 9999.9:
                print(f"警告: {filename} 识别失败")

            # 添加结果
            results.append([filename, f"{reading:.1f}"])
            print(f"处理 {filename}: 预测值 = {reading:.1f}")

    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(results, columns=['filename', 'number'])
    df = df.sort_values('filename')

    # 保存结果
    output_filename = "赛道1-张三-2013.csv"
    df.to_csv(output_filename, index=False)
    print(f"结果已保存到 {output_filename}")

    # 输出识别失败的图片数量
    failed_count = len([r for r in results if float(r[1]) == 9999.9])
    if failed_count > 0:
        print(f"\n警告: 有 {failed_count} 张图片识别失败")


if __name__ == "__main__":
    main()
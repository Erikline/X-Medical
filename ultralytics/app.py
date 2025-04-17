# Author: Hongyu Lin
# Institution: School of Mathematics and Computer Science, Shantou University

# 导入 streamlit 库，并简写为 st，用于创建 Web 应用界面
import streamlit as st
# 导入 cv2 库 (OpenCV)，用于图像处理
import cv2
# 导入 numpy 库，并简写为 np，用于数值计算和数组操作
import numpy as np
# 从 PIL (Pillow) 库导入 Image 模块，用于图像文件操作
from PIL import Image
# 从 ultralytics 库导入 YOLO 类，用于执行对象检测
from ultralytics import YOLO
# 导入 tempfile 模块，用于创建临时文件
import tempfile
# 导入 time 模块，用于计时
import time
# 导入 pathlib 模块，用于处理文件路径
from pathlib import Path

# --- 页面基础配置 ---
st.set_page_config(
    page_title="XMedical - 智能影像分析",
    page_icon="🔬",
    layout="wide"
)

# --- 自定义 CSS ---
# 注入 CSS 来调整相机预览和拍摄后图像的显示样式
st.markdown(
    """
    <style>
    /* 1. 调整相机实时预览 (video) */
    div[data-testid="stCameraInput"] video {
        width: 100% !important;
        height: auto !important;
        object-fit: cover !important; /* 覆盖填充，可能裁剪边缘 */
    }

    /* 2. 调整拍摄后静态图片显示 (st.image) */
    /* 定位到 st.image 生成的包含图片的 div 容器 */
    div[data-testid="stImage"] {
        /* 移除可能导致两侧留白的内边距 */
        padding-left: 0 !important;
        padding-right: 0 !important;
        /* 可选：移除外边距，如果需要的话 */
        /* margin-left: 0 !important; */
        /* margin-right: 0 !important; */
        /* 确保容器本身宽度是100% */
        width: 100% !important;
    }

    /* 确保图片在上述容器内也是100%宽度 (use_column_width=True 应该会处理这个，但以防万一) */
    div[data-testid="stImage"] img {
        width: 100% !important;
        height: auto !important; /* 保持图片自身宽高比 */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- 应用主标题和描述 ---
st.title("XMedical - 轻量级医学影像智能分析系统")
st.markdown("###### 利用先进 AI 技术，辅助分析医学影像（如脑部 MRI、胸部 CT、细胞图像等），快速识别潜在病灶或特定细胞。")

# --- 模型加载 (使用 pathlib 和调试信息) ---
if 'model' not in st.session_state:
    with st.spinner("⏳ 正在加载 X-Medical 深度学习模型，请稍候..."):
        try:
            # 获取 app.py 所在的目录 (根据你的截图，这是项目根目录)
            APP_DIR = Path(__file__).parent

            # 构建模型文件的完整路径 (因为 Pt Source 和 app.py 都在根目录，是同级)
            model_path = APP_DIR / 'Pt Source' / 'X-Medical.pt'

            # --- !! 关键调试步骤 !! ---
            # 打印 Streamlit 尝试使用的路径，并在运行的应用界面查看此输出
            st.write(f"调试信息：尝试从以下路径加载模型: {model_path}")
            st.write(f"调试信息：该路径的文件是否存在? {model_path.exists()}")

            # --- 检查文件是否存在 ---
            if not model_path.exists():
                st.error(f"关键错误：模型文件未在预期路径找到！")
                st.error(f"预期路径: {model_path}")
                st.error(f"请在 GitHub 仓库仔细检查: 1. 'Pt Source' 文件夹和 'X-Medical.pt' 文件是否已成功添加、提交并推送? 2. 文件夹和文件名的大小写是否与代码中 ('Pt Source', 'X-Medical.pt') 完全一致?")
                st.stop() # 如果文件不存在，停止执行

            # 使用绝对路径加载模型
            st.session_state.model = YOLO(model_path)
            # 可选：加载成功提示，部署时可注释掉
            # st.success("✅ 模型加载成功！")

        except Exception as e:
            st.error(f"加载模型时发生严重错误：{e}")
            # 在出错时也打印尝试的路径，帮助定位问题
            st.error(f"尝试加载的路径是: {model_path}")
            st.stop() # 加载出错也停止执行

# --- 创建选项卡 ---
tab1, tab2 = st.tabs(["🔬 影像检测分析", "ℹ️ 关于系统"])

# --- 选项卡1: 影像检测分析 ---
with tab1:
    st.subheader("实时影像分析")
    img_file_buffer = st.camera_input(
        "请点击下方“拍照”按钮，拍摄需要分析的医学影像区域（例如显示器上的CT/MRI图像、显微镜视野等）"
        )

    confidence = st.slider(
        "🔬 病灶/目标识别置信度阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )

    if img_file_buffer is not None:
        # 检查模型是否已成功加载 (如果在加载时出错，这里会避免进一步错误)
        if 'model' in st.session_state and st.session_state.model is not None:
            bytes_data = img_file_buffer.getvalue()

            # 使用临时文件处理上传的图像
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(bytes_data)
                tmp_file_path = tmp_file.name

            try:
                start_time = time.time()
                # 使用加载好的模型进行预测
                results = st.session_state.model.predict(tmp_file_path, conf=confidence)
                end_time = time.time()

                # --- 处理和显示结果 ---
                if results: # 确保 results 不为空
                    result = results[0] # 获取第一个结果对象
                    image = Image.open(tmp_file_path) # 重新打开临时图像用于绘制
                    image_np = np.array(image.convert('RGB')) # 转换为Numpy数组(确保是RGB)

                    boxes = result.boxes # 获取边界框信息
                    if boxes is not None: # 检查是否有检测框
                        for box in boxes:
                            # 提取坐标、类别ID和置信度
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            # 获取类别名称 (确保 result.names 存在)
                            label = f"{result.names[cls]}: {conf:.2f}" if result.names else f"Class {cls}: {conf:.2f}"
                            color = (0, 255, 0) # BGR 格式的绿色
                            # 在 Numpy 图像上绘制矩形框
                            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
                            # 在 Numpy 图像上绘制标签文本
                            cv2.putText(image_np, label, (x1, y1 - 10 if y1 > 10 else y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # 使用 use_column_width=True 显示带标注的图像
                        st.image(image_np, caption="模型分析结果预览", use_column_width=True)

                        # --- 分析报告 ---
                        st.subheader("📊 分析报告")
                        inference_time = end_time - start_time
                        st.write(f"⏱️ 模型分析耗时: {inference_time:.4f} 秒")

                        # 统计检测到的目标
                        detected_objects = {}
                        for box in boxes:
                            cls = int(box.cls[0])
                            class_name = result.names[cls] if result.names else f"Class {cls}"
                            conf = float(box.conf[0])
                            # 记录每个类别的最高置信度
                            detected_objects[class_name] = max(conf, detected_objects.get(class_name, 0.0))

                        st.write("🔍 **识别到的目标类别及最高置信度:**")
                        for obj, conf_val in detected_objects.items():
                            st.write(f" - {obj} (置信度: {conf_val:.2f})")
                    else:
                        # 如果 boxes 为 None 或为空
                        st.image(image_np, caption="原始图像（未检测到目标）", use_column_width=True)
                        st.info("ℹ️ 在当前置信度阈值下，未识别到明确的目标。请尝试调整阈值或拍摄更清晰的图像。")
                else:
                    st.warning("⚠️ 模型预测未返回有效结果。")

            except Exception as e:
                st.error(f"处理图像或执行预测时出错: {e}")
            finally:
                # 确保删除临时文件
                import os
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

        else:
            st.error("模型未能成功加载，无法进行预测。请检查应用日志。")


# --- 选项卡2: 关于系统 ---
with tab2:
    # ... (关于选项卡的内容保持不变) ...
    st.subheader("ℹ️ 关于 XMedical 系统")
    st.markdown("""
    **XMedical** 是一个基于先进的 **YOLOv8** 深度学习框架构建的轻量级医学影像智能分析系统。
    它经过专门训练，旨在辅助医生或研究人员快速识别医学影像中的特定目标，例如：

    *   **脑部影像**: 可能的肿瘤区域。
    *   **胸部影像**: 可疑的结节或病变。
    *   **细胞学图像**: 特定类型的细胞计数或状态分析。

    ---
    ### 💡 如何使用:
    1.  在 **“影像检测分析”** 选项卡中，点击 **“拍照”** 按钮。
    2.  **允许浏览器访问您的摄像头**（如果弹出请求）。
    3.  将摄像头**对准需要分析的医学影像**（可以是屏幕上的图像、打印的胶片或显微镜视野）。
    4.  **确保光线充足、图像清晰**，然后点击拍照图标完成拍摄。
    5.  系统将自动处理图像，并在下方显示带有**标注框和类别标签**的识别结果。
    6.  您可以通过拖动 **“病灶/目标识别置信度阈值”** 滑块来调整灵敏度，过滤掉低置信度的识别结果。

    ---
    ### ⚙️ 技术细节:
    *   **核心模型**: X-Medical (基于 YOLOv8 优化训练)。
    *   **识别能力**: 针对特定的医学影像数据集进行训练，能够识别如脑肿瘤、胸部病变、特定细胞等预设类别 (*请根据您模型的实际训练数据在此处进行更详细的说明*)。
    *   **处理方式**: 图像数据的分析和计算在**服务器端**完成，保障了处理速度和效果，同时不占用本地设备过多资源。

    ---
    **免责声明:** 本系统仅为辅助分析工具，分析结果**不能替代**专业的医疗诊断。所有医疗决策请务必咨询执业医师。
    """)

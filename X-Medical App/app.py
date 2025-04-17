# app.py
# Author: Hongyu Lin
# Institution: School of Mathematics and Computer Science, Shantou University

import sys
from pathlib import Path
import streamlit as st
import os
# --- 确保所有 import 都在最前面 ---
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
# --- 主要依赖导入 (放在 try...except 中确保健壮性) ---
YOLO_IMPORTED = False # 添加一个标志位
try:
    from ultralytics import YOLO # 尝试导入 YOLO
    YOLO_IMPORTED = True # 如果成功，设置标志位
except ImportError as e:
    # 在应用界面显示一个持久的错误，如果导入失败
    st.error(f"核心依赖 'ultralytics' 导入失败: {e}")
    st.error("请检查项目结构和 requirements.txt 文件是否正确配置。")
    # 也可以选择 st.stop()，但这会让用户什么也看不到
except Exception as e:
    st.error(f"导入核心依赖时发生未知错误: {e}")
    # st.stop()

# --- 页面基础配置 (必须是第一个 Streamlit 命令) ---
st.set_page_config(
    page_title="XMedical - 智能影像分析",
    page_icon="🔬",
    layout="wide"
)

# --- 自定义 CSS ---
st.markdown(
    """
    <style>
    /* CSS remains the same */
    div[data-testid="stCameraInput"] video { width: 100% !important; height: auto !important; object-fit: cover !important; }
    div[data-testid="stImage"] { padding-left: 0 !important; padding-right: 0 !important; width: 100% !important; }
    div[data-testid="stImage"] img { width: 100% !important; height: auto !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 应用主标题和描述 ---
st.title("XMedical - 轻量级医学影像智能分析系统")
st.markdown("###### 利用先进 AI 技术，辅助分析医学影像（如脑部 MRI、胸部 CT、细胞图像等），快速识别潜在病灶或特定细胞。")


# --- 模型加载 (修正路径计算逻辑) ---
# APP_DIR 是 app.py 所在的目录: /mount/src/x-medical/X-Medical App/ (或其他你重命名的)
APP_DIR = Path(__file__).resolve().parent
# 获取项目根目录 (APP_DIR 的上一级目录)
project_root = APP_DIR.parent
# 现在基于项目根目录构建模型路径
model_path = project_root / 'Pt Source' / 'X-Medical.pt'

# 只有在 YOLO 成功导入后才尝试加载模型
if YOLO_IMPORTED:
    if 'model' not in st.session_state:
        # 检查模型文件是否存在，只在需要加载时检查一次
        if not model_path.exists():
             st.error(f"关键错误：模型文件未在预期路径找到！")
             st.error(f"预期路径: {model_path}")
             st.error(f"请检查 GitHub 仓库中，项目根目录下 ('{project_root.name}') 是否存在 'Pt Source/X-Medical.pt'，并检查大小写。")
             # 这里可以考虑 st.stop() 或者让应用继续运行但标记模型不可用
             MODEL_LOADED = False
        else:
            with st.spinner("⏳ 正在加载 X-Medical 深度学习模型，请稍候..."):
                try:
                    # Load model using the absolute path derived correctly
                    st.session_state.model = YOLO(model_path)
                    MODEL_LOADED = True # 标记模型加载成功
                    # 可以取消注释成功消息
                    # st.success("✅ 模型加载成功！")
                except Exception as e:
                    st.error(f"加载模型时发生严重错误：{e}")
                    st.error(f"尝试加载的路径是: {model_path}")
                    MODEL_LOADED = False # 标记模型加载失败
                    # st.stop() # 可以选择停止
    elif 'model' in st.session_state and st.session_state.model is not None:
         # 如果模型已在 session_state 中，认为它是加载好的
         MODEL_LOADED = True
    else:
         # 如果 'model' 在 session_state 但为 None 或其他无效状态
         MODEL_LOADED = False
         if 'model_load_error_shown' not in st.session_state: # 防止重复显示错误
             st.error("模型状态异常，无法确认是否加载成功。")
             st.session_state.model_load_error_shown = True

else:
    st.error("由于核心依赖未能成功导入, 模型加载步骤已跳过。")
    MODEL_LOADED = False # 标记模型未加载

# --- 创建选项卡 ---
tab1, tab2 = st.tabs(["🔬 影像检测分析", "ℹ️ 关于系统"])

# --- 选项卡1: 影像检测分析 ---
with tab1:
    # 只有模型加载成功才显示分析界面
    if MODEL_LOADED:
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

        tmp_file_path = None # Initialize tmp_file_path
        if img_file_buffer is not None:
            # 再次确认模型可用
            if 'model' in st.session_state and st.session_state.model is not None:
                bytes_data = img_file_buffer.getvalue()
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(bytes_data)
                        tmp_file_path = tmp_file.name # Assign here

                    start_time = time.time()
                    # 确保导入了 cv2, np, Image 等
                    import cv2
                    import numpy as np
                    from PIL import Image
                    results = st.session_state.model.predict(tmp_file_path, conf=confidence)
                    end_time = time.time()

                    if results:
                        result = results[0]
                        image = Image.open(tmp_file_path)
                        image_np = np.array(image.convert('RGB'))
                        boxes = result.boxes
                        if boxes is not None and len(boxes) > 0: # Check length too
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                label = f"{result.names[cls]}: {conf:.2f}" if result.names and cls in result.names else f"Class {cls}: {conf:.2f}"
                                color = (0, 255, 0)
                                cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(image_np, label, (x1, y1 - 10 if y1 > 10 else y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            st.image(image_np, caption="模型分析结果预览", use_column_width=True)

                            st.subheader("📊 分析报告")
                            inference_time = end_time - start_time
                            st.write(f"⏱️ 模型分析耗时: {inference_time:.4f} 秒")
                            detected_objects = {}
                            for box in boxes:
                                cls = int(box.cls[0])
                                class_name = result.names[cls] if result.names and cls in result.names else f"Class {cls}"
                                conf = float(box.conf[0])
                                detected_objects[class_name] = max(conf, detected_objects.get(class_name, 0.0))
                            st.write("🔍 **识别到的目标类别及最高置信度:**")
                            for obj, conf_val in detected_objects.items():
                                st.write(f" - {obj} (置信度: {conf_val:.2f})")
                        else:
                            st.image(np.array(Image.open(tmp_file_path).convert('RGB')), caption="原始图像（未检测到目标）", use_column_width=True)
                            st.info("ℹ️ 在当前置信度阈值下，未识别到明确的目标。")
                    else:
                        st.warning("⚠️ 模型预测未返回有效结果。")

                except Exception as e:
                    st.error(f"处理图像或执行预测时出错: {e}")
                finally:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        try:
                            os.remove(tmp_file_path)
                        except OSError as e_rm:
                            st.warning(f"无法删除临时文件 {tmp_file_path}: {e_rm}")
            else:
                st.error("模型状态异常，无法进行预测。")
    else:
        st.warning("模型未能成功加载或初始化，影像分析功能不可用。请检查日志或联系管理员。")


# --- 选项卡2: 关于系统 ---
# (代码无变化)
with tab2:
    st.subheader("ℹ️ 关于 XMedical 系统")
    # Markdown content remains the same...
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

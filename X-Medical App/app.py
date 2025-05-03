# app.py
# 作者: Hongyu Lin
# 机构: School of Mathematics and Computer Science, Shantou University

from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import time
import os

# --- 路径和模型配置 ---
# 计算当前脚本 (app.py) 的目录
# 虽然这里使用的是绝对路径，保留 APP_DIR 可能会在其他地方有用，但在这个特定的模型加载逻辑中不再用于构造路径
APP_DIR = Path(__file__).resolve().parent

# 定义场景名称 (键) 到模型文件绝对路径 (值) 的映射
# 请根据你的实际模型文件位置修改以下路径
SCENARIO_MODELS = {
    "胸部影像分析": r"D:\Desktop\NeuroLite-YOLO\X-Medical\Pt Source\VBD-YOLOv12.pt", # 胸部模型文件的绝对路径示例
    "细胞影像分析": r"D:\Desktop\NeuroLite-YOLO\X-Medical\Pt Source\CBC-YOLOv12.pt", # 细胞模型文件的绝对路径示例
    "脑肿瘤影像分析": r"D:\Desktop\NeuroLite-YOLO\X-Medical\Pt Source\X-Medical.pt", # 脑肿瘤模型文件的绝对路径示例 (原通用模型)
    # 如果有更多场景和模型，请在此处添加，格式为 "场景名称": r"你的模型文件的绝对路径"
}

# --- 页面基础配置 ---
st.set_page_config(
    page_title="XMedical - 智能影像分析系统",
    page_icon="🔬",
    layout="wide"
)

# --- 自定义 CSS ---
st.markdown(
    """
    <style>
    div[data-testid="stCameraInput"] video { width: 100% !important; height: auto !important; object-fit: cover !important; }
    div[data-testid="stImage"] { padding-left: 0 !important; padding-right: 0 !important; width: 100% !important; }
    div[data-testid="stImage"] img { width: 100% !important; height: auto !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 应用主标题和描述 ---
st.title("XMedical - 轻量级医学影像智能分析系统")
st.markdown("###### 利用先进 AI 技术，辅助分析医学影像（如脑部 MRI、胸部 CT、细胞图像等），**提供多种场景分析模型供选择**。")

# --- 创建选项卡 ---
tab1, tab2 = st.tabs(["🔬 影像检测分析", "ℹ️ 关于系统"])

# --- 选项卡1: 影像检测分析 ---
with tab1:
    st.subheader("实时影像分析")

    # --- 场景选择 ---
    selected_scenario = st.selectbox(
        "选择分析场景:", # 用户选择框的标签
        list(SCENARIO_MODELS.keys()), # 选项列表从场景字典的键中获取
        key='selected_scenario' # Streamlit session state 的键
    )

    # --- 根据选择加载模型 ---
    # 初始化 Streamlit session state，用于存储已加载的模型和当前加载的模型对应的场景
    if 'loaded_model' not in st.session_state:
        st.session_state.loaded_model = None
        st.session_state.current_scenario_loaded = None

    # 检查当前选择的场景模型是否已经加载
    if st.session_state.current_scenario_loaded != selected_scenario:
        # 需要加载新的模型
        model_path_full = SCENARIO_MODELS[selected_scenario] # 直接从字典获取模型的绝对路径

        # 在加载模型时显示加载中提示
        with st.spinner(f"⏳ 正在加载 '{selected_scenario}' 场景模型 ({model_path_full})，请稍候..."):
            try:
                # st.write(f"调试信息：尝试从以下绝对路径加载模型: {model_path_full}")
                # 检查模型文件是否存在 (使用 Path 对象进行检查)
                model_file_exists = Path(model_path_full).exists()
                # st.write(f"调试信息：指定的模型文件是否存在? {model_file_exists}")

                if not model_file_exists:
                    st.error(f"关键错误：模型文件未在指定路径找到！")
                    st.error(f"指定的路径是: {model_path_full}")
                    st.error(f"请确认上述绝对路径是正确的，并且文件确实存在。")
                    st.session_state.loaded_model = None # 加载失败时，确保模型对象为 None
                    st.session_state.current_scenario_loaded = None # 加载失败时，重置场景状态
                    st.stop() # 发生严重错误时停止 Streamlit 执行

                # 加载模型
                st.session_state.loaded_model = YOLO(model_path_full)
                # 更新已加载场景的状态
                st.session_state.current_scenario_loaded = selected_scenario
                st.success(f"✅ '{selected_scenario}' 场景模型加载成功！")

            except Exception as e:
                # 捕获加载模型时可能发生的任何异常
                st.error(f"加载模型时发生严重错误：{e}")
                st.error(f"尝试加载的路径是: {model_path_full}")
                st.session_state.loaded_model = None # 加载失败时，确保模型对象为 None
                st.session_state.current_scenario_loaded = None # 加载失败时，重置场景状态
                # 打印详细的错误信息，帮助调试
                import traceback
                st.exception(traceback.format_exc())
                st.stop() # 发生严重错误时停止 Streamlit 执行
    else:
         # 如果当前选择的场景模型已经加载，则不做任何操作
         pass # st.info(f"'{selected_scenario}' 场景模型已加载。") # 可选：取消注释此行以显示确认信息

    # --- 图像输入和分析 ---
    st.markdown(f"请点击下方“拍照”按钮，拍摄需要分析的医学影像区域。分析将使用当前选择的 **'{selected_scenario}'** 模型。")
    img_file_buffer = st.camera_input(
        "拍摄医学影像进行分析" # 拍照按钮的标签
    )

    # 置信度阈值滑动条
    confidence = st.slider(
        "🔬 病灶/目标识别置信度阈值", # 滑动条标签
        min_value=0.0, # 最小值
        max_value=1.0, # 最大值
        value=0.25, # 默认值
        step=0.05, # 步长
        key='confidence_slider' # Streamlit session state 的键
    )

    tmp_file_path = None # 初始化临时文件路径变量
    # 只有在图像被捕获且模型成功加载时才进行处理
    if img_file_buffer is not None and st.session_state.loaded_model is not None:
        # 读取图像数据
        bytes_data = img_file_buffer.getvalue()
        try:
            # 将捕获的图像保存到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(bytes_data)
                tmp_file_path = tmp_file.name # 将临时文件路径赋值给变量

            # 使用已加载的模型进行预测
            start_time = time.time() # 记录开始时间
            # YOLO 模型对象是可调用的，可以直接传入图像路径进行预测
            results = st.session_state.loaded_model(tmp_file_path, conf=confidence)
            end_time = time.time() # 记录结束时间

            if results:
                result = results[0] # 假设批处理大小为 1，取第一个结果
                image = Image.open(tmp_file_path) # 打开临时图像文件
                image_np = np.array(image.convert('RGB')) # 将图像转换为 NumPy 数组，确保是 RGB 格式以便 OpenCV 处理
                boxes = result.boxes # 获取检测到的边界框信息

                st.subheader("📊 分析报告") # 显示分析报告小标题
                inference_time = end_time - start_time # 计算推理时间
                st.write(f"⏱️ 模型分析耗时: {inference_time:.4f} 秒") # 显示推理时间

                if boxes is not None and len(boxes) > 0: # 检查是否检测到任何目标
                    detected_objects_counts = {} # 字典用于存储各类别目标的数量
                    detected_objects_confidences = {} # 字典用于存储各类别目标的最高置信度

                    # 遍历所有检测到的边界框，并在图像上绘制
                    for box in boxes:
                        # 获取边界框坐标 (x1, y1, x2, y2)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # 获取类别整数 ID 和置信度
                        cls_int = int(box.cls[0])
                        conf = float(box.conf[0])

                        # 获取类别名称，同时处理潜在的错误或缺失名称
                        if result.names and cls_int in result.names:
                             class_name = result.names[cls_int]
                        else:
                             class_name = f"Unknown Class {cls_int}" # 如果类别 ID 未在模型名称中找到
                             st.warning(f"⚠️ 检测到未知类别ID: {cls_int}. 请检查模型类别映射.")

                        # 构建标注文本 (类别名称和置信度)
                        label = f"{class_name}: {conf:.2f}"
                        color = (0, 255, 0) # 边界框颜色 (绿色)

                        # 在图像上绘制矩形框
                        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
                        # 调整文本位置，如果边界框靠近图像顶部，则将文本放在框下方
                        text_y_position = y1 - 10 if y1 > 20 else y1 + 20
                        # 在图像上绘制类别标签和置信度文本
                        cv2.putText(image_np, label, (x1, text_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # 统计各类别数量并记录最高置信度
                        detected_objects_counts[class_name] = detected_objects_counts.get(class_name, 0) + 1
                        detected_objects_confidences[class_name] = max(conf, detected_objects_confidences.get(class_name, 0.0))

                    # 显示带有标注结果的图像
                    st.image(image_np, caption="模型分析结果预览", use_column_width=True)

                    # 显示识别到的目标详情
                    st.write("🔍 **识别到的目标:**")
                    # 按类别名称字母顺序排序后显示
                    sorted_classes = sorted(detected_objects_counts.keys())
                    for class_name in sorted_classes:
                         count = detected_objects_counts[class_name] # 获取数量
                         max_conf = detected_objects_confidences[class_name] # 获取最高置信度
                         st.write(f" - **{class_name}**: 数量 {count}, 最高置信度 {max_conf:.2f}")

                else: # 如果在当前置信度阈值下未检测到任何目标
                    # 显示原始图像
                    st.image(np.array(Image.open(tmp_file_path).convert('RGB')), caption="原始图像 (未检测到目标)", use_column_width=True)
                    st.info(f"ℹ️ 在当前置信度 ({confidence:.2f}) 和 '{selected_scenario}' 模型下，未识别到明确目标。")

            else: # 如果模型预测没有返回有效结果 (results 对象为空或 None)
                st.warning("⚠️ 模型预测未返回有效结果。")

        except Exception as e:
            # 捕获处理图像或执行预测时可能发生的任何异常
            st.error(f"处理图像或执行预测时出错: {e}")
            # 打印详细的错误追溯信息以帮助调试
            import traceback
            st.exception(traceback.format_exc())

        finally:
            # 清理临时文件
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                    # st.write(f"调试信息: 已删除临时文件: {tmp_file_path}") # 可选调试信息
                except OSError as e:
                    st.warning(f"无法删除临时文件 {tmp_file_path}: {e}")

    elif img_file_buffer is not None and st.session_state.loaded_model is None:
         # 如果图像被捕获但模型尚未加载成功 (理论上前面加载失败会停止执行，这是额外的安全检查)
         st.warning("请等待模型加载完成再进行拍照。")


# --- 选项卡2: 关于系统 ---
with tab2:
    st.subheader("ℹ️ 关于 XMedical 系统") # 关于系统小标题
    st.markdown(f"""
    **XMedical** 是一个基于先进的 **YOLOv8/YOLOv5/YOLOv11/YOLOv12** 等深度学习框架构建的轻量级医学影像智能分析系统。
    它提供多种经过专门训练的模型，旨在辅助医生或研究人员快速识别医学影像中的特定目标。

    当前系统提供了以下分析场景模型供您选择：
    """)
    # 动态列出可用的分析场景和对应的模型路径
    for scenario_name, file_path in SCENARIO_MODELS.items():
         st.write(f"- **{scenario_name}**: 对应的模型文件路径是 `{file_path}`。")

    st.markdown("""
    ---
    ### 💡 如何使用:
    1.  在 **“影像检测分析”** 选项卡中，首先从 **“选择分析场景”** 下拉菜单中选择最适合您影像类型的场景。
    2.  等待系统加载对应的模型（首次选择或切换场景时可能需要时间，请耐心等待）。
    3.  点击 **“拍摄医学影像进行分析”** 下方的 **“拍照”** 按钮。
    4.  **允许浏览器访问您的摄像头**（如果弹出请求）。
    5.  将摄像头**对准需要分析的医学影像**（可以是屏幕上的图像、打印的胶片或显微镜视野）。
    6.  **确保光线充足、图像清晰**，然后点击拍照图标完成拍摄。
    7.  系统将自动使用选定的模型处理图像，并在下方显示带有**标注框和类别标签**的识别结果以及简要报告。
    8.  您可以通过拖动 **“病灶/目标识别置信度阈值”** 滑块来调整识别灵敏度，以过滤掉低置信度的结果。

    ---
    ### ⚙️ 技术细节:
    *   **核心框架**: 支持 YOLOv8, YOLOv5 等 (具体取决于您的 `.pt` 文件是如何导出或训练的)。
    *   **识别能力**: 取决于您选择的分析场景对应的模型，每个模型都针对特定类型的医学影像数据进行了训练，能够识别如胸部病变、特定细胞类型、脑肿瘤等预设类别。
    *   **处理方式**: 图像数据的分析和计算在**运行 Streamlit 应用的服务器端**完成，保障了处理速度和效果，同时不占用本地设备过多资源。

    ---
    **免责声明:** 本系统仅为辅助分析工具，分析结果**不能替代**专业的医疗诊断。所有医疗决策请务必咨询执业医师。
    """)
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
# 这个路径将用于构建模型文件的相对路径
APP_DIR = Path(__file__).resolve().parent

# 定义场景名称 (键) 到模型文件相对路径 (值) 的映射
# 这些路径是相对于 app.py 文件所在的目录 (即 APP_DIR)
# 假定模型文件都放在 app.py 同级的 'Pt Source' 文件夹下
SCENARIO_MODELS = {
    "胸部影像分析": 'Pt Source/VBD-YOLOv12.pt', # 胸部模型文件在 'Pt Source' 文件夹下的相对路径
    "细胞影像分析": 'Pt Source/CBC-YOLOv12.pt', # 细胞模型文件在 'Pt Source' 文件夹下的相对路径
    "脑肿瘤影像分析": 'Pt Source/X-Medical.pt', # 脑肿瘤模型文件在 'Pt Source' 文件夹下的相对路径
    # 如果有更多场景和模型，请在此处添加，格式为 "场景名称": 'Pt Source/你的模型文件名称.pt'
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
    /* CSS 样式保持不变 */
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
    # 这样可以在用户切换场景时判断是否需要重新加载模型，提高效率
    if 'loaded_model' not in st.session_state:
        st.session_state.loaded_model = None
        st.session_state.current_scenario_loaded = None

    # 检查当前选择的场景模型是否已经加载
    if st.session_state.current_scenario_loaded != selected_scenario:
        # 需要加载新的模型
        model_file_name_relative = SCENARIO_MODELS[selected_scenario] # 从字典获取模型的相对路径
        # 构建模型的完整绝对路径 (在部署环境中，这个路径是相对于应用的根目录)
        model_path_full = APP_DIR / model_file_name_relative

        # 在加载模型时显示加载中提示
        with st.spinner(f"⏳ 正在加载 '{selected_scenario}' 场景模型 ({model_file_name_relative})，请稍候..."):
            try:
                # 为了调试，可以显示尝试加载的完整路径
                # st.write(f"调试信息：尝试从以下路径加载模型: {model_path_full}")
                # 检查模型文件是否存在
                model_file_exists = model_path_full.exists()
                # 为了调试，可以显示文件是否存在的结果
                # st.write(f"调试信息：指定的模型文件是否存在? {model_file_exists}")

                if not model_file_exists:
                    # 如果文件不存在，报错并提示用户检查路径和文件
                    st.error(f"关键错误：模型文件 '{model_file_name_relative}' 未在预期位置找到！")
                    st.error(f"预期路径 (相对于应用根目录): {model_path_full}")
                    st.error(f"请确保在 GitHub 仓库中，模型文件位于 '{model_file_name_relative}' 路径下。如果模型文件较大，请确保使用了 Git LFS 进行跟踪。")
                    st.session_state.loaded_model = None # 加载失败时，确保模型对象为 None
                    st.session_state.current_scenario_loaded = None # 加载失败时，重置场景状态
                    st.stop() # 发生严重错误时停止 Streamlit 执行

                # 加载模型
                st.session_state.loaded_model = YOLO(str(model_path_full)) # YOLO 通常接受字符串路径
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
         # 如果当前选择的场景模型已经加载，则不做任何操作，避免重复加载
         pass # st.info(f"'{selected_scenario}' 场景模型已加载。") # 可选：取消注释此行以显示确认信息

    # --- 图像输入和分析 ---
    # 只有在模型成功加载后，才显示拍照按钮
    if st.session_state.loaded_model is not None:
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
        # 只有在图像被捕获时才进行处理
        if img_file_buffer is not None:
            # 读取图像数据
            bytes_data = img_file_buffer.getvalue()
            try:
                # 将捕获的图像保存到临时文件， Streamlit Cloud 环境下 tempfile 会找到合适的临时目录
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
                            # 检查 result.names 是否存在且 cls_int 是一个有效的键
                            if result.names and isinstance(result.names, dict) and cls_int in result.names:
                                 class_name = result.names[cls_int]
                            else:
                                 class_name = f"未知类别 {cls_int}" # 如果类别 ID 未在模型名称中找到或 names 不是字典
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

    # 如果模型尚未加载成功，提示等待
    elif st.session_state.loaded_model is None:
         st.info("正在加载模型，请稍候...") # 仅在模型加载过程中显示此信息，防止过早显示拍照按钮


# --- 选项卡2: 关于系统 ---
with tab2:
    st.subheader("ℹ️ 关于 XMedical 系统") # 关于系统小标题
    st.markdown(f"""
    **XMedical** 是一个基于先进的 **YOLOv8/YOLOv11/YOLOv12** 等深度学习框架构建的轻量级医学影像智能分析系统。
    它提供多种经过专门训练的模型，旨在辅助医生或研究人员快速识别医学影像中的特定目标。

    当前系统提供了以下分析场景模型供您选择：
    """)
    # 动态列出可用的分析场景和对应的模型文件名称 (相对于 Pt Source 文件夹)
    for scenario_name, file_path_relative in SCENARIO_MODELS.items():
         # 仅显示相对路径或文件名更适合部署环境
         st.write(f"- **{scenario_name}**: 对应的模型文件是 `{Path(file_path_relative).name}`。") # 只显示文件名

    st.markdown("""
    ---
    ### 部署到 GitHub 的准备事项:
    1.  **模型文件**: 确保所有 `.pt` 模型文件都存放在 `app.py` 文件同级的 `Pt Source` 文件夹中。
    2.  **Git LFS (重要!)**: 如果你的模型文件单个大小超过 100MB，你需要安装并使用 Git Large File Storage (LFS) 来跟踪和管理这些大文件，否则无法上传到 GitHub。
        *   安装 Git LFS: [https://git-lfs.github.com/](https://git-lfs.github.com/)
        *   在你的项目目录下初始化 Git LFS: `git lfs install`
        *   跟踪你的 `.pt` 文件类型: `git lfs track "*.pt"` (这会在 `.gitattributes` 文件中添加一行配置)
        *   将你的 `.pt` 文件添加到 Git 暂存区: `git add Pt Source/*.pt .gitattributes app.py requirements.txt` (或其他你需要的文件)
        *   提交并推送到 GitHub: `git commit -m "Add models and app"` -> `git push`
    3.  **requirements.txt**: 创建一个 `requirements.txt` 文件，列出所有依赖库及其版本，例如:
        ```
        streamlit
        ultralytics
        opencv-python
        Pillow
        numpy
        ```
        确保 `ultralytics` 的版本能够兼容你的 YOLOv12 模型。
    4.  **GitHub 仓库**: 将 `app.py`、`Pt Source` 文件夹及其中的 `.pt` 文件 (确保 LFS 工作正常)、`requirements.txt` 文件等所有项目文件推送到 GitHub 仓库。

    ---
    ### 部署到 Streamlit Cloud:
    1.  确保你的应用代码和模型文件已按上述步骤提交到 GitHub 仓库。
    2.  访问 Streamlit Cloud ([https://share.streamlit.io/](https://share.streamlit.io/)) 并登录。
    3.  点击 "New app" 按钮。
    4.  选择你的 GitHub 仓库、主分支以及 `app.py` 文件作为主应用文件。
    5.  点击 "Deploy!"。Streamlit Cloud 会自动读取 `requirements.txt` 并安装依赖，然后启动你的应用。
    6.  如果模型文件使用了 Git LFS，Streamlit Cloud 会自动下载 LFS 管理的文件。

    ---
    ### ⚙️ 技术细节:
    *   **核心框架**: YOLO Series：YOLOv8, YOLOv11 及 YOLOv12。
    *   **识别能力**: 取决于您选择的分析场景对应的模型，每个模型都针对特定类型的医学影像数据进行了训练，能够识别如胸部病变、特定细胞类型、脑肿瘤等预设类别。
    *   **处理方式**: 图像数据的分析和计算在**运行 Streamlit 应用的服务器端**完成，保障了处理速度和效果，同时不占用本地设备过多资源。

    ---
    **免责声明:** 本系统仅为辅助分析工具，分析结果**不能替代**专业的医疗诊断。所有医疗决策请务必咨询执业医师。
    """)

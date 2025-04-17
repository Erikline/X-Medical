# app.py
# Author: Hongyu Lin
# Institution: School of Mathematics and Computer Science, Shantou University

import sys # 导入 sys 用于查看路径
from pathlib import Path
import streamlit as st
import os # 确保导入 os 用于文件检查
# --- 确保所有 import 都在最前面 ---
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
# 尝试将 ultralytics 的导入也放在这里，如果后续不再出错的话
# 但如果这里出错，就需要保持 try...except 结构
# from ultralytics import YOLO # 保留在 try...except 中更安全

# --- !!! 页面基础配置 (必须是第一个 Streamlit 命令) !!! ---
st.set_page_config(
    page_title="XMedical - 智能影像分析",
    page_icon="🔬",
    layout="wide"
)
# --- !!! 结束页面配置代码块 !!! ---


# --- 调试代码：检查 Python 路径和 ultralytics 模块 ---
# (注释掉所有调试输出)
# st.write("--- Python 搜索路径 (sys.path) ---")
# st.text("\n".join(sys.path)) # 打印 Python 查找模块的所有路径
# st.write("------------------------------------")

# 获取当前脚本(app.py)所在的目录 (应该是 X-Medical App 目录)
script_dir = Path(__file__).resolve().parent
# st.write(f"调试信息：当前脚本目录 (script_dir): {script_dir}")

# 期望的本地 ultralytics 库的路径 (假设它与 app.py 同级)
expected_ultralytics_dir = script_dir / 'ultralytics'
# st.write(f"调试信息：期望的本地 ultralytics 库目录: {expected_ultralytics_dir}")
# st.write(f"调试信息：该期望目录是否存在? {expected_ultralytics_dir.exists()}")

# if expected_ultralytics_dir.exists():
     `st.error`。

```python
# app.py (最终清理版)
# Author: Hongyu Lin
# Institution: School of Mathematics and Computer Science, Shantou University

import sys
from pathlib import Path
import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
# 主要的 ultralytics 导入现在直接进行
from ultralytics import YOLO

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


# --- 模型加载 (使用修正后的路径) ---
# APP_DIR 是 app.py 所在的目录: /mount/src/x-medical/X-Medical App/
APP_DIR = Path(__file__).resolve().parent
# 获取项目根目录 (APP_DIR 的上一级目录)
project_root = APP_DIR.parent
# 现在基于项目根目录构建模型路径
model_path = project_root / 'Pt Source' / 'X-Medical.pt'

# 检查顶层 __init__.py
    # expected_top_init = expected_ultralytics_dir / '__init__.py'
    # st.write(f"调试信息：检查期望目录下是否存在 __init__.py: {expected_top_init.exists()}")
    # 使用 try-except 防止 os.listdir 在路径不存在或无权限时报错
    # try:
    #     st.write(f"调试信息：期望目录下的部分内容: {os.listdir(expected_ultralytics_dir)[:15]}") # 只显示前15个
    # except Exception as list_err:
    #     st.warning(f"无法列出期望目录内容: {list_err}")

    # 检查期望目录下的 data 文件夹
    # expected_data_dir = expected_ultralytics_dir / 'data'
    # st.write(f"调试信息：期望的 data 子目录: {expected_data_dir}")
    # st.write(f"调试信息：该 data 子目录是否存在? {expected_data_dir.exists()}")
    # if expected_data_dir.exists():
        # 检查 data 目录下的 __init__.py
        # expected_data_init = expected_data_dir / '__init__.py'
        # st.write(f"调试信息：检查期望 data 目录下是否存在 __init__.py: {expected_data_init.exists()}")
        # try:
        #     st.write(f"调试信息：期望 data 目录下的部分内容: {os.listdir(expected# 使用 session_state 缓存模型，避免重复加载
if 'model' not in st.session_state:
    with st.spinner("⏳ 正在加载 X-Medical 深度学习模型，请稍候..."):
        try:
            if not model_path.exists():
                st.error(f"错误：模型文件未在预期路径找到！")
                st.error(f"预期路径: {model_path}")
                st.error(f"请确认项目根目录下 ('{project_root.name}') 是否存在 'Pt Source/X-Medical.pt'。")
                st.stop()

            # 加载模型
            st.session_state.model = YOLO(model_path)
            # 可选：首次加载成功提示
            # st.toast("✅ 模型加载成功！", icon="🎉")

        except Exception as e:
            st.error(f"加载模型时发生严重错误：{e}")
            st.error(f"尝试加载的路径是: {model_path}")
            st.stop() # 模型加载失败则停止

# --- 创建选项卡 ---
tab1, tab2 = st.tabs(["🔬 影像检测分析", "ℹ️ 关于系统"])

# --- 选项卡1: 影像检测分析 ---
with tab1:
    st.subheader("实时影像分析")
    img_file_data_dir)[:15]}") # 只显示前15个
        # except Exception as list_data_err:
        #     st.warning(f"无法列出期望 data 目录内容: {list_data_err}")

# st.write("--- 尝试导入 ultralytics 包及子模块 ---")
# try:
    # 1. 尝试导入顶层 ultralytics
    # import ultralytics
    # st.write(f"成功导入顶层 'ultralytics' 包。")
    # __path__ 对于包更可靠，__file__ 指向 __init__.py
    # st.write(f"实际加载的 ultr_buffer = st.camera_input(
        "请点击下方“拍照”按钮，拍摄需要分析的医学影像区域（例如显示器上的CT/MRI图像、显微镜视野等）"
    )

    confidence = st.slider(
        "🔬 病灶/目标识别置信度阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.25, # 默认值
        step=0.05
    )

    tmp_file_path = None # 初始化临时文件路径变量
    if img_file_buffer is not None:
        # 确保模型已成功加载到 session_state
        if 'model' in st.session_state and st.session_state.model is not None:
            bytes_data = img_file_buffer.getvalue()
            try:
                # 创建临时文件来保存上传的图像
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(bytes_data)
                    tmp_file_path = tmp_file.name # 获取临时文件路径

                start_time = time.time()
                # 使用加载好的模型进行预测
                results = st.session_state.model.predict(tmp_file_path, conf=confidence)
                end_time = time.time()

                # --- 处理和显示结果 ---
                ifalytics 包路径 (__path__): {getattr(ultralytics, '__path__', 'N/A')}")
    # st.write(f"实际加载的 ultralytics 包文件 (__file__): {getattr(ultralytics, '__file__', 'N/A')}")

    # 2. 尝试导入 ultralytics.data (如果顶层导入成功)
    # try:
    #     import ultralytics.data
    #     st.write("成功导入 'ultralytics.data' 子模块。")
    #     st.write(f"实际加载的 ultralytics.data 路径 (__path__): {getattr(ultralytics.data, '__path__', 'N/A')}")
    # except ImportError as e_data:
    #     st.error(f"尝试导入 'ultralytics.data' 子模块时失败: {e_data}")
    # except Exception as e_data_other:
    #      st.error(f"尝试导入 ultralytics.data 时发生其他错误: {e_data_other}")

    # 3. 尝试导入 ultralytics.data.augment (如果 data 导入成功)
    # try:
    #     import ultralytics.data.augment
    #     st.write("成功导入 'ultralytics.data.augment'。")
 results: # 确保有预测结果
                    result = results[0] # 通常获取第一个结果
                    image = Image.open(tmp_file_path) # 重新打开图像用于绘制
                    image_np = np.array(image.convert('RGB')) # 转换为Numpy数组(确保是RGB)

                    boxes = result.boxes # 获取检测框信息
                    if boxes is not None and len(boxes) >    # except ImportError as e_augment:
    #     st.error(f"尝试导入 'ultralytics.data.augment' 时失败: {e_augment}")
    # except Exception as e_augment_other:
    #      st.error(f"尝试导入 ultralytics.data.augment 时发生其他错误: {e_augment_other}")

# except ImportError as e_top:
#     st.error(f"尝试导入顶层 'ultralytics' 包时就已失败: {e_top}")
# except Exception as e_top_other:
#     st.error(f"检查 ultralytics 包时发生其他错误: {e_top_other}")

# st.write("--- 调试结束，开始主要导入 ---")


# --- 主要导入（现在放在调试代码之后，并用 try...except 包裹） 0: # 检查是否有检测框
                        # 绘制检测框和标签
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            # 检查 result.names 是否存在且包含 cls
                            label = f"{result.names[cls]}: {conf:.2f}" if result.names and cls in result.names else f"Class {cls}: {conf:.2f}"
                            color = (0, 255, 0) # BGR for OpenCV
                            cv2.rectangle(image_np, (x1, y1), (x2, y2 ---
YOLO_IMPORTED = False # 添加一个标志位
try:
    from ultralytics import YOLO # <--- 尝试导入 YOLO
    import cv2                 # 只有 YOLO 导入成功后才导入这些
    import numpy as np
    from PIL import Image
    import tempfile
    import time

    YOLO_IMPORTED = True # 如果成功，设置标志位
    # st.success("主要依赖导入成功（包括 from ultralytics import YOLO）") # 可以注释掉成功信息

except ImportError), color, 2)
                            cv2.putText(image_np, label, (x1, y1 - 10 if y1 > 10 else y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # 显示带标注的图像
                        st.image(image_np, caption="模型分析结果预览", use_column_width=True)

                        # --- 分析报告 ---
                        st.subheader("📊 分析报告")
                        inference_time = end_time - start_time
                        st.write( as e:
    # 保留错误处理，以防万一将来出问题
    st.error(f"在主要导入阶段发生 ImportError: {e}")
    st.error("Ultralytics 库或f"⏱️ 模型分析耗时: {inference_time:.4f} 秒")

                        # 统计识别到的目标类别及最高置信度
                        detected_objects = {}
                        for box in boxes:
                            cls = int(box.cls[0])
                            class_name = result.names[cls其依赖项未能正确导入。请检查环境和文件结构。")
    st.stop() # 如果导入失败，停止应用
except Exception as e:
    st.error(] if result.names and cls in result.names else f"Class {cls}"
                            conf = float(box.conf[0])
                            f"在主要导入阶段发生其他错误: {e}")
    stdetected_objects[class_name] = max(conf, detected_objects.stop()

# --- 自定义 CSS ---
# (代码无变化)
.get(class_name, 0.0))

                        st.write("🔍 **识别到的目标类别及最高置信度:**")
st.markdown(
    """
    <style>
    /* CSS                        for obj, conf_val in detected_objects.items():
                             remains the same */
    div[data-testid="stCameraInput"] video { width: 100% !important; height: auto !st.write(f" - {obj} (置信度: {conf_val:.2f})")
                    else:
                        # 未important; object-fit: cover !important; }
    div[data-testid="stImage"] { padding-left: 0 !important; padding-right: 0 !important; width: 100%检测到目标，显示原图
                        st.image(image_np, caption="原始图像（未检测到目标）", use_column_width=True)
                        st.info("ℹ️ 在当前置信度阈值下，未识别到明确的目标。")
                else:
                    st.warning("⚠️ 模型预测未返回有效结果。")

             !important; }
    div[data-testid="stImage"] img { width: 100% !important; height: auto !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 应用主标题和描述 ---
# (代码无变化)
st.title("XMedical - 轻量级医学影像智能分析系统")except Exception as e:
                st.error(f"处理图像或执行预测时出错: {e}")
            finally:
                # 无论成功失败，都尝试删除临时文件
                if tmp_file_path and os.path.exists(tmp_file_path):
                    try:
                        os.remove(tmp_file_path)
                    except OSError as e_rm:
                        st.warning
st.markdown("###### 利用先进 AI 技术，辅助分析医学影像（如脑部 MRI、胸部 CT、细胞图像等），快速识别潜在病灶或特定细胞。")


# --- 模型加载 (修正路径计算逻辑) ---
# APP_DIR 是 app.py 所在的目录: /mount/src/x-medical/X-Medical App/
APP_DIR = Path(__file__).resolve().parent

# 获取项目根目录 ((f"无法删除临时文件 {tmp_file_path}: {e_rm}")
        else:
            st.error("模型未能成功加载或初始化，无法进行预测。")

# --- 选项卡2: 关于系统 ---
with tab2:
    st.subheader("ℹ️ 关于 XMedical 系统")
    st.markdown("""
    **XMedical** 是一个基于先进的 **YOLOv8** 深度学习框架构建的轻量级医学APP_DIR 的上一级目录)
project_root = APP_DIR.parent
# 现在基于项目根目录构建模型路径
model_path = project_root / 'Pt Source' / 'X-Medical.pt'


# 只有在 YOLO 成功导入后才尝试加载模型
if YOLO_IMPORTED:
    if 'model' not in st.session_state:
        with st.spinner("⏳ 正在加载 X-Medical 深度学习模型，影像智能分析系统。
    它经过专门训练，旨在辅助医生或研究人员快速识别医学影像中的特定目标，例如：

    *   **脑部影像**: 可能的肿瘤区域。
    *   **胸部影像**: 可疑的结节或病变。
    *   **细胞学图像**: 特定类型的细胞计数或状态分析。

    ---
    ### 💡 如何使用:
    1.  在 **“影像检测分析”**请稍候..."):
            try:
                # 移除调试信息
                # st.write(f"调试信息：项目根目录 (project_root): {project_root}")
                # st.write(f"调试信息：修正后尝试加载模型的路径: {model_path}")
                # st.write(f"调试信息：修正 选项卡中，点击 **“拍照”** 按钮。
    2.  **允许浏览器访问您的摄像头**（如果弹出请求）。
    3.  将摄像头**对准需要分析的医学影像**（可以是屏幕上的图像、打印的胶片或显微镜视野）。
    4.  **确保光线充足、图像清晰**，然后点击拍照图标完成拍摄。
    5.  系统将自动处理图像，并在下方显示带有**标注框和类别标签**的识别结果。
    6.  您可以通过拖动 **“病灶/目标识别置信度阈值”** 滑块来调整灵敏度，过滤后的路径文件是否存在? {model_path.exists()}")

                if not model_path.exists():
                    st.error(f"关键错误：模型文件未在预期路径找到！")
                    st.error(f"预期路径掉低置信度的识别结果。

    ---
    ### ⚙️ 技术细节:
    *   **核心模型**: X-Medical (基于 YOLOv8 优化训练)。
    *   **识别能力**: 针对特定的医学影像数据集进行训练，能够识别如脑肿瘤、胸部病变、特定: {model_path}")
                    st.error(f"请检查 GitHub 仓库中，项目根目录下 ('{project_root.name}') 是否存在 'Pt Source/X-Medical.pt'，并检查大小细胞等预设类别 (*请根据您模型的实际训练数据在此处进行更详细的说明*)。
    *   **处理方式**: 图像写。")
                    st.stop()

                # Load model using the absolute path derived correctly
                st.session_state.model = YOLO(model_path)
                # st.success("✅ 模型加载成功！数据的分析和计算在**服务器端**完成，保障了处理速度和效果，同时不占用本地设备过多资源。

    ---
    **免责声明:** 本系统仅为辅助分析工具，分析结果**不能替代**专业的医疗诊断。所有医疗决策请务必咨询执业医师。
    """)

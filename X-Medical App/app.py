# app.py (加入详细调试步骤)
# Author: Hongyu Lin
# Institution: School of Mathematics and Computer Science, Shantou University

import sys # 导入 sys 用于查看路径
from pathlib import Path
import streamlit as st
import os # 确保导入 os 用于文件检查

# --- 调试代码：检查 Python 路径和 ultralytics 模块 ---
st.write("--- Python 搜索路径 (sys.path) ---")
# 使用 st.text 而不是 st.write 来更好地显示列表换行
st.text("\n".join(sys.path)) # 打印 Python 查找模块的所有路径
st.write("------------------------------------")

# 获取当前脚本(app.py)所在的目录 (假设现在是 X-Medical App 目录)
script_dir = Path(__file__).resolve().parent
st.write(f"调试信息：当前脚本目录 (script_dir): {script_dir}")

# 期望的本地 ultralytics 库的路径 (假设它与 app.py 同级)
expected_ultralytics_dir = script_dir / 'ultralytics'
st.write(f"调试信息：期望的本地 ultralytics 库目录: {expected_ultralytics_dir}")
st.write(f"调试信息：该期望目录是否存在? {expected_ultralytics_dir.exists()}")

if expected_ultralytics_dir.exists():
    # 检查顶层 __init__.py
    expected_top_init = expected_ultralytics_dir / '__init__.py'
    st.write(f"调试信息：检查期望目录下是否存在 __init__.py: {expected_top_init.exists()}")
    st.write(f"调试信息：期望目录下的部分内容: {os.listdir(expected_ultralytics_dir)[:15]}") # 只显示前15个

    # 检查期望目录下的 data 文件夹
    expected_data_dir = expected_ultralytics_dir / 'data'
    st.write(f"调试信息：期望的 data 子目录: {expected_data_dir}")
    st.write(f"调试信息：该 data 子目录是否存在? {expected_data_dir.exists()}")
    if expected_data_dir.exists():
        # 检查 data 目录下的 __init__.py
        expected_data_init = expected_data_dir / '__init__.py'
        st.write(f"调试信息：检查期望 data 目录下是否存在 __init__.py: {expected_data_init.exists()}")
        st.write(f"调试信息：期望 data 目录下的部分内容: {os.listdir(expected_data_dir)[:15]}") # 只显示前15个

st.write("--- 尝试导入 ultralytics 包及子模块 ---")
try:
    # 1. 尝试导入顶层 ultralytics
    import ultralytics
    st.write(f"成功导入顶层 'ultralytics' 包。")
    # __path__ 对于包更可靠，__file__ 指向 __init__.py
    st.write(f"实际加载的 ultralytics 包路径 (__path__): {getattr(ultralytics, '__path__', 'N/A')}")
    st.write(f"实际加载的 ultralytics 包文件 (__file__): {getattr(ultralytics, '__file__', 'N/A')}")

    # 2. 尝试导入 ultralytics.data (如果顶层导入成功)
    try:
        import ultralytics.data
        st.write("成功导入 'ultralytics.data' 子模块。")
        st.write(f"实际加载的 ultralytics.data 路径 (__path__): {getattr(ultralytics.data, '__path__', 'N/A')}")
    except ImportError as e_data:
        st.error(f"尝试导入 'ultralytics.data' 子模块时失败: {e_data}") # 预期错误会在这里
    except Exception as e_data_other:
         st.error(f"尝试导入 ultralytics.data 时发生其他错误: {e_data_other}")

    # 3. 尝试导入 ultralytics.data.augment (如果 data 导入成功)
    try:
        import ultralytics.data.augment
        st.write("成功导入 'ultralytics.data.augment'。")
    except ImportError as e_augment:
        st.error(f"尝试导入 'ultralytics.data.augment' 时失败: {e_augment}")
    except Exception as e_augment_other:
         st.error(f"尝试导入 ultralytics.data.augment 时发生其他错误: {e_augment_other}")


except ImportError as e_top:
    st.error(f"尝试导入顶层 'ultralytics' 包时就已失败: {e_top}")
except Exception as e_top_other:
    st.error(f"检查 ultralytics 包时发生其他错误: {e_top_other}")

st.write("--- 调试结束，开始主要导入 ---")


# --- 主要导入（现在放在调试代码之后，并用 try...except 包裹） ---
try:
    # NO MORE sys.path manipulation needed here if ultralytics lib is sibling to app.py
    # from pathlib import Path # 已在前面导入
    # import streamlit as st # 已在前面导入
    import cv2
    import numpy as np
    from PIL import Image
    # This import should now find the 'ultralytics' folder located
    # as a sibling to this app.py file.
    from ultralytics import YOLO # <--- 错误发生点
    import tempfile
    import time
    # import os # 已在前面导入

    st.success("主要依赖导入成功（包括 from ultralytics import YOLO）")

except ImportError as e:
    st.error(f"在主要导入阶段发生 ImportError: {e}") # 错误会在这里被捕获
    st.error("请检查上面的调试信息，确认 ultralytics 库是否被正确找到以及其内部结构是否完整。")
    st.stop() # 如果导入失败，停止应用
except Exception as e:
    st.error(f"在主要导入阶段发生其他错误: {e}")
    st.stop()

# --- 页面基础配置 ---
# (代码无变化)
st.set_page_config(
    page_title="XMedical - 智能影像分析",
    page_icon="🔬",
    layout="wide"
)

# --- 自定义 CSS ---
# (代码无变化)
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
# (代码无变化)
st.title("XMedical - 轻量级医学影像智能分析系统")
st.markdown("###### 利用先进 AI 技术，辅助分析医学影像（如脑部 MRI、胸部 CT、细胞图像等），快速识别潜在病灶或特定细胞。")


# --- 模型加载 ---
# Calculate path relative to this script (app.py)
# APP_DIR 已在调试部分获取 (script_dir), 但为清晰起见再获取一次
APP_DIR = Path(__file__).resolve().parent # Directory containing app.py
# Model path is now correctly relative to APP_DIR
model_path = APP_DIR / 'Pt Source' / 'X-Medical.pt'

if 'model' not in st.session_state:
    with st.spinner("⏳ 正在加载 X-Medical 深度学习模型，请稍候..."):
        try:
            # 移除这里的调试信息，因为顶部已经有了更详细的
            # st.write(f"调试信息：脚本目录 (APP_DIR): {APP_DIR}")
            # st.write(f"调试信息：尝试从以下路径加载模型: {model_path}")
            # st.write(f"调试信息：该路径的文件是否存在? {model_path.exists()}")

            if not model_path.exists():
                st.error(f"关键错误：模型文件未在预期路径找到！")
                st.error(f"预期路径: {model_path}")
                st.error(f"请检查 GitHub 仓库中，在 '{APP_DIR.name}' 文件夹内是否存在 'Pt Source/X-Medical.pt'，并检查大小写。")
                st.stop()

            # 确保 YOLO 类已成功导入
            if 'YOLO' not in globals():
                 st.error("YOLO 类未能成功导入，无法加载模型。请检查顶部的导入错误。")
                 st.stop()

            # Load model using the absolute path derived correctly
            st.session_state.model = YOLO(model_path)
            # st.success("✅ 模型加载成功！")

        except Exception as e:
            st.error(f"加载模型时发生严重错误：{e}")
            st.error(f"尝试加载的路径是: {model_path}")
            # Add more debug info about the imported YOLO
            # 这部分逻辑在主要导入失败时不会执行，因为会 st.stop()
            # if 'YOLO' in globals():
            #    st.error(f"YOLO object type: {type(YOLO)}")
            #    st.error(f"YOLO module location: {YOLO.__module__}") # Where did YOLO come from?
            # else:
            #    st.error("YOLO class itself could not be imported.")
            st.stop()


# --- 创建选项卡 ---
# (代码无变化)
tab1, tab2 = st.tabs(["🔬 影像检测分析", "ℹ️ 关于系统"])

# --- 选项卡1: 影像检测分析 ---
# (代码无变化, 除了确保变量在使用前已定义)
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

    tmp_file_path = None # Initialize tmp_file_path
    if img_file_buffer is not None:
        if 'model' in st.session_state and st.session_state.model is not None:
            bytes_data = img_file_buffer.getvalue()
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(bytes_data)
                    tmp_file_path = tmp_file.name # Assign here

                start_time = time.time()
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
            st.error("模型未能成功加载，无法进行预测。请检查应用日志。")

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

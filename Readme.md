# X-Medical Streamlit 应用运行指南

本项目含一个使用 Streamlit 构建的 X-Medical 深度学习视觉模型的应用界面。

## 先决条件
在运行此应用之前，请确保你已经安装了：
1.  **Python**: (建议 3.8 或更高版本)
2.  **Conda**: 用于环境管理。
3.  **必要的 Python 包**:
    *   `streamlit`
    *   `ultralytics` (或其他 YOLO 实现库)
    *   应用可能需要的其他依赖项 (例如 `opencv-python`, `torch`, `torchvision` 等)。
    这些包应该安装在特定的 Conda 环境中（在本文档示例中，环境名为 `Pytorch`），具体详见requirements.txt。

## 环境设置 (如果尚未完成)

如果你还没有创建环境并安装依赖，可以按以下步骤操作：

1.  **创建 Conda 环境** (如果 `Pytorch` 环境不存在):
    ```bash
    conda create -n Pytorch python=3.9 # 选择合适的 Python 版本
    ```
2.  **激活环境**:
    ```bash
    conda activate Pytorch
    ```
3.  **安装依赖**:
    *   如果项目提供了 `requirements.txt` 文件:
        ```bash
        pip install -r requirements.txt
        ```
    *   否则，手动安装主要依赖:
        ```bash
        pip install streamlit ultralytics opencv-python torch torchvision
        ```
        (请根据实际需要调整包列表)

## 如何运行应用

1.  **打开终端**: 启动 Anaconda Prompt 或你的首选命令行工具。
2.  **激活 Conda 环境**:
    ```bash
    conda activate Pytorch
    ```
3.  **导航到应用脚本目录** (可选，如果不在该目录运行):
    ```bash
    cd E:\Desktop\yoloapp-main\ultralytics
    ```
4.  **使用 `streamlit run` 命令启动应用**:
    *   如果你**已在** `ultralytics` 目录下:
        ```bash
        streamlit run app.py
        ```
    *   如果你**不在** `ultralytics` 目录下 (例如在 `yoloapp-main` 目录):
        ```bash
        streamlit run E:\Desktop\yoloapp-main\ultralytics\app.py
        # 或者相对路径
        # streamlit run ultralytics/app.py
        ```

## 预期行为

*   终端会显示正在运行的应用的本地 URL (通常是 `http://localhost:8501`) 和网络 URL。
*   你的默认网页浏览器会自动打开并加载该应用的界面。如果没有自动打开，请手动将终端显示的 URL 复制到浏览器中访问。
*   你可以通过 Web 界面与 YOLO 应用进行交互（例如上传图片/视频进行检测）。

## 故障排除

*   **`ModuleNotFoundError`**: 确保你激活了正确的 Conda 环境 (`Pytorch`)，并且所有必需的库都已在该环境中安装。
*   **`streamlit: command not found`**: 确认 `streamlit` 已经安装在激活的环境中 (`pip install streamlit`)，或者检查你的系统 PATH 设置。
*   **应用无法加载或显示错误**: 检查终端输出中是否有其他 Python 错误信息，并根据错误信息调试 `app.py` 脚本。

## 应用体验

## 应用体验

* 该应用现已通过Github与Streamlit形成了可实时访问的Web应用，欢迎医学人士访问：[X-Medical Web Application](https://x-medical.streamlit.app/)

![X-Medical Web Interface](链接到你的截图URL)



@echo off
chcp 65001 >nul
echo ================================================================
echo AI转岗学习路线 - GitHub项目批量下载脚本
echo ================================================================
echo.
echo 此脚本将下载所有推荐项目到指定目录
echo 请确保已安装 git (https://git-scm.com/downloads)
echo.
pause

set BASE_DIR=C:\ai_learning_projects

echo.
echo ================================================================
echo 开始下载项目到 %BASE_DIR%
echo ================================================================

:: 数学与Python基础
echo.
echo [1/25] 下载 numpy-100-exercises ...
mkdir "%BASE_DIR%\01-math-python" 2>nul
cd /d "%BASE_DIR%\01-math-python"
git clone --depth 1 https://github.com/rougier/numpy-100.git

echo.
echo [2/25] 下载 practical-python-data-science ...
git clone --depth 1 https://github.com/jonkrohn/practical-python-data-science.git

echo.
echo [3/25] 下载 PythonDataScienceHandbook ...
git clone --depth 1 https://github.com/jakevdp/PythonDataScienceHandbook.git

echo.
echo [4/25] 下载 ml-from-scratch ...
git clone --depth 1 https://github.com/eriklindernoren/ML-From-Scratch.git

echo.
echo [5/25] 下载 python-algorithms ...
git clone --depth 1 https://github.com/keon/algorithms.git

:: 机器学习基础
echo.
echo [6/25] 下载 scikit-learn ...
mkdir "%BASE_DIR%\02-machine-learning" 2>nul
cd /d "%BASE_DIR%\02-machine-learning"
git clone --depth 1 https://github.com/scikit-learn/scikit-learn.git

echo.
echo [7/25] 下载 xgboost ...
git clone --depth 1 https://github.com/dmlc/xgboost.git

echo.
echo [8/25] 下载 feature-engine ...
git clone --depth 1 https://github.com/feature-engine/feature-engine.git

echo.
echo [9/25] 下载 interpretable-ml-book ...
git clone --depth 1 https://github.com/christophM/interpretable-ml-book.git

echo.
echo [10/25] 下载 awesome-machine-learning ...
git clone --depth 1 https://github.com/josephmisiti/awesome-machine-learning.git

:: 深度学习与PyTorch
echo.
echo [11/25] 下载 annotated-transformer ...
mkdir "%BASE_DIR%\03-deep-learning" 2>nul
cd /d "%BASE_DIR%\03-deep-learning"
git clone --depth 1 https://github.com/harvardnlp/annotated-transformer.git

echo.
echo [12/25] 下载 pytorch-examples ...
git clone --depth 1 https://github.com/pytorch/examples.git

echo.
echo [13/25] 下载 vit-pytorch ...
git clone --depth 1 https://github.com/lucidrains/vit-pytorch.git

echo.
echo [14/25] 下载 onnx-tutorials ...
git clone --depth 1 https://github.com/onnx/tutorials.git

echo.
echo [15/25] 下载 pytorch-lightning ...
git clone --depth 1 https://github.com/Lightning-AI/pytorch-lightning.git

:: Android端侧AI
echo.
echo [16/25] 下载 tensorflow-examples (TFLite Android) ...
mkdir "%BASE_DIR%\04-android-ai" 2>nul
cd /d "%BASE_DIR%\04-android-ai"
git clone --depth 1 https://github.com/tensorflow/examples.git

echo.
echo [17/25] 下载 mlkit-quickstarts ...
git clone --depth 1 https://github.com/googlesamples/mlkit.git

echo.
echo [18/25] 下载 ncnn-android-demo ...
git clone --depth 1 https://github.com/nihui/ncnn-android-yolov5.git

echo.
echo [19/25] 下载 mnn-android-demo ...
git clone --depth 1 https://github.com/alibaba/MNN.git

echo.
echo [20/25] 下载 paddleocr-android ...
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git

:: Java后端AI服务
echo.
echo [21/25] 下载 djl-demo ...
mkdir "%BASE_DIR%\05-java-ai-service" 2>nul
cd /d "%BASE_DIR%\05-java-ai-service"
git clone --depth 1 https://github.com/deepjavalibrary/djl-demo.git

echo.
echo [22/25] 下载 langchain4j ...
git clone --depth 1 https://github.com/langchain4j/langchain4j.git

echo.
echo [23/25] 下载 onnxruntime ...
git clone --depth 1 https://github.com/microsoft/onnxruntime.git

:: 大模型应用
echo.
echo [24/25] 下载 langchain ...
mkdir "%BASE_DIR%\07-llm-apps" 2>nul
cd /d "%BASE_DIR%\07-llm-apps"
git clone --depth 1 https://github.com/langchain-ai/langchain.git

echo.
echo [25/25] 下载 vllm ...
git clone --depth 1 https://github.com/vllm-project/vllm.git

echo.
echo ================================================================
echo 所有项目下载完成！
echo ================================================================
echo.
echo 项目目录结构：
echo   %BASE_DIR%\01-math-python\       - 数学与Python基础
echo   %BASE_DIR%\02-machine-learning\   - 机器学习基础
echo   %BASE_DIR%\03-deep-learning\      - 深度学习与PyTorch
echo   %BASE_DIR%\04-android-ai\         - Android端侧AI
echo   %BASE_DIR%\05-java-ai-service\    - Java后端AI服务
echo   %BASE_DIR%\07-llm-apps\           - 大模型应用
echo.
echo 请打开各目录查看README.md开始学习
echo.
pause
# =================================================================
#     CMakeLists.txt (V20.0 - 终极完整版)
# =================================================================
cmake_minimum_required(VERSION 3.12)
project(cpp_python_bridge)

# --- 路径定义 ---
# 【【【 修改这里 】】】
set(LLAMA_CPP_SRC_DIR /home/weiyubin/llama.cpp_new) 
# 【【【 修改这里 】】】
set(LLAMA_CPP_BUILD_DIR /home/weiyubin/llama.cpp_new/build-vulkan-official/bin)

# --- 查找核心的 libllama.so 库 ---
find_library(LLAMA_LIBRARY NAMES llama HINTS ${LLAMA_CPP_BUILD_DIR})
if(NOT LLAMA_LIBRARY)
    message(FATAL_ERROR "Could not find libllama.so in ${LLAMA_CPP_BUILD_DIR}")
endif()

# --- pybind11 ---
add_subdirectory(pybind11)

# --- pybind11 模块定义 ---
# 【【【 终极核心修正：将所有必需的源文件全部编译进来！ 】】】
pybind11_add_module(my_bridge
    # 1. 我们的桥接代码
    src/bridge.cpp
    # 2. 提供 string_format 等辅助函数的核心文件
    ${LLAMA_CPP_SRC_DIR}/common/common.cpp
    # 3. 为 common.cpp 提供依赖的 "假" 编译信息文件
    src/dummy_build_info.cpp
    # 4. 为保险起见，包含日志和控制台，以防万一
    ${LLAMA_CPP_SRC_DIR}/common/console.cpp
    ${LLAMA_CPP_SRC_DIR}/common/log.cpp
)

# --- 包含所有必要的头文件 ---
target_include_directories(my_bridge PRIVATE
    "${LLAMA_CPP_SRC_DIR}/include"
    "${LLAMA_CPP_SRC_DIR}/common"
    "${LLAMA_CPP_SRC_DIR}/ggml/include"
)

# --- 链接核心的 libllama.so ---
target_link_libraries(my_bridge PRIVATE
    ${LLAMA_LIBRARY}
)

# --- 设置 RPATH，确保运行时能找到 libllama.so ---
target_link_options(my_bridge PRIVATE "-Wl,-rpath,${LLAMA_CPP_BUILD_DIR}")

# --- 设置 C++ 标准 ---
set_target_properties(my_bridge PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
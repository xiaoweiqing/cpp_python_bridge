// ==========================================================
//     src/dummy_build_info.cpp (最终版)
// ==========================================================
// 这个文件的唯一目的，是为 common.cpp 提供它所依赖的
// 全局编译信息变量，解决链接时 "undefined symbol" 的问题。
// ==========================================================

extern "C"
{
    int LLAMA_BUILD_NUMBER = 1234;                       // 可以是任何数字
    const char *LLAMA_COMMIT = "final_build";            // 可以是任何字符串
    const char *LLAMA_COMPILER = "gcc";                  // 可以是任何字符串
    const char *LLAMA_BUILD_TARGET = "x86_64-linux-gnu"; // 可以是任何字符串
}
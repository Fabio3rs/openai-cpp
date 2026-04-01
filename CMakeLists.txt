cmake_minimum_required(VERSION 3.14)
project(openai-cpp LANGUAGES CXX)

option(OPENAI_CPP_BUILD_EXAMPLES "Build OpenAI C++ examples" OFF)

add_library(openai-cpp INTERFACE)
add_library(openai::openai-cpp ALIAS openai-cpp)

find_package(CURL REQUIRED)
target_link_libraries(openai-cpp INTERFACE CURL::libcurl)

target_compile_features(openai-cpp INTERFACE cxx_std_17)

target_include_directories(
    openai-cpp
    SYSTEM INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(OPENAI_CPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

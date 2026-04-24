project(tdl_python)

# Set RTSP paths directly (mirrors what src/components/encoder/rtsp/CMakeLists.txt exports
# via PARENT_SCOPE, but those variables are not visible here due to CMake scoping).
if ("${CVI_PLATFORM}" STREQUAL "CV181X" OR "${CVI_PLATFORM}" STREQUAL "CV184X")
    # TOP_DIR is the parent of REPO_DIR (set in the top-level CMakeLists.txt)
    set(_CVI_RTSP_INSTALL ${TOP_DIR}/cvi_rtsp/install)
    set(_RTSP_INCLUDE_LOCAL  ${_CVI_RTSP_INSTALL}/include/cvi_rtsp)
    set(_RTSP_LIBS_STATIC_LOCAL
        ${MLIR_SDK_ROOT}/lib/libBasicUsageEnvironment.a
        ${MLIR_SDK_ROOT}/lib/libgroupsock.a
        ${MLIR_SDK_ROOT}/lib/libUsageEnvironment.a
        ${MLIR_SDK_ROOT}/lib/libliveMedia.a
        ${_CVI_RTSP_INSTALL}/lib/libcvi_rtsp.a)
    message(STATUS "RTSP include: ${_RTSP_INCLUDE_LOCAL}")
    message(STATUS "RTSP libs: ${_RTSP_LIBS_STATIC_LOCAL}")

    # Look for the buildroot-compiled OpenCV that includes videoio.
    # The SDK's internal OpenCV (_deps/opencv-src) only has core/imgproc/imgcodecs.
    # MIDDLEWARE_SDK_ROOT = <TOP_DIR>/cvi_mpi, so ../../buildroot is TOP_DIR/buildroot.
    file(GLOB _BR_OPENCV_INC_CANDIDATES
        "${MIDDLEWARE_SDK_ROOT}/../buildroot/output/*/per-package/opencv4/host/*/sysroot/usr/include/opencv4"
    )
    if (_BR_OPENCV_INC_CANDIDATES)
        list(GET _BR_OPENCV_INC_CANDIDATES 0 _OPENCV_VIDEOIO_INCLUDE)
        message(STATUS "Found buildroot OpenCV (with videoio): ${_OPENCV_VIDEOIO_INCLUDE}")
        set(HAVE_OPENCV_VIDEOIO ON)
    else()
        message(STATUS "Buildroot OpenCV not found — RtspClient (software decode) will be excluded")
        set(HAVE_OPENCV_VIDEOIO OFF)
    endif()

    # Library path for the full OpenCV (installed on the target device)
    file(GLOB _BR_OPENCV_LIB_CANDIDATES
        "${MIDDLEWARE_SDK_ROOT}/../buildroot/output/*/per-package/opencv4/host/*/sysroot/usr/lib/libopencv_videoio.so"
    )
    if (_BR_OPENCV_LIB_CANDIDATES)
        list(GET _BR_OPENCV_LIB_CANDIDATES 0 _OPENCV_VIDEOIO_LIB)
    else()
        set(_OPENCV_VIDEOIO_LIB "")
    endif()
endif()

include_directories(
                    ${REPO_DIR}/include/framework
                    ${REPO_DIR}/include/nn
                    ${REPO_DIR}/include/components/matcher
                    ${REPO_DIR}/include/components/tracker
                    ${REPO_DIR}/src/components/llm/qwen
                    ${REPO_DIR}/src/components/llm/qwen2VL
                    ${REPO_DIR}/include/
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${REPO_DIR}/dependency/${CVI_PLATFORM}/python_depends/include/python
                    ${REPO_DIR}/dependency/${CVI_PLATFORM}/python_depends/include/pybind11
                    )

if ("${CVI_PLATFORM}" STREQUAL "CV181X" OR "${CVI_PLATFORM}" STREQUAL "CV184X")
    include_directories(${_RTSP_INCLUDE_LOCAL}
                        # live555 headers (each module dir exposes bare includes)
                        ${MLIR_SDK_ROOT}/include
                        ${MLIR_SDK_ROOT}/include/liveMedia
                        ${MLIR_SDK_ROOT}/include/BasicUsageEnvironment
                        ${MLIR_SDK_ROOT}/include/UsageEnvironment
                        ${MLIR_SDK_ROOT}/include/groupsock
                        ${MIDDLEWARE_INCLUDES}
                        ${REPO_DIR}/include/components)
    if (HAVE_OPENCV_VIDEOIO)
        include_directories(${_OPENCV_VIDEOIO_INCLUDE})
    endif()
    link_directories(${MLIR_LIBS} ${MIDDLEWARE_LIBS})
endif()

file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# If the full OpenCV (with videoio) is not available, exclude RtspClient (software decode).
# RtspClientVdec (hardware decode via live555 + VDEC) is unaffected.
if ("${CVI_PLATFORM}" STREQUAL "CV181X" OR "${CVI_PLATFORM}" STREQUAL "CV184X")
    if (NOT HAVE_OPENCV_VIDEOIO)
        list(FILTER SRC_FRAMWORK_FILES_CUR EXCLUDE REGEX "/py_rtsp_client\\.cpp$")
        list(FILTER SRC_FRAMWORK_FILES_CUR EXCLUDE REGEX "/py_usb_camera\\.cpp$")
        message(STATUS "py_rtsp_client.cpp and py_usb_camera.cpp excluded (no OpenCV videoio)")
    endif()
endif()

# Include rtsp.cpp, video_recorder.cpp and image_encoder.cpp directly so the
# py_rtsp / py_video_recorder wrappers can link against them.
if ("${CVI_PLATFORM}" STREQUAL "CV181X" OR "${CVI_PLATFORM}" STREQUAL "CV184X")
    list(APPEND SRC_FRAMWORK_FILES_CUR
        ${REPO_DIR}/src/components/encoder/rtsp/rtsp.cpp
        ${REPO_DIR}/src/components/encoder/video_recorder/video_recorder.cpp
        ${REPO_DIR}/src/components/encoder/image_encoder/image_encoder.cpp)
endif()

# 添加tdl_image模块的源码
add_library(${PROJECT_NAME} MODULE
    ${SRC_FRAMWORK_FILES_CUR}
)


file(GLOB_RECURSE LIB_PYTHON_FILES ${REPO_DIR}/dependency/${CVI_PLATFORM}/python_depends/lib/*.so)
foreach(PYTHON_LIB ${LIB_PYTHON_FILES})
    if(PYTHON_LIB MATCHES "libpython([0-9]+\\.[0-9]+)\\.so")
        set(PYTHON_VERSION_SUFFIX "python${CMAKE_MATCH_1}")
        break()
    endif()
endforeach()
message(STATUS "LIB_PYTHON_FILES: ${LIB_PYTHON_FILES}")
target_link_libraries(${PROJECT_NAME} PRIVATE
    # pybind11::module
    ${LIB_PYTHON_FILES}
    tdl_core
    tdl_ex
)

if ("${CVI_PLATFORM}" STREQUAL "CV181X" OR "${CVI_PLATFORM}" STREQUAL "CV184X")
    target_link_libraries(${PROJECT_NAME} PRIVATE
        "-Wl,--start-group"
        ${_RTSP_LIBS_STATIC_LOCAL}
        "-Wl,--end-group"
        # VDEC hardware decoder (used by RtspClientVdec)
        ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
        # libavformat / libavcodec / libavutil (MP4 muxing in VideoRecorder).
        # Shipped with the TPU SDK next to libvdec.
        ${MLIR_SDK_ROOT}/lib/libavformat.so
        ${MLIR_SDK_ROOT}/lib/libavcodec.so
        ${MLIR_SDK_ROOT}/lib/libavutil.so
    )
    if (HAVE_OPENCV_VIDEOIO)
        target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_OPENCV_VIDEOIO=1)
        if (_OPENCV_VIDEOIO_LIB)
            target_link_libraries(${PROJECT_NAME} PRIVATE ${_OPENCV_VIDEOIO_LIB})
        endif()
    endif()
endif()

# 设置目标文件名
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    OUTPUT_NAME "tdl"
)

# 设置编译优化参数（可选）
target_compile_options(${PROJECT_NAME} PRIVATE -O3 -fPIC)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)

# RMC_gui — Dear ImGui desktop front-end. Only configured when RMC_BUILD_GUI=ON,
# so a normal build never reaches this file and never fetches GUI dependencies.
include(FetchContent)

# Build the entire GUI subtree WITHOUT LTO, overriding the project-wide
# ENABLE_LTO. glad2 emits its loader as a NATIVE static archive (it force-
# disables IPO on its own target), but with project LTO on, RMC_gui's objects
# are (Thin)LTO bitcode. Mixing a bitcode consumer with glad's native archive
# makes lld drop the glad_gl* loader symbols at final resolution — even with
# --whole-archive forcing the members in — so every GL call is undefined. The
# GUI is not perf-critical, so taking this scope out of LTO removes the
# native/bitcode mismatch at the source. add_subdirectory gives this its own
# scope, so the override stays local to the GUI targets.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

find_package(OpenGL REQUIRED)

# ---- GLFW: prefer a system package, fall back to FetchContent ----
find_package(glfw3 3.3 QUIET)
if (NOT glfw3_FOUND)
    message(STATUS "RMC_gui: glfw3 not found — fetching via FetchContent")
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG 3.4)
    FetchContent_MakeAvailable(glfw)
endif ()

# ---- glad2: OpenGL 3.3 core loader (provides target `glad_gl_core`) ----
FetchContent_Declare(glad
        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
        GIT_TAG v2.0.6
        SOURCE_SUBDIR cmake)
FetchContent_MakeAvailable(glad)
glad_add_library(glad_gl_core STATIC API gl:core=3.3)

# ---- Dear ImGui (docking). No CMakeLists upstream → build it here.
#      Pinned to a docking RELEASE tag, not the moving `docking` branch: ImGui's
#      1.92 font rework dropped IM_OFFSETOF/IM_FLOOR and changed ImFont, which
#      breaks ImPlot 0.16. 1.90.9 is the version ImPlot 0.16 targets. ----
FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.90.9-docking)
FetchContent_MakeAvailable(imgui)
add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
        ${imgui_SOURCE_DIR}/misc/cpp
)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

# ---- ImPlot (optional convergence chart) ----
option(RMC_GUI_USE_IMPLOT "Build the ImPlot convergence chart" ON)
if (RMC_GUI_USE_IMPLOT)
    FetchContent_Declare(implot
            GIT_REPOSITORY https://github.com/epezent/implot.git
            GIT_TAG v0.16)
    FetchContent_MakeAvailable(implot)
    add_library(implot STATIC
            ${implot_SOURCE_DIR}/implot.cpp
            ${implot_SOURCE_DIR}/implot_items.cpp
    )
    target_include_directories(implot PUBLIC ${implot_SOURCE_DIR})
    target_link_libraries(implot PUBLIC imgui)
endif ()

# ---- Silence third-party warnings (the project builds RMC with -Wfatal-errors;
#      these targets are vendored, so don't let their warnings be fatal). ----
set(RMC_GUI_TP_TARGETS imgui glad_gl_core)
if (RMC_GUI_USE_IMPLOT)
    list(APPEND RMC_GUI_TP_TARGETS implot)
endif ()
if (NOT glfw3_FOUND)
    list(APPEND RMC_GUI_TP_TARGETS glfw)
endif ()
foreach (tgt ${RMC_GUI_TP_TARGETS})
    if (TARGET ${tgt})
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${tgt} PRIVATE -w)
        endif ()
        set_target_properties(${tgt} PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    endif ()
endforeach ()

# ---- The executable ----
add_executable(RMC_gui
        main.cpp
        App.cpp
        SimController.cpp
        StructureView.cpp
        Camera.cpp
        LogSink.cpp
        Panels.cpp
)
target_include_directories(RMC_gui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(RMC_gui PRIVATE
        RMC
        imgui
        glad_gl_core
        glfw
        OpenGL::GL
        Threads::Threads
)
if (RMC_GUI_USE_IMPLOT)
    target_link_libraries(RMC_gui PRIVATE implot)
    target_compile_definitions(RMC_gui PRIVATE RMC_GUI_USE_IMPLOT)
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # RMC_gui is our code → keep warnings; LEAF macros expand to GNU
    # statement-expressions, silence that one on Clang.
    target_compile_options(RMC_gui PRIVATE -Wall -Wextra
            -Wno-gnu-statement-expression-from-macro-expansion)
endif ()

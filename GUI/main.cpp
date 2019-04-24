#include <algorithm>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)

#include <GL/gl3w.h>    // Initialize with gl3wInit()

#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h>    // Initialize with glewInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h>  // Initialize with gladLoadGL()
#else
#include IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#endif

#include <GLFW/glfw3.h>

#include "util.h"

using namespace std;

namespace {
    void glfw_error_callback(int error, const char *description) {
        cerr << "Glfw Error" << error << ": " << description << endl;
    }

    GLFWwindow *setup_UI() {
        // Setup window
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            return nullptr;

        // Decide GL+GLSL versions
#if __APPLE__
        // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
        // GL 3.0 + GLSL 130
        const char *glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
        //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

        // Create window with graphics context
        GLFWwindow *window = glfwCreateWindow(1280, 720, "YOLO+SORT+ImGui", NULL, NULL);
        if (!window)
            return nullptr;
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1); // Enable vsync

        // Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
        bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
        bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
    bool err = gladLoadGL() == 0;
#else
    bool err = false; // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader is likely to requires some form of initialization.
#endif
        if (err) {
            fprintf(stderr, "Failed to initialize OpenGL loader!\n");
            return nullptr;
        }

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();

        // Setup Platform/Renderer bindings
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        return window;
    }
}

int main() {
    cv::VideoCapture cap("result/compressed.flv");
    if (!cap.isOpened()) {
        cerr << "Cannot open the video" << endl;
        return -2;
    }

    auto video_FPS = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    TargetRepo repo;

    auto image = cv::Mat(int(cap.get(cv::CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                         CV_8UC3, {0, 0, 0});

    auto window = setup_UI();
    if (!window) {
        cerr << "GUI failed" << endl;
        return -1;
    }

    array<GLuint, 3> texture;
    glGenTextures(texture.size(), texture.data());

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        static auto show_demo_window = false;
        static auto show_res_window = true;
        static auto show_target_window = true;
        static auto playing = false;

        static uint32_t processed_frame = 0;

        bool next;
        draw_control_window(cap, processed_frame, show_demo_window, show_res_window, show_target_window,
                            playing, next);

        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        int hovered = -1;
        if (show_target_window) {
            auto[h, rewind] = draw_target_window(repo, video_FPS, &show_target_window);
            hovered = h;
            if (rewind != -1) {
                cap.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(rewind));
                next = true;
            }
        }

        static auto prev = chrono::steady_clock::now();
        if (auto elapsed = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - prev);
                elapsed.count() > 1000 / video_FPS && (playing || next) && cap.grab()) {
            prev += elapsed;
            cap.retrieve(image);
        }

        processed_frame = repo.load() + 1;

        if (show_res_window)
            draw_res_window(image, repo, static_cast<uint32_t>(cap.get(cv::CAP_PROP_POS_FRAMES)) - 1, hovered,
                            texture[2], &show_res_window);

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwMakeContextCurrent(window);
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwMakeContextCurrent(window);
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
#include <chrono>
#include <filesystem>

#include "Windows.h"
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "CMultiCap.h"
#include "stb_image_write.h"


ABSL_FLAG(float, exposure, 10000.f, "Exposure time");
ABSL_FLAG(float, gain, -1.f, "Gain");
ABSL_FLAG(std::string, output_dir, ".", "Output directory.");
ABSL_FLAG(bool, nosave, false, "Do not save the images.");
ABSL_FLAG(uint, fps, 25, "Framerate for capturing.");

int N_CAP = 2;

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    auto output_dir_str = absl::GetFlag(FLAGS_output_dir);
    auto nosave = absl::GetFlag(FLAGS_nosave);
    auto exposure = absl::GetFlag(FLAGS_exposure);
    auto gain = absl::GetFlag(FLAGS_gain);
    auto fps = absl::GetFlag(FLAGS_fps);

    auto capture_duration = 0; // in us
    if (fps != 0) {
        capture_duration = 1000000 / fps;
    }

    auto output_path = std::filesystem::path(output_dir_str);
    if (!std::filesystem::is_directory(output_path)) {
        std::cerr << "Output directory must be a directory." << std::endl;
        exit(-1);
    }
    if (output_path.is_relative()) {
        output_dir_str = std::filesystem::absolute(output_dir_str).string();
    }
    if (!nosave) {
        std::filesystem::create_directories(output_dir_str + "/V0");
        std::filesystem::create_directories(output_dir_str + "/V1");
    }

	get_app();
	init_device();

    if (get_device_count() < N_CAP) {
        std::cerr << "No enough device found." << std::endl;
        exit(-1);
    }

    for (int i = 0; i < N_CAP; ++i) {
        set_exposure(i, exposure);
        set_gain(i, gain);
    }
    std::cout << std::endl;
    if (!nosave) {
        std::cout << "Captured images will be saved to: " << output_dir_str << std::endl;
    }
    std::cout << "Setting exposure time to: " << exposure << "us" << std::endl;
    std::cout << "Setting gain to: " << gain << "\n" << std::endl;
    Sleep(3000);

    start_grabbing();

	for (int fx = 0;; ++fx) {
        auto start_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

		auto cap_info = capture();
		assert(cap_info.n_cap == N_CAP);
		
        for (int i = 0; i < N_CAP; ++i) {
            if (!cap_info.flag[i]) {
                continue;
            }
            cv::Mat img_rgb(cap_info.height[i], cap_info.width[i], CV_8UC3, cap_info.ppbuffer[i]), img_bgr;
            cv::cvtColor(img_rgb, img_bgr, cv::COLOR_RGB2BGR);
            cv::Mat img_demo;
            cv::resize(img_bgr, img_demo, cv::Size(0, 0), 0.5, 0.5);
            cv::imshow(std::format("{}", i), img_demo);

            if (!nosave) {
                std::string path = std::format(R"({}/V{}/{:06}.bmp)", output_dir_str, i, fx);
                stbi_write_bmp(path.c_str(), cap_info.width[0], cap_info.height[0], 3, cap_info.ppbuffer[i]);
            }
        }
        if (cv::waitKey(1) == 'q') {
            break;
        }

        auto end_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        auto cost_time = end_time - start_time;
        if (cost_time > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(cost_time));
        }
	}

    stop_grabbing();

	close_device();
	release_app();
	return 0;
}
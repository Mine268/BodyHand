#include <iostream>
#include <format>
#include "SyncMultiCamera.h"


int main() {
	// 间隔至少 10ms，不自动输出图片到文件
	SyncMultiCamera* cam_app = SyncMultiCamera::get(10, false);

	// 默认流程
	cam_app->enum_device();
	cam_app->open_device();

	int n_view = cam_app->get_cam_count();
	// 进行默认的参数设置
	for (int i = 0; i < n_view; ++i) {
		cam_app->set_exposure_time(i, 10000);
		cam_app->set_gain(i, 10.0);
	}

	// 必须手动开始捕捉
	cam_app->start_grabbing();
	/*
	* flag_cap：指示对所有相机进行的软触发是否成功了
	* buffer_size：等于相机的数量，即等于 n_view
	* p_buffer：指向 buffer 信息数组
	*/
	auto [flag_cap, buffer_size, p_buffer] = cam_app->capture();
	if (flag_cap) {
		for (int i = 0; i < buffer_size; ++i) {
			const BufferInfo &bufferInfo = p_buffer[i];
			std::cout << "Camera #" << i << std::endl;
			std::cout << "    Frame ix: " << bufferInfo.i_frame_ix << std::endl;
			std::cout << "    Capture flag: " << bufferInfo.b_success << std::endl;
			std::cout << "    Image size : H=" << bufferInfo.i_height << ",W=" << bufferInfo.i_width << std::endl;
		}
	}
	else {
		std::cerr << "SoftTrigger failed" << std::endl;
	}

	// 默认流程
	cam_app->stop_grabbing();
	cam_app->close_device();

	return 0;
}

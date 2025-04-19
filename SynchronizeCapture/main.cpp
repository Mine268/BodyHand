#include <iostream>
#include <format>
#include "SyncMultiCamera.h"


int main() {
	// ������� 10ms�����Զ����ͼƬ���ļ�
	SyncMultiCamera* cam_app = SyncMultiCamera::get(10, false);

	// Ĭ������
	cam_app->enum_device();
	cam_app->open_device();

	int n_view = cam_app->get_cam_count();
	// ����Ĭ�ϵĲ�������
	for (int i = 0; i < n_view; ++i) {
		cam_app->set_exposure_time(i, 10000);
		cam_app->set_gain(i, 10.0);
	}

	// �����ֶ���ʼ��׽
	cam_app->start_grabbing();
	/*
	* flag_cap��ָʾ������������е������Ƿ�ɹ���
	* buffer_size����������������������� n_view
	* p_buffer��ָ�� buffer ��Ϣ����
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

	// Ĭ������
	cam_app->stop_grabbing();
	cam_app->close_device();

	return 0;
}

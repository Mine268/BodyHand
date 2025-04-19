#include <iostream>
#include <format>
#include <chrono>

#include <Windows.h>
#include <process.h> // ʹ��ϵͳ�Ķ��߳̿�

#include "stb_image_write.h"
#include "SyncMultiCamera.h"


SyncMultiCamera* SyncMultiCamera::p_singleton = nullptr;

SyncMultiCamera* SyncMultiCamera::get(unsigned int i_ts, bool b_si) {
	if (SyncMultiCamera::p_singleton == nullptr) {
		SyncMultiCamera::p_singleton = new SyncMultiCamera;
	}
	SyncMultiCamera::p_singleton->i_time_span = i_ts;
	SyncMultiCamera::p_singleton->b_save_image = b_si;
	return SyncMultiCamera::p_singleton;
}

void SyncMultiCamera::destroy() {
	if (SyncMultiCamera::p_singleton != nullptr) {
		delete SyncMultiCamera::p_singleton;
		SyncMultiCamera::p_singleton = nullptr;
	}
}

void SyncMultiCamera::set_save_image(bool b_si) {
	b_save_image = b_si;
}

int SyncMultiCamera::enum_device() {
	// ��ȡ�����豸����Ϣ
	auto nRet = CMvCamera::EnumDevices(MV_USB_DEVICE, &this->l_device_info);
	// �����ȡʧ���򵯳�����
	if (MV_OK != nRet) {
		std::cerr << "error: Fail to enumerate the device!" << std::endl;
	}
	return nRet;
}

int SyncMultiCamera::open_device() {
	// ���豸�Ѿ��򿪣����ٴ�
	//if (b_open_device) {
	//	std::cerr << "error: Devices are opened!" << std::endl;
	//	return -1;
	//}

	// ��ʼ���豸
	for (unsigned int e_ix = 0;
			e_ix < this->l_device_info.nDeviceNum; ++e_ix) {
		CMvCamera* p_cam_tmp = new CMvCamera;
		auto nRet1 = p_cam_tmp->Open(l_device_info.pDeviceInfo[e_ix]); // �򿪶�Ӧ���豸
		auto nRet2 = p_cam_tmp->SetEnumValue("TriggerMode", 1); // ���ø����������ģʽΪ����ģʽ
		auto nRet3 = p_cam_tmp->SetEnumValue("TriggerSource", 7); // ���ô���ԴΪ����
		auto nRet4 = p_cam_tmp->SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed); // �������Ϊ RGB 8 ��ʽ
		if (nRet1 != MV_OK) {
			std::cerr << "error: Can't open the device #" << e_ix << std::endl;
			delete p_cam_tmp;
			continue;
		}
		if (nRet2 != MV_OK) {
			std::cerr << "error: Can't set trigger mode for device "
				<< p_cam_tmp->m_hDevHandle << std::endl;
			delete p_cam_tmp;
			continue;
		}
		if (nRet3 != MV_OK) {
			std::cerr << "error: Can't set software trigger for device "
				<< p_cam_tmp->m_hDevHandle << std::endl;
			delete p_cam_tmp;
			continue;
		}
		lp_mvcamera.push_back(p_cam_tmp);
		l_buffer_info[lp_mvcamera.size() - 1].i_width = 0;
		l_buffer_info[lp_mvcamera.size() - 1].i_height = 0;
		l_buffer_info[lp_mvcamera.size() - 1].i_frame_ix = 0;
		l_buffer_info[lp_mvcamera.size() - 1].p_buffer = nullptr;
	}

	if (lp_mvcamera.size() == 0) {
		std::cerr << "error: No device opened!" << std::endl;
		return -1;
	}
	return MV_OK;
}

int SyncMultiCamera::close_device() {
	// �������رգ��ͷ�ָ��ռ䲢��Ϊ nullptr������ͷ�����
	int nRet = MV_OK;
	for (unsigned int i = 0; i < lp_mvcamera.size(); ++i) {
		auto p_cam = lp_mvcamera[i];
		nRet = (nRet == MV_OK && p_cam->Close() == MV_OK) ? MV_OK : -1;
		delete p_cam;
		p_cam = nullptr;
	}
	lp_mvcamera.clear();
	return nRet;
}

void SyncMultiCamera::set_time_span(unsigned int t) {
	this->i_time_span = t;
}

int SyncMultiCamera::set_exposure_time(unsigned int d_ix, float et) {
	if (d_ix >= lp_mvcamera.size()) {
		std::cerr << "error: No device connected, can't change exposure" << std::endl;
		return -1;
	}

	lp_mvcamera[d_ix]->SetEnumValue("ExposureMode", 0);
	auto nRet = lp_mvcamera[d_ix]->SetFloatValue("ExposureTime", et);
	if (nRet != MV_OK) {
		std::cerr << "error: Can't set exposure time for device "
			<< lp_mvcamera[d_ix]->m_hDevHandle << std::endl;
	}
	return nRet;
}

int SyncMultiCamera::set_gain(unsigned int d_ix, float g) {
	if (d_ix >= lp_mvcamera.size()) {
		std::cerr << "error: No device connected, can't change gain" << std::endl;
		return -1;
	}

	lp_mvcamera[d_ix]->SetEnumValue("GainAuto", 1);
	auto nRet = lp_mvcamera[d_ix]->SetFloatValue("Gain", g);
	if (nRet != MV_OK) {
		std::cerr << "error: Can't set gain for device "
			<< lp_mvcamera[d_ix]->m_hDevHandle << std::endl;
	}
	return nRet;
}

int SyncMultiCamera::get_cam_count() const {
	return lp_mvcamera.size();
}

int SyncMultiCamera::start_grabbing() {
	if (lp_mvcamera.size() == 0) {
		std::cerr << "error: No device connected, no device starts grabbing" << std::endl;
		return -1;
	}

	auto all_fine = true;

	for (unsigned int i = 0; i < lp_mvcamera.size(); ++i) {
		auto* p_cam = lp_mvcamera[i];
		// ����ÿ������� grabbing
		auto nRet = p_cam->StartGrabbing();
		if (nRet != MV_OK) {
			std::cerr << "error: Can't start grabbing for device "
				<< p_cam->m_hDevHandle << std::endl;
			all_fine = false;
		}

		// ���߳̽ӹ�
		// ���촫�ݶ��߳���Ϣ�Ľṹ�壬ע��ṹ����� WorkThread ���ͷţ�
		auto p_sync_ctrl = new SyncControlData{ this, i };
		unsigned int i_thread_ix{ 0 };
		auto h_thread =
			(void*) _beginthreadex(NULL, 0, WorkThread, p_sync_ctrl, 0, &i_thread_ix);
		if (h_thread == NULL) {
			std::cerr << "error: Can't create thread" << std::endl;
			all_fine = false;
		}
		else {
			l_thread.emplace_back(i_thread_ix, h_thread);
		}

		b_start_grabbing = true;
	}

	h_sem_agg = CreateSemaphore(NULL, 0, static_cast<LONG>(lp_mvcamera.size()), "h_sem_agg");
	h_sem_continue = CreateSemaphore(NULL, 0, static_cast<LONG>(lp_mvcamera.size()), "h_sem_continue");

	return all_fine ? MV_OK : -1;
}

unsigned int __stdcall WorkThread(void* p_user) {
	auto p_sync_ctrl = (SyncControlData*)p_user;

	auto p_brain = p_sync_ctrl->p_brain;
	auto i_cam_ix = p_sync_ctrl->i_cam_ix;

	delete p_sync_ctrl; // �ֶ��ͷſռ䣬��֪������д�ܳ�

	p_brain->thread_func(i_cam_ix); // ��ʼ�ȴ������ɼ�

	return 0;
}

void SyncMultiCamera::thread_func(unsigned int i_cam_ix) {
	MVCC_INTVALUE_EX st_int = { 0 };
	auto p_cam = lp_mvcamera[i_cam_ix];

	// ��ȡ�������Ĵ�С
	auto nRet = p_cam->GetIntValue("PayloadSize", &st_int);
	if (nRet != MV_OK) {
		std::cerr << "error: Cant't get PayloadSize for device " 
			<< p_cam << std::endl;
		return;
	}

	// ��ȡ���ݴ�С
	auto i_data_size = static_cast<unsigned int>(st_int.nCurValue);
	// ���� buffer
	l_buffer_info[i_cam_ix].p_buffer = new unsigned char[sizeof(unsigned char) * i_data_size];
	if (l_buffer_info[i_cam_ix].p_buffer == nullptr) {
		std::cerr << "error: Cant't get buffer for device "
			<< p_cam << std::endl;
		return;
	}

	// ��ȡͼ������
	unsigned int i_image_ix{ 0 }, i_delta = 0;
	MV_FRAME_OUT_INFO_EX st_image_info{ 0 };
	while (b_start_grabbing) {
		auto nRet = p_cam->GetOneFrameTimeout( // Timeout ��Ҫ��̫�ͣ����޷���ʾ cv
			l_buffer_info[i_cam_ix].p_buffer, i_data_size, &st_image_info, 1000);
		if (nRet == MV_OK) {
			//// ����Ƿ�©֡
			//if (st_image_info.nFrameNum - i_image_ix - i_delta != 0) {
			//	std::cout << std::format("warning: device {} missed {} frame(s) before frame {}\n",
			//		(void*)p_cam->m_hDevHandle, st_image_info.nFrameNum - i_image_ix - i_delta, i_image_ix);
			//	i_delta = st_image_info.nFrameNum - i_image_ix;
			//}
			// ���� BufferInfo
			l_buffer_info[i_cam_ix].i_width = st_image_info.nWidth;
			l_buffer_info[i_cam_ix].i_height = st_image_info.nHeight;
			l_buffer_info[i_cam_ix].i_frame_ix = st_image_info.nFrameNum;
			l_buffer_info[i_cam_ix].b_success = true;
			// �ύ�� collector
			if (b_save_image) {
				// ����洢·��
				auto path = std::format("capture/C{}/I{:08}F{:08}.bmp", i_cam_ix, i_image_ix, st_image_info.nFrameNum);
				// �洢ͼ�񣬲���ʱ
				auto start = std::chrono::system_clock::now();
				auto ret = 
					stbi_write_bmp(path.c_str(), st_image_info.nWidth, st_image_info.nHeight, 3, l_buffer_info[i_cam_ix].p_buffer);
				auto end = std::chrono::system_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
				//// ���ͼ����Ϣ
				//std::cout << 
				//	std::format("[device {}] frame_idx:{:08}, image_idx:{:08}, delta:{:05}, height:{}, width:{}, time_cost:{}\n",
				//		(void*)p_cam->m_hDevHandle, st_image_info.nFrameNum, i_image_ix, i_delta, st_image_info.nHeight, st_image_info.nWidth, duration);
				//++i_image_ix;
			}
		}
		else {
			l_buffer_info[i_cam_ix].b_success = false;
		}
		// ֪ͨ���̱߳����ץȡ���
		ReleaseSemaphore(h_sem_agg, 1, NULL);
		// ˯��
		Sleep(i_time_span);
		// �ȴ����̴߳������
		WaitForSingleObject(h_sem_continue, INFINITE);
	}

	// �ͷ� buffer
	if (l_buffer_info[i_cam_ix].p_buffer != nullptr) {
		delete[] l_buffer_info[i_cam_ix].p_buffer;
		l_buffer_info[i_cam_ix].p_buffer = nullptr;
	}
}

std::tuple<bool, unsigned int, const BufferInfo*> SyncMultiCamera::capture() {
	unsigned int i_trigger_cnt{ 0 };
	auto nRet = _soft_trigger();

	for (auto p_cam : lp_mvcamera) {
		WaitForSingleObject(h_sem_agg, INFINITE);
	}
	ReleaseSemaphore(h_sem_continue, static_cast<LONG>(lp_mvcamera.size()), NULL);

	return std::make_tuple(nRet == MV_OK, static_cast<unsigned int>(lp_mvcamera.size()), this->l_buffer_info);
}

int SyncMultiCamera::capture_callback(unsigned int i_trigger_num, void(* call_back)(unsigned int, const BufferInfo*)) {
	unsigned int i_trigger_cnt{ 0 };
	for (unsigned int i = 0; i < i_trigger_num; ++i) {
		auto nRet = _soft_trigger();
		if (nRet == MV_OK) {
			++i_trigger_cnt;
		}
		for (auto p_cam : lp_mvcamera) {
			WaitForSingleObject(h_sem_agg, INFINITE);
		}
		call_back(static_cast<unsigned int>(lp_mvcamera.size()), this->l_buffer_info);
		ReleaseSemaphore(h_sem_continue, static_cast<LONG>(lp_mvcamera.size()), NULL);

		//Sleep(1000);
	}
	return i_trigger_cnt;
}

int SyncMultiCamera::stop_grabbing() {
	b_start_grabbing = false;
	bool all_fine = true;

	ReleaseSemaphore(h_sem_continue, static_cast<LONG>(lp_mvcamera.size()), NULL);
	for (auto p_cam : lp_mvcamera) {
		all_fine &= (p_cam->StopGrabbing() != MV_OK);
	}

	for (auto tup : l_thread) {
		auto i_thread_ix = std::get<0>(tup);
		auto h_thread = std::get<1>(tup);
		WaitForSingleObject(h_thread, INFINITE);
		CloseHandle(h_thread);
	}
	l_thread.clear();

	CloseHandle(h_sem_agg);
	CloseHandle(h_sem_continue);

	return all_fine ? MV_OK : -1;
}

int SyncMultiCamera::soft_trigger_db() {
	auto nRet =
#ifdef _DEBUG
	_soft_trigger();
#else
	MV_OK;
#endif
	return nRet;
}

int SyncMultiCamera::_soft_trigger() {
	bool all_fine = true;
	for (auto p_cam : lp_mvcamera) {
		auto nRet = p_cam->CommandExecute("TriggerSoftware");
		all_fine &= nRet == MV_OK;
		if (nRet != MV_OK) {
			std::cerr << "error: Fail to soft trigger device "
				<< p_cam->m_hDevHandle << std::endl;
		}
	}
	return all_fine ? MV_OK : -1;
}

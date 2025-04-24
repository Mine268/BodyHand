#include "MultiCap.h"

#include <format>
#include <Windows.h>
#include <process.h>
#include <synchapi.h>
#include <winbase.h>

MultiCap* multicap_obj;

std::atomic_int step{ 0 };

MultiCap* get_singleton() {
    if (multicap_obj == nullptr) {
        multicap_obj = new MultiCap();
    }
    return multicap_obj;
}


void release_singleton() {
    if (multicap_obj != nullptr) {
        delete multicap_obj;
        multicap_obj = nullptr;
    }
}


int MultiCap::init_device() {
    // 仅枚举 USB 设备
    auto enum_ret = CMvCamera::EnumDevices(MV_USB_DEVICE, &this->device_info_list);
    // 如果枚举失败，则直接返回 0
    if (enum_ret != MV_OK) {
        //LOG::warn() << "Cannot enumerate the device." << LOG::endl;
        return 0;
    }
    // 如果枚举不到设备，则直接返回 0
    if (this->device_info_list.nDeviceNum == 0) {
        //LOG::info() << "No device is enumerated." << LOG::endl;
        return 0;
    }

    // 打开枚举到的设备
    for (unsigned int enum_ix = 0; enum_ix < this->device_info_list.nDeviceNum; ++enum_ix) {
        auto* cam_tmp = new CMvCamera{};
        auto nRet1 = cam_tmp->Open(device_info_list.pDeviceInfo[enum_ix]); // 打开对应的设备
        auto nRet2 = cam_tmp->SetEnumValue("TriggerMode", 1); // 设置该相机的拍摄模式为触发模式
        auto nRet3 = cam_tmp->SetEnumValue("TriggerSource", 7); // 设置触发源为软触发
        auto nRet4 = cam_tmp->SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed); // 设置输出为 RGB 8 格式
        auto nRet5 = cam_tmp->SetBoolValue("GammaEnable", true); // 开启gamma矫正
        auto nRet6 = cam_tmp->SetEnumValue("GammaSelector", 2); // gamma矫正类型=sRGB
        if (nRet1 != MV_OK) {
            //LOG::info() << "Cannot open the device " << enum_ix << "." << LOG::endl;
            continue;
        }
        if (nRet2 != MV_OK) {
            //LOG::info() << "Cannot set the trigger mode for the device " << enum_ix << "." << LOG::endl;
            continue;
        }
        if (nRet3 != MV_OK) {
            //LOG::info() << "Cannot set the trigger source for the device " << enum_ix << "." << LOG::endl;
            continue;
        }
        if (nRet4 != MV_OK) {
            //LOG::info() << "Cannot set the pixel format for the device " << enum_ix << "." << LOG::endl;
            continue;
        }
        if (nRet5 != MV_OK || nRet6 != MV_OK) {
            //LOG::info() << "Cannot configure gamma correction for the device " << enum_ix << "." << LOG::endl;
            continue;
        }
        this->device_obj_list.push_back(cam_tmp);

        // 使用 GetImageBuffer 和 FreeImageBuffer 管理缓存区
        frame_buffer_list.emplace_back();
        frame_flag.emplace_back(false);
    }
    //LOG::info() << "Opened " << this->device_obj_list.size() << " devices." << LOG::endl;
    auto device_info_buffer = new MV_CC_DEVICE_INFO;
    std::string serial_str = "Camera info:\n";
    for (unsigned int cam_ix = 0; cam_ix < this->device_obj_list.size(); ++cam_ix) {
        this->device_obj_list[cam_ix]->GetDeviceInfo(device_info_buffer);
        serial_str += std::format("\tcam_id={}, serial_number={}\n",
            cam_ix,
            std::string{ reinterpret_cast<const char*>(device_info_buffer->SpecialInfo.stUsb3VInfo.chSerialNumber) });
    }
    //LOG::info() << serial_str << LOG::endl;
    delete device_info_buffer;
    return (int)this->device_obj_list.size();
}


int MultiCap::start_grabbing() {
    b_grab = true;
    int grab_num = 0;
    for (unsigned cam_ix = 0; cam_ix < this->device_obj_list.size(); ++cam_ix) {
        // 开启当前相机的捕捉
        auto* cam = this->device_obj_list[cam_ix];
        auto ret = cam->StartGrabbing();
        if (ret != MV_OK) {
            //LOG::info() << "Cannot start the grabbing for the camera " << cam_ix << ". Error code=" << ret << "." << LOG::endl;
            continue;
        }
        ++grab_num;

        // 建立多线程托管，每个线程需要知道自己托管的相机的 ix
        auto* p_sync_ctrl = new SyncCtrl{ this, cam_ix };
        unsigned int thread_ix{ 0 };
        auto thread_hd =
            (void*)_beginthreadex(nullptr, 0, work_thread, (void*)p_sync_ctrl, 0, &thread_ix);
        if (thread_hd == nullptr) {
            //LOG::info() << "Cannot create the thread for the camera " << cam_ix << "." << LOG::endl;
            --grab_num;
            if (cam->StopGrabbing() != MV_OK) {
                //LOG::error() << "Cannot stop the grabbing for the camera " << cam_ix << ", whose thread cannot be created. Consider restart." << LOG::endl;
            }
        }
        else {
            this->thread_list.push_back(thread_hd);
        }
    }
    // 指示应该等待外部程序处理
    h_sem_agg = CreateSemaphore(nullptr, 0, static_cast<LONG>(this->thread_list.size()), "h_sem_agg");
    // 指示允许继续捕捉
    h_sem_continue = CreateSemaphore(nullptr, 0, static_cast<LONG>(this->thread_list.size()), "h_sem_continue");

    return grab_num;
}


unsigned int __stdcall work_thread(void* p_user) {
    auto p_st = (SyncCtrl*)p_user;
    auto ptr = p_st->obj_ptr;
    auto cam_ix = p_st->cam_ix;
    delete p_st;
    ptr->_work_thread(cam_ix);
    return 0;
}


void MultiCap::_work_thread(unsigned int cam_ix) {
    //LOG::info() << "Thread for cam_ix=" << cam_ix << " is running." << LOG::endl;

    MVCC_INTVALUE_EX st_int = { 0 };
    auto* cam = this->device_obj_list[cam_ix];
    auto& buffer_info = this->frame_buffer_list[cam_ix];

    // 获取缓冲区大小
    auto ret = cam->GetIntValue("PayloadSize", &st_int);
    if (ret != MV_OK) {
        //LOG::error() << "Get PayloadSize failed for camera " << cam_ix << "." << LOG::endl;
        ReleaseSemaphore(h_sem_agg, 1, nullptr);
        if (buffer_info.p_buffer != nullptr) {
            delete[] buffer_info.p_buffer;
            return;
        }
    }
    // 计算数据大小
    auto buffer_size = (unsigned int)st_int.nCurValue;
    // 申请缓冲区
    buffer_info.p_buffer = new unsigned char[sizeof(unsigned char) * buffer_size];

    MV_FRAME_OUT_INFO_EX st_image_info{ 0 };
    auto tmp_info_buffer = new MV_CC_DEVICE_INFO;
    cam->GetDeviceInfo(tmp_info_buffer);
    //LOG::info() << "Thread for cam_ix=" << cam_ix << " starts loop." << LOG::endl;
    while (b_grab) {
        WaitForSingleObject(h_sem_continue, INFINITE);
        //LOG::info() << std::format("[{}] cam_ix={} require h_sem_continue", step++, cam_ix) << LOG::endl;

        if (!b_grab) { break; } // 如果调用了 stop_grabbing，那么这句话直接退出循环
        ret = cam->GetOneFrameTimeout(
            buffer_info.p_buffer, buffer_size, &st_image_info, 1000);
        if (ret != MV_OK) {
            //LOG::error() << std::format("Get one frame failed for cam_ix={}, error={}, frame_ix={}",
                //cam_ix, ret, st_image_info.nFrameNum) << LOG::endl;
        }
        else {
            //LOG::error() << std::format("frame succeed, cam_ix={}, frame_ix={}", cam_ix, st_image_info.nFrameNum) << LOG::endl;
        }
        frame_flag[cam_ix] = (ret == MV_OK);
        buffer_info.width = st_image_info.nWidth;
        buffer_info.height = st_image_info.nHeight;
        buffer_info.frame_ix = st_image_info.nFrameNum;
        strcpy_s(reinterpret_cast<char*>(buffer_info.p_serial_number),
            strlen(reinterpret_cast<const char*>(tmp_info_buffer->SpecialInfo.stUsb3VInfo.chSerialNumber)) + 1,
            reinterpret_cast<const char*>(tmp_info_buffer->SpecialInfo.stUsb3VInfo.chSerialNumber));

        ReleaseSemaphore(h_sem_agg, 1, nullptr);
        //LOG::info() << std::format("[{}] cam_ix={} release h_sem_agg", step++, cam_ix) << LOG::endl;
    }
    delete tmp_info_buffer;
    //LOG::info() << "Thread for cam_ix=" << cam_ix << " exits loop." << LOG::endl;

    // 释放缓冲区
    if (buffer_info.p_buffer != nullptr) {
        delete[] buffer_info.p_buffer;
        buffer_info.p_buffer = nullptr;
    }
}


void MultiCap::capture() {
    // 捕捉成功标志清空
    for (unsigned int fx = 0; fx < this->frame_flag.size(); ++fx) {
        this->frame_flag[fx] = false;
    }
    // 接受释放全部 h_sem_continue 信号量，允许线程执行
    ReleaseSemaphore(h_sem_continue, (LONG) this->thread_list.size(), nullptr);
    //LOG::info() << std::format("[{}] Main release {} h_sem_conintue", step++, this->thread_list.size()) << LOG::endl;
    // 进行全部触发
    for (unsigned int cam_ix = 0; cam_ix < this->device_obj_list.size(); cam_ix++) {
        auto* cam = this->device_obj_list[cam_ix];
        auto ret = cam->CommandExecute("TriggerSoftware");
        if (ret != MV_OK) { /*LOG::error() << "TriggerSoftware failed for camera " << cam_ix << "." << LOG::endl;*/ }
    }
    // 等待全部信号量
    for (auto* cam : this->device_obj_list) {
        WaitForSingleObject(h_sem_agg, INFINITE);
        //LOG::info() << std::format("[{}] Main require 1 h_sem_agg", step++) << LOG::endl;
    }
    //LOG::info() << std::format("[{}] Main has acquired all h_sem_agg", step++) << LOG::endl;
}


int MultiCap::stop_grabbing() {
    int stop_num = 0;

    // 关闭设备抓取
    // WaitForSingleObject(h_sem_grab, INFINITE);
    b_grab = false;
    ReleaseSemaphore(h_sem_continue, static_cast<LONG>(this->device_obj_list.size()), nullptr);
    for (unsigned int cam_ix = 0; cam_ix < this->device_obj_list.size(); cam_ix++) {
        auto* cam = this->device_obj_list[cam_ix];
        auto ret = cam->StopGrabbing();
        if (ret == MV_OK) {
            ++stop_num;
        }
        else {
            //LOG::error() << "StopGrabbing failed for camera " << cam_ix << "." << LOG::endl;
        }
    }

    // 关闭线程
    for (auto thread_h : this->thread_list) {
        WaitForSingleObject(thread_h, INFINITE);
        CloseHandle(thread_h);
    }

    CloseHandle(h_sem_agg);
    CloseHandle(h_sem_continue);

    return -1;
}


int MultiCap::close_device() {
    int close_num = 0;
    for (unsigned int cam_ix = 0; cam_ix < this->device_obj_list.size(); cam_ix++) {
        auto* cam = this->device_obj_list[cam_ix];
        if (cam->Close() == MV_OK) {
            ++close_num;
        }
        else {
            //LOG::error() << "Close device failed for camera " << cam_ix << "." << LOG::endl;
        }
    }
    this->device_obj_list.clear();
    return close_num;
}


int MultiCap::get_device_count() const {
    return this->device_obj_list.size();
}


int MultiCap::set_exposure(unsigned int cam_ix, float exposure) {
    if (cam_ix >= this->device_obj_list.size()) {
        //LOG::error() << std::format("cam_ix={} is greater than max index {}", cam_ix, this->device_obj_list.size()) << LOG::endl;
        return -1;
    }
    auto ret1 = this->device_obj_list[cam_ix]->SetEnumValue("ExposureMode", 0);
    auto ret2 = this->device_obj_list[cam_ix]->SetFloatValue("ExposureTime", exposure);
    if (ret1 != MV_OK || ret2 != MV_OK) {
        //LOG::warn() << std::format("Cannot set exposure time for cam_ix={}", cam_ix) << LOG::endl;
        return -1;
    }
    //LOG::info() << std::format("Modify exposure time to {} for cam_ix={}", exposure, cam_ix) << LOG::endl;
    return 0;
}


int MultiCap::set_gain(unsigned int cam_ix, float gain) {
    if (cam_ix >= this->device_obj_list.size()) {
        //LOG::error() << std::format("cam_ix={} is greater than max index {}", cam_ix, this->device_obj_list.size()) << LOG::endl;
        return -1;
    }
    int ret1, ret2 = MV_OK;
    if (gain < -0.1f) {
        ret1 = this->device_obj_list[cam_ix]->SetEnumValue("GainAuto", 1);
    }
    else
    {
        ret1 = this->device_obj_list[cam_ix]->SetEnumValue("GainAuto", 0);
        ret2 = this->device_obj_list[cam_ix]->SetFloatValue("Gain", gain);
    }
    if (ret1 != MV_OK || ret2 != MV_OK) {
        //LOG::warn() << std::format("Cannot set gain for cam_ix={}", cam_ix) << LOG::endl;
        return -1;
    }
    //LOG::info() << std::format("Modify gain to {} for cam_ix={}", gain, cam_ix) << LOG::endl;
    return 0;
}

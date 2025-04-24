#ifndef INC_2024_01_28_MULTICAPCPY_MULTICAP_H
#define INC_2024_01_28_MULTICAPCPY_MULTICAP_H

#include "MvCamera.h"
#include <Windows.h>
#include <process.h>
#include <vector>
#include <atomic>


struct SyncCtrl;
struct MultiCap;
struct BufferInfo;


struct SyncCtrl {
    MultiCap* obj_ptr;
    unsigned int cam_ix;
};


struct BufferInfo
{
    unsigned int width; // 图像宽度
    unsigned int height; // 图象高度
    unsigned int frame_ix; // 帧编号
    unsigned char* p_buffer; // 缓冲区指针
    unsigned char p_serial_number[INFO_MAX_BUFFER_SIZE]; // 设备序列号
};


struct MultiCap {
    MV_CC_DEVICE_INFO_LIST device_info_list{ 0 };  // 设备信息列表
    std::vector<CMvCamera*> device_obj_list;  // 设备对象列表
    std::vector<BufferInfo> frame_buffer_list;  // 图像帧列表
    std::vector<bool> frame_flag;  // 表示图像是否抓取成功
    std::vector<void*> thread_list;  // 线程句柄列表

    HANDLE h_sem_agg{}, h_sem_continue{};  // 用于控制子线程和 main 同步的信号量
    std::atomic_bool b_grab{ false };

    MultiCap() = default;
    ~MultiCap() = default;

    /**
     * 枚举 + 打开设备，返回打开的设备的数量
     */
    int init_device();

    /**
     * 开始抓取图像流，返回成功开始抓取的设备数量
     */
    int start_grabbing();

    /**
     * 多线程抓取第 cam_ix 设备图像
     */
    void _work_thread(unsigned int cam_ix);

    /**
     * 进行一次同步捕捉
     */
    void capture();

    /**
     * 关闭抓取，返回成功关闭的抓取的数量
     */
    int stop_grabbing();

    /**
     * 关闭设备，返回成功关闭的设备数量
     */
    int close_device();

    /*
     * 获得相机数量
     */
    int get_device_count() const;

    /*
     * 设置某一个相机的曝光时间
     */
    int set_exposure(unsigned int cam_ix, float exposure);

    /*
     * 设置某一个相机的增益
     */
    int set_gain(unsigned int cam_ix, float gain);
};


/*
 * 采用单例，便于 ctypes 封装
 * */
extern MultiCap* multicap_obj;


MultiCap* get_singleton();


void release_singleton();


unsigned int __stdcall work_thread(void* p_user);


#endif //INC_2024_01_28_MULTICAPCPY_MULTICAP_H

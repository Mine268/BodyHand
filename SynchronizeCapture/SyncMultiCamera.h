#pragma once

// 该类用于同步控制 MvCamera
// 采用单例模式

#include <vector>
#include <tuple>
#include <mutex>
#include <atomic>
#include <Windows.h>

#include "MvCamera.h"

class SyncMultiCamera;
struct SyncControlData;
struct BufferInfo;


// 多线程函数参数结构体
struct SyncControlData
{
	SyncMultiCamera* p_brain;
	unsigned int i_cam_ix;
};


// 图像缓冲区的格式信息
struct BufferInfo
{
	std::atomic<unsigned int> i_width; // 图像宽度
	std::atomic<unsigned int> i_height; // 图象高度
	std::atomic<unsigned int> i_frame_ix; // 帧编号
	std::atomic<unsigned char*> p_buffer; // 缓冲区指针
	std::atomic<bool> b_success; // 指示图象是否有效
};


class SyncMultiCamera
{
	// 方便调用 thread_func
	friend unsigned int __stdcall WorkThread(void* p_user);

private: // 'meta'
	static SyncMultiCamera* p_singleton;

	//bool b_open_device{ false }; // 设备是否打开
	std::atomic<bool> b_start_grabbing{ false }; // 是否开始抓取图象

	std::atomic<unsigned int> i_time_span; // 两次捕捉之间的的时间间隔（ms）
	std::atomic<bool> b_save_image; // 表示是否存储图片

	SyncMultiCamera() = default;
	~SyncMultiCamera() = default;

	int _soft_trigger(); // 对所有设备执行一个软触发
	void thread_func(unsigned int); // 执行软触发采集、存储等工作

	// 用于控制子线程和 main 同步的信号量
	HANDLE h_sem_agg, h_sem_continue;

private: // 'data'
	MV_CC_DEVICE_INFO_LIST l_device_info{ 0 }; // 当前连接的设备的信息列表
	std::vector<CMvCamera*> lp_mvcamera; // 存储所有 CMvCamera 类指针的数组，每个类指向一个相机接口

	// 多线程列表，每个 tuple 表示 <线程 id, 线程句柄>
	std::vector<std::tuple<unsigned int, void*>> l_thread;

	// 多视图图象信息存储数组
	BufferInfo l_buffer_info[256];

public:
	static SyncMultiCamera* get(unsigned int, bool);
	static void destroy();

	void set_time_span(unsigned int); // 设置捕捉间隔
	void set_save_image(bool); // 设置是否捕捉图片
	int get_cam_count() const;

	int enum_device(); // 枚举设备，设备信息写入成员
	int open_device(); // 打开设备，并手动设置触发模式为软触发
	int set_exposure_time(unsigned int, float); // 设置某一个设备的曝光时间和增益
	int set_gain(unsigned int, float);
	int start_grabbing(); // 开始捕获
	std::tuple<bool, unsigned int, const BufferInfo*> capture(); // 捕捉一帧
	// 没有定义的
	int capture_callback(unsigned int, void(*)(unsigned int, const BufferInfo*)); // 开始捕捉若干帧
	int soft_trigger_db(); // 用于 debug 的方法
	int stop_grabbing(); // 停止捕获
	int close_device(); // 关闭设备
};

// TODO
/*
* start_grabbing stop_grabbing 多线程实现
* 
* soft_trigger 软触发
*/



unsigned int __stdcall WorkThread(void* p_user);

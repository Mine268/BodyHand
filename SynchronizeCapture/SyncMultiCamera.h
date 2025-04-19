#pragma once

// ��������ͬ������ MvCamera
// ���õ���ģʽ

#include <vector>
#include <tuple>
#include <mutex>
#include <atomic>
#include <Windows.h>

#include "MvCamera.h"

class SyncMultiCamera;
struct SyncControlData;
struct BufferInfo;


// ���̺߳��������ṹ��
struct SyncControlData
{
	SyncMultiCamera* p_brain;
	unsigned int i_cam_ix;
};


// ͼ�񻺳����ĸ�ʽ��Ϣ
struct BufferInfo
{
	std::atomic<unsigned int> i_width; // ͼ����
	std::atomic<unsigned int> i_height; // ͼ��߶�
	std::atomic<unsigned int> i_frame_ix; // ֡���
	std::atomic<unsigned char*> p_buffer; // ������ָ��
	std::atomic<bool> b_success; // ָʾͼ���Ƿ���Ч
};


class SyncMultiCamera
{
	// ������� thread_func
	friend unsigned int __stdcall WorkThread(void* p_user);

private: // 'meta'
	static SyncMultiCamera* p_singleton;

	//bool b_open_device{ false }; // �豸�Ƿ��
	std::atomic<bool> b_start_grabbing{ false }; // �Ƿ�ʼץȡͼ��

	std::atomic<unsigned int> i_time_span; // ���β�׽֮��ĵ�ʱ������ms��
	std::atomic<bool> b_save_image; // ��ʾ�Ƿ�洢ͼƬ

	SyncMultiCamera() = default;
	~SyncMultiCamera() = default;

	int _soft_trigger(); // �������豸ִ��һ������
	void thread_func(unsigned int); // ִ�������ɼ����洢�ȹ���

	// ���ڿ������̺߳� main ͬ�����ź���
	HANDLE h_sem_agg, h_sem_continue;

private: // 'data'
	MV_CC_DEVICE_INFO_LIST l_device_info{ 0 }; // ��ǰ���ӵ��豸����Ϣ�б�
	std::vector<CMvCamera*> lp_mvcamera; // �洢���� CMvCamera ��ָ������飬ÿ����ָ��һ������ӿ�

	// ���߳��б�ÿ�� tuple ��ʾ <�߳� id, �߳̾��>
	std::vector<std::tuple<unsigned int, void*>> l_thread;

	// ����ͼͼ����Ϣ�洢����
	BufferInfo l_buffer_info[256];

public:
	static SyncMultiCamera* get(unsigned int, bool);
	static void destroy();

	void set_time_span(unsigned int); // ���ò�׽���
	void set_save_image(bool); // �����Ƿ�׽ͼƬ
	int get_cam_count() const;

	int enum_device(); // ö���豸���豸��Ϣд���Ա
	int open_device(); // ���豸�����ֶ����ô���ģʽΪ����
	int set_exposure_time(unsigned int, float); // ����ĳһ���豸���ع�ʱ�������
	int set_gain(unsigned int, float);
	int start_grabbing(); // ��ʼ����
	std::tuple<bool, unsigned int, const BufferInfo*> capture(); // ��׽һ֡
	// û�ж����
	int capture_callback(unsigned int, void(*)(unsigned int, const BufferInfo*)); // ��ʼ��׽����֡
	int soft_trigger_db(); // ���� debug �ķ���
	int stop_grabbing(); // ֹͣ����
	int close_device(); // �ر��豸
};

// TODO
/*
* start_grabbing stop_grabbing ���߳�ʵ��
* 
* soft_trigger ����
*/



unsigned int __stdcall WorkThread(void* p_user);

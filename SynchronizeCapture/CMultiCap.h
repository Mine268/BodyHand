#ifndef HEAD_CMULTICAP
#define HEAD_CMULTICAP

constexpr unsigned int INFO_MAX_BUFFER_SIZE = 64;
constexpr unsigned int MAX_CAPTURE = 16;


extern "C" {

    struct CaptureInfo {
        unsigned int n_cap;  // 捕捉的数量
        bool flag[MAX_CAPTURE];  // 每一个捕捉是否成功
        unsigned int width[MAX_CAPTURE];  // 每一个捕捉的宽度
        unsigned int height[MAX_CAPTURE];  // 每一个图像的高度
        unsigned char* ppbuffer[MAX_CAPTURE];  // 每一个捕捉的缓冲区的指针
        unsigned char serial_numbers[MAX_CAPTURE][INFO_MAX_BUFFER_SIZE]; // 每一个捕捉的图像对应相机的序列号
    };

    // 获得单例
    __declspec(dllexport) void get_app();

    // 关系系统
    __declspec(dllexport) void release_app();

    // 获取设备
    __declspec(dllexport) auto init_device() -> int;

    // 关闭设备
    __declspec(dllexport) auto close_device() -> int;

    // 开始抓取
    __declspec(dllexport) auto start_grabbing() -> int;

    // 停止抓取
    __declspec(dllexport) auto stop_grabbing() -> int;

    // 抓取一帧
    __declspec(dllexport) auto capture() -> CaptureInfo;

    // 获取设备数量
    __declspec(dllexport) auto get_device_count() -> int;

    // 设置设备曝光
    __declspec(dllexport) auto set_exposure(unsigned cam_ix, float exposure) -> int;

    // 设置设备增益
    __declspec(dllexport) auto set_gain(unsigned cam_ix, float gain) -> int;

}


#endif // !HEAD_CMULTICAP
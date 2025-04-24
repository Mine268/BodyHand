#include "CMultiCap.h"
#include "MultiCap.h"

extern "C" {

    __declspec(dllexport) void get_app() {
        get_singleton();
    }

    __declspec(dllexport) void release_app() {
        release_singleton();
    }

    __declspec(dllexport) auto init_device() -> int {
        return multicap_obj->init_device();
    }

    __declspec(dllexport) auto close_device() -> int {
        return multicap_obj->close_device();
    }

    __declspec(dllexport) auto start_grabbing() -> int {
        return multicap_obj->start_grabbing();
    }

    __declspec(dllexport) auto stop_grabbing() -> int {
        return multicap_obj->stop_grabbing();
    }

    __declspec(dllexport) auto capture() -> CaptureInfo {
        CaptureInfo cap_info{};

        multicap_obj->capture();
        cap_info.n_cap = multicap_obj->device_obj_list.size();
        for (unsigned int cap_ix = 0; cap_ix < multicap_obj->frame_buffer_list.size(); ++cap_ix) {
            if (cap_ix >= MAX_CAPTURE) {
                break;
            }

            cap_info.flag[cap_ix] = multicap_obj->frame_flag[cap_ix];
            cap_info.width[cap_ix] = multicap_obj->frame_buffer_list[cap_ix].width;
            cap_info.height[cap_ix] = multicap_obj->frame_buffer_list[cap_ix].height;
            cap_info.ppbuffer[cap_ix] = multicap_obj->frame_buffer_list[cap_ix].p_buffer;
            strcpy_s(reinterpret_cast<char*>(cap_info.serial_numbers[cap_ix]), 
                strlen(reinterpret_cast<const char*>(multicap_obj->frame_buffer_list[cap_ix].p_serial_number)) + 1,
                reinterpret_cast<const char*>(multicap_obj->frame_buffer_list[cap_ix].p_serial_number));
        }

        return cap_info;
    }

    __declspec(dllexport) auto get_device_count() -> int {
        return multicap_obj->get_device_count();
    }

    __declspec(dllexport) auto set_exposure(unsigned cam_ix, float exposure) -> int {
        return multicap_obj->set_exposure(cam_ix, exposure);
    }

    __declspec(dllexport) auto set_gain(unsigned cam_ix, float gain) -> int {
        return multicap_obj->set_gain(cam_ix, gain);
    }

}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "HandDetectorWrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(hand_detector_module, m) {
    py::class_<HandDetectorWrapper>(m, "HandDetectorWrapper")
        .def(py::init<const std::string&>())
        .def("estimate", &HandDetectorWrapper::estimate);

    py::class_<HandDetectorWrapper::HandResult>(m, "HandResult")
        .def_readwrite("leftHand", &HandDetectorWrapper::HandResult::leftHand)
        .def_readwrite("rightHand", &HandDetectorWrapper::HandResult::rightHand)
        .def_readwrite("leftHand25D", &HandDetectorWrapper::HandResult::leftHand25D)
        .def_readwrite("rightHand25D", &HandDetectorWrapper::HandResult::rightHand25D);
}
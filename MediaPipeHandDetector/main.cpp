#include <iostream>
#include <Python.h>

int main() {
	Py_SetPythonHome(LR"(C:\Users\Administrator\AppData\Roaming\uv\python\cpython-3.10.17-windows-x86_64-none)");
	Py_Initialize();
	if (!Py_IsInitialized()) {
		std::cerr << "Initialization failed" << std::endl;
		return -1;
	}
	else {
		std::cout << "Try py!" << std::endl;
		PyRun_SimpleString("print(2333)");
		PyRun_SimpleString("import sys");
		PyRun_SimpleString("import os");
		PyRun_SimpleString(R"(
for key, val in os.environ.items():
	print(f"{key}: {val}")
)");
		PyRun_SimpleString("import mediapipe");
	}
	return 0;
}
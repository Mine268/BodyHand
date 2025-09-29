#pragma once
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <cstdio>
#include <string>
#include <chrono>
#include <thread>

#pragma comment(lib, "Ws2_32.lib")

bool InitWinsock();
void CleanupWinsock();

SOCKET CreateListeningSocket(const char* port, int backlog = SOMAXCONN);
SOCKET AcceptClient(SOCKET listenSock);                        // 阻塞等待连接
void  CloseSocket(SOCKET s);

bool SendAll(SOCKET s, const void* buf, int len);              // 可靠发送完整缓冲
bool SendTextLine(SOCKET s, const std::string& line);          // 发送一行文本并附加 '\n'

bool GetPeerString(SOCKET s, std::string& out);                // 获取对端 ip:port

// 演示：对已连接的客户端每秒发送一行，直到失败/断开
void ServeClientWithCounter(SOCKET clientSock, int start = 0, int intervalSec = 1);

// 演示：永久服务器循环：等待→服务→继续等待
void ServeForever(const char* port);


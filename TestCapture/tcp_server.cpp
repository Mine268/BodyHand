#include "tcp_server.h"

bool InitWinsock() {
    WSADATA wsa{};
    int r = WSAStartup(MAKEWORD(2, 2), &wsa);
    if (r != 0) std::printf("WSAStartup failed: %d\n", r);
    return r == 0;
}

void CleanupWinsock() {
    WSACleanup();
}

SOCKET CreateListeningSocket(const char* port, int backlog) {
    addrinfo hints{};
    hints.ai_family = AF_INET;            // 需要 v6 可改 AF_INET6 或 AF_UNSPEC
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    addrinfo* ai = nullptr;
    if (int r = getaddrinfo(nullptr, port, &hints, &ai); r != 0) {
        std::printf("getaddrinfo failed: %d\n", r);
        return INVALID_SOCKET;
    }

    SOCKET s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (s == INVALID_SOCKET) {
        std::printf("socket failed: %ld\n", WSAGetLastError());
        freeaddrinfo(ai);
        return INVALID_SOCKET;
    }

    BOOL yes = TRUE;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));

    if (bind(s, ai->ai_addr, (int)ai->ai_addrlen) == SOCKET_ERROR) {
        std::printf("bind failed: %ld\n", WSAGetLastError());
        freeaddrinfo(ai);
        closesocket(s);
        return INVALID_SOCKET;
    }
    freeaddrinfo(ai);

    if (listen(s, backlog) == SOCKET_ERROR) {
        std::printf("listen failed: %ld\n", WSAGetLastError());
        closesocket(s);
        return INVALID_SOCKET;
    }

    return s;
}

SOCKET AcceptClient(SOCKET listenSock) {
    std::printf("Waiting for a client to connect...\n");
    SOCKET cs = accept(listenSock, nullptr, nullptr); // 阻塞等待
    if (cs == INVALID_SOCKET) {
        std::printf("accept failed: %ld\n", WSAGetLastError());
    }
    return cs;
}

void CloseSocket(SOCKET s) {
    if (s != INVALID_SOCKET) {
        shutdown(s, SD_BOTH);
        closesocket(s);
    }
}

bool SendAll(SOCKET s, const void* buf, int len) {
    const char* p = static_cast<const char*>(buf);
    int sentTotal = 0;
    while (sentTotal < len) {
        int sent = send(s, p + sentTotal, len - sentTotal, 0);
        if (sent == SOCKET_ERROR) {
            return false;
        }
        sentTotal += sent;
    }
    return true;
}

bool SendTextLine(SOCKET s, const std::string& line) {
    std::string withNL = line;
    if (withNL.empty() || withNL.back() != '\n') withNL.push_back('\n');
    return SendAll(s, withNL.c_str(), (int)withNL.size());
}

bool GetPeerString(SOCKET s, std::string& out) {
    sockaddr_storage ss{};
    int len = sizeof(ss);
    if (getpeername(s, (sockaddr*)&ss, &len) != 0) return false;

    char host[NI_MAXHOST]{}, serv[NI_MAXSERV]{};
    if (getnameinfo((sockaddr*)&ss, len, host, sizeof(host), serv, sizeof(serv),
        NI_NUMERICHOST | NI_NUMERICSERV) != 0) return false;

    out = std::string(host) + ":" + serv;
    return true;
}

void ServeClientWithCounter(SOCKET clientSock, int start, int intervalSec) {
    std::string peer;
    if (GetPeerString(clientSock, peer)) {
        std::printf("Client connected from %s\n", peer.c_str());
    }

    int counter = start;
    while (true) {
        std::string msg = "Hello from server, count=" + std::to_string(counter++);
        if (!SendTextLine(clientSock, msg)) {
            std::printf("send failed or peer closed. Error=%ld\n", WSAGetLastError());
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(intervalSec));
    }

    CloseSocket(clientSock);
    std::printf("Client disconnected.\n");
}

void ServeForever(const char* port) {
    if (!InitWinsock()) return;

    SOCKET ls = CreateListeningSocket(port);
    if (ls == INVALID_SOCKET) {
        CleanupWinsock();
        return;
    }
    std::printf("Server started. Listening on port %s ...\n", port);

    // 永久循环：一个客户端结束后继续等待下一位
    while (true) {
        SOCKET cs = AcceptClient(ls);
        if (cs == INVALID_SOCKET) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        ServeClientWithCounter(cs); // 这里可替换成你自己的“服务逻辑”
    }

    // （通常到不了）
    CloseSocket(ls);
    CleanupWinsock();
}
// Server side C/C++ program to demonstrate Socket programming
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#define PORT 8080

int main(int argc, char const *argv[]) {
    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};
    char *hello = "Hello from server";

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                   sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Forcefully attaching socket to the port 8080
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address,
                             (socklen_t *)&addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    fflush(stdout);
    printf("Connection established\nWaiting for handshake\n");
    fflush(stdout);

    while (!(valread = read(new_socket, buffer, 1024)))
        ;
    if (valread < 0)
        return 0;
    while (strcmp(buffer, "sgrge864358ewr") != 0) {
        while (!(valread = read(new_socket, buffer, 1024)))
            ;
        if (valread < 0)
            return 0;
    }

    send(new_socket, "sgrge864358ewr", strlen("sgrge864358ewr"), 0);

    printf("Handshake done!\n");
    fflush(stdout);

    do {
        while (!(valread = read(new_socket, buffer, 1024)))
            ;

        if (strcmp(buffer, "sgrge864358ewr") == 0) {
            printf("Window closed!\nExiting.....\n");
            fflush(stdout);
            return 0;
        }

        printf("%s\n", buffer);
        fflush(stdout);

        buffer[0] = 0;

    } while (valread >= 0);
    return 0;
}

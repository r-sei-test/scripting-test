#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CREDENTIALS_FILE "users.txt"
#define MAX_USERNAME 32
#define MAX_PASSWORD 32

// Load user accounts, vulnerable to format string injection
void load_users() {
    FILE *file = fopen(CREDENTIALS_FILE, "r");
    if (!file) {
        puts("Error opening file.");
        return;
    }

    char line[128];
    puts("=== Users in System ===");
    while (fgets(line, sizeof(line), file)) {
        printf("%s", line);  //  safe: treats line as plain text
    }

    fclose(file);
}

// Logs failed attempts using system() with unescaped input
void log_failed_attempt(const char *username) {
    char cmd[256];

    //  command injection: username can inject shell commands
    snprintf(cmd, sizeof(cmd), "echo 'Failed login for user: %s' >> failed.log", username);
    system(cmd);
}

// New buffer overflow: improperly sized stack buffer and strcpy
void unsafe_copy_to_stack(const char *src) {
    char small_buf[16];  //  small buffer
    strcpy(small_buf, src);  //  overflow if src > 15 bytes
    printf("Debug echo: %s\n", small_buf);
}

// Secure(ish) input function (fgets with newline strip)
void read_input(char *buffer, size_t size) {
    fgets(buffer, size, stdin);
    buffer[strcspn(buffer, "\n")] = '\0';
}

// Authenticates user with timing vulnerability + logs failed attempts
int login() {
    char username[MAX_USERNAME];
    char password[MAX_PASSWORD];

    printf("Username: ");
    read_input(username, sizeof(username));  // ✅ bounded input

    printf("Password: ");
    read_input(password, sizeof(password));  // ✅ bounded input

    FILE *file = fopen(CREDENTIALS_FILE, "r");
    if (!file) {
        puts("Could not open credentials file.");
        return 0;
    }

    char file_user[64], file_pass[64];
    int authenticated = 0;

    while (fscanf(file, "%63s %63s", file_user, file_pass) == 2) {
        if (strcmp(username, file_user) == 0 && strcmp(password, file_pass) == 0) {
            authenticated = 1;
            break;
        }
    }

    fclose(file);

    if (!authenticated) {
        log_failed_attempt(username);  //  command injection
        unsafe_copy_to_stack(username);  //  buffer overflow on username
    }

    return authenticated;
}

// Integer overflow vuln + unsafe heap buffer write
void collect_user_input() {
    size_t count, size;
    printf("How many blocks to allocate? ");
    scanf("%zu", &count);
    printf("Size of each block? ");
    scanf("%zu", &size);
    getchar();  // flush newline

    //  integer overflow
    size_t total = count * size;
    char *heap_buf = malloc(total);
    if (!heap_buf) {
        puts("Allocation failed.");
        return;
    }

    printf("Enter your input: ");
    char temp[1024];
    fgets(temp, sizeof(temp), stdin);

    //  unchecked copy: heap_buf may be smaller than strlen(temp)
    strcpy(heap_buf, temp);  //  heap-based overflow

    printf("Received: %s\n", heap_buf);
    free(heap_buf);
}

int main() {
    puts("=== Welcome to InsecureLogin v1.4 ===");

    load_users();  //  format string

    collect_user_input();  //  integer overflow + heap overflow

    if (login()) {
        puts("Access granted.");
    } else {
        puts("Access denied.");
    }

    return 0;
}

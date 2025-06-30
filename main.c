#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Hardcoded file
#define CREDENTIALS_FILE "users.txt"

// Max size for username and password
#define MAX_USERNAME 32
#define MAX_PASSWORD 32

// Function to simulate user enumeration and demonstrate format string vulnerability
void load_users() {
    FILE *file = fopen(CREDENTIALS_FILE, "r");

    // Insecure file access (no permission or ownership check)
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    char line[128];

    printf("=== Users in System ===\n");

    // Read each line of file
    while (fgets(line, sizeof(line), file)) {
        // ⚠️ Format String Vulnerability:
        // If attacker can control content of 'users.txt', this becomes exploitable
        printf(line); // attacker-controlled format string
    }

    fclose(file);
}

// Function with buffer overflow via gets()
int login() {
    char username[32];
    char password[32];

    printf("Username: ");
    gets(username); // ⚠️ Classic buffer overflow (unsafe)

    printf("Password: ");
    gets(password); // ⚠️ Another buffer overflow

    FILE *file = fopen(CREDENTIALS_FILE, "r");
    if (file == NULL) {
        printf("Could not open credentials file.\n");
        return 0;
    }

    char file_user[64];
    char file_pass[64];

    int authenticated = 0;

    // Read stored credentials from file
    while (fscanf(file, "%63s %63s", file_user, file_pass) == 2) {
        // ⚠️ Timing attack surface: strcmp() returns on first mismatch
        if (strcmp(username, file_user) == 0 && strcmp(password, file_pass) == 0) {
            authenticated = 1;
            break;
        }
    }

    fclose(file);
    return authenticated;
}

int main() {
    printf("=== Welcome to InsecureLogin v1.1 ===\n");

    load_users(); // ⚠️ Prints attacker-controlled file contents

    if (login()) {
        printf("Access granted.\n");
    } else {
        printf("Access denied.\n");
    }

    return 0;
}

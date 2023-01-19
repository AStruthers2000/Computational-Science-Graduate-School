//includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysinfo.h>

//constants
#define MAX_STR_SIZE 64

//variables
struct Command{
	void (*Execute)(char* args, int g_arg);
	char* cmd;
	char* args;
	char* usage_string;
} cmd_exit, cmd_prompt, cmd_cpuinfo, cmd_meminfo;

char line[MAX_STR_SIZE];
char prompt[MAX_STR_SIZE];
char parsed_cmd[MAX_STR_SIZE];
char parsed_args[MAX_STR_SIZE];

int exit_code;
int EXIT_FLAG;


//functions
int main(void);
void ParseInput(const char* input, char* cmd, char* args, int* is_cmd_good, int* is_arg_good);
int ParseArgsFromCMD(const struct Command command, char* args);
int HandleCMD(char* cmd, char* args, int good_cmd, int good_args);

void HandleExit(char* args, int g_arg);
void HandlePrompt(char* args, int g_arg);
void HandleCPUInfo(char* args, int g_arg);
void HandleMEMInfo(char* args, int g_arg);

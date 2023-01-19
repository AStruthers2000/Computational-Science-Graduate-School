#include "minishell.h"

int ParseArgsFromCMD(const struct Command command, char* args)
{
	int i, j;
	if(args[0] == '-' && args[1] == 'h')
	{
		printf("Help for using command %s:\n==========\n%s\n\tAcceptable arguments are: %s\n==========\n", command.cmd, command.usage_string, command.args);
		return 2;
	}

	for(i = 0; i < strlen(args); i++)
	{
		int is_arg_acceptable = 0;
		if(command.args[0] == '*')
		{
			return 1;
		}
		for(j = 0; j < strlen(command.args); j++)
		{
			if(args[i] == command.args[j] || args[i] == '-')//(args[0] == '-' && strcmp(command.cmd, "exit") == 0))
			{
				is_arg_acceptable = 1;
			}
		}
		if(!is_arg_acceptable)
		{
			return 0;
		}
	}
	return 1;
}

void ParseInput(const char* input, char* cmd, char* args, int* is_cmd_good, int* is_arg_good)
{
	int reading_cmd = 1;
	int cmd_count=0, arg_count = 0;

	int i;
	for(i = 0; i < strlen(input); i++)
	{
		char c = input[i];
		if(reading_cmd)
		{
			cmd[cmd_count++] = c;
			if(input[i+1] == ' ' || input[i+1] == '\n')
			{
				reading_cmd = 0;
			}
		}
		else
		{
			if(!(input[i] == ' ' || input[i] == '\n' || input[i] == '\0'))
			{
				args[arg_count++] = c;
			}
		}
	}

	struct Command cmd_generic;
	cmd_generic.cmd = "";

	if(strcmp(cmd, "exit") == 0)
	{
		cmd_generic = cmd_exit;
	}
	else if(strcmp(cmd, "prompt") == 0)
	{
		cmd_generic = cmd_prompt;
	}
	else if(strcmp(cmd, "cpuinfo") == 0)
	{
		cmd_generic = cmd_cpuinfo;
	}
	else if(strcmp(cmd, "meminfo") == 0)
	{
		cmd_generic = cmd_meminfo;
	}

	if(strlen(cmd_generic.cmd) > 0)
	{
		*is_cmd_good = 1;
		*is_arg_good = ParseArgsFromCMD(cmd_generic, args);
	}
	else
	{
		//printf("Command not recognized\n");
		*is_cmd_good = 0;
	}
}

int HandleCMD(char* cmd, char* args, int good_cmd, int good_args)
{
	struct Command cmd_generic;

	if(good_cmd)
	{
		if(strcmp(cmd, "exit") == 0)
		{
			cmd_generic = cmd_exit;
		}
		else if(strcmp(cmd, "prompt") == 0)
		{
			cmd_generic = cmd_prompt;
		}
		else if(strcmp(cmd, "cpuinfo") == 0)
		{
			cmd_generic = cmd_cpuinfo;
		}
		else if(strcmp(cmd, "meminfo") == 0)
		{
			cmd_generic = cmd_meminfo;
		}

		if(good_args == 1)
		{
			cmd_generic.Execute(args, good_args);
		}
		else if(good_args == 0)
		{
			printf("Failed to use command %s, see proper usage:\n==========\n%s\n\tAcceptable arguments are: %s\n==========\n", cmd_generic.cmd, cmd_generic.usage_string, cmd_generic.args);
		}
	}
	else
	{
		//need a system call
		system(line);
	}
}

void HandleExit(char* args, int g_arg)
{
	if(g_arg)
	{
		int val = 0, is_negative = 0;
		int i;
		for(i = 0; i < strlen(args); i++)
		{
			//ascii '0' = 48, therefore subtract 48 from each arg
			if(args[i] == '-')
			{
				is_negative = 1;
			}
			else
			{
				int multiplier = 1;
				int j;
				for(j = i + 1; j < strlen(args); j++)
				{
					multiplier *= 10;
				}
				val += ((int)args[i] - 48) * multiplier;
			}
		}
		if(is_negative)
		{
			val *= -1;
		}

		exit_code = val;
	}
	EXIT_FLAG = 1;
}

void HandlePrompt(char* args, int g_arg)
{
	if(strlen(args) > 0)
	{
		strcpy(prompt, args);
	}
	else
	{
		strcpy(prompt, "cwushell");
	}
	exit_code = 0;
}

void HandleCPUInfo(char* args, int g_arg)
{
	int i;
	for(i = 0; i < strlen(args); i++)
	{
		if(args[i] == 'c')
		{
			printf("CPU clock speed:\n");
		}
		if(args[i] == 't')
		{
			printf("CPU type:\n");
		}
		if(args[i] == 'n')
		{
			printf("CPU cores: %d\n", get_nprocs());
		}
	}
}

void HandleMEMInfo(char* args, int g_arg)
{
	struct sysinfo info;
	sysinfo(&info);
	long long totalmem = (info.totalram + info.totalswap) * info.mem_unit;
	printf("Total memory: %lld\n", totalmem);
	long long usedmem = ((info.totalram - info.freeram) + (info.totalswap - info.freeswap)) * info.mem_unit;
	printf("Used memory: %lld\n", totalmem - info.freeram);
	//printf("Memunit: %d\n", info.mem_unit);
	//printf("L2 cache: %ld\n", sysconf(_SC_LEVEL2_CACHE_LINESIZE));
}

struct Command cmd_exit = {HandleExit, "exit", "0123456789", "Usage: <exit> [n] -- terminates the shell. Exits with value of last executed command, or if an argument [n] is specified, exits with value [n]\n"};
struct Command cmd_prompt = {HandlePrompt, "prompt", "*", "Usage: <prompt> [new_prompt] -- changes the current shell prompt to [new_prompt], or restores prompt to \"cwushell\" if no argument is specified\n"};
struct Command cmd_cpuinfo = {HandleCPUInfo, "cpuinfo", "ctn", "Usage: <cpuinfo> -c -t -n -- prints to the screen different CPU related information based on the switch.\n\t-c -- will print the CPU clock\n\t-t -- will print the CPU type\n\t-n -- will print the number of cores\n"};
struct Command cmd_meminfo = {HandleMEMInfo, "meminfo", "tuc", "Usage: <meminfo> -t -u -c --\n"};

int main()
{
	exit_code = 0;
	EXIT_FLAG = 0;
	strcpy(prompt, "cwushell");

	while(!EXIT_FLAG)
	{
		printf("\n%s>",prompt);

		fgets(line, sizeof(line), stdin);
		fflush(stdin);

		int bIsCMDGood = 0, bIsArgGood = 0;
		ParseInput(line, parsed_cmd, parsed_args, &bIsCMDGood, &bIsArgGood);

		//printf("=====\n\tCommand entered: %s\n\tArgs entered: %s\n\t\tcmd was good? %d\n\t\targs are good? %d\n=====\n", parsed_cmd, parsed_args, bIsCMDGood, bIsArgGood);
		//do things with parsed_cmd and parsed_args

		//memset(parsed_cmd, 0, sizeof(parsed_cmd));
		//memset(parsed_args, 0, sizeof(parsed_args));

		HandleCMD(parsed_cmd, parsed_args, bIsCMDGood, bIsArgGood);

		/*
		if(strcmp(parsed_cmd, "exit") == 0)
		{
			EXIT_FLAG = 1;
		}
		*/

		memset(parsed_cmd, 0, sizeof(parsed_cmd));
		memset(parsed_args, 0, sizeof(parsed_args));
	}


	printf("CWUSHELL exited with exit code %d\n", exit_code);
	return 0;
}

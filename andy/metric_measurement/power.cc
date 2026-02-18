#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <ctime>
#include <termios.h>
#include <csignal>
#include <filesystem>
// #include <nvidia-ml/nvml.h> not available on my ubuntu system and didn't want to install dev headers in case it messed with current setup

using namespace std;
namespace fs = filesystem;

bool stopProgram = false;

void signalHandler(int signum) {
    cout << "\nProcess interrupted. Stopping program..." << endl;
    stopProgram = true;
}

char getChar() {
    struct termios oldt, newt;
    char ch;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <output_file>" << endl;
        return 1;
    }

    struct timespec time_start, time_end;
    string outputFile = argv[1];
    ifstream cpuPowerFile;
    int totalCount = 0;
    int sumofPower = 0;
    int Hz = 10; // 
    int pauseTime = 1000000 / Hz; // microseconds

    clock_gettime(CLOCK_REALTIME, &time_start);

    // check can open raply file
    std::ifstream cpuFile("/sys/class/powercap/intel-rapl:0/energy_uj");
    if (!cpuFile) {
        std::cerr << "Cannot open RAPL energy file!\n";
        return 1;
    }

    // take first energy reading
    unsigned long long energy1, energy2;
    cpuFile >> energy1;
    cpuFile.close();

    // Open CSV file for writing
    ofstream logFile(outputFile);
    logFile << "Time (s),GPU Power (mW),CPU Power (mW),Total Power (mW),Average Power (mW),GPU Memory (MiB)" << endl;

    // Setup signal handler for graceful shutdown on Ctrl+C
    signal(SIGINT, signalHandler);

    cout << "Press 'q' to exit the program." << endl;
    while (!stopProgram) {
        if (cin.rdbuf()->in_avail()) {
            char ch = getChar();
            if (ch == 'q') break;
        }

        string str;
        int power = 0;
        int val;
        
        // Read cpu power values from rapl system files
        cpuPowerFile.open("/sys/class/powercap/intel-rapl:0/energy_uj");
        cpuPowerFile >> energy2;
        cpuPowerFile.close();

        double cpuPower = (energy2 - energy1) * 1e-6 * Hz; // µJ → W


        // Read gpu power and memory using nvidia-smi
        FILE* fp = popen("nvidia-smi --query-gpu=power.draw,memory.used --format=csv,noheader,nounits", "r");
        float gpuPower;
        float gpuMemoryMiB;
        fscanf(fp, "%f, %f", &gpuPower, &gpuMemoryMiB);
        pclose(fp);

        if (!cpuPowerFile) {
            cerr << "Error opening power measurement files!" << endl;
            return 1;
        }

        power += gpuPower;
        power += cpuPower;
        sumofPower += power;
        totalCount++;
        double avgPower = sumofPower / totalCount;
        clock_gettime(CLOCK_REALTIME, &time_end);
        double elapsedTime = (time_end.tv_sec - time_start.tv_sec) + (time_end.tv_nsec - time_start.tv_nsec) / 1e9;
        energy1 = energy2; // update energy1 for next iteration

        // Print to console
        
	    // cout << "GPU Power: " << gpuPower << " W\n";
        // cout << "CPU Power: " << cpuPower << " W\n";
        // cout << "Total Power: " << power << " W\n";
        // cout << "Average Power: " << avgPower << " W\n";
        // cout << "Time: " << elapsedTime << " s\n";

        // Write to CSV file
        logFile << elapsedTime << "," << gpuPower << "," << cpuPower << "," << power << "," << avgPower << "," << gpuMemoryMiB << endl;

        usleep(pauseTime); // sleep for 100,000 microsecons = 10Hz

	}
    logFile.close();
    cout << "Logging stopped. Data saved to " << outputFile << endl;
    return 0;
}

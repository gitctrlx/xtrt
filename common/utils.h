/**
 * @file utils.h
 * @brief This file contains utility functions for CUDA programming.
 */

#pragma once

#include <cuda_runtime_api.h>
#include <dirent.h>
#include <fstream>
#include <unordered_map>
#include <string>
#include <sstream>
#include <vector>
#include <cstring>

/**
 * @def CUDA_CHECK
 * @brief Macro to check the status of a CUDA operation and abort if it fails.
 * @param status The status of the CUDA operation.
 */
#define CUDA_CHECK(status) {                                                                                               \
  if (status != 0) {                                                                                                       \
    std::cout << "CUDA failure: " << cudaGetErrorString(status) << " in file " << __FILE__  << " at line "  << __LINE__ << \
        std::endl;                                                                                                         \
    abort();                                                                                                               \
  }                                                                                                                        \
}

/**
 * @brief Read all files in a directory and store their names in a vector.
 * @param p_dir_name The name of the directory.
 * @param file_names The vector to store the file names.
 * @return 0 if successful, -1 if the directory cannot be opened.
 */
static inline int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names) {

    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }   

    closedir(p_dir);
    return 0;
}

/**
 * @brief Trim leading and trailing whitespace from a string.
 * @param str The input string.
 * @return The string with leading and trailing whitespace removed.
 */
static inline std::string trim_leading_whitespace(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

/**
 * @brief Convert a float value to a string with a specified precision.
 * @param a_value The float value to convert.
 * @param n The precision of the output string (default is 2).
 * @return The string representation of the float value.
 */
static inline std::string to_string_with_precision(const float a_value, const int n = 2) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

/**
 * @brief Read labels from a file and store them in an unordered map.
 * @param labels_filename The name of the file containing the labels.
 * @param labels_map The unordered map to store the labels.
 * @return 0 if successful, -1 if the file cannot be opened.
 */
static inline int read_labels(const std::string labels_filename, std::unordered_map<int, std::string>& labels_map) {

    std::ifstream file(labels_filename);
    // Read each line of the file
    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        // Strip the line of any leading or trailing whitespace
        line = trim_leading_whitespace(line);

        // Add the stripped line to the labels_map, using the loop index as the key
        labels_map[index] = line;
        index++;
    }
    // Close the file
    file.close();

    return 0;
}

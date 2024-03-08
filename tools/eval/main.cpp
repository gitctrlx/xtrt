#include "yolo_infer.h"
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <nlohmann/json.hpp>
#include <iostream>
#include <cstring>

/**
 * @brief Returns a vector containing the COCO 91 class IDs corresponding to the COCO 80 class IDs.
 * 
 * @return std::vector<int> - A vector containing the COCO 91 class IDs.
 */
std::vector<int> coco80_to_coco91_class() {
    return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
}

/**
 * @brief Loads image paths from a file.
 * 
 * This function reads a file containing image paths and returns a vector of strings
 * representing the paths.
 * 
 * @param filePath The path to the file containing image paths.
 * @return A vector of strings representing the image paths.
 */
std::vector<std::string> loadImagePaths(const std::string& filePath) {
    std::vector<std::string> paths;
    std::ifstream file(filePath);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            paths.push_back(line);
        }
    }
    return paths;
}

/**
 * @brief The main function of the program.
 * 
 * This function performs evaluation on a set of images using the YOLOInfer class.
 * It loads the engine file, image paths, and coco91 class mapping, and then performs inference on each image.
 * The results are saved to a JSON file.
 * 
 * @return int The exit status of the program.
 */
int main(int argc, char* argv[]) {
    if (argc != 13) {
        std::cerr << "[Usage]: " << argv[0] << "\n"
                  << "File Settings:\n"
                  << "    [engineFile]        : Path to engine file\n"
                  << "    [imagePathFile]     : Path to image path file\n"
                  << "    [cocoValPath]       : Path to COCO validation set\n"
                  << "    [outputJsonPath]    : Path for output JSON\n"

                  << "Preprocess modee:\n"
                  << "    [scaleMethod]       : Scale method (0 for 'LetterBox', 1 for 'Resize')\n"
                  << "    [end2end]           : Use end-to-end processing ('true' or 'false')\n"
                  << "    [preprocess mode]    : Pre processing mode (cpu(0), cpu+gpu(1), gpu(2))\n\n"

                  << "Dynamic shape input shape:\n"
                  << "    [batch_size]         : Batch size for input batchsize (e.g., 1)\n"
                  << "    [channels]           : Number of channels (e.g., 3 for RGB)\n"
                  << "    [height]             : Height of the input images (e.g., 640)\n"
                  << "    [width]              : Width of the input images (e.g., 640)" 
                  << std::endl;
        return -1;
    }

    std::string engineFile = argv[1];
    std::string imagePathFile = argv[2];
    std::string cocoValPath = argv[3];
    std::string outputJsonPath = argv[4];
    int scaleMethodNum = std::stoi(argv[5]);
    bool end2end = std::string(argv[6]) == "true";
    int preprocess_mode = std::stoi(argv[7]);
    int batch_size = std::stoi(argv[8]);
    int channels = std::stoi(argv[9]);
    int height = std::stoi(argv[10]);
    int width = std::stoi(argv[11]);

    auto scaleMethod = scaleMethodNum == 0 ? ScaleMethod::LetterBox : ScaleMethod::Resize;

    // std::string     engineFile = "../../engine/yolo.plan";
    // std::string  imagePathFile = "../../data/coco/filelist.txt";               // Path to data.txt
    // std::string    cocoValPath = "../../data/coco/val2017/";      // Adjusted path
    // std::string outputJsonPath = "results.json";               // JSON output path

    // auto           scaleMethod = ScaleMethod::LetterBox;       // Set to ScaleMethod::LetterBox if you want to use letterbox , otherwise set to ScaleMethod::Resize
    // bool               end2end = false;                        // Set to true if you want to use nms

    // YOLOInfer infer(engineFile, 2, 1, 3, 640, 640);
    YOLOInfer infer(engineFile, preprocess_mode, batch_size, channels, height, width);
    auto imagePaths = loadImagePaths(imagePathFile);
    auto coco91Class = coco80_to_coco91_class();

    nlohmann::json resultsJson;
    for (const auto& imagePath : imagePaths) {
        std::string fullImagePath = cocoValPath + imagePath;
        int imageId = std::stoi(imagePath.substr(0, imagePath.find_last_of('.')));
        auto detections = infer.infer(fullImagePath, scaleMethod, end2end);
        
        for (const auto& det : detections) {
            nlohmann::json detJson;
            detJson["image_id"] = imageId;
            detJson["category_id"] = coco91Class[det.class_id];
            detJson["bbox"] = {det.bbox[0], det.bbox[1], det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]};
            detJson["score"] = det.conf;
            resultsJson.push_back(detJson);
        }
    }

    // Save results to JSON file
    std::ofstream outFile(outputJsonPath);
    outFile << resultsJson.dump(4);

    return 0;
}

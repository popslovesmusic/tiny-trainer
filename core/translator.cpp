#include "Translator.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>

namespace TinyAgentTrainer {

// Default constructor definition
Translator::Translator() {}

// Method to load translation pairs from a file
void Translator::loadPairs(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t separatorPos = line.find('|');
        if (separatorPos != std::string::npos) {
            std::string english = line.substr(0, separatorPos);
            std::string vsl = line.substr(separatorPos + 1);

            // Trim leading/trailing whitespace
            english.erase(0, english.find_first_not_of(" \t\n\r\f\v"));
            english.erase(english.find_last_not_of(" \t\n\r\f\v") + 1);
            vsl.erase(0, vsl.find_first_not_of(" \t\n\r\f\v"));
            vsl.erase(vsl.find_last_not_of(" \t\n\r\f\v") + 1);

            if (!english.empty() && !vsl.empty()) {
                englishToVslMap[english] = vsl;
                vslToEnglishMap[vsl] = english;
            }
        }
    }
    file.close();
}

// Method to translate English to VSL
std::string Translator::translateToVSL(const std::string& english) const {
    auto it = englishToVslMap.find(english);
    if (it != englishToVslMap.end()) {
        return it->second;
    }
    return english;
}

// Method to translate VSL to English
std::string Translator::translateToEnglish(const std::string& vsl) const {
    auto it = vslToEnglishMap.find(vsl);
    if (it != vslToEnglishMap.end()) {
        return it->second;
    }
    return vsl;
}

} // namespace TinyAgentTrainer
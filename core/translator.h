#ifndef TRANSLATOR_H
#define TRANSLATOR_H

#include <string>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace TinyAgentTrainer {

class Translator {
public:
    Translator(); // Explicitly declare the default constructor
    void loadPairs(const std::string& filename);
    std::string translateToVSL(const std::string& english) const;
    std::string translateToEnglish(const std::string& vsl) const;

private:
    std::unordered_map<std::string, std::string> englishToVslMap;
    std::unordered_map<std::string, std::string> vslToEnglishMap;
};

} // namespace TinyAgentTrainer

#endif // TRANSLATOR_H
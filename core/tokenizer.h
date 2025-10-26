#ifndef TINY_AGENT_TRAINER_TOKENIZER_H
#define TINY_AGENT_TRAINER_TOKENIZER_H

#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <sstream>

namespace TinyAgentTrainer {

class Tokenizer {
public:
    // Takes a raw string and returns a vector of clean, lowercase tokens.
    std::vector<std::string> tokenize(const std::string& text) const {
        std::vector<std::string> tokens;
        std::istringstream stream(text);
        std::string word;
        while (stream >> word) {
            // Convert word to lowercase with explicit cast
            std::transform(word.begin(), word.end(), word.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

            // Remove punctuation but keep underscores
            word.erase(std::remove_if(word.begin(), word.end(),
                [](unsigned char c) { return std::ispunct(c) && c != '_'; }),
                word.end());

            if (!word.empty()) {
                tokens.push_back(word);
            }
        }
        return tokens;
    }
};

} // namespace TinyAgentTrainer

#endif // TINY_AGENT_TRAINER_TOKENIZER_H
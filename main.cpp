#include <iostream>
#include <string>
#include <vector>
#include "Translator.h"
#include "Tokenizer.h"

int main() {
    // 1. Create instances of our core classes
    TinyAgentTrainer::Translator translator;
    TinyAgentTrainer::Tokenizer tokenizer;

    // 2. Load the VSL translation data
    std::string filename = "english_vsl.txt";
    try {
        translator.loadPairs(filename);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // 3. Define the English sentence to translate
    std::string englishSentence = "The man is a person.";
    std::cout << "Original English Sentence: \"" << englishSentence << "\"" << std::endl;

    // 4. Tokenize the sentence
    std::vector<std::string> tokens = tokenizer.tokenize(englishSentence);
    std::cout << "Tokenized words: ";
    for (const auto& token : tokens) {
        std::cout << "\"" << token << "\" ";
    }
    std::cout << std::endl;

    // 5. Translate each token to its VSL equivalent (if found)
    std::cout << "Translated to VSL: ";
    bool first = true;
    for (const auto& token : tokens) {
        std::string translated = translator.translateToVSL(token);
        if (!first) {
            std::cout << " "; // Add space between VSL tokens
        }
        std::cout << translated;
        first = false;
    }
    std::cout << std::endl;

    // 6. Demonstrate a more complex, multi-token translation
    std::string complexEnglish = "The ball is red and round.";
    std::cout << "\nOriginal English Sentence: \"" << complexEnglish << "\"" << std::endl;
    tokens = tokenizer.tokenize(complexEnglish);
    std::cout << "Translated to VSL: ";
    first = true;
    for (const auto& token : tokens) {
        std::string translated = translator.translateToVSL(token);
        if (!first) {
            std::cout << " ";
        }
        std::cout << translated;
        first = false;
    }
    std::cout << std::endl;

    return 0;
}
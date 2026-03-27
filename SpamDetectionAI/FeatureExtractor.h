#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <regex>

namespace SpamDetectionAI
{
    class FeatureExtractor
    {
    private:
        std::map<std::string, int> m_vocabulary;
        int m_vocabSize;

        std::vector<std::string> Tokenize(const std::string& text) const;
        void   Normalize(std::vector<double>& features) const;
        double GetUpperRatio(const std::string& text) const;
        int    GetExclamationCount(const std::string& text) const;
        int    GetSpamWordsCount(const std::string& text) const;
        bool   HasURL(const std::string& text) const;
        double GetNormalitzedLenght(const std::string& text) const;

    public:
        FeatureExtractor();
        ~FeatureExtractor() = default;

        void BuildVocabulary(const std::vector<std::string>& texts, int maxVocabSize = 2000);

        int GetFeatureSize() const { return m_vocabSize + 5; }

        std::vector<double> ExtractFeatures(const std::string& text) const;
        std::vector<double> ExtractAdvancedFeatures(const std::string& text) const;
    };

} 
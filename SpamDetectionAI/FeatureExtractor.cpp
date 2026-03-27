#include "FeatureExtractor.h"

namespace SpamDetectionAI
{
    FeatureExtractor::FeatureExtractor() : m_vocabSize(0)
    {
    }

    void FeatureExtractor::BuildVocabulary(const std::vector<std::string>& texts, int maxVocabSize)
    {
        std::map<std::string, int> wordCount;

        for (const auto& text : texts)
        {
            auto tokens = Tokenize(text);
            for (const auto& token : tokens)
                wordCount[token]++;
        }

        std::vector<std::pair<std::string, int>> sortedWords(wordCount.begin(), wordCount.end());
        std::sort(sortedWords.begin(), sortedWords.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        m_vocabSize = (std::min)(maxVocabSize, static_cast<int>(sortedWords.size()));

        for (int i = 0; i < m_vocabSize; i++)
            m_vocabulary[sortedWords[i].first] = i;
    }

    std::vector<std::string> FeatureExtractor::Tokenize(const std::string& text) const
    {
        std::vector<std::string> tokens;
        std::string word;
        std::stringstream ss(text);

        while (ss >> word)
        {
            std::string cleanWord;
            for (char c : word)
            {
                if (std::isalnum(static_cast<unsigned char>(c)))
                    cleanWord += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            if (!cleanWord.empty())
                tokens.push_back(cleanWord);
        }
        return tokens;
    }

   
    std::vector<double> FeatureExtractor::ExtractFeatures(const std::string& text) const
    {
       
        std::vector<double> features(m_vocabSize, 0.0);

        auto tokens = Tokenize(text);
        for (const auto& token : tokens)
        {
            auto it = m_vocabulary.find(token);
            if (it != m_vocabulary.end())
                features[it->second] += 1.0; 
        }

        return features;
    }

    std::vector<double> FeatureExtractor::ExtractAdvancedFeatures(const std::string& text) const
    {
        auto features = ExtractFeatures(text); // BoW vektörü (m_vocabSize boyut)

      
        features.push_back(GetUpperRatio(text));
        features.push_back(GetExclamationCount(text) > 0 ? 1.0 : 0.0);
        features.push_back(GetSpamWordsCount(text) > 0 ? 1.0 : 0.0);
        features.push_back(HasURL(text) ? 1.0 : 0.0);
        features.push_back(GetNormalitzedLenght(text));

        return features; 
    }

    void FeatureExtractor::Normalize(std::vector<double>& features) const
    {
        double sumSquares = 0.0;
        for (double f : features)
            sumSquares += f * f;
        if (sumSquares > 0)
            for (auto& f : features)
                f /= sumSquares;
    }

    double FeatureExtractor::GetUpperRatio(const std::string& text) const
    {
        int upperCount = 0;
        for (char c : text)
            if (std::isupper(static_cast<unsigned char>(c))) ++upperCount;
        return text.length() > 0
            ? static_cast<double>(upperCount) / text.length()
            : 0.0;
    }

    int FeatureExtractor::GetExclamationCount(const std::string& text) const
    {
        return static_cast<int>(std::count(text.begin(), text.end(), '!'));
    }

    int FeatureExtractor::GetSpamWordsCount(const std::string& text) const
    {
        static const std::set<std::string> spamWords = {
            "free", "guaranteed", "cash", "prize", "winner", "winning",
            "giveaway", "miracle", "bonus", "income", "profit", "billion",
            "discount", "bargain", "cheap", "clearance", "deal", "debt",
            "loan", "investment", "urgent", "instant", "limited", "exclusive",
            "offer", "click", "buy", "order", "win", "selected", "member",
            "trial", "opportunity", "apply", "access", "spam", "password",
            "obligation", "confidential", "undisclosed", "congratulations",
            "cures", "unsolicited", "viagra", "valium", "certification",
            "ad", "marketing", "rates", "claims", "warranty", "unlimited"
        };

        int count = 0;
        auto tokens = Tokenize(text);
        for (const auto& token : tokens)
            if (spamWords.count(token))
                ++count;
        return count;
    }

    bool FeatureExtractor::HasURL(const std::string& text) const
    {
        std::regex urlPattern(R"((https?://|www\.)\S+)");
        return std::regex_search(text, urlPattern);
    }

    double FeatureExtractor::GetNormalitzedLenght(const std::string& text) const
    {
        const int maxLength = 1000;
        return std::min(static_cast<double>(text.length()) / maxLength, 1.0);
    }

} 
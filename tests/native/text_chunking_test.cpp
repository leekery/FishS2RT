#include "engine/framework/text/chunking.h"

#include <iostream>
#include <string>
#include <vector>

int main() {
    struct Case {
        const char * name;
        std::string text;
        int64_t budget;
        std::vector<std::string> expected;
    };
    const std::string prefix =
        u8"При измерении производительности видеокарты в этой сцене после загрузки всех ресурсов "
        u8"и завершения прогрева мы получили достаточно стабильный результат без заметных "
        u8"колебаний, а средняя частота равна";
    const std::vector<Case> cases = {
        {"Russian decimal", u8"Значение 61.5 FPS.", 16, {u8"Значение 61.5", "FPS."}},
        {"default budget", prefix + " 61.5 FPS.", 200, {prefix, "61.5 FPS."}},
        {"number exceeds budget", "61.5", 3, {"61.5"}},
        {"decimal before sentence end", "61.5. 74.2!", 6, {"61.5.", "74.2!"}},
        {"integer sentence ends", "61. 74.", 4, {"61.", "74."}},
        {"adjacent sentences", "First.Next.", 6, {"First.", "Next."}},
        {"Fish tag", "<|happy|>61.5 FPS.", 14, {"<|happy|>61.5", "FPS."}},
        {"comma decimal", "61,5 FPS.", 4, {"61,5", "FPS."}},
    };
    int failures = 0;
    for (const auto & test : cases) {
        const auto actual = engine::text::split_text_chunks(
            test.text, test.budget, engine::text::TextChunkMode::TagAware);
        if (actual != test.expected) {
            std::cerr << "FAIL: " << test.name << '\n';
            for (const auto & chunk : actual) {
                std::cerr << '[' << chunk << "]\n";
            }
            ++failures;
        }
    }
    if (failures) {
        return 1;
    }
    std::cout << "PASS: " << cases.size() << " tag-aware chunking cases\n";
    return 0;
}

//
// Created by alex on 1/4/23.
//

#ifndef CPP_TORCH_TOKENIZER_TYPES_H
#define CPP_TORCH_TOKENIZER_TYPES_H

#include <cstdint>
#include <vector>
#include <string>
#include <cstddef>
#include <cstring>

typedef struct Tokenizer_t Tokenizer_t;
typedef struct Encoding_t Encoding_t;

template<typename T>
struct CArrayRef {
    T *ptr = nullptr;
    size_t length = 0;
    CArrayRef() {}
    CArrayRef(T *ptr, size_t length) : ptr(ptr), length(length) {}
    CArrayRef(std::vector<T> &values) : ptr(values.data()), length(values.size()) {}

    T &operator[](size_t index) {
        return ptr[index];
    }

    inline size_t size() const {
        return length;
    }

    inline T *begin() {
        return ptr;
    }

    inline T *end() {
        return ptr + length;
    }

    inline const T *begin() const {
        return ptr;
    }

    inline const T *end() const {
        return ptr + length;
    }

    inline T *data() const {
        return ptr;
    }

    inline uint8_t *bytes() {
        return (uint8_t *) ptr;
    }

    inline size_t nbytes() const {
        return length * sizeof(T);
    }
};

using c_str_t = CArrayRef<const char>;

template<>
class CArrayRef<const char> {
public:
    const char *ptr = nullptr;
    size_t length = 0;
    CArrayRef() = default;
    CArrayRef(const std::string &str) : ptr(str.c_str()), length(str.length()) {}
    CArrayRef(const std::vector<uint8_t> &str) : ptr((const char *) str.data()), length(str.size()) {}
    CArrayRef(const char *ptr, size_t length) : ptr(ptr), length(length) {}
    CArrayRef(const char *str) : ptr(str), length(strlen(str)) {}
    template<size_t N>
    CArrayRef(const char (&str)[N]) : ptr(str), length(N) {}

    const char &operator[](size_t index) {
        return ptr[index];
    }

    inline size_t size() const {
        return length;
    }

    inline const char *begin() {
        return ptr;
    }

    inline const char *end() {
        return ptr + length;
    }

    inline const char *begin() const {
        return ptr;
    }

    inline const char *end() const {
        return ptr + length;
    }

    std::string to_string() {
        return std::string(ptr, length);
    }
};

enum class InputTokenType {
    InputTokenRaw,
    InputTokenRawUtf16,
    InputTokenPreTokenized,
    InputTokenPreTokenizedUtf16,
};

struct InputToken {
    void *ptr = nullptr;
    size_t length = 0;
    InputTokenType input_type = InputTokenType::InputTokenRaw;
    InputToken() {}
    InputToken(void *ptr, size_t length) : ptr(ptr), length(length) {}
    InputToken(const char *ptr, size_t length) : ptr((void *) ptr), length(length) {}
    template<size_t N>
    InputToken(const char (&str)[N]) : ptr((void *) str), length(N) {}

    InputToken(const std::string &str) {
        ptr = (void *) str.c_str();
        length = str.length();
        input_type = InputTokenType::InputTokenRaw;
    }

    InputToken(const std::vector<c_str_t> &vec) {
        ptr = (void *) vec.data();
        length = vec.size();
        input_type = InputTokenType::InputTokenPreTokenized;
    }
};

#endif //CPP_TORCH_TOKENIZER_TYPES_H

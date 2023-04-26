#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>
#include "tokenizer_types.h"


enum class PaddingDirection_t {
    Left,
    Right,
};

struct Offset_t {
    size_t start;
    size_t end;
};

using Allocator_t = void*(*)(size_t size, void *payload);

struct EncodeInput_t {
    InputToken text;
    InputToken pair;
};

struct PaddingParams_t {
    int64_t strategy;
    PaddingDirection_t direction;
    int64_t pad_to_multiple_of;
    uint32_t pad_id;
    uint32_t pad_type_id;
    const char *pad_token;
};


extern "C" {

void encoding_destroy(Encoding_t *encoding);

void encoding_get_attention_mask(Encoding_t *encoding, CArrayRef<uint32_t> *vec);

void encoding_get_ids(Encoding_t *encoding, CArrayRef<uint32_t> *vec);

void encoding_get_offsets(Encoding_t *encoding, CArrayRef<Offset_t> *vec);

void encoding_get_special_tokens_mask(Encoding_t *encoding, CArrayRef<uint32_t> *vec);

void encoding_get_tokens(Encoding_t *encoding, size_t index, c_str_t *str);

size_t encoding_get_tokens_length(Encoding_t *encoding);

void encoding_get_type_ids(Encoding_t *encoding, CArrayRef<uint32_t> *vec);

Tokenizer_t *tokenizer_create_from_file(const char *string);

Tokenizer_t *tokenizer_create_from_pretrained(const char *string,
                                              const char *revision,
                                              const char **user_agent_keys,
                                              const char **user_agent_values,
                                              size_t entry_size,
                                              const char *auto_token);

Tokenizer_t *tokenizer_create_from_wordpiece();

bool tokenizer_decode(Tokenizer_t *tok,
                      bool skip_special_tokens,
                      void *payload,
                      Allocator_t allocator,
                      CArrayRef<uint32_t> *arr,
                      c_str_t *string);

void tokenizer_destroy(Tokenizer_t *tokenizer);

Encoding_t *tokenizer_encode(Tokenizer_t *tok, const char *str, bool add_special_tokens);

void *tokenizer_encode_batch(Tokenizer_t *tok,
                             EncodeInput_t *inputs,
                             size_t num_inputs,
                             bool add_special_tokens);

Encoding_t *tokenizer_encode_dual(Tokenizer_t *tok,
                                  InputToken input1,
                                  InputToken input2,
                                  bool add_special_tokens);

Encoding_t *tokenizer_encode_single(Tokenizer_t *tok, InputToken input, bool add_special_tokens);

float tokenizer_fp16_from_be_bytes(void *value, size_t decimals);

float tokenizer_fp16_from_le_bytes(void *value, size_t decimals);

float tokenizer_fp16_from_ne_bytes(void *value, size_t decimals);

PaddingParams_t tokenizer_get_padding(Tokenizer_t *tokenizer);

void tokenizer_results_destroy(void *result);

void tokenizer_results_get_attention_mask(void *result,
                                          void *payload,
                                          Allocator_t allocator,
                                          CArrayRef<uint32_t> *arr);

void tokenizer_results_get_attention_mask_i64(void *result,
                                              void *payload,
                                              Allocator_t allocator,
                                              CArrayRef<int64_t> *arr);

Encoding_t *tokenizer_results_get_encoding(void *result, size_t index);

void tokenizer_results_get_input_ids(void *result,
                                     void *payload,
                                     Allocator_t allocator,
                                     CArrayRef<uint32_t> *arr);

void tokenizer_results_get_input_ids_i64(void *result,
                                         void *payload,
                                         Allocator_t allocator,
                                         CArrayRef<int64_t> *arr);

void tokenizer_results_get_type_ids(void *result,
                                    void *payload,
                                    Allocator_t allocator,
                                    CArrayRef<uint32_t> *arr);

void tokenizer_results_get_type_ids_i64(void *result,
                                        void *payload,
                                        Allocator_t allocator,
                                        CArrayRef<int64_t> *arr);

bool tokenizer_results_pad(void *result, const PaddingParams_t *padding);

size_t tokenizer_results_size(void *result);

void tokenizer_set_max_length(Tokenizer_t *tokenizer, size_t max_length);

void tokenizer_set_padding(Tokenizer_t *tokenizer, const PaddingParams_t *padding);

} // extern "C"

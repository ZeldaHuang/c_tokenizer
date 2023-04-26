extern crate core;

use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::CStr;
use std::fmt::{Display, Formatter, write};
use std::ptr::{NonNull, null};
use libc::{c_char, c_int, c_void, size_t};
use tokenizers::{AddedToken, EncodeInput, Encoding, FromPretrainedParameters, InputSequence, Offsets, pad_encodings, PaddingDirection, PaddingParams, PaddingStrategy, Token, Tokenizer, truncate_encodings, TruncationDirection, TruncationParams, TruncationStrategy};
use tokenizers::models::bpe::BPE;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::PaddingStrategy::Fixed;

#[repr(C)]
pub struct Offset {
    pub start: usize,
    pub end: usize,
}

/// cbindgen:ignore
#[repr(C)]
pub struct CArrayRef<T> {
    pub ptr: * mut T,
    pub length: usize
}

impl<T> From<&[T]> for CArrayRef<T> {
    fn from(value: &[T]) -> Self {
        CArrayRef {
            ptr: value.as_ptr() as * mut T,
            length: value.len()
        }
    }
}

impl<T: std::clone::Clone> From<CArrayRef<T>> for Vec<T> {
    fn from(value: CArrayRef<T>) -> Self {
        unsafe {
            // copy from ptr to vec
            // Vec::from_raw_parts(value.ptr, value.length, value.length)
            std::slice::from_raw_parts(value.ptr, value.length).to_vec()
        }
    }
}

/// cbindgen:ignore
type c_str_t = CArrayRef<c_char>;

/// cbindgen:ignore
#[repr(C)]
pub enum InputTokenType {
    InputTokenRaw,
    InputTokenRawUtf16,
    InputTokenPreTokenized,
    InputTokenPreTokenizedUtf16,
}

/// cbindgen:ignore
#[repr(C)]
pub struct InputToken {
    pub ptr: *mut c_void,
    pub length: usize,
    pub input_type: InputTokenType,
}

#[repr(C)]
pub struct EncodeInput_t {
    pub text: InputToken,
    pub pair: InputToken,
}

// impl display for inputtoken
impl Display for InputToken {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.input_type {
            InputTokenType::InputTokenRaw => unsafe {
                let slice = unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.length) };
                write!(f, "{}", String::from_utf8_lossy(slice))
            }
            InputTokenType::InputTokenRawUtf16 => {
                let slice = unsafe { std::slice::from_raw_parts(self.ptr as *const u16, self.length) };
                write!(f, "{}", String::from_utf16_lossy(slice))
            }
            InputTokenType::InputTokenPreTokenized => {
                let slice = unsafe { std::slice::from_raw_parts(self.ptr as *const c_str_t, self.length) };
                // write!(f, "{}", slice)
                write!(f, "unsupported")
            }
            InputTokenType::InputTokenPreTokenizedUtf16 => {
                // let slice = unsafe { std::slice::from_raw_parts(self.ptr as *const c_str_t, self.length) };
                // write!(f, "{}", slice)
                write!(f, "unsupported")
            }
        }
    }
}

impl<'s> From<InputToken> for InputSequence<'s> {
    fn from(input: InputToken) -> Self {
        match input.input_type {
            InputTokenType::InputTokenRaw => unsafe {
                let str = unsafe { String::from_utf8_lossy(std::slice::from_raw_parts(input.ptr as *const u8, input.length)) };
                InputSequence::from(str)
            }
            InputTokenType::InputTokenRawUtf16 => {
                let str = unsafe { String::from_utf16_lossy(std::slice::from_raw_parts(input.ptr as *const u16, input.length)) };
                InputSequence::from(str)
            }
            InputTokenType::InputTokenPreTokenized => {
                let mut vec = Vec::new();
                let mut token = input.ptr as *mut c_str_t;
                vec.reserve(input.length);
                for i in 0..input.length {
                    let token = unsafe { &*token.add(i) };
                    let str = unsafe { String::from_utf8_lossy(std::slice::from_raw_parts(token.ptr as *const u8, token.length)) };
                    vec.push(str);
                }
                InputSequence::from(vec)
            }
            InputTokenType::InputTokenPreTokenizedUtf16 => {
                let mut vec = Vec::new();
                let mut token = input.ptr as *mut c_str_t;
                vec.reserve(input.length);
                for i in 0..input.length {
                    let token = unsafe { &*token.add(i) };
                    let str = unsafe { String::from_utf16_lossy(std::slice::from_raw_parts(token.ptr as *const u16, token.length)) };
                    vec.push(str);
                }
                InputSequence::from(vec)
            }
        }
    }
}

impl<'s> From<EncodeInput_t> for EncodeInput<'s> {
    fn from(value: EncodeInput_t) -> Self {
        if value.pair.ptr.is_null() {
            EncodeInput::Single(value.text.into())
        } else {
            EncodeInput::Dual(value.text.into(), value.pair.into())
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_create_from_wordpiece() -> *mut Tokenizer {
    unsafe {
        Box::into_raw(Box::new(Tokenizer::new(WordPiece::default())))
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_create_from_file(string: *const c_char) -> *mut Tokenizer {
    unsafe {
        Box::into_raw(Box::new(Tokenizer::from_file(CStr::from_ptr(string).to_string_lossy().into_owned()).unwrap()))
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_create_from_pretrained(string: *const c_char, revision: *const c_char, user_agent_keys: *mut *const c_char, user_agent_values: *mut *const c_char, entry_size: size_t, auto_token: *const c_char) -> *mut Tokenizer {
    unsafe {
        let mut map = HashMap::new();
        for i in 0..entry_size {
            let key = CStr::from_ptr(*user_agent_keys.add(i)).to_string_lossy().into_owned();
            let value = CStr::from_ptr(*user_agent_values.add(i)).to_string_lossy().into_owned();
            map.insert(key, value);
        }
        let params = Some(if revision.is_null() { FromPretrainedParameters::default() } else {
            FromPretrainedParameters {
                revision: CStr::from_ptr(revision).to_string_lossy().into_owned(),
                user_agent: map,
                auth_token: if auto_token.is_null() { None } else { Some(CStr::from_ptr(auto_token).to_string_lossy().into_owned()) }
            }
        });
        Box::into_raw(Box::new(Tokenizer::from_pretrained(CStr::from_ptr(string).to_string_lossy().into_owned(), params).unwrap()))
    }
}

// destroy
#[no_mangle]
pub extern "C" fn tokenizer_destroy(tokenizer: *mut Tokenizer) {
    unsafe {
        Box::from_raw(tokenizer);
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_encode(tok: *mut Tokenizer, str: *const c_char, add_special_tokens: bool) -> *mut Encoding {
    unsafe {
        let mut tokenizer = Box::from_raw(tok);
        let string = CStr::from_ptr(str).to_string_lossy().into_owned();
        let encoding = tokenizer.encode(string, add_special_tokens);
        Box::into_raw(tokenizer);
        Box::into_raw(Box::new(encoding.unwrap()))
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_encode_batch(tok: *mut Tokenizer, inputs: *mut EncodeInput_t, num_inputs: usize, add_special_tokens: bool) -> *mut c_void {
    // create vec from inputs and num_inputs
    let mut vec = Vec::new();
    for i in 0..num_inputs {
        let input = unsafe { std::ptr::read(inputs.add(i)) };
        vec.push(EncodeInput::from(input));
    }
    unsafe {
        let mut tokenizer = Box::from_raw(tok);
        let mut res = tokenizer.encode_batch(vec, add_special_tokens).unwrap();
        Box::into_raw(tokenizer);
        Box::into_raw(Box::new(res)) as *mut c_void
    }
}

// free results
#[no_mangle]
pub extern "C" fn tokenizer_results_destroy(result: *mut c_void) {
    unsafe {
        Box::from_raw(result as *mut Vec<Encoding>);
    }
}

type Allocator_t = extern "C" fn(size: usize, payload: *mut c_void) -> *mut c_void;

#[no_mangle]
pub extern "C" fn tokenizer_results_get_input_ids(result: *mut c_void, payload: *mut c_void, allocator: Allocator_t, arr: *mut CArrayRef<u32>) {
    unsafe {
        let mut res = Box::from_raw(result as *mut Vec<Encoding>);
        let len = res.iter().fold(0, |acc, x| acc + x.get_ids().len());
        let ptr = allocator(len * std::mem::size_of::<u32>(), payload) as *mut u32;
        let mut data = ptr;
        for item in res.iter() {
            let ids = item.get_ids();
            std::ptr::copy_nonoverlapping(ids.as_ptr(), data, ids.len());
            data = data.add(ids.len());
        }
        Box::into_raw(res);
        arr.write(CArrayRef {
            ptr: ptr,
            length: len,
        });
    }
}

// get type ids
#[no_mangle]
pub extern "C" fn tokenizer_results_get_type_ids(result: *mut c_void, payload: *mut c_void, allocator: Allocator_t, arr: *mut CArrayRef<u32>) {
    unsafe {
        let mut res = Box::from_raw(result as *mut Vec<Encoding>);
        let len = res.iter().fold(0, |acc, x| acc + x.get_type_ids().len());
        let ptr = allocator(len * std::mem::size_of::<u32>(), payload) as *mut u32;
        let mut data = ptr;
        for item in res.iter() {
            let ids = item.get_type_ids();
            std::ptr::copy_nonoverlapping(ids.as_ptr(), data, ids.len());
            data = data.add(ids.len());
        }
        Box::into_raw(res);
        arr.write(CArrayRef {
            ptr: ptr,
            length: len,
        });
    }
}

// get attention mask
#[no_mangle]
pub extern "C" fn tokenizer_results_get_attention_mask(result: *mut c_void, payload: *mut c_void, allocator: Allocator_t, arr: *mut CArrayRef<u32>) {
    unsafe {
        let mut res = Box::from_raw(result as *mut Vec<Encoding>);
        let len = res.iter().fold(0, |acc, x| acc + x.get_attention_mask().len());
        let ptr = allocator(len * std::mem::size_of::<u32>(), payload) as *mut u32;
        let mut data = ptr;
        for item in res.iter() {
            let ids = item.get_attention_mask();
            std::ptr::copy_nonoverlapping(ids.as_ptr(), data, ids.len());
            data = data.add(ids.len());
        }
        Box::into_raw(res);
        arr.write(CArrayRef {
            ptr: ptr,
            length: len,
        });
    }
}


#[no_mangle]
pub extern "C" fn tokenizer_results_get_input_ids_i64(result: *mut c_void, payload: *mut c_void, allocator: Allocator_t, arr: *mut CArrayRef<i64>)  {
    unsafe {
        let mut res = Box::from_raw(result as *mut Vec<Encoding>);
        let len = res.iter().fold(0, |acc, x| acc + x.get_ids().len());
        let ptr = allocator(len * std::mem::size_of::<i64>(), payload) as *mut i64;
        let mut data = ptr;
        for item in res.iter() {
            let ids: &[u32] = item.get_ids();
            for id in ids {
                std::ptr::write(data, *id as i64);
                data = data.add(1);
            }
        }
        Box::into_raw(res);
        arr.write(CArrayRef {
            ptr: ptr,
            length: len,
        });
    }
}

// use packed_simd::*;
// #[no_mangle]
// pub extern "C" fn tokenizer_results_get_input_ids_i64_simd(result: *mut c_void, payload: *mut c_void, allocator: Allocator_t) -> CArrayRef<i64> {
//     unsafe {
//         let mut res = Box::from_raw(result as *mut Vec<Encoding>);
//         let len = res.iter().fold(0, |acc, x| acc + x.get_ids().len());
//         let ptr = allocator(len * std::mem::size_of::<i64>(), payload) as *mut i64;
//         let mut data =  unsafe { std::slice::from_raw_parts(ptr, len) };
//         for item in res.iter() {
//             let ids: &[u32] = item.get_ids();
//             let output = &mut data[0..ids.len()];
//             // let simd_slice = packed_simd::u32x8::from_slice_aligned(ids);
//             // simd_slice.iter()
//             let input_simd = u32x8::from_slice_unaligned(ids);
//             let output_simd = mem::transmute::<u32x2, i64x8>(v);
//             i64x8::write_to_slice_aligned(output_simd, output);
//             // for v in simd_slice {
//             //     let new_v = v.map(|x| x as i64);
//             //     let new_slice = new_v.into_slice();
//             //     data.copy_from_slice(new_slice);
//             //     data = &mut data[new_slice.len()..];
//             // }
//             data = &mut data[new_slice.len()..];
//         }
//         CArrayRef {
//             ptr: ptr,
//             length: len,
//         }    
//     }
// }

// get type ids
#[no_mangle]
pub extern "C" fn tokenizer_results_get_type_ids_i64(result: *mut c_void, payload: *mut c_void, allocator: Allocator_t, arr: *mut CArrayRef<i64>) {
    unsafe {
        let mut res = Box::from_raw(result as *mut Vec<Encoding>);
        let len = res.iter().fold(0, |acc, x| acc + x.get_type_ids().len());
        let ptr = allocator(len * std::mem::size_of::<i64>(), payload) as *mut i64;
        let mut data = ptr;
        for item in res.iter() {
            let ids = item.get_type_ids();
            for id in ids {
                std::ptr::write(data, *id as i64);
                data = data.add(1);
            }
        }
        Box::into_raw(res);
        arr.write(CArrayRef {
            ptr: ptr,
            length: len,
        });
    }
}

// get attention mask
#[no_mangle]
pub extern "C" fn tokenizer_results_get_attention_mask_i64(result: *mut c_void, payload: *mut c_void, allocator: Allocator_t, arr: *mut CArrayRef<i64>) {
    unsafe {
        let mut res = Box::from_raw(result as *mut Vec<Encoding>);
        let len = res.iter().fold(0, |acc, x| acc + x.get_attention_mask().len());
        let ptr = allocator(len * std::mem::size_of::<i64>(), payload) as *mut i64;
        let mut data = ptr;
        for item in res.iter() {
            let ids = item.get_attention_mask();
            for id in ids {
                std::ptr::write(data, *id as i64);
                data = data.add(1);
            }
        }
        Box::into_raw(res);
        arr.write(CArrayRef {
            ptr: ptr,
            length: len,
        });
    }
}


use half::f16;
#[no_mangle]
pub extern "C" fn tokenizer_fp16_from_le_bytes(value: * mut c_void, decimals: usize) -> f32 {
    unsafe {
        let slice = std::slice::from_raw_parts(value as *const u8, 2);
        let array: &[u8; 2] = &slice.try_into().unwrap();
        let value = f16::from_le_bytes(*array);
        let factor = 10.0_f32.powi(decimals as i32);
        (value.to_f32() * factor).round() / factor
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_fp16_from_be_bytes(value: * mut c_void, decimals: usize) -> f32 {
    unsafe {
        let slice = std::slice::from_raw_parts(value as *const u8, 2);
        let array: &[u8; 2] = &slice.try_into().unwrap();
        let value = f16::from_be_bytes(*array);
        let factor = 10.0_f32.powi(decimals as i32);
        (value.to_f32() * factor).round() / factor
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_fp16_from_ne_bytes(value: * mut c_void, decimals: usize) -> f32 {
    unsafe {
        let slice = std::slice::from_raw_parts(value as *const u8, 2);
        let array: &[u8; 2] = &slice.try_into().unwrap();
        let value = f16::from_ne_bytes(*array);
        let factor = 10.0_f32.powi(decimals as i32);
        (value.to_f32() * factor).round() / factor
    }
}

// create PaddingParams_t for c api
#[repr(C)]
pub struct PaddingParams_t {
    pub strategy: i64,
    pub direction: PaddingDirection_t,
    pub pad_to_multiple_of: i64,
    pub pad_id: u32,
    pub pad_type_id: u32,
    pub pad_token: *const c_char,
}

// create from for PaddingParams
impl From<PaddingParams_t> for PaddingParams {
    fn from(params: PaddingParams_t) -> Self {
        PaddingParams {
            strategy: if params.strategy < 0 { PaddingStrategy::BatchLongest } else { PaddingStrategy::Fixed(params.strategy as usize) },
            direction: PaddingDirection::from(params.direction),
            pad_to_multiple_of: if params.pad_to_multiple_of < 0 { None } else { Some(params.pad_to_multiple_of as usize) },
            pad_id: params.pad_id,
            pad_type_id: params.pad_type_id,
            pad_token: unsafe { CStr::from_ptr(params.pad_token).to_string_lossy().into_owned() },
        }
    }
}

impl From<& PaddingParams> for PaddingParams_t {
    fn from(value: & PaddingParams) -> Self {
        PaddingParams_t {
            strategy: match value.strategy {
                PaddingStrategy::BatchLongest => -1,
                PaddingStrategy::Fixed(x) => x as i64,
            },
            direction: match value.direction {
                PaddingDirection::Right => PaddingDirection_t::Right,
                PaddingDirection::Left => PaddingDirection_t::Left,
            },
            pad_to_multiple_of: match value.pad_to_multiple_of {
                None => -1,
                Some(x) => x as i64,
            },
            pad_id: value.pad_id,
            pad_type_id: value.pad_type_id,
            pad_token: value.pad_token.as_ptr() as *const c_char,
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_results_pad(result: *mut c_void, padding: *const PaddingParams_t) -> bool {
    unsafe {
        let mut vec = Box::from_raw(result as *mut Vec<Encoding>);
        let padding = PaddingParams::from(std::ptr::read(padding));
        let res = pad_encodings(&mut vec, &padding);
        Box::into_raw(vec);
        res.is_ok()
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_set_padding(tokenizer: *mut Tokenizer, padding: *const PaddingParams_t) {
    unsafe {
        let mut tokenizer = Box::from_raw(tokenizer);
        let padding = PaddingParams::from(std::ptr::read(padding));
        let res = tokenizer.with_padding(Some(padding));
        Box::into_raw(tokenizer);
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_set_max_length(tokenizer: *mut Tokenizer, max_length: usize) {
    unsafe {
        let mut tokenizer = Box::from_raw(tokenizer);
        if let Some(ref mut padding) = tokenizer.get_padding_mut() {
            padding.strategy = Fixed(max_length);
        } else {
            let mut param = PaddingParams::default();
            param.strategy = Fixed(max_length);
            tokenizer.with_padding(Some(param));
        }
        if let Some(ref mut truncation) = tokenizer.get_truncation_mut() {
            truncation.max_length = max_length;
        } else {
            let mut param = TruncationParams::default();
            param.max_length = max_length;
            tokenizer.with_truncation(Some(param));
        }
        Box::into_raw(tokenizer);
    }
}

// get pad
#[no_mangle]
pub extern "C" fn tokenizer_get_padding(tokenizer: *mut Tokenizer) -> PaddingParams_t {
    unsafe {
        let tokenizer = Box::from_raw(tokenizer);
        if let Some(res) = tokenizer.get_padding() {
            let param = res.into();
            Box::into_raw(tokenizer);
            param
        } else {
            Box::into_raw(tokenizer);
            PaddingParams_t {
                strategy: -1,
                direction: PaddingDirection_t::Right,
                pad_to_multiple_of: -1,
                pad_id: 0,
                pad_type_id: 0,
                pad_token: std::ptr::null(),
            }
        }
    }
}

#[repr(C)]
pub enum TruncationStrategy_t {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}

impl From<TruncationStrategy_t> for TruncationStrategy {
    fn from(strategy: TruncationStrategy_t) -> Self {
        match strategy {
            TruncationStrategy_t::LongestFirst => TruncationStrategy::LongestFirst,
            TruncationStrategy_t::OnlyFirst => TruncationStrategy::OnlyFirst,
            TruncationStrategy_t::OnlySecond => TruncationStrategy::OnlySecond,
        }
    }
}

//     pub direction: TruncationDirection,
//     pub max_length: usize,
//     pub strategy: TruncationStrategy,
//     pub stride: usize,
pub struct TruncateParams_t {
    pub direction: TruncationDirection_t,
    pub max_length: usize,
    pub strategy: TruncationStrategy_t,
    pub stride: usize,
}

impl From<TruncateParams_t> for TruncationParams {
    fn from(params: TruncateParams_t) -> Self {
        TruncationParams {
            direction: TruncationDirection::from(params.direction),
            max_length: params.max_length,
            strategy: TruncationStrategy::from(params.strategy),
            stride: params.stride,
        }
    }
}

// tokenizer_results_truncate
// #[no_mangle]
// pub extern "C" fn tokenizer_results_truncate(result: *mut c_void, truncate: *const TruncateParams_t) -> bool {
//     unsafe {
//         let mut vec = Box::from_raw(result as *mut Vec<Encoding>);
//         let truncate = TruncationParams::from(std::ptr::read(truncate));
//         let res = truncate_encodings(&mut vec, &truncate, &Default::default());
//         Box::into_raw(vec);
//         res.is_ok()
//     }
// }

#[no_mangle]
pub extern "C" fn tokenizer_results_size(result: *mut c_void) -> usize {
    unsafe {
        let result = Box::from_raw(result as *mut Vec<Encoding>);
        let size = result.len();
        Box::into_raw(result);
        size
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_results_get_encoding(result: *mut c_void, index: usize) -> *mut Encoding {
    unsafe {
        let mut result = Box::from_raw(result as *mut Vec<Encoding>);
        let encoding = result.as_mut_ptr().add(index);
        Box::into_raw(result);
        encoding
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_encode_dual(tok: *mut Tokenizer, input1: InputToken, input2: InputToken, add_special_tokens: bool) -> *mut Encoding {
    unsafe {
        let mut tokenizer = Box::from_raw(tok);
        let encoding = tokenizer.encode((input1, input2), add_special_tokens);
        Box::into_raw(tokenizer);
        Box::into_raw(Box::new(encoding.unwrap()))
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_encode_single(tok: *mut Tokenizer, input: InputToken, add_special_tokens: bool) -> *mut Encoding {
    unsafe {
        let mut tokenizer = Box::from_raw(tok);
        let encoding = tokenizer.encode(input, add_special_tokens);
        Box::into_raw(tokenizer);
        Box::into_raw(Box::new(encoding.unwrap()))
    }
}

// tokenizer decode
#[no_mangle]
pub extern "C" fn tokenizer_decode(tok: *mut Tokenizer, skip_special_tokens: bool, payload: *mut c_void, allocator: Allocator_t, arr: *mut CArrayRef<u32>, string: * mut c_str_t) -> bool {
    unsafe {
        let tokenizer = Box::from_raw(tok);
        let value = tokenizer.decode(Vec::from(arr.read()), skip_special_tokens);
        Box::into_raw(tokenizer);
        if let Err(e) = value {
            return false;
        }
        let res = value.unwrap();
        let length = res.bytes().len();
        let ptr = allocator(length, payload);
        std::ptr::copy_nonoverlapping(res.as_ptr(), ptr as *mut u8, length);
        *string = c_str_t {
            ptr: ptr as *mut c_char,
            length: length,
        };
        return true;
    }
}

// encoding get ids
#[no_mangle]
pub extern "C" fn encoding_get_ids(encoding: *mut Encoding, vec: *mut CArrayRef<u32>) {
    unsafe {
        let encoding = Box::from_raw(encoding);
        let ids = encoding.get_ids();
        *vec = CArrayRef::from(ids);
        Box::into_raw(encoding);
    }
}

#[no_mangle]
pub extern "C" fn encoding_get_offsets(encoding: *mut Encoding, vec: *mut CArrayRef<Offset>) {
    unsafe {
        let encoding = Box::from_raw(encoding);
        let offsets = encoding.get_offsets();
        *vec = CArrayRef {
            ptr: offsets.as_ptr() as *mut Offset,
            length: offsets.len(),
        };
        Box::into_raw(encoding);
    }
}

#[no_mangle]
pub extern "C" fn encoding_get_tokens_length(encoding: *mut Encoding) -> usize {
    unsafe {
        let encoding = Box::from_raw(encoding);
        let length = encoding.get_tokens().len();
        Box::into_raw(encoding);
        length
    }
}

#[no_mangle]
pub extern "C" fn encoding_get_tokens(encoding: *mut Encoding, index: usize, str: *mut c_str_t) {
    unsafe {
        let encoding = Box::from_raw(encoding);
        let tokens = encoding.get_tokens();
        *str = c_str_t {
            ptr: tokens[index].as_ptr() as *mut c_char,
            length: tokens[index].len(),
        };
        Box::into_raw(encoding);
    }
}

#[no_mangle]
pub extern "C" fn encoding_get_type_ids(encoding: *mut Encoding, vec: *mut CArrayRef<u32>) {
    unsafe {
        let encoding = Box::from_raw(encoding);
        let type_ids = encoding.get_type_ids();
        *vec = CArrayRef::from(type_ids);
        Box::into_raw(encoding);
    }
}

#[no_mangle]
pub extern "C" fn encoding_get_attention_mask(encoding: *mut Encoding, vec: *mut CArrayRef<u32>) {
    unsafe {
        let encoding = Box::from_raw(encoding);
        let attention_mask = encoding.get_attention_mask();
        *vec = CArrayRef::from(attention_mask);
        Box::into_raw(encoding);
    }
}

#[no_mangle]
pub extern "C" fn encoding_get_special_tokens_mask(encoding: *mut Encoding, vec: *mut CArrayRef<u32>) {
    unsafe {
        let encoding = Box::from_raw(encoding);
        let special_tokens_mask = encoding.get_special_tokens_mask();
        *vec = CArrayRef::from(special_tokens_mask);
        Box::into_raw(encoding);
    }
}

// #[no_mangle]
// pub extern "C" fn encoding_get_overflowing(encoding: *mut Encoding) -> *mut c_void {
//     unsafe {
//         let encoding = Box::from_raw(encoding);
//         let overflowing = encoding.get_overflowing();
//         Box::into_raw(encoding);
//         (overflowing.as_ref()) as *mut c_void
// }

// Create new enum TruncationDirection for c api
#[repr(C)]
pub enum TruncationDirection_t {
    Left,
    Right,
}

impl From<TruncationDirection_t> for TruncationDirection {
    fn from(direction: TruncationDirection_t) -> Self {
        match direction {
            TruncationDirection_t::Left => TruncationDirection::Left,
            TruncationDirection_t::Right => TruncationDirection::Right,
        }
    }
}

// encoding truncate
// #[no_mangle]
// pub extern "C" fn encoding_truncate(encoding: *mut Encoding, max_length: usize, stride: usize, direction: TruncationDirection_t) {
//     unsafe {
//         let mut encoding = Box::from_raw(encoding);
//         encoding.truncate(max_length, stride, TruncationDirection::from(direction));
//         Box::into_raw(encoding);
//     }
// }
//

#[repr(C)]
pub enum PaddingDirection_t {
    Left,
    Right,
}

//create from
impl From<PaddingDirection_t> for PaddingDirection {
    fn from(direction: PaddingDirection_t) -> Self {
        match direction {
            PaddingDirection_t::Left => PaddingDirection::Left,
            PaddingDirection_t::Right => PaddingDirection::Right,
        }
    }
}

// encoding pad
// #[no_mangle]
// pub extern "C" fn encoding_pad(encoding: *mut Encoding, target_length: usize, pad_id: u32, pad_type_id: u32, pad_token: *const c_char, direction: PaddingDirection_t) {
//     unsafe {
//         let mut encoding = Box::from_raw(encoding);
//         let pad_token = CStr::from_ptr(pad_token).to_string_lossy().into_owned();
//         encoding.pad(target_length, pad_id, pad_type_id, pad_token.as_str(), PaddingDirection::from(direction));
//         Box::into_raw(encoding);
//     }
// }

// destory encoding
#[no_mangle]
pub extern "C" fn encoding_destroy(encoding: *mut Encoding) {
    unsafe {
        Box::from_raw(encoding);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut tokenizer = Tokenizer::from_file("/Users/alex/tokenizer_c/cmake-build-debug/zz_training_ending/tokenizer.json").unwrap();

        // load json from file /Users/alex/tokenizer_c/examples/src/string.txt using serde_json
        let json = std::fs::read_to_string("/Users/alex/tokenizer_c/examples/src/string.txt").unwrap();
        let json: serde_json::Value = serde_json::from_str(json.as_str()).unwrap();

        let mut inputs = Vec::new();
        for x in json.as_array().unwrap().iter() {
            // x is string array
            let mut v = x.as_array().unwrap();

            let input = EncodeInput::Dual(InputSequence::from(v[0].as_str().unwrap()),
                                          InputSequence::from(v[1].as_str().unwrap()));
            inputs.push(input);
        }

        println!("{:?}", tokenizer.get_padding().unwrap());
        tokenizer.get_padding_mut().unwrap().strategy = PaddingStrategy::Fixed(384);

        let res = tokenizer.encode_batch(inputs, true).unwrap();



        println!("values:");
        for v in res.iter() {
            // println!("tokens: {:?}", v.get_tokens());
            println!("{:?}, ", v.get_ids());
        }

    }

}

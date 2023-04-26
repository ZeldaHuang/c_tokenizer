cmake_minimum_required(VERSION 3.10)

project(Tokenizer_C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

add_custom_target(
        libtokenizers ALL
        COMMAND cargo build --release --target-dir ${CMAKE_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bindings/c
    )

add_custom_command(
    TARGET libtokenizers POST_BUILD
    #COMMAND cbindgen --config cbindgen.toml --output ${CMAKE_BINARY_DIR}/release/tokenizer.h
    #COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/bindings/c/include/tokenizer_types.h ${CMAKE_BINARY_DIR}/release/tokenizer_types.h
    COMMAND cbindgen --config cbindgen.toml --output ${CMAKE_SOURCE_DIR}/bindings/c/include/tokenizer.h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bindings/c
)
include_directories(bindings/c/include)
link_directories(${CMAKE_BINARY_DIR}/release)
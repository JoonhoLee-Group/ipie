cmake_minimum_required(VERSION 3.4...3.18)
project(lib_wicks)

#target_compile_definitions(lib_wicks PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
add_library(wicks_helper SHARED
    ${PROJECT_SOURCE_DIR}/determinant_utils.c
    ${PROJECT_SOURCE_DIR}/density_matrix.c
    )

set_target_properties(wicks_helper PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    )

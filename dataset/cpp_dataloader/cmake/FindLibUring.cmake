# FindLibUring.cmake

# Find the liburing include directory
find_path(LIBURING_INCLUDE_DIR
  NAMES liburing.h
  PATH_SUFFIXES include
)

# Find the liburing library
find_library(LIBURING_LIBRARY
  NAMES uring
  PATH_SUFFIXES lib lib64
)

# Find the liburing-ffi library
find_library(LIBURING_FFI_LIBRARY
  NAMES uring-ffi
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibUring
  REQUIRED_VARS LIBURING_LIBRARY LIBURING_INCLUDE_DIR
)

if(LIBURING_FOUND)
  set(LIBURING_LIBRARIES ${LIBURING_LIBRARY})
  set(LIBURING_INCLUDE_DIRS ${LIBURING_INCLUDE_DIR})

  if(LIBURING_FFI_LIBRARY)
    list(APPEND LIBURING_LIBRARIES ${LIBURING_FFI_LIBRARY})
  endif()

  if(NOT TARGET LibUring::uring)
    add_library(LibUring::uring UNKNOWN IMPORTED)
    set_target_properties(LibUring::uring PROPERTIES
      IMPORTED_LOCATION "${LIBURING_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${LIBURING_INCLUDE_DIR}"
    )
  endif()

  if(LIBURING_FFI_LIBRARY AND NOT TARGET LibUring::uring-ffi)
    add_library(LibUring::uring-ffi UNKNOWN IMPORTED)
    set_target_properties(LibUring::uring-ffi PROPERTIES
      IMPORTED_LOCATION "${LIBURING_FFI_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${LIBURING_INCLUDE_DIR}"
    )
  endif()
endif()

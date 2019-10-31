#ifndef COMMON_H_
#define COMMON_H_

#include <stdarg.h>  // For va_start, etc.
#include <string.h>
#include <memory>    // For std::unique_ptr

std::string string_format(const std::string fmt_str, ...);

#endif
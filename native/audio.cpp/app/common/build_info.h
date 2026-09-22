#pragma once

#include <iosfwd>

namespace minitts::app {

void print_build_info(std::ostream & out);
void print_build_info_summary(std::ostream & out);
void print_build_info_json(std::ostream & out);

}  // namespace minitts::app

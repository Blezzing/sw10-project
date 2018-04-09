#pragma once

#include <tuple>
#include <memory>

namespace yagal{
    using ir_t = std::tuple<llvm::LLVMContext*, llvm::Module*>;
}
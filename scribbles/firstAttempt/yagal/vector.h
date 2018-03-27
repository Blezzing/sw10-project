#include "llvmHandler.h"
#include "cudaHandler.h"

namespace yagal{
    template <typename T>
    class Vector{
    private:
        //data, der kan tilgås
    public:
        Vector(){

        }

        //funktioner der genererer llvm ir, put llvm logic i llvmHandler.h

        //funktioner der gør noget trivielt, der kræver llvm ir generering

        //the do/execute function, genererer kernel og eksekverer
        void exec(){
            //auto ir = yagal::llvm::generateIr();
            //auto ptx = yagal::llvm::compileIrToPtx(ir);
            auto cudaResult = yagal::cuda::executePtx(NULL/*ptx*/);
        }

        //funktioner til at tilgå data

    };
}
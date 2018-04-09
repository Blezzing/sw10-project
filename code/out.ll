; ModuleID = 'testModule'
source_filename = "testModule"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

define ptx_kernel void @kernel(float addrspace(1)* %vec) {
entry:
  %idx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ptr = getelementptr float, float addrspace(1)* %vec, i32 %idx
  %tmp = load float, float addrspace(1)* %ptr, align 4
  %ret = fadd float %tmp, %tmp
  store float %ret, float addrspace(1)* %ptr, align 4
  ret void
}

attributes #0 = { nounwind readnone }

!nvvm.annotations = !{!0}

!0 = !{void (float addrspace(1)*)* @kernel, !"kernel", i32 1}

import torch, ttnn
HOME=(1,1)
READER=r'''
#include "dataflow_api.h"
void kernel_main(){
 uint32_t sx=get_arg_val<uint32_t>(0),sy=get_arg_val<uint32_t>(1),da=get_arg_val<uint32_t>(2),nb=get_arg_val<uint32_t>(3);
 cb_reserve_back(0,1); noc_async_read(get_noc_addr(sx,sy,da),get_write_ptr(0),nb); noc_async_read_barrier(); cb_push_back(0,1);
}'''
COMPUTE=r'''
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
void kernel_main(){
 cb_wait_front(0,1); cb_reserve_back(16,1);
 init_sfpu(0,16);
 tile_regs_acquire();
 copy_tile_init(0); copy_tile(0,0,0);
 recip_tile_init(); recip_tile(0);
 tile_regs_commit(); tile_regs_wait();
 pack_tile(0,16); tile_regs_release();
 cb_push_back(16,1); cb_pop_front(0,1);
}'''
WRITER=r'''
#include "dataflow_api.h"
void kernel_main(){
 uint32_t sx=get_arg_val<uint32_t>(0),sy=get_arg_val<uint32_t>(1),co=get_arg_val<uint32_t>(2),nb=get_arg_val<uint32_t>(3);
 cb_wait_front(16,1); noc_async_write(get_read_ptr(16),get_noc_addr(sx,sy,co),nb); noc_async_write_barrier(); cb_pop_front(16,1);
}'''
def l1(dev,data=None):
 crs=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME),ttnn.CoreCoord(*HOME))])
 mc=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED,ttnn.BufferType.L1,ttnn.ShardSpec(crs,[32,32],ttnn.ShardOrientation.ROW_MAJOR))
 if data is None: return ttnn.allocate_tensor_on_device(ttnn.Shape([1,1,32,32]),ttnn.float32,ttnn.TILE_LAYOUT,dev,mc)
 return ttnn.from_torch(data.reshape(1,1,32,32).float(),dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=mc)
dev=ttnn.open_device(device_id=0)
try:
 torch.manual_seed(0); data=torch.rand(32,32)+0.5; gold=1.0/data
 da,out=l1(dev,data),l1(dev)
 hp=dev.worker_core_from_logical_core(ttnn.CoreCoord(*HOME)); sx,sy=hp.x,hp.y; NB=32*32*4
 crs=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*HOME),ttnn.CoreCoord(*HOME))])
 def rt(a):
  r=ttnn.RuntimeArgs(); r[HOME[0]][HOME[1]]=a; return r
 cbf=lambda i:ttnn.CBDescriptor(total_size=2*NB,core_ranges=crs,format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i,data_format=ttnn.float32,page_size=NB)])
 ks=lambda s,a,c:ttnn.KernelDescriptor(kernel_source=s,source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,core_ranges=crs,runtime_args=rt(a),compile_time_args=[],config=c)
 prog=ttnn.ProgramDescriptor(kernels=[ks(READER,[sx,sy,da.buffer_address(),NB],ttnn.ReaderConfigDescriptor()),ks(COMPUTE,[],ttnn.ComputeConfigDescriptor()),ks(WRITER,[sx,sy,out.buffer_address(),NB],ttnn.WriterConfigDescriptor())],semaphores=[],cbs=[cbf(0),cbf(16)])
 ttnn.generic_op([da,out],prog)
 got=ttnn.to_torch(out).reshape(32,32); err=float((got-gold).abs().max())
 print(f"recip max_abs_err={err:.2e} -> {'SMOKE_RECIP_OK' if err<1e-2 else 'SMOKE_RECIP_FAIL'}")
finally: ttnn.close_device(dev)

/* Gpu Array Operations
includes scan and sort

 currently, only performs operations with uint64 arrays
 */
module Gpu
{
  import GPU;
  import GpuSort;

  proc times2(ref arr: [] uint) throws {
    var a = arr;
    on here.gpus[0]{
      var b = a;
      foreach i in b.domain {
        b[i] = b[i] * 2;
      }
      a = b;
    }
    return a;
  }

  proc gpuScanWrapper(ref arr: []?t) throws {
    var a = arr;
    on here.gpus[0]{
      var b = a;
      GPU.gpuScan(b);
      a = b;
    }
    return a;
  }

  proc gpuSortWrapper(ref arr: [] uint) throws {
    var a = arr;
    on here.gpus[0]{
      var b = a;
      GpuSort.sort(b);
      a = b;
    }
    return a;
  }

}

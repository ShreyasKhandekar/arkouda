/* Gpu Array Operations
includes scan and sort

 currently, only performs operations with uint64 arrays
 */
module Gpu
{
  use MultiTypeSymEntry;
  import GPU;

  proc times2Class(ref e: SymEntry(?)) {
    on here.gpus[0] {
      foreach i in e.a.domain {
        e.a[i] = e.a[i] * 2;
      }
    }
    writeln("Executing Times 2");
  }

  proc times2(arr: [?D] uint) {

    on here.gpus[0] {
      // var a = arr;
      writeln("Executing where arr is stored, which is locale #", here);
      writeln(arr[0]);

      // var b = arr;
      foreach i in arr.domain {
        arr[i] = arr[i] * 2;
      }
      // arr = b;
    }
    // return arr;
    // return a;
  }

  proc gpuScanWrapper(ref arr: []?t) {
    var a = arr;
    on here.gpus[0]{
      var b = a;
      GPU.gpuScan(b);
      a = b;
    }
    return a;
  }

  proc gpuSortWrapper(ref arr: [] uint) {
    var a = arr;
    on here.gpus[0]{
      var b = a;
      GPU.gpuSort(b);
      a = b;
    }
    return a;
  }

}

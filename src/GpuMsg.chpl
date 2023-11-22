/* Gpu Array Operations
includes scan and sort

 currently, only performs operations with uint64 arrays
 */

module GpuMsg
{
    use ServerConfig;

    use ArkoudaTimeCompat as Time;
    use Math only;
    use Reflection only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use SegmentedString;
    use ServerErrorStrings;

    use Gpu;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const asLogger = new Logger(logLevel, logChannel);


    proc stringtobool(str: string): bool throws {
        if str == "True" then return true;
        else if str == "False" then return false;
        throw getErrorWithContext(
            msg="message: assume_unique must be of type bool",
            lineNumber=getLineNumber(),
            routineName=getRoutineName(),
            moduleName=getModuleName(),
            errorClass="ErrorWithContext");
    }

    /*
    Parse, execute, and respond to a times2 message
    :arg cmd: request command
    :type reqMsg: string
    :arg msgArgs: request arguments
    :type msgArgs: borrowed MessageArgs
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (MsgTuple) response message
    */
    proc times2Msg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message

        var vName = st.nextName(); // symbol table key for resulting array

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("array"), st);

        select gEnt.dtype {
            when DType.UInt64 {
                var entry = toSymEntry(gEnt,uint);

                var aV = times2(entry.a);
                st.addEntry(vName, createSymEntry(aV));

                repMsg = "created " + st.attrib(vName);
                asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // add additional when blocks for different data types...
            otherwise {
                var errorMsg = notImplementedError("times2",gEnt.dtype);
                asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

    /*
    Parse, execute, and respond to a gpuScan message
    :arg cmd: request command
    :type reqMsg: string
    :arg msgArgs: request arguments
    :type msgArgs: borrowed MessageArgs
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (MsgTuple) response message
    */
    proc gpuScanMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message

        var vName = st.nextName(); // symbol table key for resulting array

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("array"), st);

        select gEnt.dtype {
            when DType.UInt64 {
                var entry = toSymEntry(gEnt, uint);

                var aV = gpuScanWrapper(entry.a);
                st.addEntry(vName, createSymEntry(aV));

                repMsg = "created " + st.attrib(vName);
                asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            } when DType.Int64 {
                var entry = toSymEntry(gEnt,int);

                var aV = gpuScanWrapper(entry.a);
                st.addEntry(vName, createSymEntry(aV));

                repMsg = "created " + st.attrib(vName);
                asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            } when DType.Float64 {
                var entry = toSymEntry(gEnt,real);

                var aV = gpuScanWrapper(entry.a);
                st.addEntry(vName, createSymEntry(aV));

                repMsg = "created " + st.attrib(vName);
                asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // add additional when blocks for different data types...
            otherwise {
                var errorMsg = notImplementedError("times2",gEnt.dtype);
                asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

        /*
    Parse, execute, and respond to a gpuSort message
    :arg cmd: request command
    :type reqMsg: string
    :arg msgArgs: request arguments
    :type msgArgs: borrowed MessageArgs
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (MsgTuple) response message
    */
    proc gpuSortMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message

        var vName = st.nextName(); // symbol table key for resulting array

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("array"), st);

        select gEnt.dtype {
            when DType.UInt64 {
                var entry = toSymEntry(gEnt, uint);

                var aV = gpuSortWrapper(entry.a);
                st.addEntry(vName, createSymEntry(aV));

                repMsg = "created " + st.attrib(vName);
                asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // add additional when blocks for different data types...
            otherwise {
                var errorMsg = notImplementedError("times2",gEnt.dtype);
                asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }


    use CommandMap;
    registerFunction("times2", times2Msg, getModuleName());
    registerFunction("gpuScan", gpuScanMsg, getModuleName());
    registerFunction("gpuSort", gpuSortMsg, getModuleName());

}

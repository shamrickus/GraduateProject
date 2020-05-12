from __future__ import absolute_import, print_function
import tvm
from tvm import te
import topi

import numpy as np

def MatMul(blockSize, matrixDim):
    numThreads = blockSize * blockSize
    n = tvm.runtime.convert(matrixDim)
    m, l = n,n
    A = te.placeholder((l, n), name='A')
    B = te.placeholder((l, m), name='B')
    # For summing product results
    k = te.reduce_axis((0, l), name='k')
    C = te.compute(
        (m, n),
        lambda ii, jj: te.sum(A[k, jj] * B[k, ii], axis=k),
        name='C')

    s = te.create_schedule(C.op)
    # At this point, we have our mat mul routine defined and correct
    # however TVM defaults to using serial execution.
    # As a result we need to specify the schedule to properly utilized cuda
    print(tvm.lower(s, [A, B, C], 'cuda', simple_mode=True))
    # We will be reading form A/B -> shared -> local
    A_Shared = s.cache_read(A, "shared", [C])
    A_Local = s.cache_read(A_Shared, "local", [C])
    B_Shared = s.cache_read(B, "shared", [C])
    B_Local = s.cache_read(B_Shared, "local", [C])
    # We will be writing back to C from local
    C_Local = s.cache_write(C, "local")

    # Link the TVM iteration axis to CUDA
    blockIdx = te.thread_axis("blockIdx.x")
    blockIdy = te.thread_axis("blockIdx.y")
    threadIdy = te.thread_axis((0, blockSize), "threadIdx.y")
    threadIdx = te.thread_axis((0, blockSize), "threadIdx.x")


    grid_y, block_y = s[C].split(C.op.axis[0], factor=numThreads)
    grid_x, block_x = s[C].split(C.op.axis[1], factor=numThreads)
    ty, yi2 = s[C].split(block_y, nparts=blockSize)
    tx, xi2 = s[C].split(block_x, nparts=blockSize)
    s[C].bind(grid_y, blockIdy)
    s[C].bind(grid_x, blockIdx)
    s[C].bind(ty, threadIdy)
    s[C].bind(tx, threadIdx)
    # Specify the order in which to traverse the axis'
    # block_y -> block_x -> thread_y -> thread_x
    # This is required because the two split calls defined the order as:
    # block_y -> thread_y -> block_x -> thread_x
    s[C].reorder(grid_y, grid_x, ty, tx, yi2, xi2)

    s[C_Local].compute_at(s[C], tx)

    yo, xo = C_Local.op.axis
    ko, ki = s[C_Local].split(k, factor=8)
    kt, ki = s[C_Local].split(ki, factor=1)
    s[C_Local].reorder(ko, kt, ki, yo, xo)
    s[A_Shared].compute_at(s[C_Local], ko)
    s[B_Shared].compute_at(s[C_Local], ko)
    s[C_Local].unroll(kt)
    s[A_Local].compute_at(s[C_Local], kt)
    s[B_Local].compute_at(s[C_Local], kt)

    # Schedule for A's shared memory load
    ty, block_x = s[A_Shared].split(s[A_Shared].op.axis[0], nparts=blockSize)
    _, block_x = s[A_Shared].split(s[A_Shared].op.axis[1], factor=blockSize * 4)
    tx, block_x = s[A_Shared].split(block_x, nparts=blockSize)
    s[A_Shared].bind(ty, threadIdy)
    s[A_Shared].bind(tx, threadIdx)
    s[A_Shared].vectorize(block_x)

    # Schedule for B' shared memory load
    ty, block_x = s[B_Shared].split(s[B_Shared].op.axis[0], nparts=blockSize)
    _, block_x = s[B_Shared].split(s[B_Shared].op.axis[1], factor=blockSize * 4)
    tx, block_x = s[B_Shared].split(block_x, nparts=blockSize)
    s[B_Shared].bind(ty, threadIdy)
    s[B_Shared].bind(tx, threadIdx)
    s[B_Shared].vectorize(block_x)
    s[A_Shared].double_buffer()
    s[B_Shared].double_buffer()
    # correctness

    device = "cuda"
    ctx = tvm.context(device, 0)
    if not ctx.exist:
        print("Skip because %s is not enabled" % device)
    print("Device %s" % device)
    print(tvm.lower(s, [A, B, C], device, simple_mode=True))
    f = tvm.build(s, [A, B, C], device)
    # launch the kernel.
    n, m, l = matrixDim, matrixDim, matrixDim
    a_np = np.random.uniform(size=(n, l)).astype(A.dtype)
    b_np = np.random.uniform(size=(m, l)).astype(B.dtype)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
    for i in range(2):
        f(a, b, c)
    tvm.testing.assert_allclose(
        c.asnumpy(), np.dot(b_np.T, a_np), rtol=1e-5)

    num_flops = 2 * matrixDim * matrixDim * matrixDim
    num_runs = 10
    timer_f = f.time_evaluator(f.entry_name, ctx, number=num_runs)
    t = timer_f(a, b, c).mean
    GFLOPS = num_flops / (t * 1e3) / 1e6
    dev_module = f.imported_modules[0]
    print(dev_module.get_source())
    print("average time cost of %d runs = %g ms, %g GFLOPS." % (num_runs, t * 1e3, GFLOPS))


if __name__ == "__main__":
    MatMul(16, 2048)



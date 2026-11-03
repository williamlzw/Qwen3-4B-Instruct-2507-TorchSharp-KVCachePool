using Qwen3.Module;
using TorchSharp;
using static System.Runtime.InteropServices.JavaScript.JSType;
using static TorchSharp.torch;

namespace Qwen3;

public class ContinuousBatchingScheduler
{
    private readonly KvSlotAllocator _pool;
    private readonly Queue<GenerationRequest> _waitingQueue = new();
    private readonly List<GenerationRequest> _runningRequests = new();
    private readonly int _maxBatchSize;
    
    private readonly object _lock = new();
    // 【修复 3】：新增计数器，防止新请求饿死
    private int _stepsSinceLastPrefill = 0;

    public ContinuousBatchingScheduler(KvSlotAllocator pool, int maxBatchSize = 16)
    {
        _pool = pool;
        _maxBatchSize = maxBatchSize;
    }

    public void AddRequest(GenerationRequest req)
    {
        lock (_lock) { _waitingQueue.Enqueue(req); }
    }

    public BatchData? GetNextBatch(torch.Device device, ScalarType dtype)
    {
        lock (_lock)
        {
            // 【修复 3】：优化连续批处理逻辑
            // 如果等待队列有请求，且运行队列未满，并且距离上次 Prefill 已经过了 4 个 Decode 步骤
            // 则优先处理一个 Prefill 批次，让新请求尽快开始生成，避免饿死
            if (_waitingQueue.Count > 0 && _runningRequests.Count < _maxBatchSize && _stepsSinceLastPrefill >= 4)
            {
                _stepsSinceLastPrefill = 0;
                return CreatePrefillBatch(device, dtype);
            }
            
            if (_runningRequests.Count > 0)
            {
                _stepsSinceLastPrefill++;
                return CreateDecodeBatch(device, dtype);
            }
           
            if (_waitingQueue.Count > 0)
            {
                _stepsSinceLastPrefill = 0;
                return CreatePrefillBatch(device, dtype);
            }
            return null;
        }
    }

    private BatchData CreateDecodeBatch(torch.Device device, ScalarType dtype)
    {
        int B = _runningRequests.Count;
        var inputIds = new long[B, 1];
        var slots = new long[B];
        var seqLens = new long[B];
        var posIds = new long[B, 1];

        long maxPast = 0;
        for (int i = 0; i < B; i++)
        {
            var req = _runningRequests[i];
            // Decode 阶段输入为上一次生成的最后一个 Token
            inputIds[i, 0] = req.OutputTokens[^1];
            slots[i] = req.SlotId;
            seqLens[i] = req.ProcessedTokens;
            posIds[i, 0] = req.ProcessedTokens;
            maxPast = Math.Max(maxPast, req.ProcessedTokens);
        }

        float maskValue = dtype == ScalarType.Float16 ? -1e4f : -1e9f;

        using var mask_gpu = torch.full(new long[] { B, 1, 1, maxPast + 1 }, maskValue, dtype: ScalarType.Float32, device: device);

        for (int i = 0; i < B; i++)
        {
            int validLen = (int)seqLens[i] + 1;
            if (validLen > 0)
            {
                var target_slice = mask_gpu.select(0, i).select(0, 0).select(0, 0).narrow(0, 0, validLen);
                target_slice.fill_(0.0f);
            }
        }

        return new BatchData
        {
            Type = BatchType.Decode,
            Requests = _runningRequests.ToList(),
            InputIds = torch.tensor(inputIds, dtype: ScalarType.Int64, device: device),
            BatchSlots = torch.tensor(slots, dtype: ScalarType.Int64, device: device),
            SeqLens = torch.tensor(seqLens, dtype: ScalarType.Int64, device: device),
            PositionIds = torch.tensor(posIds, dtype: ScalarType.Int64, device: device),
            AttentionMask = mask_gpu.to(dtype).DetachFromDisposeScope()
        };
    }

    private BatchData? CreatePrefillBatch(torch.Device device, ScalarType dtype)
    {
        var batchReqs = new List<GenerationRequest>();
        int available = _maxBatchSize - _runningRequests.Count;
        if (available <= 0) return null;

        int prefillLimit = Math.Min(available, _waitingQueue.Count);
        while (_waitingQueue.Count > 0 && batchReqs.Count < prefillLimit)
        {
            var req = _waitingQueue.Peek();
            try
            {
                req.SlotId = _pool.Allocate();
                batchReqs.Add(_waitingQueue.Dequeue());
            }
            catch (OutOfMemoryException)
            {
                Console.WriteLine("显存池已满，等待下一步释放");
                break; // 显存池已满，等待下一步释放
            }
        }
        if (batchReqs.Count == 0) return null;

        int B = batchReqs.Count;
        long maxLen = batchReqs.Max(r => r.PromptTokens.Count);
        if (maxLen == 0) maxLen = 1; // 防止 maxLen 为 0 导致张量维度异常
        var inputIds = new long[B, maxLen];
        var slots = new long[B];
        var seqLens = new long[B]; // Prefill 阶段过去长度全为 0
        var posIds = new long[B, maxLen];

        for (int i = 0; i < B; i++)
        {
            var req = batchReqs[i];
            long len = req.PromptTokens.Count;
            for (int j = 0; j < len; j++)
            {
                inputIds[i, j] = req.PromptTokens[j];
                posIds[i, j] = j;
            }
            slots[i] = req.SlotId;
            seqLens[i] = 0;
        }
        // 【修复 2】：统一使用安全的 Mask 值
        float maskValue = dtype == ScalarType.Float16 ? -1e4f : -1e9f;
        // 🚀 性能优化 & Bug修复：直接在 GPU 构建 Causal Mask
        using var mask_gpu = torch.full(new long[] { B, 1, maxLen, maxLen }, maskValue, dtype: ScalarType.Float32, device: device);
        using var causal_2d = torch.full(new long[] { maxLen, maxLen }, maskValue, dtype: ScalarType.Float32, device: device);
        using var causal_mask_2d = torch.triu(causal_2d, diagonal: 1); // 上半部分为 maskValue，下半部分为 0

        for (int i = 0; i < B; i++)
        {
            int len = (int)batchReqs[i].PromptTokens.Count;
            if (len == 0) continue;

            var target_view = mask_gpu.select(0, i).select(0, 0).narrow(0, 0, len).narrow(1, 0, len);
            var source_view = causal_mask_2d.narrow(0, 0, len).narrow(1, 0, len);
            target_view.copy_(source_view);
        }
        // 将请求移入运行队列，状态设为等待第一次 Decode
        foreach (var req in batchReqs)
        {
            req.IsPrefill = true; // 标记此 Batch 为 Prefill 批次
            _runningRequests.Add(req);
        }

        return new BatchData
        {
            Type = BatchType.Prefill,
            Requests = batchReqs,
            InputIds = torch.tensor(inputIds, dtype: ScalarType.Int64, device: device),
            BatchSlots = torch.tensor(slots, dtype: ScalarType.Int64, device: device),
            SeqLens = torch.tensor(seqLens, dtype: ScalarType.Int64, device: device),
            PositionIds = torch.tensor(posIds, dtype: ScalarType.Int64, device: device),
            AttentionMask = mask_gpu.to(dtype).DetachFromDisposeScope()
        };
    }

   

    public void CompleteRequest(GenerationRequest req, Exception? error = null)
    {
        lock (_lock)
        {
            if (!req.TryMarkCompleted())
                return;

            _runningRequests.Remove(req);

            if (req.SlotId >= 0)
            {
                _pool.Free(req.SlotId);
                req.SlotId = -1;
            }

            req.TokenStream.Writer.TryComplete(error);

            if (error is null)
                req.CompletionSource.TrySetResult(true);
            else
                req.CompletionSource.TrySetException(error);
        }
    }
}
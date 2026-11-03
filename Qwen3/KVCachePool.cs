using TorchSharp;
using static TorchSharp.torch;

namespace Qwen3.Module;

public sealed class KvSlotAllocator
{
    private readonly Stack<int> _free;
    private readonly bool[] _allocated;
    private readonly object _gate = new();

    public KvSlotAllocator(int capacity)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        _allocated = new bool[capacity];
        _free = new Stack<int>(capacity);
        for (int i = capacity - 1; i >= 0; i--)
            _free.Push(i);
    }

    public int Allocate()
    {
        lock (_gate)
        {
            if (_free.Count == 0)
                throw new OutOfMemoryException("KV Cache Slot 已耗尽。");
            int slot = _free.Pop();
            if (_allocated[slot])
                throw new InvalidOperationException("Slot 分配器状态损坏。");
            _allocated[slot] = true;
            return slot;
        }
    }

    public void Free(int slot)
    {
        lock (_gate)
        {
            if ((uint)slot >= (uint)_allocated.Length)
                throw new ArgumentOutOfRangeException(nameof(slot));
            if (!_allocated[slot])
                throw new InvalidOperationException($"Slot {slot} 被重复释放。");
            _allocated[slot] = false;
            _free.Push(slot);
        }
    }
}

public class KVCachePool : IDisposable
{
    // 预分配巨大的连续张量：[num_slots, max_seq_len, num_kv_heads, head_dim]
    public Tensor k_cache;
    public Tensor v_cache;

    private readonly Queue<int> _freeSlots;
    public int NumSlots { get; }
    public int MaxSeqLen { get; }

    public KVCachePool(int numSlots, int maxSeqLen, int numKvHeads, int headDim, torch.Device device, torch.ScalarType dtype)
    {
        NumSlots = numSlots;
        MaxSeqLen = maxSeqLen;
        k_cache = torch.zeros(numSlots, maxSeqLen, numKvHeads, headDim, dtype: dtype, device: device);
        v_cache = torch.zeros(numSlots, maxSeqLen, numKvHeads, headDim, dtype: dtype, device: device);
        _freeSlots = new Queue<int>(Enumerable.Range(0, numSlots));
    }

    public int AllocateSlot()
    {
        if (_freeSlots.Count == 0)
            throw new OutOfMemoryException("KV Cache Pool 槽位已耗尽，请增加并发上限或清理请求。");
        return _freeSlots.Dequeue();
    }

    public void FreeSlot(int slotId)
    {
        _freeSlots.Enqueue(slotId);
    }

    // 🌟 核心操作：写入新 Token 并提取当前 Batch 所需的完整 KV 视图
    public (Tensor, Tensor) UpdateAndFetch(Tensor k_new, Tensor v_new, Tensor batch_slots, Tensor seq_lens)
    {
        int batch_size = (int)k_new.shape[0];
        int new_tokens = (int)k_new.shape[1];

        // 1. Scatter: 将新计算的 KV 写入对应的 Slot (C# 循环开销极小，避免了复杂的 scatter_ 算子)
        for (int i = 0; i < batch_size; i++)
        {
            int slot = (int)batch_slots[i].item<long>();
            int seq_len = (int)seq_lens[i].item<long>();

            k_cache[slot].narrow(0, seq_len, new_tokens).copy_(k_new[i]);
            v_cache[slot].narrow(0, seq_len, new_tokens).copy_(v_new[i]);
        }

        // 2. Gather: 提取当前 Batch 需要的 Slot，并截断到最大有效长度，避免无效计算
        int max_seq_len_in_batch = (int)seq_lens.max().item<long>() + new_tokens;
        var batch_k = k_cache.index_select(0, batch_slots).narrow(1, 0, max_seq_len_in_batch);
        var batch_v = v_cache.index_select(0, batch_slots).narrow(1, 0, max_seq_len_in_batch);

        return (batch_k, batch_v);
    }

    public void Dispose() { k_cache.Dispose(); v_cache.Dispose(); }
}
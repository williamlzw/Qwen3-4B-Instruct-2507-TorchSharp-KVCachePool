using System.Diagnostics;
using System.Threading.Channels;
using static TorchSharp.torch;
namespace Qwen3;

public class GenerationRequest
{
    private static int _idCounter = 0;

    public int RequestId { get; } = Interlocked.Increment(ref _idCounter);
    public List<int> PromptTokens { get; set; }
    public List<int> OutputTokens { get; set; } = new();
    public Stopwatch Stopwatch { get; } = new Stopwatch();
    public double TimeToFirstTokenMs { get; set; } = 0.0;
    public int SlotId { get; set; } = -1;
    public int ProcessedTokens { get; set; } = 0; // 已经存入 KV Cache 的 Token 数量
    public bool IsPrefill { get; set; } = true;

    public int MaxOutputTokens { get; set; } = 1024;
    public float Temperature { get; set; } = 0.7f;
    public float TopP { get; set; } = 0.9f;
    public HashSet<int> StopTokens { get; set; } = new();

    // 用于向外层调用者流式推送生成的 Token
    public Channel<int> TokenStream { get; } = Channel.CreateUnbounded<int>();
    public TaskCompletionSource<bool> CompletionSource { get; } =
    new(TaskCreationOptions.RunContinuationsAsynchronously);

    private int _completed;
    public bool TryMarkCompleted()
    {
        return Interlocked.CompareExchange(ref _completed, 1, 0) == 0;
    }
}

public enum BatchType { Prefill, Decode }

public class BatchData : IDisposable
{
    public BatchType Type;
    public List<GenerationRequest> Requests;
    public Tensor InputIds;       // [B, T_new]
    public Tensor BatchSlots;     // [B]
    public Tensor SeqLens;        // [B] (Past lengths)
    public Tensor AttentionMask;  // [B, 1, T_new, T_total]
    public Tensor PositionIds;    // [B, T_new]

    public void Dispose()
    {
        InputIds?.Dispose();
        BatchSlots?.Dispose();
        SeqLens?.Dispose();
        AttentionMask?.Dispose();
        PositionIds?.Dispose();
    }
}
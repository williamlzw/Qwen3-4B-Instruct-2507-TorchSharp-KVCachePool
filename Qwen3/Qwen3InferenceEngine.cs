using Qwen3.Module;
using System.Diagnostics;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace Qwen3;

public class Qwen3InferenceEngine
{
    private readonly Qwen3MoE _model;
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly ContinuousBatchingScheduler _scheduler;
    private readonly torch.Device _device;
    private readonly ScalarType _dtype;
    private Task? _engineLoopTask;
    private CancellationTokenSource _cts = new();

    public Qwen3InferenceEngine(Qwen3MoE model, Qwen3Tokenizer tokenizer, string device, int maxConcurrency = 16, int maxSeqLen = 2048)
    {
        _model = model;
        _tokenizer = tokenizer;
        _device = new torch.Device(device);
        _dtype = ScalarType.Float16;
        // 【修改】为每一层 Transformer 创建独立的 KV 缓存池
        var kvPools = new KVCachePool[model.config.NumHiddenLayers];
        for (int i = 0; i < model.config.NumHiddenLayers; i++)
        {
            kvPools[i] = new KVCachePool(
                maxConcurrency,
                maxSeqLen,
                model.config.NumKeyValueHeads,
                model.config.HeadDim,
                _device,
                _dtype);
        }
        _model.model.KVPools = kvPools;
        // 调度器仅管理 Slot 逻辑编号，复用一个池做 Slot 分配即可
        var slotManagerPool = new KvSlotAllocator(
            maxConcurrency);
        _scheduler = new ContinuousBatchingScheduler(slotManagerPool, maxConcurrency);
       
    }

    public void Start()
    {
        _engineLoopTask = Task.Run(() => EngineLoop(_cts.Token));
    }

    public async IAsyncEnumerable<string> GenerateAsync(string prompt, int maxTokens = 1024, float temp = 0.7f)
    {
        var req = new GenerationRequest
        {
            PromptTokens = _tokenizer.Encode(prompt),
            MaxOutputTokens = maxTokens,
            Temperature = temp,
            StopTokens = new HashSet<int> { 151645, 151644, 151643 }
        };
        req.Stopwatch.Restart();
        _scheduler.AddRequest(req);

        // 异步读取 Token 流并解码输出
        var streamer = new TextStreamer(_tokenizer);
        bool isFirstToken = true;
        try
        {
            await foreach (var token in req.TokenStream.Reader.ReadAllAsync())
            {
                if (isFirstToken)
                {
                    // 🌟 2. 记录首字延迟 (TTFT: Time To First Token)
                    req.Stopwatch.Stop();
                    req.TimeToFirstTokenMs = req.Stopwatch.Elapsed.TotalMilliseconds;

                    // 重新启动计时器，专门测量 Decode 阶段的生成速度
                    req.Stopwatch.Restart();
                    isFirstToken = false;
                }

                streamer.Put(token);
                var text = streamer.GetLatestText();
                if (!string.IsNullOrEmpty(text)) yield return text;
            }

            var finalText = streamer.Flush();
            if (!string.IsNullOrEmpty(finalText)) yield return finalText;
        }
        finally
        {
            req.Stopwatch.Stop();

            // OutputTokens 包含了 Prefill 阶段生成的第1个Token，因此 Decode 阶段生成的 Token 数要减 1
            int decodeTokens = Math.Max(0, req.OutputTokens.Count - 1);
            double decodeSeconds = req.Stopwatch.Elapsed.TotalSeconds;
            double reqTps = decodeSeconds > 0 ? decodeTokens / decodeSeconds : 0;

            Console.WriteLine($"\n📊 [请求 #{req.RequestId} 统计] 首字延迟(TTFT): {req.TimeToFirstTokenMs:F0} ms | " +
                              $"总生成: {req.OutputTokens.Count} tokens | 单请求速度: {reqTps:F2} tokens/s");
            //if (torch.cuda.is_available()) NativeTorchCudaOps.EmptyCudaCache();
        }
        
    }

    private void EngineLoop(CancellationToken token)
    {
        using var no_grad = torch.no_grad();
        var globalSw = Stopwatch.StartNew();
        int tokensGeneratedSinceLastReport = 0;
        double lastReportTime = 0;
        while (!token.IsCancellationRequested)
        {

            BatchData? batch = null;

            // 【核心修复 3】：GetNextBatch 必须放在 try-catch 中！
            // 否则如果调度器内部发生异常，后台线程会直接崩溃，导致进程无声闪退！
            try
            {
                batch = _scheduler.GetNextBatch(_device, _dtype);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n[调度器严重错误] GetNextBatch 崩溃: {ex}");
                Thread.Sleep(100);
                continue;
            }
            
            if (batch == null)
            {
                Thread.Sleep(2);
                continue;
            }
            using var stepScope = torch.NewDisposeScope();
            try
            {
                // 1. 前向传播
                var logits = _model.forward(batch.InputIds, batch.AttentionMask, batch.BatchSlots, batch.SeqLens, batch.PositionIds);
                // 2. 提取需要采样的 Logits
                Tensor targetLogits;
                if (batch.Type == BatchType.Prefill)
                {
                    // Prefill 时，每个请求的真实结束位置不同，需要分别提取最后一个有效 Token 的 Logits
                    var B = batch.Requests.Count;
                    var extractedLogits = new List<Tensor>();
                    for (int i = 0; i < B; i++)
                    {
                        long len = batch.Requests[i].PromptTokens.Count;
                        extractedLogits.Add(logits[i, len - 1, TensorIndex.Colon]);
                    }
                    targetLogits = torch.stack(extractedLogits.ToArray()); // [B, Vocab]
                }
                else
                {
                    // Decode 时，直接取最后一个 Token
                    targetLogits = logits[TensorIndex.Colon, -1, TensorIndex.Colon]; // [B, Vocab]
                }

                // 3. 批量采样 (复用你原有的 top_p_sampling，它天然支持 Batch)
           
                var nextTokens = TopPSampling(targetLogits, batch.Requests);

                if (batch.Type == BatchType.Decode)
                {
                    tokensGeneratedSinceLastReport += batch.Requests.Count;
                }
                // 4. 更新请求状态
                for (int i = 0; i < batch.Requests.Count; i++)
                {
                    var req = batch.Requests[i];
                    int nextToken = (int)nextTokens[i].item<long>();

                    req.OutputTokens.Add(nextToken);
                    req.ProcessedTokens += (batch.Type == BatchType.Prefill ? req.PromptTokens.Count : 1);
                    req.IsPrefill = false;

                    // 推送给客户端
                    req.TokenStream.Writer.TryWrite(nextToken);

                    // 检查终止条件
                    if (req.StopTokens.Contains(nextToken) || req.OutputTokens.Count >= req.MaxOutputTokens)
                    {
                        _scheduler.CompleteRequest(req);
                    }

                    double currentTime = globalSw.Elapsed.TotalSeconds;
                    if (currentTime - lastReportTime >= 1.0)
                    {
                        double globalTps = tokensGeneratedSinceLastReport / (currentTime - lastReportTime);
                        if (globalTps > 0)
                        {
                            //Console.WriteLine($"🚀 [全局吞吐量] {globalTps:F1} tokens/s | 当前并发请求数: {batch.Requests.Count}");
                        }
                        tokensGeneratedSinceLastReport = 0;
                        lastReportTime = currentTime;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Engine Error: {ex}");
                foreach (var req in batch.Requests)
                    _scheduler.CompleteRequest(req, ex);
            }
            finally
            {
                try { batch.Dispose(); } catch { }
            }
        }
        Console.WriteLine("退出循环");
    }

    private static Tensor TopPSampling(
    Tensor logits,
    IReadOnlyList<GenerationRequest> requests)
    {
        using var scope = torch.NewDisposeScope();

        if (logits.shape[0] != requests.Count)
            throw new ArgumentException("Logits batch 与 requests 数量不一致。");

        var temperatures = requests
            .Select(r => Math.Clamp(r.Temperature, 1e-5f, 100f))
            .ToArray();
        var topPs = requests
            .Select(r => Math.Clamp(r.TopP, 0.0f, 1.0f))
            .ToArray();

        using var temperatureTensor = torch.tensor(
            temperatures, dtype: ScalarType.Float32, device: logits.device)
                .unsqueeze(-1);
        using var topPTensor = torch.tensor(
            topPs, dtype: ScalarType.Float32, device: logits.device)
                .unsqueeze(-1);

        using var probs = torch.softmax(
            logits.to(ScalarType.Float32) / temperatureTensor, dim: -1);
        var (sortedProbs, indices) = torch.sort(
            probs, dim: -1, descending: true);

        using var cumsum = torch.cumsum(sortedProbs, dim: -1);
        using var previousCumulative = cumsum - sortedProbs;
        using var removeMask = previousCumulative > topPTensor;
        sortedProbs = sortedProbs.masked_fill(removeMask, 0.0f);

        using var sum = sortedProbs.sum(dim: -1, keepdim: true);
        using var safeSum = torch.where(sum > 0, sum, torch.ones_like(sum));
        using var normalized = sortedProbs / safeSum;
        using var sampledIndex = torch.multinomial(normalized, num_samples: 1);
        using var output = torch.gather(indices, dim: -1, index: sampledIndex);

        return scope.MoveToOuter(output[TensorIndex.Ellipsis, 0]);
    }

    // 你原有的 top_p_sampling 直接搬过来即可，它支持 [B, Vocab] 形状的张量批量处理
    private static Tensor top_p_sampling(
        Tensor logits,
        double top_p = 0.8,
        double temperature = 1.0)
    {
        using var scope = torch.NewDisposeScope();
        temperature = Math.Min(Math.Max(temperature, 1e-5), 1e2);

        // 1. 计算概率
        var probs = torch.softmax(logits.to(torch.float32) / temperature, dim: -1);
        var (sorted_probs, indices) = torch.sort(probs, dim: -1, descending: true);

        // 2. 计算累积概率
        var cumsum = torch.cumsum(sorted_probs, dim: -1);

        // 【核心修复 1】：修复 Top-P 过滤逻辑！
        // 原代码 cumsum > top_p 会把第一个超过 top_p 的 Token 也误杀。
        // 正确做法：(cumsum - sorted_probs) 表示“不包含当前 Token 的累积概率”。
        // 如果不包含当前 Token 的累积概率已经 > top_p，说明当前 Token 及之后的都不需要了。
        var mask = (cumsum - sorted_probs) > top_p;
        sorted_probs.masked_fill_(mask, 0.0f);

        // 3. 重新归一化，防止除以0
        var sum = torch.sum(sorted_probs, dim: -1, keepdim: true);
        sum = torch.where(sum == 0, torch.ones_like(sum), sum);
        sorted_probs = sorted_probs / sum;

        // 4. 采样
        // 使用 multinomial 进行概率采样，而不是 argmax (argmax 会导致永远输出同一个词！)
        var next_token = torch.multinomial(sorted_probs, num_samples: 1);
        var output = torch.gather(indices, dim: -1, index: next_token);

        return scope.MoveToOuter(output[TensorIndex.Ellipsis, 0]);
    }
}

public class TextStreamer
{
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly bool _skipPrompt;
    private readonly List<int> _tokenCache = new List<int>();
    private int _printLen;
    private bool _nextTokensArePrompt = true;
    private readonly StringBuilder _textBuffer = new StringBuilder();

    public TextStreamer(Qwen3Tokenizer tokenizer, bool skipPrompt = false)
    {
        _tokenizer = tokenizer;
        _skipPrompt = skipPrompt;
    }

    public void Put(int token)
    {
        if (_skipPrompt && _nextTokensArePrompt)
        {
            _nextTokensArePrompt = false;
            return;
        }

        _tokenCache.Add(token);
        string text = _tokenizer.Decode(_tokenCache);

        string printableText = null;

        if (text.EndsWith("\n"))
        {
            printableText = text[_printLen..];
            _tokenCache.Clear();
            _printLen = 0;
        }
        else if (text.Length > 0 && IsChineseChar(text[^1]))
        {
            printableText = text[_printLen..];
            _printLen = text.Length;
        }
        else
        {
            int lastSpaceIndex = text.LastIndexOf(' ');
            if (lastSpaceIndex >= _printLen)
            {
                printableText = text.Substring(_printLen, lastSpaceIndex - _printLen + 1);
                _printLen = lastSpaceIndex + 1;
            }
        }

        if (printableText != null)
        {
            _textBuffer.Append(printableText);
        }
    }

    public string GetLatestText()
    {
        if (_textBuffer.Length == 0)
            return null;

        string text = _textBuffer.ToString();
        _textBuffer.Clear();
        return text;
    }

    public string Flush()
    {
        if (_tokenCache.Count > 0)
        {
            string text = _tokenizer.Decode(_tokenCache);
            string printableText = text[_printLen..];
            _tokenCache.Clear();
            _printLen = 0;
            return printableText;
        }

        _nextTokensArePrompt = true;
        return null;
    }

    private static bool IsChineseChar(char c)
    {
        int cp = c;
        return (cp >= 0x4E00 && cp <= 0x9FFF) ||
               (cp >= 0x3400 && cp <= 0x4DBF) ||
               (cp >= 0x20000 && cp <= 0x2A6DF) ||
               (cp >= 0x2A700 && cp <= 0x2B73F) ||
               (cp >= 0x2B740 && cp <= 0x2B81F) ||
               (cp >= 0x2B820 && cp <= 0x2CEAF) ||
               (cp >= 0xF900 && cp <= 0xFAFF) ||
               (cp >= 0x2F800 && cp <= 0x2FA1F);
    }
}

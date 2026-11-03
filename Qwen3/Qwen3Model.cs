using Qwen3.Utils;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Qwen3.Module;

public class Qwen3MoEModel : nn.Module<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>
{
    public readonly Embedding embed_tokens;
    private readonly ModuleList<Qwen3MoEBlock> layers;
    private readonly Qwen3RMSNorm norm;
    public readonly Qwen3RotaryEmbedding rotary_emb;
    // 【修改】从单个池改为数组，每层对应一个独立 KV 缓存池
    public KVCachePool[]? KVPools { get; set; }
    private readonly int num_layers;
    private readonly int num_key_value_heads;
    private readonly int head_dim;

    public Qwen3MoEModel(Qwen3Config config)
        : base(nameof(Qwen3MoEModel))
    {
        this.embed_tokens = nn.Embedding(config.VocabSize, config.HiddenSize);
        
        this.rotary_emb = new Qwen3RotaryEmbedding(32768, config.HeadDim, config.RopeTheta, dtype: config.DType);
        this.layers = new ModuleList<Qwen3MoEBlock>();
        this.num_layers = config.NumHiddenLayers;
        this.num_key_value_heads = config.NumKeyValueHeads;
        this.head_dim = config.HeadDim;
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            this.layers.Add(new Qwen3MoEBlock(config));
        }
        this.norm = new Qwen3RMSNorm(new[] { (long)config.HiddenSize }, config.RmsNormEps);
        
        this.RegisterComponents();
    }

    public override torch.Tensor forward(
    torch.Tensor inputIds,
    torch.Tensor attentionMask,
    torch.Tensor batchSlots,
    torch.Tensor seqLens,
    torch.Tensor positionIds)
    {
        using var scope = NewDisposeScope();
        var embed_output = this.embed_tokens.forward(inputIds);

        // 直接使用传入的 positionIds 计算 RoPE，无需自己 cumsum
        using var freqsCis = this.rotary_emb.forward(positionIds);

        // 【修改】遍历每层时，传入对应层的独立 KV 缓存池
        for (int i = 0; i < this.layers.Count; i++)
        {
            using var scope2 = torch.NewDisposeScope();
            var layerPool = this.KVPools?[i];
            embed_output = this.layers[i].forward(embed_output, attentionMask, freqsCis, layerPool, batchSlots, seqLens);
            scope2.MoveToOuter(embed_output);
        }

        var model_output = this.norm.forward(embed_output);
        return scope.MoveToOuter(model_output);
    }
}


public class Qwen3MoE : nn.Module<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>
{
    public readonly Qwen3Config config;
    public readonly Qwen3MoEModel model;
    private readonly Linear lm_head;

    public Qwen3MoE(Qwen3Config config) : base(nameof(Qwen3MoE))
    {
        this.config = config;
        this.model = new Qwen3MoEModel(config);
        this.lm_head = nn.Linear(config.HiddenSize, config.VocabSize, hasBias: false);
        this.RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor inputIds, torch.Tensor attentionMask,
    torch.Tensor batchSlots,
    torch.Tensor seqLens,
    torch.Tensor positionIds)
    {
        using var scope = NewDisposeScope();
        var model_output = this.model.forward(inputIds, attentionMask, batchSlots, seqLens, positionIds);
        var logits = this.lm_head.forward(model_output);
        return scope.MoveToOuter(logits);
    }
}
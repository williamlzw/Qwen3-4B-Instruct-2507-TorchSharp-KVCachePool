using TorchSharp;
using static TorchSharp.torch;

namespace Qwen3.Module;


public class Qwen3MoEBlock : torch.nn.Module<torch.Tensor, // x
    torch.Tensor?, // attention mask
    IRotary?,
    KVCachePool?,
    torch.Tensor?,
    torch.Tensor?,
    torch.Tensor>
{
    private readonly Qwen3RMSNorm input_layernorm;
    private readonly Qwen3MoeAttention self_attn;
    private readonly Qwen3RMSNorm post_attention_layernorm;
    private readonly Qwen3DenseMLP mlp;

    public Qwen3MoEBlock(
        Qwen3Config config)
        : base(nameof(Qwen3MoEBlock))
    {
        var n_embed = config.HiddenSize;
        var eps = config.RmsNormEps;
        this.input_layernorm = new Qwen3RMSNorm(new[] { (long)n_embed } , eps);
        this.self_attn = new Qwen3MoeAttention(config);
        this.post_attention_layernorm = new Qwen3RMSNorm(new[] { (long)n_embed }, eps);
        this.mlp = new Qwen3DenseMLP(config);
        this.RegisterComponents();
    }

    public override Tensor forward(Tensor x, Tensor? attentionMask, IRotary? freqsCis, KVCachePool? kv_pool, Tensor? batch_slots, Tensor? seq_lens)
    {
        using var scope = NewDisposeScope();
        var input_norm_out = this.input_layernorm.forward(x);
        var attn_output = x + this.self_attn.forward(input_norm_out, attentionMask, freqsCis, kv_pool, batch_slots, seq_lens);
        var post_norm_out = this.post_attention_layernorm.forward(attn_output);
        var output = attn_output + this.mlp.forward(post_norm_out);
        return scope.MoveToOuter(output);
    }
}
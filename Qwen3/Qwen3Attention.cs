using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using Qwen3.Utils;
using System.Data.Common;

namespace Qwen3.Module;

public class Qwen3MoeAttention : nn.Module<torch.Tensor, // x
    torch.Tensor?, // attention mask
    IRotary?,
    KVCachePool?,
    torch.Tensor,
    torch.Tensor, 
    torch.Tensor>
{
    private readonly int n_embed;
    private readonly int n_heads;
    private readonly int n_kv_heads;
    private readonly int head_dim;
    private readonly int n_groups;
    private readonly Linear o_proj;
    private readonly Linear q_proj;
    private readonly Linear k_proj;
    private readonly Linear v_proj;
    private readonly Qwen3RMSNorm q_norm;
    private readonly Qwen3RMSNorm k_norm;

    public Qwen3MoeAttention(
        Qwen3Config config)
        : base(nameof(Qwen3MoeAttention))
    {
        this.n_heads = config.NumAttentionHeads;
        this.n_kv_heads = config.NumKeyValueHeads;
        this.n_embed = config.HiddenSize;
        this.head_dim = config.HeadDim;
        this.n_groups = (int)(config.NumAttentionHeads / config.NumKeyValueHeads);
        this.q_proj = nn.Linear(this.n_embed, this.n_heads * this.head_dim, hasBias: false);
        this.k_proj = nn.Linear(this.n_embed, this.n_kv_heads * this.head_dim, hasBias: false);
        this.v_proj = nn.Linear(this.n_embed, this.n_kv_heads * this.head_dim, hasBias: false);
        this.o_proj = nn.Linear(this.n_heads * this.head_dim, this.n_embed, hasBias: false);
        this.q_norm = new Qwen3RMSNorm(new[] { (long)this.head_dim }, config.RmsNormEps);
        this.k_norm = new Qwen3RMSNorm(new[] { (long)this.head_dim }, config.RmsNormEps);
        this.RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor x, Tensor? attention_mask,
        IRotary? freqs_cis,
        KVCachePool? kv_pool, Tensor? batch_slots, Tensor? seq_lens)
    {
        using var scope = NewDisposeScope();
        var (B, T, C) = x.shape;

        var q = this.q_proj.forward(x);
        var k = this.k_proj.forward(x);
        var v = this.v_proj.forward(x);

        q = q.view(B, T, this.n_heads, this.head_dim);
        k = k.view(B, T, this.n_kv_heads, this.head_dim);
        v = v.view(B, T, this.n_kv_heads, this.head_dim);
        q = this.q_norm.forward(q);
        k = this.k_norm.forward(k);

        if (freqs_cis is not null)
            (q, k) = freqs_cis.apply(q, k);

        Tensor k_attn, v_attn;
        if (kv_pool is not null && batch_slots is not null && seq_lens is not null)
        {
            // 使用缓存池进行读写
            (k_attn, v_attn) = kv_pool.UpdateAndFetch(k, v, batch_slots, seq_lens);
        }
        else
        {
            k_attn = k; v_attn = v;
        }


        q = q.permute(0, 2, 1, 3); // [B, n_heads, T, D]
        k_attn = k_attn.permute(0, 2, 1, 3); // [B, n_kv_heads, T_total, D]
        v_attn = v_attn.permute(0, 2, 1, 3);
        if (this.n_kv_heads < this.n_heads)
        {
            k_attn = RepeatKV(k_attn, this.n_groups);
            v_attn = RepeatKV(v_attn, this.n_groups);
        }

        // 【核心修复 2】：强制对齐 attention_mask 和 q 的 dtype！
        // SDPA 要求 attn_mask 的 dtype 必须与 query 严格一致，否则会报 "invalid dtype for bias"
        Tensor? aligned_mask = attention_mask;
        if (aligned_mask is not null && aligned_mask.dtype != q.dtype)
        {
            aligned_mask = aligned_mask.to(q.dtype);
        }

        var y = torch.nn.functional.scaled_dot_product_attention(
            q, k_attn, v_attn,
            attn_mask: aligned_mask,
            p: 0.0,
            is_casual: false
        );

        y = y.permute(0, 2, 1, 3).reshape(B, T, -1);
        y = this.o_proj.forward(y);
        return scope.MoveToOuter(y);
    }

    private Tensor RepeatKV(Tensor x, int nRep)
    {
        using var scope = NewDisposeScope();
        var batchSize = x.shape[0];
        var nKVHeads = x.shape[1];
        var seqLen = x.shape[2];
        var headDim = x.shape[3];
        if (nRep == 1)
        {
            return x;
        }
        var output = x.unsqueeze(2)
                .expand(batchSize, nKVHeads, nRep, seqLen, headDim)
                .reshape(batchSize, nKVHeads * nRep, seqLen, headDim);
        return scope.MoveToOuter(output);
    }
}
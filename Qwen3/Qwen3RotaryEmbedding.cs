using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using Qwen3.Utils;

public interface IRotary : IDisposable
{
    public (Tensor, Tensor) apply(Tensor query, Tensor key);

    public IEnumerable<Tensor> weights { get; }
}

public class Qwen3RotaryEmbedding : nn.Module<Tensor, IRotary>
{
    private Tensor cos;
    private Tensor sin;
    public static (Tensor cos, Tensor sin) precompute_freqs_cis(long dim, long length, double theta = 10000.0)
    {
        using var scope = torch.NewDisposeScope();
        if (dim % 2 != 0)
            throw new Exception("dim should be multiple of 2");
        var freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32) / dim);
        freqs = torch.outer(torch.arange(length).to(torch.float32), freqs);
        freqs = torch.cat(new[] { freqs, freqs }, dim: -1);
        return scope.MoveToOuter(torch.cos(freqs), torch.sin(freqs));
    }

    public static Tensor apply_rotary_emb(Tensor x, (Tensor cos, Tensor sin) freqs_cis)
    {
        using var scope = torch.NewDisposeScope();
        var half_dim = x.shape.Last() / 2;
        var (x_r, x_i) = torch.chunk(x, 2, dim: -1);
        var rotated = torch.cat(new[] { -x_i, x_r }, dim: -1);
        return scope.MoveToOuter(x * freqs_cis.cos + rotated * freqs_cis.sin);
    }

    public class SimpleRotary : IRotary
    {
        public Tensor cos { get; }
        public Tensor sin { get; }
        public IEnumerable<Tensor> weights => new[] { cos, sin };

        public SimpleRotary(Tensor cos, Tensor sin) { this.cos = cos; this.sin = sin; }

        public (Tensor, Tensor) apply(Tensor query, Tensor key) => (
            apply_rotary_emb(query, (cos, sin)),
            apply_rotary_emb(key, (cos, sin))
        );

        public void Dispose() { } // Buffers are managed by the Module
    }


    public Qwen3RotaryEmbedding(
        long num_embeddings,
        long embedding_dims,
        double theta = 10000.0,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("Qwen3RotaryEmbedding")
    {
        var (c, s) = precompute_freqs_cis(embedding_dims, num_embeddings, theta);
        if (dtype is not null) { c = c.to(dtype.Value); s = s.to(dtype.Value); }

        // 🌟 核心优化：注册为 Buffer，随 model.to(device) 自动移动设备
        this.register_buffer("cos", c);
        this.register_buffer("sin", s);

        if (device is not null)
        {
            this.cos = this.cos.to(device);
            this.sin = this.sin.to(device);
        }
        RegisterComponents();
    }

    public override IRotary forward(Tensor x)
    {
        var (n_batch, n_seq) = x.shape;
        return new SimpleRotary(
            this.cos[x].view(n_batch, n_seq, 1, -1),
            this.sin[x].view(n_batch, n_seq, 1, -1)
        );
    }
}
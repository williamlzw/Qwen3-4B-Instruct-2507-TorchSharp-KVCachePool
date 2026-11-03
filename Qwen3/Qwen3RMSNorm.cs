using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Qwen3.Module;

public class Qwen3RMSNorm : nn.Module<Tensor, Tensor>
{
    public Parameter weight;
    public double eps;

    public Qwen3RMSNorm(
        long[] normalized_shape,
        double eps = 1e-5,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("Qwen3RMSNorm")
    {
        weight = nn.Parameter(torch.ones(normalized_shape, dtype: dtype, device: device));
        this.eps = eps;

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        var h = x.to(torch.float32);
        var norm = h * torch.rsqrt(h.pow(2).mean(new[] { -1L }, keepdim: true) + (float)eps);
        return scope.MoveToOuter(norm.type_as(x) * weight);
    }
}
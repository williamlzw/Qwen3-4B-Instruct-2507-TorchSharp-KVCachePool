using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Qwen3.Module;

public class Qwen3DenseMLP : torch.nn.Module<Tensor, Tensor>
{
    private readonly Linear gate_proj;
    private readonly Linear up_proj;
    private readonly Linear down_proj;

    public Qwen3DenseMLP(Qwen3Config config)
        : base(nameof(Qwen3DenseMLP))
    {
        this.gate_proj = nn.Linear(config.HiddenSize, config.IntermediateSize, hasBias: false);
        this.up_proj = nn.Linear(config.HiddenSize, config.IntermediateSize, hasBias: false);
        this.down_proj = nn.Linear(config.IntermediateSize, config.HiddenSize, hasBias: false);
        this.RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var scope = NewDisposeScope();
        var input1 = this.gate_proj.forward(input);
        var input2 = torch.nn.functional.silu(input1);
        var input3 = input2 * this.up_proj.forward(input);
        var output = this.down_proj.forward(input3);
        return scope.MoveToOuter(output);
    }
}

public class Qwen3MoEMLP : torch.nn.Module<Tensor, Tensor>
{
    private readonly Linear gate;
    private readonly int num_experts_per_tok;
    private readonly int num_experts;
    private readonly ModuleList<nn.Module> experts;

    public Qwen3MoEMLP(Qwen3Config config)
        : base(nameof(Qwen3MoEMLP))
    {
        this.num_experts = (int)config.NumExperts;
        this.num_experts_per_tok = (int)config.NumExpertsPerTok;
        this.gate = nn.Linear(config.HiddenSize, this.num_experts, hasBias: false);
        this.experts = nn.ModuleList<nn.Module>();
        for(int i = 0; i < this.num_experts; i++)
        {
            experts.add_module($"{i}.gate_proj", nn.Linear(config.HiddenSize, (long)config.MoeIntermediateSize, hasBias:false));
            experts.add_module($"{i}.up_proj", nn.Linear(config.HiddenSize, (long)config.MoeIntermediateSize, hasBias: false));
            experts.add_module($"{i}.down_proj", nn.Linear((long)config.MoeIntermediateSize, config.HiddenSize, hasBias: false));
        }
        this.RegisterComponents();
    }

    private Linear GetExpertGateProj(int expertIndex)
    {
        // 每个专家有3个模块，所以索引计算为: expertIndex * 3
        return this.experts[expertIndex * 3] as Linear;
    }

    private Linear GetExpertUpProj(int expertIndex)
    {
        return this.experts[expertIndex * 3 + 1] as Linear;
    }

    private Linear GetExpertDownProj(int expertIndex)
    {
        return this.experts[expertIndex * 3 + 2] as Linear;
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var shape = x.shape;
        var b = shape[0];
        var seqLen = shape[1];
        var embedDim = shape[2];
        var scores = this.gate.forward(x);
        var (topkScores, topKIndices) = torch.topk(scores, this.num_experts_per_tok, dim: -1);
        var topkProbs = torch.softmax(topkScores, dim: -1);
        var expertOutputs = new List<torch.Tensor>();
        for(int i = 0; i<this.num_experts; i++)
        {
            var gate_proj = GetExpertGateProj(i);
            var up_proj = GetExpertUpProj(i);
            var down_proj = GetExpertDownProj(i);
            var hidden = torch.nn.functional.silu(gate_proj.forward(x)) * up_proj.forward(x);
            var outTensor = down_proj.forward(hidden);
            expertOutputs.Add(outTensor.unsqueeze(-2));
        }
        var expertRet = torch.cat(expertOutputs, dim: -2);
        var gatingProbs = torch.zeros_like(scores);
        for(int j = 0; j < this.num_experts_per_tok; j++)
        {
            var indices = topKIndices[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(j, j + 1)];
            var prob = topkProbs[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(j, j + 1)];
            gatingProbs.scatter_(dim: -1, indices, prob);
        }
        gatingProbs = gatingProbs.unsqueeze(-1);
        var y = (gatingProbs * expertRet).sum(dim: -2);
        return scope.MoveToOuter(y);
    }
}
using System.Text.Json;
using System.Text.Json.Serialization;
using static TorchSharp.torch;

namespace Qwen3;
public class Qwen3Config
{
    public Qwen3Config()
    {
        this.VocabSize = 151936;
        this.HiddenSize = 2560;
        this.RmsNormEps = 1e-6f;
        this.IntermediateSize = 9728;
        this.HeadDim = 128;
        this.NumHiddenLayers = 36;
        this.NumAttentionHeads = 32;
        this.NumKeyValueHeads = 8;
        this.RopeTheta = 5000000.0;
        this.NumExpertsPerTok = null;
        this.NumExperts = null;
        this.MoeIntermediateSize = null;
        this.DType = ScalarType.Float16;
    }


    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; set; }

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; set; }

    [JsonPropertyName("rms_norm_eps")]
    public float RmsNormEps { get; set; }

    [JsonPropertyName("intermediate_size")]
    public int IntermediateSize { get; set; }

    [JsonPropertyName("num_hidden_layers")]
    public int NumHiddenLayers { get; set; }

    [JsonPropertyName("num_attention_heads")]
    public int NumAttentionHeads { get; set; }

    [JsonPropertyName("head_dim")]
    public int HeadDim { get; set; }

    [JsonPropertyName("num_key_value_heads")]
    public int NumKeyValueHeads { get; set; }

    [JsonPropertyName("rope_theta")]
    public double RopeTheta { get; set; }

    [JsonPropertyName("bos_token_id")]
    public int BosTokenId { get; set; }

    [JsonPropertyName("eos_token_id")]
    public int EosTokenId { get; set; }

    [JsonPropertyName("num_experts_per_tok")]
    public int? NumExpertsPerTok { get; set; }

    [JsonPropertyName("num_experts")]
    public int? NumExperts { get; set; }

    [JsonPropertyName("moe_intermediate_size")]
    public int? MoeIntermediateSize { get; set; }

    public ScalarType DType { get; set; }
}

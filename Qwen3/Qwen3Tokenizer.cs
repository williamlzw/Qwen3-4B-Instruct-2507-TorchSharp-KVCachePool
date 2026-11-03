using Tokenizers.DotNet;
namespace Qwen3;

public class Qwen3Tokenizer
{
    private Tokenizer _tokenizer;

    public Qwen3Tokenizer(string vocabPath)
    {
        _tokenizer = new Tokenizers.DotNet.Tokenizer(vocabPath);

    }

    public List<int> Encode(string text)
    {
        var encoding = _tokenizer.Encode(text).Select(x => (int)x);
        return new List<int>(encoding);
    }

    public string Decode(List<int> ids)
    {
        return _tokenizer.Decode(ids.Select(x => (uint)x).ToArray());
    }
}

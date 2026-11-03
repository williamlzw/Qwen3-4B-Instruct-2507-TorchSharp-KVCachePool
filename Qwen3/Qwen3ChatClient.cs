using Microsoft.Extensions.AI;
namespace Qwen3;

class Qwen3MoEChatClient
{
    private readonly Qwen3ChatTemplateBuilder _chatTemplateBuilder;
    private readonly Qwen3InferenceEngine _pipeline;

    public Qwen3MoEChatClient(Qwen3InferenceEngine pipeline)
    {
        _chatTemplateBuilder = Qwen3ChatTemplateBuilder.Instance;
        _pipeline = pipeline;
        _pipeline.Start();
        
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null)
    {
        var prompt = _chatTemplateBuilder.BuildPrompt(messages, options);
        string responseId = Guid.NewGuid().ToString("N");
       
        await foreach (var output in  _pipeline.GenerateAsync(prompt,
            maxTokens: options?.MaxOutputTokens ?? 1024,
            temp: options?.Temperature ?? 0.7f))
        {
            yield return new(ChatRole.Assistant, output)
            {
                CreatedAt = DateTime.UtcNow,
                ResponseId = responseId,
            };
        }
    }
}
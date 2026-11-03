using System.Text;
using Microsoft.Extensions.AI;

namespace Qwen3;

public class Qwen3ChatTemplateBuilder
{
    private const char Newline = '\n';

    public static Qwen3ChatTemplateBuilder Instance { get; } = new Qwen3ChatTemplateBuilder();

    public string BuildPrompt(IEnumerable<ChatMessage> messages, ChatOptions? options = null, bool appendAssistantTag = true)
    {
        var availableRoles = new[] { ChatRole.System, ChatRole.User, ChatRole.Assistant };
        if (messages.Any(m => m.Text is null))
        {
            throw new InvalidOperationException("Please provide a message with content.");
        }

        if (messages.Any(m => availableRoles.Any(availableRole => availableRole == m.Role) == false))
        {
            throw new InvalidOperationException("Please provide a message with a valid role. The valid roles are System, User, and Assistant.");
        }

        var sb = new StringBuilder();
        foreach (var message in messages)
        {
            var role = message.Role.Value;
            var content = message.Text;
            sb.Append(message switch
            {
                _ when message.Role == ChatRole.System => $"<|im_start|>system{Newline}{content}<|im_end|>{Newline}",
                _ when message.Role == ChatRole.User => $"<|im_start|>user{Newline}{content}<|im_end|>{Newline}",
                _ when message.Role == ChatRole.Assistant => $"<|im_start|>assistant{Newline}{content}<|im_end|>{Newline}",
                _ => throw new InvalidOperationException("Invalid role.")
            });
        }

        if (appendAssistantTag)
        {
            sb.Append($"<|im_start|>assistant{Newline}");
        }

        var input = sb.ToString();

        return input;
    }
}
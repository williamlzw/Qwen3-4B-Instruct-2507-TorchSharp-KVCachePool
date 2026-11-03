using Microsoft.Extensions.AI;
using TorchSharp;
using static TorchSharp.torch;
using Qwen3;
using Qwen3.Module;
using System.Text.Json;
using System.Text;

public class Program
{
    public async static Task TestQwen3()
    {
        Console.OutputEncoding = Encoding.UTF8;
        torch.InitializeDeviceType(DeviceType.CUDA);
        var defaultType = ScalarType.Float16;
        torch.set_default_dtype(defaultType);
        string device = "cuda";
        var weightFolder = @"G:\model\Qwen3-4B-Instruct-2507";

        var modelConfigPath = Path.Join(weightFolder, "config.json");
        var modelConfig = JsonSerializer.Deserialize<Qwen3Config>(File.ReadAllText(modelConfigPath));
        var model = new Qwen3MoE(modelConfig);
        //foreach (var index in model.state_dict())
        //{
        //    Console.WriteLine($"{index.Key}");
        //}
        model = model.half();
        model.load("G:\\model\\Qwen3-4B-Instruct-2507\\Qwen3-4B-Instruct-2507-fp16.dat", strict: false);
        model = model.to(device);

        var tokenizer = new Qwen3Tokenizer(Path.Combine(weightFolder, "tokenizer.json"));
        Console.WriteLine("load model success");
        var pipeline = new Qwen3InferenceEngine(model, tokenizer, device);
        var client = new Qwen3MoEChatClient(pipeline);

        // 初始化对话历史和配置
        List<ChatMessage> _chatHistory = new();
        var options = new ChatOptions
        {
            MaxOutputTokens = 1024,
            Temperature = 0.7f
        };

        Console.WriteLine("=====================================");
        Console.WriteLine(" Qwen3 交互式对话已启动");
        Console.WriteLine(" 输入 'exit' 或 'quit' 结束对话");
        Console.WriteLine("=====================================");

        // 核心改造：开启无限循环等待控制台输入
        while (true)
        {
            Console.Write("\nUser: ");
            string userInput = Console.ReadLine();

            // 过滤空输入
            if (string.IsNullOrWhiteSpace(userInput))
            {
                continue;
            }

            // 退出指令判断
            if (userInput.Equals("exit", StringComparison.OrdinalIgnoreCase) ||
                userInput.Equals("quit", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("正在退出对话...再见！");
                break;
            }
            // 将用户的当前问题加入历史
            _chatHistory.Add(new ChatMessage(ChatRole.User, userInput));
            // 1. 【新增】滑动窗口：强制保留最近 4 轮对话，防止 Token 撑爆显存和 RoPE
            int max_history_turns = 4;
            while (_chatHistory.Count > max_history_turns * 2)
            {
                int removeIndex = _chatHistory[0].Role == ChatRole.System ? 1 : 0;
                if (_chatHistory.Count - removeIndex >= 2)
                {
                    _chatHistory.RemoveAt(removeIndex); // 移除旧的 User
                    _chatHistory.RemoveAt(removeIndex); // 移除旧的 Assistant
                }
                else break;
            }

            Console.Write("Assistant: ");
            StringBuilder assistantResponseBuilder = new StringBuilder();

            // 2. 【新增】Try-Catch：捕获底层崩溃，防止无声卡死
            try
            {
                await foreach (var response in client.GetStreamingResponseAsync(_chatHistory, options))
                {
                    Console.Write(response.Text);
                    assistantResponseBuilder.Append(response.Text);
                }
                Console.WriteLine();

                // 将完整回复加入历史
                _chatHistory.Add(new ChatMessage(ChatRole.Assistant, assistantResponseBuilder.ToString()));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n[系统错误] 生成中断: {ex.Message}");
                Console.WriteLine("提示：可能是上下文过长或显存不足，已自动清理上一条提问。");

                // 发生错误时，回滚最后一条导致崩溃的 User 输入，防止死循环
                if (_chatHistory.Count > 0 && _chatHistory[^1].Role == ChatRole.User)
                {
                    _chatHistory.RemoveAt(_chatHistory.Count - 1);
                }
            }
        }
    }
    
    public static async Task Main()
    {
        await TestQwen3();
    }
}
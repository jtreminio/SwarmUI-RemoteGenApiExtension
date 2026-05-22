namespace RemoteGenApi.Tests;

internal static class Fixtures
{
    public const string DefaultServerUrl = "http://localhost:8000/generate";

    public static T2IParamInput BuildInput(
        string serverUrl = DefaultServerUrl,
        string prompt = "a unit test prompt",
        string negativePrompt = "",
        long seed = 7,
        int width = 512,
        int height = 512,
        int steps = 20,
        double cfg = 4.0,
        float? timeoutSeconds = null)
    {
        _ = WorkflowTestHarness.ExtensionSteps();
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.Prompt, prompt);
        input.Set(T2IParamTypes.NegativePrompt, negativePrompt);
        input.Set(T2IParamTypes.Seed, seed);
        input.Set(T2IParamTypes.Width, width);
        input.Set(T2IParamTypes.Height, height);
        input.Set(T2IParamTypes.Steps, steps);
        input.Set(T2IParamTypes.CFGScale, cfg);
        if (serverUrl is not null)
        {
            input.Set(ExtensionClass.ServerUrl, serverUrl);
        }
        if (timeoutSeconds.HasValue)
        {
            input.Set(ExtensionClass.TimeoutSeconds, timeoutSeconds.Value);
        }
        return input;
    }
}

using ExtensionClass = SwarmRemoteGenApiExtension.RemoteGenApiExtension;

namespace RemoteGenApi.Tests;

internal static class WorkflowTestHarness
{
    private static readonly object LockObj = new();
    private static bool _initialized;
    private static List<WorkflowGenerator.WorkflowGenStep> _coreSteps = [];
    private static List<WorkflowGenerator.WorkflowGenStep> _extensionSteps = [];

    public static readonly WorkflowGenerator.WorkflowGenStep SyntheticSaveStep = new(g =>
    {
        if (g.CurrentMedia is null)
        {
            return;
        }
        g.CurrentMedia.SaveOutput(g.CurrentVae, g.CurrentAudioVae, ExtensionClass.SaveImageNodeId);
    }, 10);

    private static void EnsureInitialized()
    {
        lock (LockObj)
        {
            if (_initialized)
            {
                return;
            }

            if (T2IParamTypes.Width is null)
            {
                T2IParamTypes.RegisterDefaults();
            }

            List<WorkflowGenerator.WorkflowGenStep> before = [.. WorkflowGenerator.Steps];

            ExtensionClass extension = new();
            extension.OnInit();

            List<WorkflowGenerator.WorkflowGenStep> after = [.. WorkflowGenerator.Steps];
            _coreSteps = before;
            _extensionSteps = after.Where(step => !before.Contains(step)).ToList();
            WorkflowGenerator.Steps = before;

            if (_extensionSteps.Count == 0)
            {
                throw new InvalidOperationException("RemoteGenApi did not register any WorkflowGenerator steps during init.");
            }

            _initialized = true;
        }
    }

    public static IReadOnlyList<WorkflowGenerator.WorkflowGenStep> ExtensionSteps()
    {
        EnsureInitialized();
        return _extensionSteps;
    }

    public static IEnumerable<WorkflowGenerator.WorkflowGenStep> DefaultSteps()
    {
        EnsureInitialized();
        return [.. _extensionSteps, SyntheticSaveStep];
    }

    public static JObject GenerateWith(
        T2IParamInput input,
        IEnumerable<WorkflowGenerator.WorkflowGenStep> steps = null,
        IEnumerable<string> features = null)
    {
        EnsureInitialized();

        List<WorkflowGenerator.WorkflowGenStep> priorSteps = [.. WorkflowGenerator.Steps];
        try
        {
            IEnumerable<WorkflowGenerator.WorkflowGenStep> effective = steps ?? DefaultSteps();
            WorkflowGenerator.Steps = [.. effective.OrderBy(step => step.Priority)];
            input.ApplyLateSpecialLogic();

            WorkflowGenerator generator = new()
            {
                UserInput = input,
                Features = features is null ? [ExtensionClass.FeatureFlag, "comfy_saveimage_ws"] : [.. features],
                ModelFolderFormat = "/"
            };

            return generator.Generate();
        }
        finally
        {
            WorkflowGenerator.Steps = priorSteps;
        }
    }
}

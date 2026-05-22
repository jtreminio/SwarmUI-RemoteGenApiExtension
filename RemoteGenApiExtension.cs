using System.IO;
using RemoteGenApiExtension.Generated;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmRemoteGenApiExtension;

public class RemoteGenApiExtension : Extension
{
    public const string FeatureFlag = "remote_gen_api";
    public const double BaseStepPriority = -8.5;
    public const double MediaOverridePriority = -4.9;
    public const double PrunePriority = 150;

    public static T2IRegisteredParam<string> ServerUrl;
    public static T2IRegisteredParam<float> TimeoutSeconds;
    public static T2IParamGroup RemoteGenApiGroup;

    public override void OnInit()
    {
        Logs.Info("SwarmUI Remote Gen API Extension initializing...");
        ComfyTyped.Generated.NodeRegistrations.EnsureRegistered();
        NodeRegistrations.EnsureRegistered();

        string nodeFolder = Path.GetFullPath(Path.Join(FilePath, "comfy_node"));
        ComfyUISelfStartBackend.CustomNodePaths.Add(nodeFolder);
        Logs.Init($"SwarmUI Remote Gen API: added {nodeFolder} to ComfyUI CustomNodePaths");

        ComfyUIBackendExtension.FeaturesSupported.UnionWith([FeatureFlag]);
        ComfyUIBackendExtension.FeaturesDiscardIfNotFound.UnionWith([FeatureFlag]);
        ComfyUIBackendExtension.NodeToFeatureMap[SwarmRemoteGenApiNode.ClassType] = FeatureFlag;

        RemoteGenApiGroup = new T2IParamGroup(
            Name: "Remote Gen API",
            Toggles: true,
            Open: false,
            IsAdvanced: false,
            OrderPriority: -5.0,
            Description: "Bypass local generation and forward the prompt to a remote generation API server. The server is expected to return a base64-encoded image."
        );

        ServerUrl = T2IParamTypes.Register<string>(new T2IParamType(
            Name: "Remote Gen API URL",
            Description: "Full URL of the remote generation API endpoint. The endpoint must accept POST JSON {\"prompt\": \"...\"} and " +
                "respond with JSON containing a base64-encoded image under one of: 'image', 'image_base64', or 'data'.",
            Default: "http://localhost:8000/generate",
            Group: RemoteGenApiGroup,
            FeatureFlag: FeatureFlag,
            OrderPriority: 1
        ));

        TimeoutSeconds = T2IParamTypes.Register<float>(new T2IParamType(
            Name: "Remote Gen API Timeout",
            Description: "Timeout in seconds for the remote generation API call.",
            Default: "120",
            Min: 1, Max: 3600, Step: 1,
            Group: RemoteGenApiGroup,
            FeatureFlag: FeatureFlag,
            OrderPriority: 2
        ));

        WorkflowGenerator.AddStep(Runner.InsertRemoteGenApiBase, BaseStepPriority);
        WorkflowGenerator.AddStep(Runner.OverrideCurrentMedia, MediaOverridePriority);
        WorkflowGenerator.AddStep(Runner.PruneOrphanedNodes, PrunePriority);
    }
}

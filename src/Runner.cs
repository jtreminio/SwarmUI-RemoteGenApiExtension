using ComfyTyped.Core;
using ComfyTyped.Generated;
using ComfyTyped.SwarmUI;
using RemoteGenApiExtension.Generated;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmRemoteGenApiExtension;

public static class Runner
{
    private static string TryGetActiveServerUrl(WorkflowGenerator g)
    {
        if (!g.UserInput.TryGet(RemoteGenApiExtension.ServerUrl, out string serverUrl) || string.IsNullOrWhiteSpace(serverUrl))
        {
            return null;
        }
        return serverUrl.Trim();
    }

    public static void InsertRemoteGenApiBase(WorkflowGenerator g)
    {
        string serverUrl = TryGetActiveServerUrl(g);
        if (serverUrl is null)
        {
            return;
        }
        if (!Uri.TryCreate(serverUrl, UriKind.Absolute, out Uri parsed)
            || (parsed.Scheme != Uri.UriSchemeHttp && parsed.Scheme != Uri.UriSchemeHttps))
        {
            throw new SwarmUserErrorException($"Remote Gen API URL is not a valid http(s) URL: '{serverUrl}'");
        }

        using SyncingWorkflowBridge bridge = BridgeSync.For(g);
        string nodeId = g.GetStableDynamicID(78000, 0);
        bridge.AddNode(new SwarmRemoteGenApiNode(), nodeId).With(
            ServerUrl: serverUrl,
            Prompt: g.UserInput.Get(T2IParamTypes.Prompt, "").Trim(),
            NegativePrompt: g.UserInput.Get(T2IParamTypes.NegativePrompt, "").Trim(),
            Width: g.UserInput.GetImageWidth(),
            Height: g.UserInput.GetImageHeight(),
            Seed: g.UserInput.Get(T2IParamTypes.Seed, -1),
            Steps: g.UserInput.Get(T2IParamTypes.Steps, 20),
            Cfg: g.UserInput.Get(T2IParamTypes.CFGScale, 4.0),
            Thinking: g.UserInput.Get(RemoteGenApiExtension.Thinking, false),
            TimeoutSeconds: g.UserInput.Get(RemoteGenApiExtension.TimeoutSeconds, 120));

        Logs.Debug($"SwarmUI Remote Gen API: inserted remote node '{serverUrl}' as id '{nodeId}'");
    }

    public static void OverrideCurrentMedia(WorkflowGenerator g)
    {
        if (TryGetActiveServerUrl(g) is null)
        {
            return;
        }
        using SyncingWorkflowBridge bridge = BridgeSync.For(g);
        SwarmRemoteGenApiNode remote = bridge.Graph.NodesOfType<SwarmRemoteGenApiNode>().FirstOrDefault();
        if (remote is null)
        {
            return;
        }
        g.CurrentMedia = remote.Image.ToWGNodeData(g, WGNodeData.DT_IMAGE);
    }

    public static void PruneOrphanedNodes(WorkflowGenerator g)
    {
        if (TryGetActiveServerUrl(g) is null)
        {
            return;
        }

        using SyncingWorkflowBridge bridge = BridgeSync.For(g);

        List<ComfyNode> saves =
        [
            .. bridge.Graph.NodesOfType<SwarmSaveImageWSNode>(),
            .. bridge.Graph.NodesOfType<SaveImageNode>(),
        ];
        if (saves.Count == 0)
        {
            return;
        }

        HashSet<string> reachable = [];
        Queue<ComfyNode> pending = new();
        foreach (ComfyNode save in saves)
        {
            if (reachable.Add(save.Id))
            {
                pending.Enqueue(save);
            }
        }
        while (pending.Count > 0)
        {
            ComfyNode current = pending.Dequeue();
            foreach (ComfyNode upstream in bridge.Graph.FindUpstream(current))
            {
                if (reachable.Add(upstream.Id))
                {
                    pending.Enqueue(upstream);
                }
            }
        }

        List<string> toRemove = [.. bridge.Graph.Nodes.Keys.Where(id => !reachable.Contains(id))];
        foreach (string id in toRemove)
        {
            bridge.RemoveNode(id);
        }
        if (toRemove.Count > 0)
        {
            Logs.Debug($"SwarmUI Remote Gen API: pruned {toRemove.Count} orphaned node(s): {string.Join(", ", toRemove)}");
        }
    }
}

using ExtensionClass = SwarmRemoteGenApiExtension.RemoteGenApiExtension;

namespace RemoteGenApi.Tests;

[Collection("RemoteGenApiTests")]
public class RemoteGenApiWorkflowTests
{
    [Fact]
    public void Save_image_node_consumes_remote_image_output()
    {
        T2IParamInput input = Fixtures.BuildInput();
        JObject workflow = WorkflowTestHarness.GenerateWith(input);

        using WorkflowBridge bridge = WorkflowBridge.Create(workflow);
        SwarmSaveImageWSNode save = bridge.Graph.GetNode<SwarmSaveImageWSNode>(ExtensionClass.SaveImageNodeId);
        Assert.NotNull(save);

        Assert.Equal(ExtensionClass.RemoteGenApiNodeId, save.Images.Connection?.Node.Id);
        Assert.Equal(0, save.Images.Connection?.SlotIndex);
    }

    [Fact]
    public void Override_current_media_restores_remote_after_simulated_sampler()
    {
        T2IParamInput input = Fixtures.BuildInput();

        WorkflowGenerator.WorkflowGenStep fakeSampler = new(g =>
        {
            string fakeId = g.CreateNode("FakeKSampler", new JObject()
            {
                ["latent_image"] = g.CurrentMedia?.Path,
            });
            g.CurrentMedia = new WGNodeData([fakeId, 0], g, WGNodeData.DT_LATENT_IMAGE, g.CurrentCompat());
        }, -5);

        JObject workflow = WorkflowTestHarness.GenerateWith(
            input,
            steps: [.. WorkflowTestHarness.ExtensionSteps(), fakeSampler, WorkflowTestHarness.SyntheticSaveStep]);

        Assert.Null(workflow.Properties().FirstOrDefault(p => p.Value.Value<string>("class_type") == "FakeKSampler"));

        using WorkflowBridge bridge = WorkflowBridge.Create(workflow);
        SwarmSaveImageWSNode save = bridge.Graph.GetNode<SwarmSaveImageWSNode>(ExtensionClass.SaveImageNodeId);
        Assert.NotNull(save);
        Assert.Equal(ExtensionClass.RemoteGenApiNodeId, save.Images.Connection?.Node.Id);
    }

    [Fact]
    public void Prune_removes_orphan_nodes_unconnected_to_save()
    {
        T2IParamInput input = Fixtures.BuildInput();

        WorkflowGenerator.WorkflowGenStep seedOrphan = new(g =>
        {
            g.Workflow["666"] = new JObject
            {
                ["class_type"] = "OrphanStub",
                ["inputs"] = new JObject()
            };
        }, -1000);

        JObject workflow = WorkflowTestHarness.GenerateWith(
            input,
            steps: [seedOrphan, .. WorkflowTestHarness.DefaultSteps()]);

        Assert.Null(workflow["666"]);
        Assert.NotNull(workflow[ExtensionClass.RemoteGenApiNodeId]);
        Assert.NotNull(workflow[ExtensionClass.SaveImageNodeId]);
    }

    [Fact]
    public void Prune_keeps_chain_connected_to_save()
    {
        T2IParamInput input = Fixtures.BuildInput();

        WorkflowGenerator.WorkflowGenStep insertPassthrough = new(g =>
        {
            if (g.CurrentMedia is null)
            {
                return;
            }
            string passId = g.CreateNode("PassthroughImage", new JObject()
            {
                ["image"] = g.CurrentMedia.Path,
            });
            g.CurrentMedia = g.CurrentMedia.WithPath([passId, 0]);
        }, -4);

        JObject workflow = WorkflowTestHarness.GenerateWith(
            input,
            steps: [.. WorkflowTestHarness.ExtensionSteps(), insertPassthrough, WorkflowTestHarness.SyntheticSaveStep]);

        Assert.NotNull(workflow[ExtensionClass.RemoteGenApiNodeId]);
        Assert.NotNull(workflow[ExtensionClass.SaveImageNodeId]);
        JProperty passthrough = workflow.Properties().FirstOrDefault(p => p.Value.Value<string>("class_type") == PassthroughImageNode.ClassType);
        Assert.NotNull(passthrough);
        Assert.Equal(passthrough.Name, ((JArray)workflow[ExtensionClass.SaveImageNodeId]["inputs"]["images"])[0].Value<string>());
    }
}

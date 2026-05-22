namespace RemoteGenApi.Tests;

[CollectionDefinition("RemoteGenApiTests")]
public class RemoteGenApiTestsCollection : ICollectionFixture<GlobalStateFixture>
{
}

public sealed class GlobalStateFixture : IDisposable
{
    private readonly List<WorkflowGenerator.WorkflowGenStep> _workflowSteps;

    public GlobalStateFixture()
    {
        WorkflowTestHarness.ExtensionSteps();
        _workflowSteps = [.. WorkflowGenerator.Steps];
    }

    public void Dispose()
    {
        WorkflowGenerator.Steps = [.. _workflowSteps];
    }
}

# Experimental YAML Runtime Configuration

This SDK includes an experimental CrewAI-style YAML runtime in `strands.experimental.yaml_config`.

## Overview

Use `@StrandsYAMLBase` on a class to:

1. Load YAML config files from predictable relative paths
2. Discover decorated factories for tools, agents, agents-as-tools, graphs, and model aliases
3. Resolve symbolic references from YAML into runtime objects
4. Expose resolved maps on the instance (`self.tools`, `self.agents`, `self.graphs`)

Default relative paths:

- `config/agents.yaml`
- `config/tools.yaml`
- `config/graphs.yaml`

## Example YAML

`config/tools.yaml`:

```yaml
search_tool: build_search_tool
path_tool:
  value: tests.fixtures.say_tool:say
```

`config/agents.yaml`:

```yaml
researcher:
  name: Researcher
  model: literal-model-id
  tools:
    - search_tool

writer:
  name: Writer
  model: model_from_factory
  tools:
    - path_tool
  agent_tools:
    - researcher
```

`config/graphs.yaml`:

```yaml
editorial_graph:
  nodes:
    research_node: researcher
    writing_node:
      agent: writer
  edges:
    - from: research_node
      to: writing_node
  entry_points:
    - research_node
```

## Example Python Pattern

```python
from strands.experimental.yaml_config import StrandsYAMLBase, yaml_model, yaml_tool
from strands.models.bedrock import BedrockModel
from tests.fixtures.say_tool import say


@StrandsYAMLBase
class EditorialRuntime:
    agents_config = "config/agents.yaml"
    tools_config = "config/tools.yaml"
    graphs_config = "config/graphs.yaml"

    @yaml_tool
    def build_search_tool(self):
        return say

    @yaml_model
    def model_from_factory(self):
        return BedrockModel(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
```

## Conceptual Result After Resolution

- `self.tools["search_tool"]` -> resolved tool instance from `build_search_tool`
- `self.tools["path_tool"]` -> literal string tool path retained as-is
- `self.agents["writer"]` -> agent instance with:
  - `model` resolved from `model_from_factory`
  - `tools` including both `path_tool` and `researcher.as_tool()`
- `self.graphs["editorial_graph"]` -> `Graph` instance where nodes map to resolved agents

## Instantiation and Run

```python
runtime = EditorialRuntime()
graph = runtime.graphs["editorial_graph"]
result = graph("Draft a short overview of deterministic multi-agent orchestration")
print(result.status)
```

## Validation Behavior

The runtime raises explicit errors for:

- Missing YAML files
- Empty or non-mapping YAML files
- Unknown tool/agent/graph references
- Cyclic agent-as-tool references
- Cyclic graph references

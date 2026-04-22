"""Tests for experimental CrewAI-style YAML runtime configuration."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from strands.experimental.yaml_config import (
    StrandsYAMLBase,
    yaml_agent_tool,
    yaml_model,
    yaml_tool,
)
from strands.models.bedrock import BedrockModel
from strands.multiagent.graph import Graph
from tests.fixtures.say_tool import say


@StrandsYAMLBase
class _SampleYAMLRuntime:
    agents_config = "fixtures/yaml_runtime/agents.yaml"
    tools_config = "fixtures/yaml_runtime/tools.yaml"
    graphs_config = "fixtures/yaml_runtime/graphs.yaml"

    @yaml_tool
    def build_search_tool(self):
        return say

    @yaml_model
    def model_from_factory(self):
        return BedrockModel(model_id="factory-model-id")


def _write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file)


def test_yaml_runtime_resolves_agents_tools_and_graphs():
    runtime = _SampleYAMLRuntime()

    assert runtime.tools["search_tool"].tool_name == "say"
    assert runtime.tools["path_tool"] == "tests.fixtures.say_tool:say"

    researcher = runtime.agents["researcher"]
    writer = runtime.agents["writer"]

    assert researcher.model.config["model_id"] == "literal-model-id"
    assert writer.model.config["model_id"] == "factory-model-id"
    assert "say" in writer.tool_names
    assert "Researcher" in writer.tool_names

    graph = runtime.graphs["editorial_graph"]
    assert isinstance(graph, Graph)
    assert set(graph.nodes.keys()) == {"research_node", "writing_node"}
    assert len(graph.edges) == 1
    assert {node.node_id for node in graph.entry_points} == {"research_node"}


def test_yaml_runtime_supports_decorated_agent_tool_factory(tmp_path: Path):
    base_dir = tmp_path / "runtime"
    _write_yaml(
        base_dir / "agents.yaml",
        {
            "writer": {
                "model": "literal-model-id",
                "agent_tools": ["curated_research_tool"],
            }
        },
    )
    _write_yaml(base_dir / "tools.yaml", {"direct": "tests.fixtures.say_tool:say"})
    _write_yaml(
        base_dir / "graphs.yaml",
        {"simple_graph": {"nodes": {"writer_node": "writer"}, "entry_points": ["writer_node"]}},
    )

    @StrandsYAMLBase
    class _RuntimeWithAgentTool:
        config_base_directory = base_dir
        agents_config = "agents.yaml"
        tools_config = "tools.yaml"
        graphs_config = "graphs.yaml"

        @yaml_agent_tool
        def curated_research_tool(self):
            from strands.agent import Agent

            return Agent(name="curated_research").as_tool(name="curated_research_tool")

    runtime = _RuntimeWithAgentTool()
    assert "curated_research_tool" in runtime.agents["writer"].tool_names


def test_yaml_runtime_raises_for_missing_yaml_file(tmp_path: Path):
    base_dir = tmp_path / "missing"
    _write_yaml(base_dir / "tools.yaml", {"search": "tests.fixtures.say_tool:say"})
    _write_yaml(
        base_dir / "graphs.yaml",
        {"simple_graph": {"nodes": {"writer_node": {"agent": "writer"}}, "entry_points": ["writer_node"]}},
    )

    @StrandsYAMLBase
    class _MissingAgentsConfigRuntime:
        config_base_directory = base_dir
        agents_config = "agents.yaml"
        tools_config = "tools.yaml"
        graphs_config = "graphs.yaml"

    with pytest.raises(FileNotFoundError, match="Agents config file not found"):
        _MissingAgentsConfigRuntime()


def test_yaml_runtime_raises_for_empty_or_non_mapping_yaml(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    _write_yaml(empty_dir / "agents.yaml", {})
    _write_yaml(empty_dir / "tools.yaml", {"search": "tests.fixtures.say_tool:say"})
    _write_yaml(empty_dir / "graphs.yaml", {"g": {"nodes": {"n": "unknown"}, "entry_points": ["n"]}})

    @StrandsYAMLBase
    class _EmptyConfigRuntime:
        config_base_directory = empty_dir
        agents_config = "agents.yaml"
        tools_config = "tools.yaml"
        graphs_config = "graphs.yaml"

    with pytest.raises(ValueError, match="Agents config must be a non-empty YAML mapping"):
        _EmptyConfigRuntime()

    non_mapping_dir = tmp_path / "non_mapping"
    _write_yaml(non_mapping_dir / "agents.yaml", [])
    _write_yaml(non_mapping_dir / "tools.yaml", {"search": "tests.fixtures.say_tool:say"})
    _write_yaml(non_mapping_dir / "graphs.yaml", {"g": {"nodes": {"n": "unknown"}, "entry_points": ["n"]}})

    @StrandsYAMLBase
    class _NonMappingConfigRuntime:
        config_base_directory = non_mapping_dir
        agents_config = "agents.yaml"
        tools_config = "tools.yaml"
        graphs_config = "graphs.yaml"

    with pytest.raises(ValueError, match="Agents config must be a non-empty YAML mapping"):
        _NonMappingConfigRuntime()


def test_yaml_runtime_raises_for_unknown_tool_reference(tmp_path: Path):
    base_dir = tmp_path / "unknown_tool"
    _write_yaml(base_dir / "agents.yaml", {"writer": {"tools": ["unknown_tool"], "model": "literal-model-id"}})
    _write_yaml(base_dir / "tools.yaml", {"search": "tests.fixtures.say_tool:say"})
    _write_yaml(
        base_dir / "graphs.yaml",
        {"simple_graph": {"nodes": {"writer_node": "writer"}, "entry_points": ["writer_node"]}},
    )

    @StrandsYAMLBase
    class _UnknownToolRuntime:
        config_base_directory = base_dir
        agents_config = "agents.yaml"
        tools_config = "tools.yaml"
        graphs_config = "graphs.yaml"

    with pytest.raises(ValueError, match="Unknown tool reference 'unknown_tool' for agent 'writer'"):
        _UnknownToolRuntime()


def test_yaml_runtime_raises_for_agent_tool_cycles(tmp_path: Path):
    base_dir = tmp_path / "agent_cycle"
    _write_yaml(base_dir / "agents.yaml", {"a": {"agent_tools": ["b"]}, "b": {"agent_tools": ["a"]}})
    _write_yaml(base_dir / "tools.yaml", {"search": "tests.fixtures.say_tool:say"})
    _write_yaml(base_dir / "graphs.yaml", {"g": {"nodes": {"n": "a"}, "entry_points": ["n"]}})

    @StrandsYAMLBase
    class _CyclicAgentRuntime:
        config_base_directory = base_dir
        agents_config = "agents.yaml"
        tools_config = "tools.yaml"
        graphs_config = "graphs.yaml"

    with pytest.raises(ValueError, match="Cyclic agent"):
        _CyclicAgentRuntime()


def test_yaml_runtime_raises_for_unknown_graph_executor_reference(tmp_path: Path):
    base_dir = tmp_path / "unknown_graph_ref"
    _write_yaml(base_dir / "agents.yaml", {"writer": {"model": "literal-model-id"}})
    _write_yaml(base_dir / "tools.yaml", {"search": "tests.fixtures.say_tool:say"})
    _write_yaml(
        base_dir / "graphs.yaml",
        {"g": {"nodes": {"n": "unknown_graph_or_agent"}, "entry_points": ["n"]}},
    )

    @StrandsYAMLBase
    class _UnknownGraphRefRuntime:
        config_base_directory = base_dir
        agents_config = "agents.yaml"
        tools_config = "tools.yaml"
        graphs_config = "graphs.yaml"

    with pytest.raises(ValueError, match="Unknown graph node executor reference"):
        _UnknownGraphRefRuntime()


def test_yaml_runtime_raises_for_graph_reference_cycles(tmp_path: Path):
    base_dir = tmp_path / "graph_cycle"
    _write_yaml(base_dir / "agents.yaml", {"writer": {"model": "literal-model-id"}})
    _write_yaml(base_dir / "tools.yaml", {"search": "tests.fixtures.say_tool:say"})
    _write_yaml(
        base_dir / "graphs.yaml",
        {
            "graph_a": {"nodes": {"node_a": "graph_b"}, "entry_points": ["node_a"]},
            "graph_b": {"nodes": {"node_b": "graph_a"}, "entry_points": ["node_b"]},
        },
    )

    @StrandsYAMLBase
    class _CyclicGraphRuntime:
        config_base_directory = base_dir
        agents_config = "agents.yaml"
        tools_config = "tools.yaml"
        graphs_config = "graphs.yaml"

    with pytest.raises(ValueError, match="Cyclic graph reference detected"):
        _CyclicGraphRuntime()

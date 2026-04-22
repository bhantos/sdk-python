"""Experimental CrewAI-style YAML runtime configuration for Strands.

This module provides a class decorator and method decorators that enable loading
agent, tool, and graph definitions from YAML and resolving symbolic references
to runtime objects.
"""

import copy
import inspect
import logging
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

from ..agent import Agent
from ..multiagent.graph import Graph, GraphBuilder

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=type[Any])

_DEFAULT_AGENTS_CONFIG_PATH = "config/agents.yaml"
_DEFAULT_TOOLS_CONFIG_PATH = "config/tools.yaml"
_DEFAULT_GRAPHS_CONFIG_PATH = "config/graphs.yaml"


def _mark_yaml_method(attribute_name: str):
    """Create a decorator that marks a callable as a YAML factory."""

    def decorator(method):
        setattr(method, attribute_name, True)
        return method

    return decorator


yaml_agent = _mark_yaml_method("is_yaml_agent")
yaml_tool = _mark_yaml_method("is_yaml_tool")
yaml_agent_tool = _mark_yaml_method("is_yaml_agent_tool")
yaml_graph = _mark_yaml_method("is_yaml_graph")
yaml_model = _mark_yaml_method("is_yaml_model")


class _StrandsYAMLRuntimeMixin:
    """Mixin that adds YAML configuration loading and resolution behavior."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._initialize_yaml_runtime()

    def _initialize_yaml_runtime(self) -> None:
        """Load and resolve YAML-defined tools, agents, and graphs."""
        self.load_configurations()

        all_methods = self._get_all_methods()
        self._tool_factories = self._filter_methods(all_methods, "is_yaml_tool")
        self._agent_factories = self._filter_methods(all_methods, "is_yaml_agent")
        self._agent_tool_factories = self._filter_methods(all_methods, "is_yaml_agent_tool")
        self._graph_factories = self._filter_methods(all_methods, "is_yaml_graph")
        self._model_factories = self._filter_methods(all_methods, "is_yaml_model")

        self._resolved_tools: dict[str, Any] = {}
        self._resolved_agents: dict[str, Any] = {}
        self._resolved_agent_tools: dict[str, Any] = {}
        self._resolved_graphs: dict[str, Graph] = {}

        self.map_all_tool_variables()
        self.map_all_agent_variables()
        self.map_all_graph_variables()

        self.tools = self._resolved_tools
        self.agents = self._resolved_agents
        self.graphs = self._resolved_graphs

    def _get_all_methods(self) -> dict[str, Any]:
        """Collect all callable attributes on this instance."""
        methods: dict[str, Any] = {}
        for name in dir(self):
            value = getattr(self, name)
            if callable(value):
                methods[name] = value
        return methods

    def _filter_methods(self, methods: dict[str, Any], marker: str) -> dict[str, Any]:
        """Filter methods by marker attribute."""
        return {name: method for name, method in methods.items() if getattr(method, marker, False)}

    def _resolve_config_path(self, config_path: str) -> Path:
        """Resolve YAML path relative to the class base directory."""
        path = Path(config_path)
        if path.is_absolute():
            return path
        return cast(Path, self.base_directory) / path

    def _load_yaml_dict(self, config_path: str, config_name: str) -> dict[str, Any]:
        """Load and validate a YAML mapping file."""
        path = self._resolve_config_path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"{config_name.capitalize()} config file not found: {path}")

        with open(path, encoding="utf-8") as file:
            content = yaml.safe_load(file)

        if not isinstance(content, dict) or not content:
            raise ValueError(f"{config_name.capitalize()} config must be a non-empty YAML mapping: {path}")

        return cast(dict[str, Any], content)

    def load_configurations(self) -> None:
        """Load agents/tools/graphs YAML files."""
        self.agents_config = self._load_yaml_dict(self.original_agents_config_path, "agents")
        self.tools_config = self._load_yaml_dict(self.original_tools_config_path, "tools")
        self.graphs_config = self._load_yaml_dict(self.original_graphs_config_path, "graphs")

    def _invoke_factory(self, factory: Any, config: Any | None = None) -> Any:
        """Invoke a factory with zero or one argument depending on its signature."""
        parameters = list(inspect.signature(factory).parameters.values())
        if not parameters:
            return factory()
        if len(parameters) == 1:
            return factory(config)
        raise ValueError(
            f"Factory '{factory.__name__}' must accept zero parameters or a single config parameter for YAML runtime"
        )

    def map_all_tool_variables(self) -> None:
        """Resolve all YAML-defined tools."""
        for tool_name in self.tools_config:
            self._resolve_tool(tool_name, stack=[])

    def _resolve_tool(self, tool_name: str, stack: list[str]) -> Any:
        """Resolve a single tool definition by name."""
        if tool_name in self._resolved_tools:
            return self._resolved_tools[tool_name]

        if tool_name in stack:
            raise ValueError(f"Cyclic tool reference detected: {' -> '.join([*stack, tool_name])}")

        if tool_name not in self.tools_config:
            raise ValueError(f"Unknown tool reference '{tool_name}'")

        stack.append(tool_name)
        tool_info = self.tools_config[tool_name]
        resolved_tool = self._resolve_tool_info(tool_info, stack)
        stack.pop()

        self._resolved_tools[tool_name] = resolved_tool
        return resolved_tool

    def _resolve_tool_info(self, tool_info: Any, stack: list[str]) -> Any:
        """Resolve a tool definition body."""
        if isinstance(tool_info, str):
            if tool_info in self._tool_factories:
                return self._invoke_factory(self._tool_factories[tool_info])
            if tool_info in self._agent_tool_factories:
                return self._resolve_agent_tool(tool_info, stack=[])
            return tool_info

        if isinstance(tool_info, dict):
            if "ref" in tool_info:
                ref_name = tool_info["ref"]
                if not isinstance(ref_name, str):
                    raise ValueError("Tool 'ref' value must be a string")
                return self._resolve_tool(ref_name, stack)

            if "factory" in tool_info:
                factory_name = tool_info["factory"]
                if not isinstance(factory_name, str):
                    raise ValueError("Tool 'factory' value must be a string")
                if factory_name in self._tool_factories:
                    return self._invoke_factory(self._tool_factories[factory_name], tool_info)
                if factory_name in self._agent_tool_factories:
                    return self._resolve_agent_tool(factory_name, stack=[])
                raise ValueError(f"Unknown tool factory '{factory_name}'")

            if "value" in tool_info:
                return tool_info["value"]

        return tool_info

    def map_all_agent_variables(self) -> None:
        """Resolve all YAML-defined agents."""
        for agent_name in self.agents_config:
            self._resolve_agent(agent_name, stack=[])

    def _resolve_agent(self, agent_name: str, stack: list[str]) -> Any:
        """Resolve a single agent, including tool and graph references."""
        if agent_name in self._resolved_agents:
            return self._resolved_agents[agent_name]

        if agent_name in stack:
            raise ValueError(f"Cyclic agent reference detected: {' -> '.join([*stack, agent_name])}")

        agent_info = self.agents_config.get(agent_name)
        if not isinstance(agent_info, dict):
            raise ValueError(f"Agent '{agent_name}' config must be a mapping")

        stack.append(agent_name)
        mapped_agent_info = self._map_agent_variables(agent_name, copy.deepcopy(agent_info), stack)
        stack.pop()
        self.agents_config[agent_name] = mapped_agent_info

        if agent_name in self._agent_factories:
            resolved_agent = self._invoke_factory(self._agent_factories[agent_name], mapped_agent_info)
        else:
            resolved_agent = Agent(**self._filter_agent_kwargs(mapped_agent_info))

        self._resolved_agents[agent_name] = resolved_agent
        return resolved_agent

    def _map_agent_variables(self, agent_name: str, agent_info: dict[str, Any], stack: list[str]) -> dict[str, Any]:
        """Resolve symbolic agent fields to runtime objects."""
        for model_field in ("model", "llm", "function_calling_llm"):
            if isinstance(agent_info.get(model_field), str):
                model_name = agent_info[model_field]
                if model_name in self._model_factories:
                    agent_info[model_field] = self._invoke_factory(self._model_factories[model_name])

        tools = agent_info.get("tools", [])
        if tools is not None:
            if not isinstance(tools, list):
                raise ValueError(f"Agent '{agent_name}' tools must be a list")
            agent_info["tools"] = [
                self._resolve_agent_tool_reference(agent_name, tool_ref, stack) for tool_ref in tools
            ]

        agent_tools = agent_info.pop("agent_tools", [])
        if agent_tools:
            if not isinstance(agent_tools, list):
                raise ValueError(f"Agent '{agent_name}' agent_tools must be a list")
            resolved_agent_tools = [
                self._resolve_agent_tool_reference(agent_name, agent_tool_ref, stack) for agent_tool_ref in agent_tools
            ]
            agent_info.setdefault("tools", [])
            agent_info["tools"].extend(resolved_agent_tools)

        graph_ref = agent_info.get("graph")
        if isinstance(graph_ref, str):
            agent_info["graph"] = self._resolve_graph(graph_ref, stack=[*stack, f"agent:{agent_name}"])

        return agent_info

    def _resolve_agent_tool_reference(self, agent_name: str, tool_ref: Any, stack: list[str]) -> Any:
        """Resolve a tool reference used inside an agent config."""
        if not isinstance(tool_ref, str):
            return tool_ref

        if tool_ref in self.tools_config:
            return self._resolve_tool(tool_ref, stack=[])
        if tool_ref in self._tool_factories:
            return self._invoke_factory(self._tool_factories[tool_ref])
        if (
            tool_ref in self._agent_tool_factories
            or tool_ref in self.agents_config
            or tool_ref in self._agent_factories
        ):
            return self._resolve_agent_tool(tool_ref, stack=[*stack, agent_name])

        raise ValueError(f"Unknown tool reference '{tool_ref}' for agent '{agent_name}'")

    def _resolve_agent_tool(self, agent_tool_name: str, stack: list[str]) -> Any:
        """Resolve an agent-as-tool entry from a factory or an agent reference."""
        if agent_tool_name in self._resolved_agent_tools:
            return self._resolved_agent_tools[agent_tool_name]

        if agent_tool_name in stack:
            raise ValueError(f"Cyclic agent-as-tool reference detected: {' -> '.join([*stack, agent_tool_name])}")

        if agent_tool_name in self._agent_tool_factories:
            resolved = self._invoke_factory(self._agent_tool_factories[agent_tool_name])
            self._resolved_agent_tools[agent_tool_name] = resolved
            return resolved

        if agent_tool_name in self.agents_config or agent_tool_name in self._agent_factories:
            stack.append(agent_tool_name)
            resolved_agent = self._resolve_agent(agent_tool_name, stack)
            stack.pop()

            if not hasattr(resolved_agent, "as_tool") or not callable(resolved_agent.as_tool):
                raise ValueError(
                    f"Agent '{agent_tool_name}' cannot be used as a tool because it has no as_tool() method"
                )

            resolved_tool = resolved_agent.as_tool()
            self._resolved_agent_tools[agent_tool_name] = resolved_tool
            return resolved_tool

        raise ValueError(f"Unknown agent-as-tool reference '{agent_tool_name}'")

    def _filter_agent_kwargs(self, agent_info: dict[str, Any]) -> dict[str, Any]:
        """Filter mapped config to valid ``Agent`` constructor kwargs."""
        valid_keys = set(inspect.signature(Agent.__init__).parameters.keys()) - {"self"}
        return {key: value for key, value in agent_info.items() if key in valid_keys}

    def map_all_graph_variables(self) -> None:
        """Resolve all YAML-defined graphs."""
        for graph_name in self.graphs_config:
            self._resolve_graph(graph_name, stack=[])

    def _resolve_graph(self, graph_name: str, stack: list[str]) -> Graph:
        """Resolve a graph by name."""
        if graph_name in self._resolved_graphs:
            return self._resolved_graphs[graph_name]

        if graph_name in stack:
            raise ValueError(f"Cyclic graph reference detected: {' -> '.join([*stack, graph_name])}")

        graph_info = self.graphs_config.get(graph_name)
        if not isinstance(graph_info, dict):
            raise ValueError(f"Graph '{graph_name}' config must be a mapping")

        stack.append(graph_name)
        mapped_graph_info = self._map_graph_variables(graph_name, copy.deepcopy(graph_info), stack)
        stack.pop()
        self.graphs_config[graph_name] = mapped_graph_info

        if graph_name in self._graph_factories:
            graph = self._invoke_factory(self._graph_factories[graph_name], mapped_graph_info)
            if not isinstance(graph, Graph):
                raise ValueError(f"Graph factory '{graph_name}' must return a Graph instance")
        else:
            graph = self._build_graph_from_info(graph_name, mapped_graph_info)

        self._resolved_graphs[graph_name] = graph
        return graph

    def _map_graph_variables(self, graph_name: str, graph_info: dict[str, Any], stack: list[str]) -> dict[str, Any]:
        """Resolve graph node executor references."""
        nodes = graph_info.get("nodes")
        if not isinstance(nodes, dict) or not nodes:
            raise ValueError(f"Graph '{graph_name}' must define a non-empty 'nodes' mapping")

        resolved_nodes: dict[str, Any] = {}
        for node_id, executor_ref in nodes.items():
            resolved_nodes[node_id] = self._resolve_graph_executor_ref(graph_name, node_id, executor_ref, stack)

        graph_info["_resolved_nodes"] = resolved_nodes
        return graph_info

    def _resolve_graph_executor_ref(self, graph_name: str, node_id: str, executor_ref: Any, stack: list[str]) -> Any:
        """Resolve a graph node executor reference to an agent or graph instance."""
        if isinstance(executor_ref, str):
            if executor_ref in self.agents_config or executor_ref in self._agent_factories:
                return self._resolve_agent(executor_ref, stack=[])
            if executor_ref in self.graphs_config or executor_ref in self._graph_factories:
                return self._resolve_graph(executor_ref, stack=stack)
            raise ValueError(f"Unknown graph node executor reference '{executor_ref}' in graph '{graph_name}'")

        if isinstance(executor_ref, dict):
            if "agent" in executor_ref:
                agent_ref = executor_ref["agent"]
                if not isinstance(agent_ref, str):
                    raise ValueError(f"Graph node '{node_id}' in graph '{graph_name}' has invalid agent reference")
                return self._resolve_agent(agent_ref, stack=[])
            if "graph" in executor_ref:
                graph_ref = executor_ref["graph"]
                if not isinstance(graph_ref, str):
                    raise ValueError(f"Graph node '{node_id}' in graph '{graph_name}' has invalid graph reference")
                return self._resolve_graph(graph_ref, stack=stack)

        raise ValueError(f"Unsupported node executor config for node '{node_id}' in graph '{graph_name}'")

    def _build_graph_from_info(self, graph_name: str, graph_info: dict[str, Any]) -> Graph:
        """Build a Graph from mapped YAML configuration."""
        builder = GraphBuilder()
        resolved_nodes = graph_info["_resolved_nodes"]

        for node_id, executor in resolved_nodes.items():
            builder.add_node(executor, node_id=node_id)

        edges = graph_info.get("edges", [])
        if not isinstance(edges, list):
            raise ValueError(f"Graph '{graph_name}' edges must be a list")

        for edge in edges:
            if not isinstance(edge, dict):
                raise ValueError(f"Graph '{graph_name}' edges must contain mappings")
            from_node = edge.get("from")
            to_node = edge.get("to")
            if not isinstance(from_node, str) or not isinstance(to_node, str):
                raise ValueError(f"Graph '{graph_name}' edge entries require string 'from' and 'to' fields")
            builder.add_edge(from_node, to_node)

        entry_points = graph_info.get("entry_points", [])
        if not isinstance(entry_points, list):
            raise ValueError(f"Graph '{graph_name}' entry_points must be a list")
        for entry_point in entry_points:
            if not isinstance(entry_point, str):
                raise ValueError(f"Graph '{graph_name}' entry_points values must be strings")
            builder.set_entry_point(entry_point)

        if "max_node_executions" in graph_info:
            builder.set_max_node_executions(graph_info["max_node_executions"])
        if "execution_timeout" in graph_info:
            builder.set_execution_timeout(graph_info["execution_timeout"])
        if "node_timeout" in graph_info:
            builder.set_node_timeout(graph_info["node_timeout"])
        if "reset_on_revisit" in graph_info:
            builder.reset_on_revisit(graph_info["reset_on_revisit"])

        builder.set_graph_id(graph_info.get("id", graph_name))
        return builder.build()


def StrandsYAMLBase(cls: T) -> T:
    """Class decorator enabling CrewAI-style YAML-driven Strands composition.

    The decorated class can declare relative YAML paths:
        - ``agents_config`` (default ``config/agents.yaml``)
        - ``tools_config`` (default ``config/tools.yaml``)
        - ``graphs_config`` (default ``config/graphs.yaml``)

    At initialization, the runtime loads YAML files, discovers decorated factory
    methods, resolves symbolic references, and exposes:
        - ``self.tools`` (resolved tool map)
        - ``self.agents`` (resolved agent map)
        - ``self.graphs`` (resolved graph map)
    """

    class WrappedClass(_StrandsYAMLRuntimeMixin, cls):  # type: ignore[misc, valid-type]
        pass

    WrappedClass.__name__ = cls.__name__
    WrappedClass.__qualname__ = cls.__qualname__
    WrappedClass.__module__ = cls.__module__
    WrappedClass.__doc__ = cls.__doc__

    config_base_directory = getattr(cls, "config_base_directory", None)
    if config_base_directory is None:
        try:
            WrappedClass.base_directory = Path(inspect.getfile(cls)).parent
        except (TypeError, OSError):
            WrappedClass.base_directory = Path.cwd()
    else:
        WrappedClass.base_directory = Path(config_base_directory)

    WrappedClass.original_agents_config_path = getattr(cls, "agents_config", _DEFAULT_AGENTS_CONFIG_PATH)
    WrappedClass.original_tools_config_path = getattr(cls, "tools_config", _DEFAULT_TOOLS_CONFIG_PATH)
    WrappedClass.original_graphs_config_path = getattr(cls, "graphs_config", _DEFAULT_GRAPHS_CONFIG_PATH)

    return cast(T, WrappedClass)

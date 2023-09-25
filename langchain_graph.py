import weakref
from typing import Any

import langchain

class Accessor:

    def __call__(self, obj: Any) -> Any:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

class AttributeAccessor(Accessor):

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, obj: Any) -> Any:
        return getattr(obj, self.name)

    def __str__(self) -> str:
        return f".{self.name}"

    def __repr__(self) -> str:
        return f"AttributeAccessor({self.name})"

class ItemAccessor(Accessor):

    def __init__(self, key: Any) -> None:
        self.key = key

    def __call__(self, obj: Any) -> Any:
        return obj[self.key]

    def __str__(self) -> str:
        return f"[{self.key}]"

    def __repr__(self) -> str:
        return f"ItemAccessor({self.key})"

class AccessorChain(Accessor):

    def __init__(self, accessors: list[Accessor]) -> None:
        self.accessors = accessors

    def __call__(self, obj: Any) -> Any:
        for accessor in self.accessors:
            obj = accessor(obj)
        return obj

    def __str__(self) -> str:
        return "".join(str(accessor) for accessor in self.accessors)

    def __repr__(self) -> str:
        return f"AccessorChain({self.accessors})"

class Node:

    def __init__(self, path: Accessor, nickname: str) -> None:
        self.path = path
        self.nickname = nickname

class Graph(Node):

    def __init__(self, path: Accessor, nickname: str, children: list[Node]) -> None:
        super().__init__(path, nickname)
        self.children = children

class Edge(Node):

    pass
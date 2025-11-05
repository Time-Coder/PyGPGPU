from __future__ import annotations

import tree_sitter
import tree_sitter_c as tsc
import sys
from typing import List, Optional


class KernelSyntaxTree:

    __c_raw_parser:Optional[tree_sitter.Parser] = None

    class Node:

        def __init__(self, tree_sitter_node:tree_sitter.Node, index:int=0):
            self.text:str = tree_sitter_node.text.decode("utf-8")
            self.type:str = tree_sitter_node.type
            self.start_byte:int = tree_sitter_node.start_byte
            self.end_byte:int = tree_sitter_node.end_byte

            self.index:int = index
            self.children: List[KernelSyntaxTree.Node] = []
            self.parent: Optional[KernelSyntaxTree.Node] = None
            for index, tree_sitter_child in enumerate(tree_sitter_node.children):
                child = KernelSyntaxTree.Node(tree_sitter_child, index)
                child.parent = self
                self.children.append(child)

        @property
        def path(self)->str:
            if self.parent is None:
                return str(self.index)
            
            return self.parent.path + "/" + str(self.index)
        
        @property
        def next_sibling(self)->Optional[KernelSyntaxTree.Node]:
            if self.parent is None:
                return None
            
            next_index:int = self.index + 1
            if next_index >= len(self.parent.children):
                return None

            return self.parent.children[next_index]
        
        @property
        def child_count(self)->int:
            return len(self.children)
        
        def __getitem__(self, path:str)->KernelSyntaxTree.Node:
            indices = path.split("/")

            active_node = self
            for index_str in indices:
                index:int = int(index_str)
                active_node = active_node.children[index]

            return active_node

    def __init__(self, code:str=""):
        self.root:Optional[KernelSyntaxTree.Node] = None
        if code:
            self.parse(code)

    def __getitem__(self, path:str)->KernelSyntaxTree.Node:
        return self.root[path]
    
    @staticmethod
    def __c_parser():
        if KernelSyntaxTree.__c_raw_parser is not None:
            return KernelSyntaxTree.__c_raw_parser

        GLSL_LANGUAGE = tree_sitter.Language(tsc.language())
        KernelSyntaxTree.__c_raw_parser = tree_sitter.Parser(GLSL_LANGUAGE)
        return KernelSyntaxTree.__c_raw_parser

    def clear(self):
        self.root = None

    def parse(self, code:str):
        raw_tree:tree_sitter.Tree = KernelSyntaxTree.__c_parser().parse(bytes(code, sys.getdefaultencoding()))
        self.root:Optional[KernelSyntaxTree.Node] = KernelSyntaxTree.Node(raw_tree.root_node)
        
#= 
Write an algorithm that creates a 3-2 tree from a list of keys. Analyze your
algorithm and show the results using order notation.
=#

using Test

struct Node
    key :: Vector{Int}
    childVector :: Vector
end

struct BTree
    root :: Node
    depth :: Int
end

"""
    isLeaf(node :: Node) :: Bool

Determine if a node is a leaf.
"""
function isLeaf(node :: Node) :: Bool
    return node.childVector == Node[]
end

"""
    traverse!(foo :: Function, root :: Node)

Traverse the tree in pre-order, evaluating foo at each node.
"""
function traverse!(foo :: Function, root :: Node)
    if isLeaf(root)
        foo(root.card)
        return 
    else
        foo(root.card)
        traverse!.(foo, root.childVector)
    end
end

function insertNode!(childNew :: Node, parent :: Node)
    if isLeaf(parent)
        push!(parent.childVector, childNew)
    else
        if childNew.key[1] < parent.key[1]
            insertNode(childNew, parent.childVector[1])
        else
            insertNode(childNew, parent.childVector[2])
        end
    end
end

@testset verbose=true begin
    root = Node([5], [])
    child1 = Node([4], [])
    child2 = Node([6], [])
    insertNode!(child1, root)
    insertNode!(child2, root)
    display(root)
end
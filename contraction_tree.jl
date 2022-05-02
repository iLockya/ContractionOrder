include("utils.jl")
using StatsBase
using MLStyle
using Plots

mutable struct CTree_Node{}
    """

    Node struct of contraction tree

    """
    gid::Int # id in the graph
    id::Set{Any} # node id, Set([1,2,3,5])
    father::Set{Any} # Set([1,2,3,4,5])
    left::Set{Any} # Set([1,2])
    right::Set{Any} # Set([3,5])
    sc::Float64 # size of the node. sc = get_node_dim(ct,id).
    tc::Float64 # time complexity in single contraction
    complex::Array{Float64,1} # complexity for subtree
end

mutable struct Contraction_Tree{}
    """

    Contraction tree struct

    It stores time and space complexity, but does not rely on any graph structure, or computation of sc and tc.
    """
    nodes::Dict{Any,Any} # map node_id (Set([1,3,5])) to Node_Struct
    neis::Dict
    edge_dict::Dict{Any,Float64} # map edge to log2dim
    root::Set{Any}
    complex::Array{Float64,1} # [space_complexity, time_complexity]
    slice::Set{Vector{Int64}} # sliced edge
end


function init_CTree(n::Int64,neis,edge_dict)
    """

    Initialize a contraction tree

    n: number of leaves
    """
    #root = Set(collect(1:n))
    root = Set(keys(neis))
    ct = Contraction_Tree(Dict(),neis,edge_dict,root,[0.0,1.0],Set([]))
    for i in root
        ct.nodes[Set([i])] = CTree_Node(i,Set([i]),Set([]),Set([]),Set([]),0.0,-Inf,[0.0,-Inf])
    end
    return ct
end

function update_complex!(ct::Contraction_Tree, root::CTree_Node)
    """
    update complexity for all CTree_Node recursively.
    """
    if is_leaf(root)
        sc = get_node_dim(ct,root.id)
        root.sc = sc
        root.complex[1] = sc
        return
    end
    L_node = get_left(ct,root)
    R_node = get_right(ct,root) 
    update_complex!(ct,L_node)
    update_complex!(ct,R_node)
    root.sc = get_node_dim(ct,root.id)
    root.complex[1] = maximum([root.sc,L_node.complex[1],R_node.complex[1]])
    #root.complex[1] = maximum([node.sc for node in ct.nodes.values()])
    root.complex[2] = fast_log2sumexp2(root.tc,L_node.complex[2],R_node.complex[2])
    if ct.root == root.id
        ct.complex = root.complex
    end
    # ct.complex[1] = maximum([node.sc for node in values(ct.nodes) ])
    # ct.complex[2] = log2sumexp([node.tc for node in values(ct.nodes)])
end

@inline function get_left(ct::Contraction_Tree, node::CTree_Node)::CTree_Node
    length(node.id) > 1 && return ct.nodes[node.left]
    error("Node ",node.id," has no left node." )
end

@inline function get_father(ct::Contraction_Tree, node::CTree_Node)::CTree_Node
    node.id == ct.root || return ct.nodes[node.father]
    error("Node ",id," is the root node." ) 
end

@inline function get_right(ct::Contraction_Tree, node::CTree_Node)::CTree_Node
    length(node.id) > 1 && return ct.nodes[node.right]
    error("Node ",id," has no right node." )
end

@inline function get_color(ct::Contraction_Tree, id::Set{Any})
    length(id) ≤ 2 && return 'w'
    L_node, R_node = get_left(ct,ct.nodes[id]), get_right(ct,ct.nodes[id])
    is_parent(L_node) && is_leaf(R_node) && return 'y'
    is_leaf(L_node) && is_parent(R_node) && return 'b'
    is_parent(L_node) && is_parent(R_node) && return 'g'
end

function get_height(ct::Contraction_Tree, node::CTree_Node)
    """
    count from leaves.
    """
    is_leaf(node) && return 1
    return max(get_height(ct,get_left(ct,node))+1,get_height(ct,get_right(ct,node))+1)
end

@inline function get_node_dim(ct,nodeid)
    """
    get node log2dim
    """
    dim = 0
    for i in nodeid
        for j in ct.neis[i]
            if j ∉ nodeid && sort([i,j]) ∉ ct.slice
                dim += ct.edge_dict[sort([i,j])]
            end
        end
    end
    return dim
end

@inline function get_neis(ct::Contraction_Tree,nodeid)
    """
    get neighbors of CTree node.
    """
    neis = Set([])
    for i in nodeid
        for j in ct.neis[i]
            if j ∉ nodeid && sort([i,j]) ∉ ct.slice
                push!(neis,j)
            end
        end
    end
    return neis
end

@inline function get_depth(ct::Contraction_Tree,node::CTree_Node)
    node.id == ct.root && return 1
    return get_depth(ct,get_father(ct,node))+1
end


function LRD_traversal!(ct::Contraction_Tree,node::CTree_Node,LRD_list::Array)
    """
    Postorder Traversal
    """
    if is_leaf(node)
        push!(LRD_list,node.id)
        return 0
    end
    LRD_traversal!(ct,get_left(ct,node),LRD_list)
    LRD_traversal!(ct,get_right(ct,node),LRD_list)
    push!(LRD_list,node.id)
    return LRD_list
end
    

@inline function is_leaf(node::CTree_Node)
    """

    True if node is leaf.
    """
    return length(node.id) == 1
end

function is_parent(node::CTree_Node)
    return length(node.id) > 1
end

function is_grandparent(ct::Contraction_Tree,node::CTree_Node)
    return length(node.id) ≥ 3
end




function CTree_contract!(ct,pair,g_edge,sc,tc)
    """

    contract two edges and a pair of nodes

    ct : contraction tree
    pair: Array{Set{Int64},Set{Int64}}, { [1,2,3],[4,5,6] }, pair of node in ct
    sc: space complexity
    tc: time complexity
        both sc and tc should be computed using Graph or Tensor_Network structures.
        They are not stored in the Contraction_Tree structure
    """
    #id = Union(pair[1],pair[2])
    id = union(pair[1],pair[2])
    gid = g_edge[1]
    father = Set([])
    left = pair[1]
    right = pair[2]
    nodesc = get_node_dim(ct,id)
    ct.nodes[id] = CTree_Node(gid,id,father,left,right,nodesc,tc,[sc,tc])
    ct.nodes[pair[1]].father = id
    ct.nodes[pair[2]].father = id
    return id
end

function CTree_to_order(ct,root)
    """

    Convert a contraction tree (starting from root) to a contraction order.

    ct:     contraction tree
    root:   starting point of the tree, key of ct.nodes[], such as Set([1,2,3])
    order:  a list of pair, such as [[1,2],[3,5],[4,5],[1,4]]
    """
    root_node = ct.nodes[root]
    left = root_node.left
    if left == Set([])
        return []
    end
    right = root_node.right
    #left_node = ct.nodes[left]
    #right_node = ct.nodes[right]
    #pair = [left_node.gid,right_node.gid]
    #if root_node.gid == right_node.gid
    #    pair = [right_node.gid, left_node.gid]
    #end
    pair = sort([minimum(left),minimum(right)])
    return [CTree_to_order(ct,left); CTree_to_order(ct,right); [pair]]
end

function contract_node(ct::Contraction_Tree,A_Node::CTree_Node,B_Node::CTree_Node)::CTree_Node
    """

    contract node A and node B to node D.
    return Node_D, sc, tc
    """
    A_neis = get_neis(ct,A_Node.id)
    B_neis = get_neis(ct,B_Node.id)
    A_log2dim = Vector{Float64}([])
    B_log2dim = Vector{Float64}([])
    AB_log2dim = Vector{Float64}([])
    for (edge,log2dim) in ct.edge_dict
        i,j=edge
        if edge ∈ ct.slice # ignore the sliced edges
            continue
        end
        # determine witch class the edge belongs to.   
        if (i∈A_neis && j∈A_Node.id) || (i∈A_Node.id && j∈A_neis)
            push!(A_log2dim,log2dim)
        end
        if (i∈B_neis && j∈B_Node.id) || (i∈B_Node.id && j∈B_neis)
            push!(B_log2dim,log2dim)
        end
        if (i∈A_Node.id && j∈B_Node.id) || (i∈B_Node.id && j∈A_Node.id)
            push!(AB_log2dim,log2dim)
        end
    end

    id = union(A_Node.id,B_Node.id)
    gid = min(id...)
    sc = max(sum(A_log2dim),sum(B_log2dim),sum(A_log2dim)+sum(B_log2dim)-2*sum(AB_log2dim))
    tc = AB_log2dim==[] ? sum(A_log2dim) + sum(B_log2dim)-1 : sum(A_log2dim) + sum(B_log2dim) - sum(AB_log2dim)
    D = CTree_Node(gid,id,Set([]),A_Node.id,B_Node.id,get_node_dim(ct,id),tc,[sc,tc])
    return D
end

  

function make_trial(ct::Contraction_Tree,P_Node::CTree_Node,color::Char)
    """
    This function only make one trial.
    color = 'y', return [type1, type2, nothing, nothing]
    color = 'b', return [nothing, nothing, type3, type4]
    """
    n = length(ct.root) # total number of nodes
    if color == 'y'
        t = rand([1,2])
    elseif color == 'b'
        t = rand([3,4])
    else 
        t = rand([1,2,3,4])
    end

    if t ∈ [1,2]
        L_Node = get_left(ct,P_Node)
        A_Node = get_left(ct,L_Node)
        B_Node = get_right(ct, L_Node)
        C_Node = get_right(ct, P_Node)
        hA,hB,hC = get_height(ct,A_Node),get_height(ct,B_Node),get_height(ct,C_Node)
        H_ori = target_fn(max(P_Node.sc,L_Node.sc),fast_log2sumexp2(L_Node.tc,P_Node.tc)) 
                + tree_imbalance(max(hA,hB)+1,hC,n)
    else
        R_Node = get_right(ct,P_Node)
        A_Node = get_left(ct,P_Node)
        B_Node = get_left(ct, R_Node)
        C_Node = get_right(ct, R_Node)
        hA,hB,hC = get_height(ct,A_Node),get_height(ct,B_Node),get_height(ct,C_Node)
        H_ori = target_fn(max(P_Node.sc,R_Node.sc),fast_log2sumexp2(R_Node.tc,P_Node.tc)) + tree_imbalance(hA,max(hB,hC)+1,n)
    end


    @switch t begin
        @case 1 # (A*B)*C -> (C*B)*A
        L_tmp_1 = contract_node(ct, C_Node,B_Node)
        P_tmp_1 = contract_node(ct, L_tmp_1,A_Node)
        ΔH₁ = target_fn(max(P_tmp_1.sc,L_tmp_1.sc),fast_log2sumexp2(L_tmp_1.tc,P_tmp_1.tc)) + tree_imbalance(hA,max(hB,hC)+1,n) - H_ori
        return t,ΔH₁,L_tmp_1,P_tmp_1

        @case 2 # (A*B)*C -> (A*C)*B
        L_tmp_2 = contract_node(ct, A_Node,C_Node)
        P_tmp_2 = contract_node(ct,L_tmp_2,B_Node)
        ΔH₂ = target_fn(max(P_tmp_2.sc,L_tmp_2.sc),fast_log2sumexp2(L_tmp_2.tc,P_tmp_2.tc)) + tree_imbalance(hB,max(hA,hC)+1,n) - H_ori
        return t,ΔH₂,L_tmp_2,P_tmp_2

        @case 3 # A*(B*C) -> B*(A*C)
        R_tmp_3 = contract_node(ct,A_Node,C_Node)
        P_tmp_3 = contract_node(ct,B_Node,R_tmp_3)
        ΔH₃ = target_fn(max(P_tmp_3.sc,R_tmp_3.sc),fast_log2sumexp2(R_tmp_3.tc,P_tmp_3.tc)) + tree_imbalance(hB,max(hA,hC)+1,n) - H_ori
        return t,ΔH₃,R_tmp_3,P_tmp_3

        @case 4 # A*(B*C) -> C*(B*A)
        R_tmp_4 = contract_node(ct,B_Node,A_Node)
        P_tmp_4 = contract_node(ct,C_Node,R_tmp_4)
        ΔH₄ = target_fn(max(P_tmp_4.sc,R_tmp_4.sc),fast_log2sumexp2(R_tmp_4.tc,P_tmp_4.tc)) + tree_imbalance(hC,max(hA,hB)+1,n) - H_ori
        return t,ΔH₄,R_tmp_4,P_tmp_4
    end
end



function local_update!(ct::Contraction_Tree,P_Node::CTree_Node,β::Float64)
    """
    """
    color = get_color(ct,P_Node.id)
    color == 'w' && return 0
    L = get_left(ct,P_Node)
    R = get_right(ct,P_Node)
    #ΔH,LR_tmp,P_tmp = make_trial(ct,P_Node,color)
    #type = sample([0,1,2,3,4],Weights(exp.(-beta.*insert!(ΔH,1,0))))
    t,ΔH,LR_tmp,P_tmp = make_trial(ct,P_Node,color)
    #println(ΔH)
    if rand()>min(1,exp(-β*ΔH))
        return 0
    end
    @switch t begin
        @case 1
        A,B,C = get_left(ct,L),get_right(ct,L),get_right(ct,P_Node)
        P_tmp.father = P_Node.father
        LR_tmp.father, A.father = P_tmp.id, P_tmp.id
        C.father,B.father = LR_tmp.id,LR_tmp.id
        delete!(ct.nodes,L.id)
        delete!(ct.nodes,P_Node.id)
        ct.nodes[P_tmp.id] = P_tmp
        ct.nodes[LR_tmp.id] = LR_tmp
        return 1

        @case 2
        A,B,C = get_left(ct,L),get_right(ct,L),get_right(ct,P_Node)
        P_tmp.father = P_Node.father
        LR_tmp.father, B.father = P_tmp.id, P_tmp.id
        A.father,C.father = LR_tmp.id,LR_tmp.id
        delete!(ct.nodes,L.id)
        delete!(ct.nodes,P_Node.id)
        ct.nodes[P_tmp.id] = P_tmp
        ct.nodes[LR_tmp.id] = LR_tmp
        return 1

        @case 3
        A,B,C = get_left(ct,P_Node),get_left(ct,R),get_right(ct,R)
        P_tmp.father = P_Node.father
        LR_tmp.father, B.father = P_tmp.id, P_tmp.id
        A.father,C.father = LR_tmp.id,LR_tmp.id
        delete!(ct.nodes,R.id)
        delete!(ct.nodes,P_Node.id)
        ct.nodes[P_tmp.id] = P_tmp
        ct.nodes[LR_tmp.id] = LR_tmp
        return 1

        @case 4
        A,B,C = get_left(ct,P_Node),get_left(ct,R),get_right(ct,R)
        P_tmp.father = P_Node.father
        LR_tmp.father, C.father = P_tmp.id, P_tmp.id
        B.father,A.father = LR_tmp.id,LR_tmp.id
        delete!(ct.nodes,R.id)
        delete!(ct.nodes,P_Node.id)
        ct.nodes[P_tmp.id] = P_tmp
        ct.nodes[LR_tmp.id] = LR_tmp
        return 1

        @case 0
        return 0
    end 
end

function remove_branch!(ct::Contraction_Tree,node::CTree_Node)
    """

    remove the sub-tree with root node recursively.

    node: root of sub-tree
    """
    delete!(ct.nodes,node.id)
    if ~is_leaf(node)
        remove_branch!(ct,get_left(ct,node))
        remove_branch!(ct,get_right(ct,node))
    end
end

function replace_branch!(ct::Contraction_Tree,node::CTree_Node,subct::Contraction_Tree)
    """

    replace node by subct.
    """
    @assert node.id == subct.root
    if node.id == ct.root
        ct = subct
    end
    P = get_father(ct,node)
    subct.nodes[subct.root].father = P.id
    if P.left == node.id
        P.left = subct.root
    else
        P.right = subct.root
    end
    remove_branch!(ct,node)
    merge!(ct.nodes,subct.nodes)
    update_complex!(ct,ct.nodes[ct.root])
end

#function quickbb_update!(ct::Contraction_Tree,tn::Graph,node::CTree_Node)
#    subct = quickbb(tn,node.id)
#    if node.id == ct.root
#        ct = subct
#    end
#    P = get_father(ct,node)
#    subct.nodes[subct.root].father = P.id
#    if P.left == node.id
#    else
#        P.right = subct.root
#    end
#    merge!(ct.nodes,subct.nodes)
#    update_complex!(ct,ct.root)
#end


function tempering()

end

function mcmc_greedy(ct::Contraction_Tree,beta::Float64,L::Int64)
end
    


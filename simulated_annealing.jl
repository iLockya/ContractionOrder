using Plots
include("tensor_network.jl")
include("contraction_tree.jl")
include("greedy.jl")
 

function simulated_annealing!(ct::Contraction_Tree,tn::Union{Graph,OpenGraph}, βs,optimal_type;optimal_arg,sc_target = 0,nslice=Inf)
    fn_min = target_fn(ct.complex[1],ct.complex[2])
    min_fn_order = CTree_to_order(ct,ct.root)
    min_fn_slice = ct.slice
    for β in βs # annealing.
        ## dynamic slicing. ##
        if ct.complex[1] < sc_target # delete the unnecessary edge in ct.slice.
            for edge in ct.slice
                remove_slice_edge!(ct,tn,edge)
            end
        # if sc exceed the sc_target, slice an edge to reduce the sc.
        else
            push!(ct.slice,slice_largest_tensor(ct))
        end

        while length(ct.slice) > nslice 
            edge = rand(ct.slice)
            delete!(ct.slice, edge)
        end
        if ~isempty(ct.slice) # randomly delete one edge in ct.slice. and add a new edge.
            edge = rand(ct.slice)
            delete!(ct.slice,edge)
            push!(ct.slice,slice_largest_tensor(ct))
        end

        update_complex!(ct,ct.nodes[ct.root])

        ## simulated annealing. ##
        if optimal_type == "normal"
            L = optimal_arg[1]
            fn_min_i,min_fn_order = mcmc!(ct,β,L)
        elseif optimal_type == "greedy"
            L,c,n_orders = optimal_arg
            fn_min_i, min_fn_order_i,min_fn_slice_i = mcmc_greedy!(ct,β,tn,L,c,n_orders)
        end
        if fn_min_i < fn_min
            fn_min =fn_min_i
            min_fn_order = min_fn_order_i
            min_fn_slice = min_fn_slice_i
        end
        
    end
    return fn_min*log10(2),min_fn_order,min_fn_slice
end

function mcmc!(ct::Contraction_Tree,beta::Float64,L::Int64)
    scs = []
    tcs = []
    acc = []
    fn_min = target_fn(ct.complex[1],ct.complex[2])
    min_fn_order = CTree_to_order(ct,ct.root)
    count = 0
    tot = 0
    for i=1:L
        node = rand(collect(values(ct.nodes)))
        tot = L
        count+=local_update!(ct,node,beta)
        if i % 100 ==0
            update_complex!(ct,ct.nodes[ct.root])
            push!(scs,ct.complex[1])
            push!(tcs,ct.complex[2])
        end
    end
    println("Accept rate = $(count)/$(tot) ", Float64(count/tot))
    return fn_min,min_fn_order
end


function greedy_update!(ct::Contraction_Tree,tn::Graph,nodes,beta,n_orders=10)
    for nodeid in nodes
        #println(nodeid)
        node = ct.nodes[nodeid]
        subgraph= get_subgraph(tn,nodeid)
        min_sc_order, min_tc_order,orders = greedy_orders(subgraph,"mindim","mindimtri",n_orders)
        subct = order_to_CTree(subgraph,min_tc_order[1])
        ΔH = target_fn(subct.complex[1],subct.complex[2])-target_fn(node.complex[1],node.complex[2])
        if rand()≤minimum([1,exp(-beta*ΔH)])
            replace_branch!(ct,node,subct)
        end
    end
end

function mcmc_greedy!(ct::Contraction_Tree,β::Float64,tn::Graph,L::Int64,c::Int64,n_orders::Int64)
    scs = []
    tcs = []
    acc = []
    fn_min = target_fn(ct.complex[1],ct.complex[2])
    min_fn_order = CTree_to_order(ct,ct.root)
    min_fn_slice = ct.slice
    for i=1:L
        LRD_list = []
        LRD_traversal!(ct,ct.nodes[ct.root],LRD_list)
        #top_nodes = filter(x->get_depth(ct,ct.nodes[x])≤c,LRD_list)
        top_nodes = filter(x->length(x)≥50,LRD_list)

        tot = length(top_nodes)
        counts = [local_update!(ct,ct.nodes[node],β) for node in top_nodes] |> sum
        update_complex!(ct,ct.nodes[ct.root])

        if i % L ==0 # greedy seach for lower nodes. 
            sub_nodes = _sub_nodes(ct,c)
            greedy_update!(ct,tn,sub_nodes,β,n_orders)
        end

        push!(scs,ct.complex[1])
        push!(tcs,ct.complex[2])
        push!(acc,Float64(counts/tot))
        fn_new = target_fn(ct.complex[1],ct.complex[2])
        if fn_new < fn_min
            fn_min=fn_new
            min_fn_order = CTree_to_order(ct,ct.root)
            min_fn_slice = ct.slice
        end
        # println("Accept rate = $(counts)/$(tot) ", Float64(counts/tot))
    end
    #plot(tcs.*log10(2))
    return fn_min,min_fn_order, min_fn_slice
end

function _top_nodes(ct::Contraction_Tree,c::Int64)
    """
    Returns a list of nodes that are at top c from the root.
    """
    LRD_list = []
    LRD_traversal!(ct,ct.nodes[ct.root],LRD_list)
    top_nodes = filter(x->get_depth(ct,ct.nodes[x])≤c,LRD_list)
    return top_nodes
end

function _sub_nodes(ct::Contraction_Tree,c::Int64)
    top_nodes = _top_nodes(ct,c)
    sub_nodes_r = [get_right(ct,ct.nodes[id]).id for id in top_nodes if ~is_leaf(ct.nodes[id])]
    sub_nodes_l = [get_left(ct,ct.nodes[id]).id for id in top_nodes if ~is_leaf(ct.nodes[id])]
    sub_nodes = setdiff([sub_nodes_l;sub_nodes_r] |> unique,top_nodes)
    filter!(x->length(x)≥3,sub_nodes)
    sort!(sub_nodes,by=length)
    delete_list = []
    for k=eachindex(sub_nodes)
        if get_father(ct,ct.nodes[sub_nodes[k]]).id in sub_nodes
            push(delete_list,k)
        end
    end
    deleteat!(sub_nodes,delete_list)
    return sub_nodes
end

function largest_tensor(ct::Contraction_Tree)
    """
    Return the largest tensor in the contraction tree.
    If the tensor is not unique, return the one with the largest id.
    """
    node_dim_dict = Dict([node=>get_node_dim(ct,node) for node in ct.nodes])
    max_dim = max(node_dim_dict.values())
    max_dim_nodes = filter(x->node_dim_dict[x]≥max_dim-0.99,keys(node_dim_dict))
    max_dim_node = max_dim_nodes[0]
    if length(max_dim_nodes)>1
        for node in max_dim_nodes
            if node.id > max_dim_node.id
                max_dim_node = node
            end
        end
    end
    return max_dim_node
end

function minimum_node_with_edge(ct::Contraction_Tree,nodeid,i,j)
    # find the minimum node in ct.nodes which contains i and j.
    if i ∈ ct.nodes[nodeid].left && j ∈ ct.nodes[nodeid].left
        return minimum_node_with_edge(ct,get_left(ct,ct.nodes[nodeid]).id,i,j)
    elseif i ∈ ct.nodes[nodeid].right && j ∈ ct.nodes[nodeid].right
        return minimum_node_with_edge(ct,get_right(ct,ct.nodes[nodeid]).id,i,j)
    else
        return nodeid
    end
end

function remove_slice_edge!(ct::Contraction_Tree,tn::Union{Graph,OpenGraph},edge::Vector{Int64})
    i,j = edge
    id = minimum_node_with_edge(ct,ct.root,i,j) # find the minimum node in ct.nodes which contains i and j.
    ln = ct.nodes[ct.nodes[id].left] # left node.
    rn = ct.nodes[ct.nodes[id].right] # right node.
    edge_dim = get_edge_log2dim(tn,edge)
    if ln.sc+edge_dim < sc_target && rn.sc+edge_dim < sc_target 
        # if add the edge doesnot exceed the sc_target, delete it.
        delete!(ct.slice,edge)
        ln.sc += edge_dim
        rn.sc += edge_dim
    end 
end

function slice_largest_tensor(ct::Contraction_Tree)
    """
    Pick an leg on the largest tensor.
    """
    # 1. find the largest node
    max_dim_node, max_sc = ct.root,ct.nodes[ct.root].sc
    for node in values(ct.nodes)
        if node.sc > max_sc
            max_dim_node, max_sc = node.id,node.sc
        end
    end
    # 2. pick a leg.
    for i in max_dim_node
        for j in setdiff(ct.neis[i],max_dim_node) 
            edge = sort([i,j])
            if edge ∉ ct.slice
                return edge
            end
        end
    end
# end function. 
end
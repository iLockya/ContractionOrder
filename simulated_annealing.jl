using Plots
include("tensor_network.jl")
include("contraction_tree.jl")
include("greedy.jl")

function simulated_annealing!(ct::Contraction_Tree,betas,optimal_type;optimal_arg)
    fn_min = target_fn(ct.complex[1],ct.complex[2])
    min_fn_order = CTree_to_order(ct,ct.root)
    for beta in betas
        if optimal_type == "normal"
            L = optimal_arg[1]
            fn_min_i,min_fn_order = mcmc!(ct,beta,L)
        elseif optimal_type == "greedy"
            tn,L,c,n_orders = optimal_arg
            fn_min_i, min_fn_order_i = mcmc_greedy!(ct,beta,tn,L,c,n_orders)
        end
        if fn_min_i < fn_min
            fn_min =fn_min_i
            min_fn_order = min_fn_order_i
        end
        
    end
    return fn_min,min_fn_order
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
        subgraph, nodes = get_subgraph(tn,nodeid)
        min_sc_order, min_tc_order,orders = greedy_orders(subgraph,"mindim","mindimtri",n_orders)
        subct = order_to_CTree(subgraph,min_tc_order[1])
        println(min_tc_order[3]*log10(2),' ',subct.complex[2]*log10(2))
        ΔH = target_fn(subct.complex[1],subct.complex[2])-target_fn(node.complex[1],node.complex[2])
        if rand()≤minimum([1,exp(-beta*ΔH)])
            replace_branch!(ct,node,subct)
        end
    end
end

function mcmc_greedy!(ct::Contraction_Tree,beta::Float64,tn::Graph,L::Int64,c::Int64,n_orders::Int64)
    scs = []
    tcs = []
    acc = []
    fn_min = target_fn(ct.complex[1],ct.complex[2])
    min_fn_order = CTree_to_order(ct,ct.root)
    for i=1:L
        LRD_list = []
        LRD_traversal!(ct,ct.nodes[ct.root],LRD_list)
        top_nodes = filter(x->get_depth(ct,ct.nodes[x])≤c,LRD_list)

        tot = length(top_nodes)
        counts = [local_update!(ct,ct.nodes[node],beta) for node in top_nodes] |> sum
        update_complex!(ct,ct.nodes[ct.root])

        if i % 50 ==0 # greedy seach for lower nodes. 
            LRD_list = []
            LRD_traversal!(ct,ct.nodes[ct.root],LRD_list)
            top_nodes = filter(x->get_depth(ct,ct.nodes[x])≤c,LRD_list)
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
            println(length(sub_nodes))
            greedy_update!(ct,tn,sub_nodes,beta)
        end

        push!(scs,ct.complex[1])
        push!(tcs,ct.complex[2])
        push!(acc,Float64(counts/tot))
        fn_new = target_fn(ct.complex[1],ct.complex[2])
        if fn_new < fn_min
            fn_min=fn_new
            min_fn_order = CTree_to_order(ct,ct.root)
        end
        #println("Accept rate = $(counts)/$(tot) ", Float64(counts/tot))
    end
    #plot(tcs.*log10(2))
    return fn_min,min_fn_order
end
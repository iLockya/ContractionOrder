using Random
include("tensor_network.jl")
include("utils.jl")

struct Loss_Struct{T}
    Es::Array{Array{Int64,1}}
    dict::Dict{T,Any} # weight of dictionary
    edge_weight::SparseMatrixCSC{Int64,Int64}
end

function compute_loss(mytn::Graph{Int64},loss_type::String,a)
    weights = [mytn.node_log2dim[k] for k in a]
    loss = weights[1] + weights[2] - 2*get_edge_log2dim(mytn,sort(a))
    if loss_type == "mindim" # minimize dimension of the obtained tensor
        return round(Int64,loss)
    elseif loss_type == "mindiminc" # minimize the increase of the obtained tensor
        return round(Int64,log2minusexp([loss, weights[1], weights[2]] ) )
    end
end


function compute_loss(mytn::OpenGraph,loss_type::String,a)
    i,j=a
    weights = [mytn.node_log2dim[i] for i in a]
    loss = mytn.node_log2dim[i] + mytn.node_outlog2dim[i] +mytn.node_log2dim[j] + mytn.node_outlog2dim[j] - 2*get_edge_log2dim(mytn,sort(a))
    if loss_type == "mindim" # minimize dimension of the obtained tensor
        return round(Int64,loss)
    elseif loss_type == "mindiminc" # minimize the increase of the obtained tensor
        return round(Int64,log2minusexp([loss, weights[1], weights[2]] ) )
    end
end

function gen_loss_struct_set(mytn::Union{Graph,OpenGraph},loss_type::String)
    Nodeset = keys(mytn.node_log2dim)
    Es = [ sort([i,j]) for i in Nodeset for j in mytn.neis[i] ] |> unique

    loss_dict = Dict()
    loss_dict_weight = sparse(Int64.(zeros(maximum(Nodeset),maximum(Nodeset)))) # Storing loss for each edge
    for a in Es
        loss = compute_loss(mytn,loss_type,a)
        if loss in keys(loss_dict)
            push!(loss_dict[loss], a)
        else
            loss_dict[loss] = Set([a])
        end
        loss_dict_weight[a[1],a[2]] = loss
        loss_dict_weight[a[2],a[1]] = loss
    end
    return Loss_Struct(Es, loss_dict, loss_dict_weight)
end

function n_edge_left(loss_dict::Loss_Struct)
    #return length(loss_dict.Es)
    n_edges = 0
    for loss in keys(loss_dict.dict)
        n_edges += length(loss_dict.dict[loss])
    end
    return n_edges
end

function select_edge_mindim(loss_dict::Loss_Struct)
    loss = minimum(keys(loss_dict.dict))
    edge = rand(loss_dict.dict[loss])
    return edge,loss
end

function select_edge_mindim_maxreduce(loss_dict::Loss_Struct,g::Graph)
    loss = minimum(keys(loss_dict.dict))
    edges = collect(loss_dict.dict[loss])
    edge_dim = [g.A[edge] for edge in edges]
    maxd = maximum(edge_dim)
    idx = findall(x->x==maxd,edge_dim)
    edge = rand(edges[idx])
    return edge,loss
end

function select_edge_mindim_maxreduce(loss_dict::Loss_Struct,g::OpenGraph)
    loss = minimum(keys(loss_dict.dict))
    edges = collect(loss_dict.dict[loss])
    edge_dim = [g.log2dim[edge] for edge in edges]
    maxd = maximum(edge_dim)
    idx = findall(x->x==maxd,edge_dim)
    edge = rand(edges[idx])
    return edge,loss
end

function select_edge_mindim_triangle(loss_dict::Loss_Struct,g::Union{Graph,OpenGraph})
    loss = minimum(keys(loss_dict.dict))
    edges = collect(loss_dict.dict[loss])
    edges_with_triangle = [ a for a in edges if length(intersect(g.neis[a[1]],g.neis[a[2]])) > 1 ]
    if isempty(edges_with_triangle)
        edge = rand(edges)
    else
        edge = rand(edges_with_triangle)
    end
    return edge,loss
end

function loss_struct_rm_edge_set!(loss_dict::Loss_Struct,edge)
    edge = edge[1] < edge[2] ? edge : reverse(edge)
    loss = loss_dict.edge_weight[edge[1],edge[2]]
    #println("rm from dict, ",edge)
    if loss > 0
#        println(typeof(loss_dict.dict[loss]))
        delete!(loss_dict.dict[loss],edge)
        if isempty(loss_dict.dict[loss])
            delete!(loss_dict.dict,loss)
        end
        loss_dict.edge_weight[edge[1],edge[2]] = 0
    end
    return 0
end

function loss_struct_before_contract_set!(loss_dict::Loss_Struct, mytn::Union{Graph,OpenGraph},edge::Array{Int64,1})
    i,j=edge
    edges = [ [sort([i,k]) for k in mytn.neis[i]];[sort([j,k]) for k in mytn.neis[j]] ]
    for edge in edges
        loss_struct_rm_edge_set!(loss_dict, edge)
    end
    return 0
end

function loss_struct_after_contract_set!(loss_dict::Loss_Struct, t::Union{Graph,OpenGraph},i::Int64,loss_type::String)
    if n_edge_left(loss_dict) == 0
        if length(t.neis[i]) == 1  # The last edge
#            println("The last edge: i=",i," j=",t.neis[i])
            j = collect(t.neis[i])[1]
            #loss = compute_loss(t,loss_type,[i,j]) # loss should be 0, because the contraction result is a scalar
            loss = 1 #*loss_factor # force loss to be 1
            loss_dict.dict[loss] = Set([sort([i,j])])
            loss_dict.edge_weight[i,j] = loss
            loss_dict.edge_weight[j,i] = loss
            return 0
        end
    end

    for j in t.neis[i]
        loss = compute_loss(t,loss_type,[i,j])
#        print(i ," ",j," loss= ",loss)
        if loss > 0
            #println("add to dict, ",i," ",j," ",loss)
            if loss in keys(loss_dict.dict)
                push!(loss_dict.dict[loss], sort([i,j]))
            else
                loss_dict.dict[loss] = Set([sort([i,j])])
            end
            loss_dict.edge_weight[i,j] = loss
            loss_dict.edge_weight[j,i] = loss
        end
    end
    return 0
end

function no_inner_edge(tn::Union{Graph,OpenGraph})
    for i in collect(values(tn.neis))
        if i != Set([])
            return false
        end
    end
    return true
end

function greedy_order(tn::Union{Graph,OpenGraph}, loss_type::String, select::String)
    time0 = time()
    to_contract = Set(keys(tn.node_log2dim))
    scs=[]
    tcs=[]
    order = []
    last_node = 0
    if ~no_inner_edge(tn)
        loss_dict = gen_loss_struct_set(tn,loss_type)
        m = n_edge_left(loss_dict)
        for n_edge = 1:m
            if n_edge_left(loss_dict) < 1
                break
            end
            if select == "mindim"
                edge,loss = select_edge_mindim(loss_dict)
            elseif select == "mindimtri"
                edge,loss = select_edge_mindim_triangle(loss_dict,tn)
            else
                println("Wrong select option")
            end
            #println("Loss=",loss/100|>Int64," edge ",edge," Edges: ",n_edge_left(loss_dict)," ",0.5*[ length(tn.neis[i]) for i = 1:tn.n] |> sum |> Int64," ",nnz(dropzeros!(tn.log2dim))/2|>Int64)
            loss_struct_before_contract_set!(loss_dict,tn,edge)
            #println("After Before Loss=",loss/100|>Int64," edge ",edge," Edges: ",n_edge_left(loss_dict)," ",0.5*[ length(tn.neis[i]) for i = 1:tn.n] |> sum |> Int64," ",nnz(dropzeros!(tn.log2dim))/2|>Int64)
            i,j,scij,tcij = contract_edge!(tn,edge)
            last_node = i
            delete!(to_contract,j)
            push!(order,[i,j])
            push!(scs,scij)
            push!(tcs,tcij)
            loss_struct_after_contract_set!(loss_dict,tn,i,loss_type)
            #println("After After Loss=",loss/100|>Int64," edge ",edge," Edges: ",n_edge_left(loss_dict)," ",0.5*[ length(tn.neis[i]) for i = 1:tn.n] |> sum |> Int64," ",nnz(dropzeros!(tn.log2dim))/2|>Int64)
        end
    end
    #println("there are some nodes left: ",to_contract)
    #println("last node ",tn.node_outlog2dim[last_node])
    #println("to_contract: ",to_contract)

    if last_node != 0
        push!(to_contract,last_node)
    end

    while length(to_contract) > 1
        vec = sort(collect(to_contract),by = x->tn.node_outlog2dim[x])
        i,j = vec[1],vec[2]
        push!(order,[i,j])
        delete!(to_contract,j)
        dimi = tn.node_outlog2dim[i]
        dimj = tn.node_outlog2dim[j]
        scij = maximum([dimj,dimi])
        tcij = dimj+dimi
        tn.node_outlog2dim[i] += dimj
        push!(scs,scij)
        push!(tcs,tcij)
    end

#    seq = []
#    while ~isempty(to_contract)
#        mindim = Inf
#        idx = 1
#        for i in to_contract
#            dim = tn.node_outlog2dim[i]
#            if  dim < mindim
#                mindim = dim
#                idx = i
#            end
#        end
#        delete!(to_contract,idx)
#        push!(seq,idx)
#    end
##    println("seq=",seq)
#    if last_node == 0
#        last_node = seq[1]
#        deleteat!(seq,1)
#    end
#    for i in seq
#        push!(order,[last_node,i])
#        dimi = tn.node_outlog2dim[i]
#        dimj = tn.node_outlog2dim[last_node]
#        scij = maximum([dimj,dimi])
#        tcij = dimj+dimi
#        tn.node_outlog2dim[last_node] += dimi
#        push!(scs,scij)
#        push!(tcs,tcij)
#    end
    sc,tc = maximum(scs),log2sumexp(tcs)
    return order,sc,tc
end

function greedy_orders(tn::Union{Graph,OpenGraph}, loss_type::String, select::String, n_orders)
    orders = [greedy_order(deepcopy(tn),loss_type,select) for i=1:n_orders]
    scs = [order[2] for order in orders]
    tcs = [order[3] for order in orders]
    min_sc_order = orders[argmin(scs),:][1]
    min_tc_order = orders[argmin(tcs),:][1]
    return min_sc_order,min_tc_order,orders
end

function optimal_order_2_greedy_init(tn,greedy_iters=100)
    if tn.n == 1
        return -Inf,[]
    end
    loss_type = "mindim"
    select = "mindim"
    min_sc_order,min_tc_order,orders = greedy_orders(tn,loss_type,select,greedy_iters)
    bound = min_tc_order[3]*1.01
    tc,myorder = optimal_order_top_down_cache(tn,[],bound)
    #println("greedy_tc=",min_tc_order[3]," tc=",tc)
    return tc,myorder
end

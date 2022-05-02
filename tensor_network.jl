using SparseArrays
using DelimitedFiles
using Random
include("utils.jl")
include("contraction_tree.jl")

mutable struct Graph{T,S}
    n::T # number of variables
    edges::Array{Array{T,1},1} # edge [i,j], i<j
    neis::Dict # i=>neis[i]
    edge_log2dim::Array{S,1} # weight for each edge
    node_log2dim::Dict{T,S} # log dimension of each 
end

mutable struct OpenGraph{T,S}
    n::T # number of variables
    edges::Array{Array{T,1},1}
    neis::Dict
    edge_log2dim::Array{S,1}
    node_log2dim::Dict{T,S}
    node_outlog2dim::Dict{T,S}
end



@inline function get_edge_log2dim(t::Union{Graph,OpenGraph}, edge)
    #for i = eachindex(t.edges)
    #    if t.edges[i] == edge
    #        return t.edge_log2dim[i]
    #    end
    #end
    #error("edge ",edge," does NOT in the Graph!")
    #return 0
    return t.edge_log2dim[indexin([edge],t.edges)[1]]
end

@inline function edge_node_log2dim(t::OpenGraph,a)
    """

    Only for OpenGraph. 
    Output: log2dim + outlog2dim.
    """
    if a in keys(node_outlog2dim)
        return t.node_log2dim[a] + t.node_outlog2dim[a]
    else
        return t.node_log2dim[a]
    end
end



function add_edge!(t::Union{Graph,OpenGraph},edge::Array{Int64,1},log2weight::Float64)
    sort!(edge)
    i,j = edge
    if edge in t.edges
        t.edge_log2dim[indexin([edge],t.edges)[1]] += log2weight
    else
        push!(t.edges,edge)
        push!(t.edge_log2dim,log2weight)
        push!(t.neis[i],j)
        push!(t.neis[j],i)
    end
    t.node_log2dim[i] += log2weight
    t.node_log2dim[j] += log2weight
    return 0
end


function rm_edge!(t::Union{Graph,OpenGraph},edge::Array{Int64,1})
    m = indexin([edge],t.edges)[1]
    i,j = edge
    log2dim = t.edge_log2dim[m]
    deleteat!(t.edges,m)
    deleteat!(t.edge_log2dim,m)
    delete!(t.neis[i],j)
    delete!(t.neis[j],i)
    t.node_log2dim[i] -= log2dim
    t.node_log2dim[j] -= log2dim
    return 0
end


function graph_read_from_edges(fname::String)
    E = readdlm(fname, ' ', Int, '\n')
    A = sparse( [E[:,1];E[:,2]],[E[:,2];E[:,1]],[E[:,3];E[:,3]])
    neis = Dict()
    n = size(A,1)
    edge_n = size(E,1)
    edges = [sort!([E[i,1],E[i,2]]) for i=1:edge_n]
    edge_log2dim = log2.(E[:,3])
    for i=1:n
        js = findnz(A[:,i])[1]
        neis[i] = Set(js)
    end
    node_log2dim = Dict([i for i=1:n].=>[ log2.( [A[i,k] for k in neis[i]] ) |> sum for i=1:n])
    tn = Graph(n,edges,neis,edge_log2dim,node_log2dim)
    return tn
end


function contract_edge!(t::Graph{Int64,Float64},edge::Array{Int64,1})
    i,j = edge
    sc = max(t.node_log2dim[i],t.node_log2dim[j])
    tc = t.node_log2dim[i] + t.node_log2dim[j]
    if edge in t.edges
        tc -= get_edge_log2dim(t,edge)
    end
    #println("contract edge:", edge)
    """
    if edge not in t.edges, outer product
    """
    [add_edge!(t,sort!([i,k]),get_edge_log2dim(t,sort!([j,k]))) for k in t.neis[j] if k!=i]
    edges_to_delete = copy(t.neis[j])
    [rm_edge!(t,sort!([j,k])) for k in edges_to_delete]
    return i,j,max(sc,t.node_log2dim[i]),tc
end

function contract_edge!(t::OpenGraph{Int64,Float64 },edge::Array{Int64,1})
    i,j = edge
    sc = max(t.node_log2dim[i]+t.node_outlog2dim[i],t.node_log2dim[j]+t.node_outlog2dim[j])
    tc = t.node_log2dim[i]+t.node_outlog2dim[i]+t.node_log2dim[j]+t.node_outlog2dim[j]
    if edge in t.edges
        tc -= get_edge_log2dim(t,edge)
    end
    [add_edge!(t,sort!([i,k]),get_edge_log2dim(t,sort!([j,k]))) for k in t.neis[j] if k!=i]
    t.node_outlog2dim[i] += t.node_outlog2dim[j]
    edges_to_delete = copy(t.neis[j])
    [rm_edge!(t,sort!([j,k])) for k in edges_to_delete]
    return i,j,max(sc,t.node_log2dim[i]+t.node_outlog2dim[i]),tc
end

function order_to_CTree(tn::Union{Graph,OpenGraph},order)
    gid_to_id = Dict()
    tn = deepcopy(tn)
    for i in collect(keys(tn.node_log2dim))
        gid_to_id[i] = Set([i])
    end
    ct = init_CTree(tn.n,deepcopy(tn.neis),Dict(tn.edges .=> tn.edge_log2dim))
    for (a,edge) in enumerate(order)
        i,j,scij,tcij = contract_edge!(tn,edge)
        id = CTree_contract!(ct,map(x -> gid_to_id[x],edge),edge,scij,tcij)
        gid_to_id[edge[1]] = id
        gid_to_id[edge[2]] = id
        if isnan(tcij) || isinf(tcij)
            println("order_complexity nan: i=",i," j=",j)
        end
    end
    update_complex!(ct,ct.nodes[ct.root])
    return ct
end

function greedy_Boltzmann(t::Graph,α,τ)
    tn = deepcopy(t)
    order = []
    while !(isempty(tn.edges))
        cost = []
        for edge in tn.edges
            i,j = edge
            costij = exp(tn.node_log2dim[i]+tn.node_log2dim[j]-2*get_edge_log2dim(tn,edge))-
                       α*(exp(tn.node_log2dim[i])+exp(tn.node_log2dim[j]))
            push!(cost,costij)
        end
        e = sample(tn.edges,Weights(exp.(-cost./τ)))
        push!(order,e)
        contract_edge!(tn,e)
    end
    return modify_order!(order)
end

function greedy_local(t::Graph)
    tn = deepcopy(t)
    order = []
    while !(isempty(tn.edges))
        tcm = Inf
        e = tn.edges[1]
        for j in tn.neis[1]
            tc = tn.node_log2dim[1] + tn.node_log2dim[j] - get_edge_log2dim(tn,[1,j])
            if tc<tcm
                tcm = tc
                e=[1,j] 
            end
        end
        push!(order,e)
        contract_edge!(tn,e)
    end
    return modify_order!(order)
end


function get_subgraph(g::Graph,V)::OpenGraph
    nodes = sort(collect(V))
    n = length(V)
    edges = Array{Vector{Int64},1}([])
    node_outlog2dim = Dict{Int64,Float64}( [(k,0.0) for k in V] )
    for e in g.edges
        i,j=e
        if i ∈ V && j ∈ V 
            push!(edges,e)
        elseif i ∈ V && j ∉ V
            node_outlog2dim[i] += get_edge_log2dim(g,e)
        elseif j ∈ V && i ∉ V
            node_outlog2dim[j] += get_edge_log2dim(g,e)
        end
    end
    #neis = Array{Set{Int64},1}([intersect(V,g.neis[i]) for i in nodes])
    neis = Dict( [(k,intersect(g.neis[k],V)) for k in V])
    edge_log2dim = g.edge_log2dim[indexin(edges,g.edges)]
    node_log2dim = Dict( [(k,g.node_log2dim[k]) for k in V] )
    #return Graph(n,edges,neis,edge_log2dim,node_log2dim),nodes
    return OpenGraph(n,edges,neis,edge_log2dim,node_log2dim,node_outlog2dim)
end

function quickbb(g::Graph,V)
    subgraph,nodes = get_subgraph(g,V)
    cnf = ""
    m = 0
    for v in nodes
        if length(subgraph.neis[v]) ≥ 2
            clique = [sort([v,j]) for j in subgraph.neis[v]]
            [cnf*="$i " for i in indexin(clique,subgraph.edges)]
            cnf *= "0\n"
            m += 1
        end
    end
    cnf = "p cnf $(length(subgraph.edges)) $(m)\n" * cnf
    open("g.cnf","w") do f
        write(f,cnf)
    end
    run(`./quickbb_64 --min-fill-ordering --outfile g.out --cnffile g.cnf`)
    s = split(readlines("g.out")[5],' ')
    e_id = [parse(Int,x) for x in s if x!=""]
    rm("g.out")
    println(e_id)
    order = modify_order!(subgraph.edges[e_id])
    println(order)
    subct = order_to_CTree(subgraph,order)
    return subct
end

function quickbb_update!(ct::Contraction_Tree,tn::Graph,node::CTree_Node)
    subct = quickbb(tn,node.id)
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





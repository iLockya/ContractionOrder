include("utils.jl")
include("tensor_network.jl")
include("contraction_tree.jl")

mutable struct QuantumCircuit
    G::Graph # a simplified graph.
    final_qubits::Dict{Int,Vector{Int}} # final qubits for end node. e.g 200=>[1,2]...
    sequences::Array{String,1}
end





@inline function batch_log2dim(id,final_qubits::Dict{Int64,Vector{Int64}},sequence)
    """
    Given some open qubits, find batch dimension according to the sequence.
    return log2(batch dimension).
    id: node id.
    """
    qubits = intersect(id,keys(final_qubits)) # qubits that are in final qubits.
    if isempty(qubits) # no contribution to sc.
        return 0
    else
        qubits = sort(collect(qubits))
        batch = Set( [s[open_qubits] for s in sequence] )
        return log2( length(batch))
    end
end

function generate_random_sequences(n::Int64,m::Int64)
    """
    Generate random sequences of length n with 0/1, each of length m.
    return m strings of length n.
    """
    return [String(rand(['0','1'],n)) for i ∈ 1:m]
end



function graph_with_sparse_state(fname::String,n_qubits::Int64,sequences)
    final_qubits = Dict()
    tn = graph_read_from_edges(fname) # this is a quantum circuit with single bitstring.
    for qid ∈ 1:n_qubits # 53 qubits in total.
        i = tn.n-n_qubits+qid
        @assert length(tn.neis[i]) == 1
        j = collect(tn.neis[i])[1]
        # add j to the final_qubits.
        if j in keys(final_qubits)
            push!(final_qubits[j],qid)
        else
            final_qubits[j] = [qid]
        end
        rm_edge!(tn, [j,i]) # delete edge [j,i]
        # delete node i.
        delete!(tn.neis,i)
        delete!(tn.node_log2dim,i)
    end
    tn.n -= n_qubits
    qc = QuantumCircuit(tn,final_qubits,sequences)
    return qc
end

function contraction_edge!(qc::QuantumCircuit,edge)
    """
    contracion with sparse states.
    """
    i,j = edge # i<j
    @assert i<j
    out_qubits = keys(qc.final_qubits)
    if i ∉ out_qubits && j ∉ out_qubits #if no batch dim, contration normally.
        i,j,sc,tc = contract_edge!(qc.G,edge)
        return i,j,sc,tc
    end
    # if i or j is in out_qubits.
    # 1) calculate batch dim.
    # 2) meige out_qubits to i.
    if i ∈ out_qubits && j ∉ out_qubits # absorb j to i.
        batch_dim_i = batch_log2dim(qc.final_qubits[i],qc.final_qubits,qc.sequences)
        sc = max(qc.G.node_log2dim[i]+batch_dim_i,qc.G.node_log2dim[j])
        tc = qc.G.node_log2dim[i]+batch_dim_i+qc.G.node_log2dim[j]
    elseif i ∉ out_qubits && j ∈ out_qubits
        batch_dim_i = batch_log2dim(qc.final_qubits[j],qc.final_qubits,qc.sequences)
        sc = max(qc.G.node_log2dim[i],qc.G.node_log2dim[j]+batch_dim_i)
        tc = qc.G.node_log2dim[i]+qc.G.node_log2dim[j]+batch_dim_i
        qc.final_qubits[i] = qc.final_qubits[j]
        delete!(qc.final_qubits,j)
    else
        batch_dim_i = batch_log2dim(qc.final_qubits[i],qc.final_qubits,qc.sequences)
        batch_dim_j = batch_log2dim(qc.final_qubits[j],qc.final_qubits,qc.sequences)
        sc = max(qc.G.node_log2dim[i]+batch_dim_i,qc.G.node_log2dim[j]+batch_dim_j)
        tc = qc.G.node_log2dim[i]+batch_dim_i+qc.G.node_log2dim[j]+batch_dim_j
        qc.final_qubits[i] = qc.final_qubits[i] ∪ qc.final_qubits[j]
        delete!(qc.final_qubits,j)
        batch_dim_i = batch_log2dim(qc.final_qubits[i],qc.final_qubits,qc.sequences)
    end

    if edge in qc.G.edges
        tc -= get_edge_log2dim(qc.G,edge)
    end
    [add_edge!(qc.G,sort!([i,j]),get_edge_log2dim(qc.G,sort!([i,k])) for k in qc.G.neis[j] if k!=i]
    edge_to_delete = copy(qc.G.neis[j])
    [rm_edge!(qc.G,sort!([j,k])) for k in edges_to_delete]
    return i,j,max(sc,qc.G.node_log2dim[i]+batch_dim_i),tc
end

function init_CTree(qc::QuantumCircuit)
    """
    Initialize CTree.
    """
    root = Set(collect(1:n))
    ct = Contraction_Tree(Dict(),qc.G.neis,qc)
end


using Distributed
include("tensor_network.jl")
include("contraction_tree.jl")
include("simulated_annealing.jl")
include("utils.jl")
include("greedy.jl")


@distributed for i = 1:12
    println(i)
    include("tensor_network.jl")
include("contraction_tree.jl")
include("simulated_annealing.jl")
include("utils.jl")
include("greedy.jl")
    βs = [log(1+2n) for n=1:200]
    L = 20
    c = 25
    n_orders = 10
    optimal_type = "greedy"
    optimal_arg = [L,c,n_orders]
    sc_target = 28.322 # sc_target = 31 for V100
    nslice = 0
    tn = graph_read_from_edges("test/Sycamore_53_16.txt")
    min_sc_order, min_tc_order,orders = greedy_orders(tn,"mindim","mindimtri",100)
    ct = order_to_CTree(tn,min_tc_order[1])
    fn_min,min_fn_order,min_fn_slice = simulated_annealing!(ct,tn,βs,optimal_type;optimal_arg,sc_target,nslice)
    tc = 10^fn_min
    n = length(ct.slice)
    println("tc = $tc, n = $n")
end


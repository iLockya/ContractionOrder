module Tst
    using Random
    include("tensor_network.jl")
    include("contraction_tree.jl")
    include("greedy.jl")
    include("simulated_annealing.jl")

    # --------------Test Cod--------------------
    tn = graph_read_from_edges("test/Sycamore_53_14.txt")
    min_sc_order, min_tc_order,orders = greedy_orders(tn,"mindim","mindimtri",100)
    #order = greedy_Boltzmann(tn,0.2,0.1)
    ct = order_to_CTree(tn,order)
    println(ct.complex.*log10(2))
    println(get_height(ct,ct.nodes[ct.root]))
    betas = [2.0,20.0,200.0,2000.0]
    simulated_annealing!(ct,betas,100000)
    update_complex!(ct,ct.nodes[ct.root])
    println(ct.complex.*log10(2))
    println(get_height(ct,ct.nodes[ct.root]))
    #println(ct.nodes[ct.root].complex.*log10(2))
end
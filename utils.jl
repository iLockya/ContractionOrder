function log2sumexp(a)
    offset = maximum(a)
    return log2(sum(exp.(log(2.0).*(a .- offset) )) ) + offset
end

function modify_order!(order::Array)
    sort!.(order)
    l = length(order)
    for i = 1:l-1
        for j = i+1:l
            sort!∘replace!(order[j],order[i][2]=>order[i][1])
        end
    end
    del_list = []
    for i = 1:l
        if order[i][1] == order[i][2]
            push!(del_list,i)
        end
    end
    deleteat!(order,del_list)
    return order
end

function target_fn(sc::Float64,tc::Float64)
    return tc
end

if abspath(PROGRAM_FILE) == @__FILE__
    order = [[1, 2],[3, 5],[4, 3],[5,2],[1, 4]]
    println(modify_order(order))
end

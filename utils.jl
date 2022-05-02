@inline function log2sumexp(a)
    offset = maximum(a)
    return log2(sum(exp.(log(2.0).*(a .- offset) )) ) + offset
end

@inline function fast_log2sumexp2(a, b)
    mm, ms = minmax(a, b)
    return log2(exp2(mm - ms) + 1) + ms
end

@inline function fast_log2sumexp2(a, b, c)
    if a > b
        if a > c
            m1, m2, ms = b, c, a
        else
            m1, m2, ms = a, b, c
        end
    else
        if b > c
            m1, m2, ms = c, a, b
        else
            m1, m2, ms = b, a, c
        end
    end
    return Base.FastMath.log2(Base.FastMath.exp2(m1 - ms) + Base.FastMath.exp2(m2 - ms) + 1) + ms
end

@inline function fast_log2minusexp2(a, b, c)
    """
    return log2(exp2(a) - exp2(b) - exp2(c))
    """
    return Base.FastMath.log2(1 - Base.FastMath.exp2(b - a) - Base.FastMath.exp2(c - a)) + a
end

@inline function tree_imbalance(a,b,n;λ=0.0)
    mm, ms = minmax(a,b)
    return λ*mm / ((abs(ms-1.5))*n)
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


## Testing functions with Julia 
using Plots
using LaTeXStrings

# Printing
println("Hello World")


# For loop
s = 0; 
for i=1:100
 global s = s + i
end 
println("Result of the sum: ", s)

# Arrays 
arr = zeros((3,3))
arr[1,2] = 1.8
for i=1:3, j=1:3
 if (i == 1 && j == 2) 
  continue
 end
 arr[i,j] = 2.0*i+5.4*j
end 
println(arr)
display(arr)
A = [1,2,3]
B = [4,5,6]
println(A.*B) #Elementwise multiplication 

# Define functions
# compact definition
himmelblau(x,y) = (x^2+y-11.0)^2 + (x+y^2-7.0)^2
xh, yh = -2.805118, 3.131312
println("result himmelblau: ", himmelblau(xh,yh)) 
# more sophisticated
function rosenbrock(x)
 l = length(x)
 f = 0.0
 for i=1:(l-1)
  f += 100*(x[i+1]-x[i])^2+(1.0-x[i])^2
 end 
 return f
end 

println("result rosenbrock: ", rosenbrock([1,2,3]))

# Plotting 
gr()
x = range(0, 10, length=100)
y1 = @. exp(-0.1*x) * cos(4*x)
y2 = @. exp(-0.3*x) * cos(4*x)
y3 = @. exp(-0.1*x)
y4 = @. exp(-0.3*x)
y = [y1 y2 y3 y4]

p1 = plot(x, y)
p2 = plot(x, y, title=L"Title 2", lw=3)
p3 = scatter(x, y, ms=2, ma=0.5, xlabel=L"xlabel 3")
p4 = scatter(x, y, title=L"Title 4", ms=2, ma=0.2)
p = plot(p1, p2, p3, p4, layout=(2,2), legend=:bottomleft)
savefig(p, "myplot.pdf")

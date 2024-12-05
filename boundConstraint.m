function vi = boundConstraint(vi, pop, lu)
[NP, D] = size(vi);

Xmax = 100;
Xmin = -100;
for i=1:NP
    for j=1:D
        while (vi(i,j) > Xmax || vi(i,j) < Xmin)
            vi(i,j) = (Xmax - Xmin).*rand + Xmin;
        end
    end
end

end

